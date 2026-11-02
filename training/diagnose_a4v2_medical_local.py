from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.aptos_dataset import APTOS_CLASS_ORDER
from models.a3_diagnostic_discriminator import A3DiagnosticDiscriminator
from training.run_aptos_convnext_baselines import build_model, build_transform, seed_everything


MINORITY = APTOS_CLASS_ORDER[1:]
DEFAULT_TEACHERS = {
    "original_ce": "outputs/aptos2019/convnext_tiny_scratch_preliminary/original_ce/best_model.pt",
    "real_seed2027": "outputs/aptos2019/a4_from_a3_adversarial_candidate/source_holdout_d4_v4/feature_teachers/seed_2027/best_model.pt",
    "real_seed2028": "outputs/aptos2019/a4_from_a3_adversarial_candidate/source_holdout_d4_v4/feature_teachers/seed_2028/best_model.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A4-v2 train-only lesion-aware real-real calibrated diagnostic")
    parser.add_argument(
        "--a3-root", default="outputs/aptos2019/a3_adversarial_candidate/source_holdout_d4_v4/balanced_train"
    )
    parser.add_argument(
        "--prior-index", default="outputs/aptos2019/structure_guidance_v3/weak_localization_priors/prior_index.csv"
    )
    parser.add_argument(
        "--output-dir", default="outputs/aptos2019/a4_from_a3_adversarial_candidate/source_holdout_d4_v4/medical_local_diagnostic"
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--reference-fraction", type=float, default=0.70)
    parser.add_argument("--bootstrap-seeds", default=",".join(str(x) for x in range(2026, 2041)))
    parser.add_argument("--deficit-threshold", type=float, default=0.03)
    parser.add_argument("--same-sign-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.clip(left @ right.T, -1.0, 1.0))).astype(np.float32)


def radii(x: np.ndarray, k: int) -> np.ndarray:
    d = distances(x, x)
    np.fill_diagonal(d, np.inf)
    return np.partition(d, kth=k - 1, axis=1)[:, k - 1]


class FeatureDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int):
        self.frame = frame.reset_index(drop=True)
        self.transform = build_transform(image_size, traditional=False)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = self.transform(Image.open(str(row.image_path)).convert("RGB"))
        prior = Image.open(str(row.prior_path)).convert("L").resize((14, 14), Image.Resampling.BILINEAR)
        prior_array = np.asarray(prior, dtype=np.float32) / 255.0
        return image, torch.from_numpy(prior_array).unsqueeze(0), int(row.label), int(index)


def load_teacher(name: str, path: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu")
    state = payload["model_state"]
    if not any(key.startswith("backbone.") for key in state):
        model = build_model(device)
        model.load_state_dict(state, strict=True)
    else:
        model = A3DiagnosticDiscriminator().to(device)
        model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def encode(model: torch.nn.Module, images: torch.Tensor, priors: torch.Tensor, original: bool):
    if original:
        x = images
        mid = None
        for index, layer in enumerate(model.features):
            x = layer(x)
            if index == 5:
                mid = x
        pooled = model.avgpool(x)
        final = torch.flatten(model.classifier[0](pooled), 1)
        logits = model.classifier[2](final)
    else:
        mid, final, logits = model.encode(images)
    if mid is None:
        raise RuntimeError("ConvNeXt features[5] unavailable")
    mask = F.interpolate(priors, size=mid.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
    inv = 1.0 - mask
    lesion = (mid * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1e-6)
    context = (mid * inv).sum((2, 3)) / inv.sum((2, 3)).clamp_min(1e-6)
    local = torch.cat(
        [F.normalize(final.float(), dim=1), F.normalize(lesion.float(), dim=1),
         F.normalize(context.float(), dim=1), F.normalize((lesion - context).float(), dim=1)], dim=1
    )
    return F.normalize(final.float(), dim=1), F.normalize(local, dim=1), logits.float()


@torch.no_grad()
def extract(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    cache: Path,
    teacher_name: str,
) -> Dict[str, np.ndarray]:
    expected = frame.image_id.astype(str).tolist()
    if cache.is_file():
        saved = np.load(cache, allow_pickle=False)
        if saved["image_ids"].astype(str).tolist() == expected:
            print(f"[{teacher_name}] reuse {cache.name}", flush=True)
            return {key: saved[key] for key in saved.files}
    loader = DataLoader(
        FeatureDataset(frame, args.image_size), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    global_features: List[np.ndarray] = []
    local_features: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    seen = 0
    for images, priors, _labels, indices in loader:
        images = images.to(device, non_blocking=True)
        priors = priors.to(device, non_blocking=True)
        with autocast(enabled=True):
            global_feature, local_feature, logits = encode(
                model, images, priors, original=not isinstance(model, A3DiagnosticDiscriminator)
            )
        global_features.append(global_feature.cpu().numpy().astype(np.float32))
        local_features.append(local_feature.cpu().numpy().astype(np.float32))
        probabilities.append(torch.softmax(logits, 1).cpu().numpy().astype(np.float32))
        seen += len(indices)
        if seen == len(frame) or seen % (args.batch_size * 20) == 0:
            print(f"[{teacher_name}] {cache.stem}: {seen}/{len(frame)}", flush=True)
    result = {
        "global_features": np.concatenate(global_features),
        "local_features": np.concatenate(local_features),
        "probabilities": np.concatenate(probabilities),
        "image_ids": frame.image_id.astype(str).to_numpy(),
        "source_ids": frame.source_image_id.astype(str).to_numpy(),
        "labels": frame.label.to_numpy(dtype=np.int64),
        "label_names": frame.label_name.astype(str).to_numpy(),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **result)
    return result


def source_excluded_coverage(
    real: np.ndarray, synthetic: np.ndarray, real_ids: np.ndarray, source_ids: np.ndarray, k: int
) -> Tuple[float, np.ndarray, np.ndarray]:
    radius = radii(real, k)
    d = distances(real, synthetic)
    d[real_ids[:, None] == source_ids[None, :]] = np.inf
    nearest = d.min(1)
    ratio = nearest / np.maximum(radius, 1e-8)
    return float(np.mean(ratio <= 1.0)), ratio, ratio > 1.0


def calibrated_bootstraps(
    real: np.ndarray,
    synthetic: np.ndarray,
    real_ids: np.ndarray,
    source_ids: np.ndarray,
    k: int,
    reference_fraction: float,
    seeds: List[int],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        ordering = rng.permutation(len(real))
        n_reference = max(k + 2, int(round(reference_fraction * len(real))))
        reference_index = ordering[:n_reference]
        query_index = ordering[n_reference:]
        reference = real[reference_index]
        radius = radii(reference, k)
        real_query = real[query_index]
        real_coverage = float(np.mean(distances(reference, real_query).min(1) <= radius))

        query_ids = set(real_ids[query_index].tolist())
        eligible = np.flatnonzero(np.asarray([value in query_ids for value in source_ids], dtype=bool))
        chosen: List[int] = []
        for image_id in real_ids[query_index]:
            candidates = eligible[source_ids[eligible] == image_id]
            if len(candidates):
                chosen.append(int(rng.choice(candidates)))
        if len(chosen) < len(query_index):
            remainder = np.setdiff1d(eligible, np.asarray(chosen, dtype=np.int64), assume_unique=False)
            need = min(len(query_index) - len(chosen), len(remainder))
            if need:
                chosen.extend(rng.choice(remainder, size=need, replace=False).tolist())
        synthetic_query = synthetic[np.asarray(chosen[: len(query_index)], dtype=np.int64)]
        synthetic_coverage = float(np.mean(distances(reference, synthetic_query).min(1) <= radius))
        rows.append(
            {
                "seed": seed,
                "reference_count": len(reference_index),
                "query_count": len(query_index),
                "real_real_coverage": real_coverage,
                "synthetic_real_coverage_size_matched": synthetic_coverage,
                "coverage_deficit": real_coverage - synthetic_coverage,
            }
        )
    return rows


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A4-v2 diagnostic requires CUDA")
    seed_everything(args.seed)
    device = torch.device("cuda")
    out = (ROOT / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    a3_root = (ROOT / args.a3_root).resolve()
    train = pd.read_csv(a3_root / "splits/train.csv")
    train = train[train.label_name.astype(str).isin(MINORITY)].copy().reset_index(drop=True)
    prior = pd.read_csv(ROOT / args.prior_index)[["image_id", "prior_path"]]
    train = train.merge(prior, on="image_id", how="left")
    train["source_image_id"] = train.image_id.astype(str)
    synthetic = pd.read_csv(a3_root / "generated_manifest.csv").copy().reset_index(drop=True)
    if "image_path" not in synthetic:
        synthetic["image_path"] = synthetic.output_path
    if train.prior_path.isna().any() or len(train) != 1300 or len(synthetic) != 3752:
        raise RuntimeError("Unexpected train/prior/A3 manifest protocol")
    bootstrap_seeds = [int(value) for value in args.bootstrap_seeds.split(",") if value.strip()]
    started = time.time()
    all_data: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = {}
    for teacher_index, (name, relative_path) in enumerate(DEFAULT_TEACHERS.items(), 1):
        write_json(out / "live_progress.json", {
            "status": "extracting", "teacher": name, "teacher_index": teacher_index,
            "teachers_total": len(DEFAULT_TEACHERS), "test_accessed": False,
        })
        model = load_teacher(name, ROOT / relative_path, device)
        real_data = extract(model, train, args, device, out / "cache" / f"{name}_real.npz", name)
        synthetic_data = extract(model, synthetic, args, device, out / "cache" / f"{name}_a3.npz", name)
        all_data[name] = (real_data, synthetic_data)
        del model
        torch.cuda.empty_cache()

    result_rows: List[Dict[str, object]] = []
    bootstrap_rows: List[Dict[str, object]] = []
    uncovered: Dict[Tuple[str, str], np.ndarray] = {}
    for teacher_name, (real_data, synthetic_data) in all_data.items():
        for feature_name in ["global_features", "local_features"]:
            for class_name in MINORITY:
                rmask = real_data["label_names"].astype(str) == class_name
                smask = synthetic_data["label_names"].astype(str) == class_name
                real_feature = normalize(real_data[feature_name][rmask])
                synthetic_feature = normalize(synthetic_data[feature_name][smask])
                real_ids = real_data["image_ids"][rmask].astype(str)
                source_ids = synthetic_data["source_ids"][smask].astype(str)
                coverage, ratio, missing = source_excluded_coverage(
                    real_feature, synthetic_feature, real_ids, source_ids, args.knn_k
                )
                rows = calibrated_bootstraps(
                    real_feature, synthetic_feature, real_ids, source_ids, args.knn_k,
                    args.reference_fraction, bootstrap_seeds,
                )
                deficits = np.asarray([row["coverage_deficit"] for row in rows])
                supported = bool(
                    np.median(deficits) >= args.deficit_threshold
                    and np.mean(deficits > 0) >= args.same_sign_fraction
                )
                result_rows.append({
                    "teacher": teacher_name,
                    "feature": feature_name,
                    "class_name": class_name,
                    "real_count": int(rmask.sum()),
                    "synthetic_count": int(smask.sum()),
                    "source_excluded_full_coverage": coverage,
                    "nearest_radius_ratio_mean": float(ratio.mean()),
                    "nearest_radius_ratio_median": float(np.median(ratio)),
                    "uncovered_rate": float(missing.mean()),
                    "bootstrap_deficit_mean": float(deficits.mean()),
                    "bootstrap_deficit_median": float(np.median(deficits)),
                    "bootstrap_deficit_ci025": float(np.quantile(deficits, 0.025)),
                    "bootstrap_deficit_ci975": float(np.quantile(deficits, 0.975)),
                    "positive_deficit_fraction": float(np.mean(deficits > 0)),
                    "teacher_class_support": supported,
                })
                for row in rows:
                    bootstrap_rows.append({"teacher": teacher_name, "feature": feature_name, "class_name": class_name, **row})
                if feature_name == "local_features":
                    uncovered[(teacher_name, class_name)] = missing
                print(
                    f"{teacher_name} {feature_name} {class_name}: full={coverage:.3f} "
                    f"deficit_med={np.median(deficits):+.3f} sign={np.mean(deficits > 0):.2f} support={supported}",
                    flush=True,
                )

    results = pd.DataFrame(result_rows)
    results.to_csv(out / "representation_results.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(out / "real_real_bootstraps.csv", index=False)
    class_decisions: Dict[str, object] = {}
    for class_name in MINORITY:
        local = results[(results.feature == "local_features") & (results.class_name == class_name)]
        support_count = int(local.teacher_class_support.sum())
        pairwise = []
        names = list(DEFAULT_TEACHERS)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairwise.append(jaccard(uncovered[(names[i], class_name)], uncovered[(names[j], class_name)]))
        class_decisions[class_name] = {
            "supporting_teachers": support_count,
            "supported": support_count >= 2,
            "uncovered_jaccard_mean": float(np.mean(pairwise)),
            "uncovered_jaccard_min": float(np.min(pairwise)),
        }
    supported_classes = [name for name, value in class_decisions.items() if value["supported"]]
    summary = {
        "status": "completed",
        "method": "three train-only medical teachers + Grad-CAM weighted local features + real-real calibration",
        "teachers": DEFAULT_TEACHERS,
        "test_accessed": False,
        "source_derived_synthetic_excluded_from_own_prototype": True,
        "bootstrap_seeds": bootstrap_seeds,
        "preregistered_thresholds": {
            "median_deficit": args.deficit_threshold,
            "positive_deficit_fraction": args.same_sign_fraction,
            "supporting_teachers_per_class": 2,
        },
        "class_decisions": class_decisions,
        "supported_classes": supported_classes,
        "controller_authorized_by_diagnostic": bool(supported_classes),
        "elapsed_minutes": (time.time() - started) / 60,
    }
    write_json(out / "diagnostic_summary.json", summary)
    write_json(out / "live_progress.json", summary)
    write_json(out / "run_config.json", {**vars(args), "teachers": DEFAULT_TEACHERS, "test_accessed": False})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
