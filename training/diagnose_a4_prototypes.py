from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.aptos_dataset import APTOS_CLASS_ORDER
from training.run_aptos_convnext_baselines import build_model, build_transform, seed_everything


MINORITY = APTOS_CLASS_ORDER[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A4 train-only intra-class prototype diagnostic")
    parser.add_argument(
        "--a3-root", default="outputs/aptos2019/a3_adversarial_candidate/source_holdout_d4_v4/balanced_train"
    )
    parser.add_argument(
        "--feature-checkpoint",
        default="outputs/aptos2019/convnext_tiny_scratch_preliminary/original_ce/best_model.pt",
    )
    parser.add_argument(
        "--output-dir", default="outputs/aptos2019/reproductions/a4_prototype_diagnostic"
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--feature-layer",
        choices=["final", "mid_stats", "final_mid_stats"],
        default="final",
        help="Train-only medical feature representation used for the sensitivity analysis.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--cluster-seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--min-cluster-size", type=int, default=8)
    parser.add_argument("--min-cluster-fraction", type=float, default=0.05)
    parser.add_argument("--min-stability-ari", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class FeatureDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int):
        self.frame = frame.reset_index(drop=True)
        self.transform = build_transform(image_size, traditional=False)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = self.transform(Image.open(str(row.image_path)).convert("RGB"))
        return image, int(row.label), str(row.image_id), str(row.label_name)


def encode(model: torch.nn.Module, images: torch.Tensor, feature_layer: str) -> Tuple[torch.Tensor, torch.Tensor]:
    x = images
    mid = None
    for index, layer in enumerate(model.features):
        x = layer(x)
        if index == 5:
            mid = x
    x = model.avgpool(x)
    x = model.classifier[0](x)
    final_feature = torch.flatten(x, 1)
    logits = model.classifier[2](final_feature)
    if mid is None:
        raise RuntimeError("ConvNeXt features[5] was not reached")
    mid_feature = torch.cat([mid.mean(dim=(2, 3)), mid.std(dim=(2, 3), unbiased=False)], dim=1)
    if feature_layer == "final":
        feature = final_feature
    elif feature_layer == "mid_stats":
        feature = mid_feature
    else:
        feature = torch.cat([final_feature, mid_feature], dim=1)
    return feature, logits


@torch.no_grad()
def extract(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    cache_path: Path,
) -> Dict[str, np.ndarray]:
    expected_ids = frame.image_id.astype(str).tolist()
    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        cached_ids = cached["image_ids"].astype(str).tolist()
        if cached_ids == expected_ids:
            return {key: cached[key] for key in cached.files}
    loader = DataLoader(
        FeatureDataset(frame, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    features: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    labels: List[int] = []
    image_ids: List[str] = []
    label_names: List[str] = []
    model.eval()
    for batch_index, (images, batch_labels, batch_ids, batch_names) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        with autocast(enabled=True):
            batch_features, logits = encode(model, images, args.feature_layer)
        features.append(F.normalize(batch_features.float(), dim=1).cpu().numpy())
        probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        labels.extend(batch_labels.tolist())
        image_ids.extend(batch_ids)
        label_names.extend(batch_names)
        if batch_index % 20 == 0 or batch_index == len(loader):
            print(f"feature extraction {cache_path.stem}: {len(image_ids)}/{len(frame)}", flush=True)
    payload = {
        "features": np.concatenate(features).astype(np.float32),
        "probabilities": np.concatenate(probabilities).astype(np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "image_ids": np.asarray(image_ids),
        "label_names": np.asarray(label_names),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    eps = 1e-12
    return float(0.5 * np.sum(p * np.log((p + eps) / (m + eps))) + 0.5 * np.sum(q * np.log((q + eps) / (m + eps))))


def fit_pca(real: np.ndarray) -> Tuple[PCA, np.ndarray]:
    max_components = min(64, real.shape[0] - 1, real.shape[1])
    solver = "randomized" if real.shape[1] > 1024 else "full"
    probe = PCA(n_components=max_components, svd_solver=solver, random_state=2026).fit(real)
    cumulative = np.cumsum(probe.explained_variance_ratio_)
    components = int(np.searchsorted(cumulative, 0.95) + 1)
    components = max(2, min(components, max_components))
    pca = PCA(n_components=components, svd_solver=solver, random_state=2026).fit(real)
    return pca, pca.transform(real).astype(np.float32)


def explore_k(
    real_projected: np.ndarray, args: argparse.Namespace, seeds: List[int]
) -> Tuple[int, List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    minimum = max(args.min_cluster_size, int(np.ceil(args.min_cluster_fraction * len(real_projected))))
    upper = min(args.k_max, max(args.k_min, len(real_projected) // minimum))
    valid: List[Dict[str, object]] = []
    for k in range(args.k_min, upper + 1):
        assignments = []
        silhouettes = []
        min_sizes = []
        inertias = []
        for seed in seeds:
            fit = KMeans(n_clusters=k, n_init=20, random_state=seed, algorithm="lloyd").fit(real_projected)
            assignments.append(fit.labels_)
            silhouettes.append(float(silhouette_score(real_projected, fit.labels_)))
            min_sizes.append(int(np.bincount(fit.labels_, minlength=k).min()))
            inertias.append(float(fit.inertia_))
        pairwise = [adjusted_rand_score(assignments[i], assignments[j]) for i, j in combinations(range(len(seeds)), 2)]
        row: Dict[str, object] = {
            "k": k,
            "silhouette_mean": float(np.mean(silhouettes)),
            "silhouette_std": float(np.std(silhouettes, ddof=1)),
            "stability_ari_mean": float(np.mean(pairwise)),
            "stability_ari_min": float(np.min(pairwise)),
            "minimum_cluster_size": int(np.min(min_sizes)),
            "minimum_required": minimum,
            "inertia_mean": float(np.mean(inertias)),
        }
        row["stable"] = bool(
            row["stability_ari_mean"] >= args.min_stability_ari
            and row["minimum_cluster_size"] >= minimum
            and row["silhouette_mean"] > 0.0
        )
        rows.append(row)
        if row["stable"]:
            valid.append(row)
    if not valid:
        return 1, rows, {"reason": "no_stable_multimode_solution", "minimum_required": minimum}
    # Prefer the most separated stable solution; a small complexity penalty prevents gratuitous micro-clusters.
    selected = max(valid, key=lambda row: float(row["silhouette_mean"]) + 0.15 * float(row["stability_ari_mean"]) - 0.01 * int(row["k"]))
    return int(selected["k"]), rows, {"reason": "best_stable_train_only_solution", **selected}


def analyze_class(
    class_name: str,
    real_feature: np.ndarray,
    synthetic_feature: np.ndarray,
    args: argparse.Namespace,
    seeds: List[int],
    output_dir: Path,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray], List[Dict[str, object]]]:
    pca, real_projected = fit_pca(real_feature)
    synthetic_projected = pca.transform(synthetic_feature).astype(np.float32)
    k, k_rows, selection = explore_k(real_projected, args, seeds)
    if k == 1:
        real_assignment = np.zeros(len(real_projected), dtype=np.int64)
        synthetic_assignment = np.zeros(len(synthetic_projected), dtype=np.int64)
        centers = real_projected.mean(axis=0, keepdims=True)
    else:
        fitted = KMeans(n_clusters=k, n_init=50, random_state=args.seed, algorithm="lloyd").fit(real_projected)
        centers = fitted.cluster_centers_.astype(np.float32)
        real_assignment = fitted.labels_.astype(np.int64)
        synthetic_assignment = fitted.predict(synthetic_projected).astype(np.int64)
    real_distance = np.linalg.norm(real_projected - centers[real_assignment], axis=1)
    synthetic_distance = np.linalg.norm(synthetic_projected - centers[synthetic_assignment], axis=1)
    real_counts = np.bincount(real_assignment, minlength=k)
    synthetic_counts = np.bincount(synthetic_assignment, minlength=k)
    real_occupancy = real_counts / real_counts.sum()
    synthetic_occupancy = synthetic_counts / synthetic_counts.sum()
    distance_p95 = np.asarray(
        [np.percentile(real_distance[real_assignment == cluster], 95) for cluster in range(k)], dtype=np.float32
    )
    covered = synthetic_counts >= np.maximum(5, np.floor(0.5 * len(synthetic_assignment) * real_occupancy)).astype(int)
    out_of_support = synthetic_distance > distance_p95[synthetic_assignment]
    summary: Dict[str, object] = {
        "class_name": class_name,
        "real_count": int(len(real_feature)),
        "synthetic_count": int(len(synthetic_feature)),
        "pca_components": int(pca.n_components_),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "selected_k": int(k),
        "selection": selection,
        "real_cluster_counts": real_counts.tolist(),
        "synthetic_cluster_counts": synthetic_counts.tolist(),
        "real_occupancy": real_occupancy.tolist(),
        "synthetic_occupancy": synthetic_occupancy.tolist(),
        "prototype_coverage": float(covered.mean()),
        "occupancy_js_divergence": js_divergence(real_occupancy, synthetic_occupancy),
        "occupancy_l1_gap": float(np.abs(real_occupancy - synthetic_occupancy).sum()),
        "maximum_occupancy_ratio": float(synthetic_occupancy.max() / max(real_occupancy.max(), 1e-12)),
        "real_nearest_prototype_distance_mean": float(real_distance.mean()),
        "synthetic_nearest_prototype_distance_mean": float(synthetic_distance.mean()),
        "synthetic_out_of_real_p95_rate": float(out_of_support.mean()),
        "real_cluster_distance_p95": distance_p95.tolist(),
    }
    artifact = {
        "pca_mean": pca.mean_.astype(np.float32),
        "pca_components": pca.components_.astype(np.float32),
        "prototype_centers": centers.astype(np.float32),
        "real_occupancy": real_occupancy.astype(np.float32),
        "real_distance_p95": distance_p95.astype(np.float32),
    }
    assignment_rows: List[Dict[str, object]] = []
    for domain, assignment, distance in [
        ("real_train", real_assignment, real_distance),
        ("a3_synthetic", synthetic_assignment, synthetic_distance),
    ]:
        for index, (cluster, value) in enumerate(zip(assignment, distance)):
            assignment_rows.append(
                {
                    "class_name": class_name,
                    "domain": domain,
                    "local_index": index,
                    "prototype": int(cluster),
                    "distance": float(value),
                    "inside_real_p95": bool(value <= distance_p95[int(cluster)]),
                }
            )
    pd.DataFrame(k_rows).to_csv(output_dir / f"{class_name}_k_stability.csv", index=False)
    return summary, artifact, assignment_rows


def make_embedding_plot(
    real: Dict[str, np.ndarray], synthetic: Dict[str, np.ndarray], summaries: List[Dict[str, object]], output: Path, seed: int
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 11))
    rng = np.random.default_rng(seed)
    for axis, class_name, summary in zip(axes.flat, MINORITY, summaries):
        rmask = real["label_names"].astype(str) == class_name
        smask = synthetic["label_names"].astype(str) == class_name
        r = real["features"][rmask]
        s = synthetic["features"][smask]
        if len(s) > 800:
            s = s[rng.choice(len(s), size=800, replace=False)]
        values = np.concatenate([r, s], axis=0)
        domain = np.asarray(["Real train"] * len(r) + ["A3 synthetic"] * len(s))
        reduced = PCA(n_components=min(32, len(values) - 1), random_state=seed).fit_transform(values)
        embedded = TSNE(
            n_components=2,
            perplexity=min(30, max(5, (len(values) - 1) // 3)),
            init="pca",
            learning_rate="auto",
            random_state=seed,
            n_iter=1000,
        ).fit_transform(reduced)
        for name, color, alpha in [("Real train", "#1f77b4", 0.75), ("A3 synthetic", "#ff7f0e", 0.30)]:
            mask = domain == name
            axis.scatter(embedded[mask, 0], embedded[mask, 1], s=10, alpha=alpha, c=color, label=name)
        axis.set_title(f"{class_name}: selected K={summary['selected_k']}, JS={summary['occupancy_js_divergence']:.3f}")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.legend(fontsize=8)
    figure.suptitle("A3 intra-class distribution diagnostic (train-only medical features)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A4 prototype diagnostic requires CUDA")
    seed_everything(args.seed)
    device = torch.device("cuda")
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    a3_root = (ROOT / args.a3_root).resolve()
    train = pd.read_csv(a3_root / "splits" / "train.csv")
    train = train[train.label_name.astype(str).isin(MINORITY)].reset_index(drop=True)
    synthetic = pd.read_csv(a3_root / "generated_manifest.csv")
    synthetic = synthetic.rename(columns={"output_path": "image_path"}) if "image_path" not in synthetic else synthetic
    synthetic = synthetic[synthetic.label_name.astype(str).isin(MINORITY)].reset_index(drop=True)
    if len(train) != 1300 or len(synthetic) != 3752:
        raise RuntimeError(f"Expected 1300 real-train minority and 3752 A3 synthetic, got {len(train)} and {len(synthetic)}")
    if set(synthetic.source_image_id.astype(str)) - set(train.image_id.astype(str)):
        raise RuntimeError("A3 synthetic manifest contains a source outside real train")
    checkpoint_path = (ROOT / args.feature_checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("pretrained") is not False:
        raise RuntimeError("Feature checkpoint must be the weights=None medical classifier")
    model = build_model(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    started = time.time()
    write_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "feature_checkpoint_resolved": str(checkpoint_path),
            "feature_checkpoint_pretrained": False,
            "feature_fit_data": "real train only; test not loaded",
            "feature_layer": args.feature_layer,
            "real_count": len(train),
            "synthetic_count": len(synthetic),
            "status": "running",
        },
    )
    write_json(output_dir / "live_progress.json", {"status": "extracting_real", "completed": 0})
    real_cache = extract(model, train, args, device, output_dir / "real_train_features.npz")
    write_json(output_dir / "live_progress.json", {"status": "extracting_a3", "completed": 1})
    synthetic_cache = extract(model, synthetic, args, device, output_dir / "a3_synthetic_features.npz")
    del model
    torch.cuda.empty_cache()
    seeds = [int(value) for value in args.cluster_seeds.split(",") if value.strip()]
    summaries: List[Dict[str, object]] = []
    assignment_rows: List[Dict[str, object]] = []
    artifacts: Dict[str, np.ndarray] = {}
    for class_index, class_name in enumerate(MINORITY, 1):
        write_json(
            output_dir / "live_progress.json",
            {"status": "clustering", "class_name": class_name, "completed": class_index - 1, "total": 4},
        )
        real_mask = real_cache["label_names"].astype(str) == class_name
        synthetic_mask = synthetic_cache["label_names"].astype(str) == class_name
        summary, artifact, rows = analyze_class(
            class_name,
            real_cache["features"][real_mask],
            synthetic_cache["features"][synthetic_mask],
            args,
            seeds,
            output_dir,
        )
        summaries.append(summary)
        assignment_rows.extend(rows)
        for key, value in artifact.items():
            artifacts[f"{class_name}__{key}"] = value
        print(
            f"{class_name}: K={summary['selected_k']} coverage={summary['prototype_coverage']:.3f} "
            f"JS={summary['occupancy_js_divergence']:.4f} out95={summary['synthetic_out_of_real_p95_rate']:.3f}",
            flush=True,
        )
    pd.DataFrame(assignment_rows).to_csv(output_dir / "prototype_assignments.csv", index=False)
    np.savez_compressed(output_dir / "prototype_model.npz", **artifacts)
    make_embedding_plot(real_cache, synthetic_cache, summaries, output_dir / "a3_intra_class_tsne.png", args.seed)
    stable_classes = [item for item in summaries if item["selected_k"] > 1]
    imbalanced_classes = [
        item
        for item in stable_classes
        if item["occupancy_js_divergence"] >= 0.02
        or item["occupancy_l1_gap"] >= 0.20
        or item["prototype_coverage"] < 1.0
    ]
    supports = len(stable_classes) >= 2 and len(imbalanced_classes) >= 1
    final = {
        "status": "completed",
        "method": f"train-only scratch-medical ConvNeXt {args.feature_layer} features; per-class PCA; multi-K multi-seed K-Means stability",
        "test_accessed": False,
        "feature_pretrained": False,
        "class_summaries": summaries,
        "stable_multimode_classes": [item["class_name"] for item in stable_classes],
        "materially_imbalanced_classes": [item["class_name"] for item in imbalanced_classes],
        "prototype_direction_supported": supports,
        "decision_rule": "at least two stable multimode classes and at least one class with train-only occupancy gap",
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    write_json(output_dir / "diagnostic_summary.json", final)
    write_json(output_dir / "live_progress.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
