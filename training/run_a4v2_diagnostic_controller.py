from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import StableDiffusionImg2ImgPipeline

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.aptos_dataset import APTOS_CLASS_ORDER
from metrics.generation_metrics import build_vgg16_feature_extractor
from models.a3_diagnostic_discriminator import A3DiagnosticDiscriminator
from training.diagnose_a4v2_medical_local import (
    DEFAULT_TEACHERS,
    FeatureDataset,
    encode,
    load_teacher,
    normalize,
    distances as medical_distances,
    radii as medical_radii,
)
from training.run_a4_adaptive_controller import (
    CandidateTask,
    candidate_passes,
    generate_round,
    make_candidate_frame,
    pairwise_distance,
    quality_thresholds,
    score_frame,
)
from training.run_aptos_convnext_baselines import seed_everything
from utils.image_quality import compute_pixel_stats


MINORITY = APTOS_CLASS_ORDER[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A4-v2 diagnosis-aware adaptive controller")
    parser.add_argument("--model-dir", default="models/pretrained/stable-diffusion-v1-5")
    parser.add_argument("--a3-lora", default="outputs/aptos2019/a3_adversarial_candidate/source_holdout_d4_v4/full_training/final_lora")
    parser.add_argument("--a3-root", default="outputs/aptos2019/a3_adversarial_candidate/source_holdout_d4_v4/balanced_train")
    parser.add_argument("--a3-quality-cache", default="outputs/aptos2019/a3_adversarial_candidate/source_holdout_d4_v4/image_quality_fid_kid_ipr/feature_cache.npz")
    parser.add_argument("--diagnostic-dir", default="outputs/aptos2019/a4_from_a3_adversarial_candidate/source_holdout_d4_v4/medical_local_diagnostic")
    parser.add_argument("--discriminator", default="outputs/aptos2019/a3_adversarial_candidate/source_holdout_d4_v4/full_training/final_online_discriminator.pt")
    parser.add_argument("--prior-index", default="outputs/aptos2019/structure_guidance_v3/weak_localization_priors/prior_index.csv")
    parser.add_argument("--output-dir", default="outputs/aptos2019/a4_from_a3_adversarial_candidate/source_holdout_d4_v4/balanced_train")
    parser.add_argument("--round-size", type=int, default=48)
    parser.add_argument("--max-attempt-factor", type=float, default=2.5)
    parser.add_argument("--operational-gap-threshold", type=float, default=0.03)
    parser.add_argument("--base-replacement-fraction", type=float, default=0.05)
    parser.add_argument("--gap-gain", type=float, default=0.35)
    parser.add_argument("--max-replacement-fraction", type=float, default=0.15)
    parser.add_argument("--quality-quantile", type=float, default=0.10)
    parser.add_argument("--distance-quantile", type=float, default=0.90)
    parser.add_argument("--teacher-quantile", type=float, default=0.10)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--score-image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-rounds", type=int, default=0)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


@torch.no_grad()
def score_medical(
    frame: pd.DataFrame,
    teachers: Dict[str, torch.nn.Module],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Dict[str, np.ndarray]]:
    loader = DataLoader(
        FeatureDataset(frame, args.score_image_size), batch_size=args.score_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    collectors = {name: {"features": [], "probabilities": []} for name in teachers}
    for images, priors, _labels, _indices in loader:
        images = images.to(device, non_blocking=True)
        priors = priors.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=True):
            for name, model in teachers.items():
                _global, local, logits = encode(model, images, priors, original=not isinstance(model, A3DiagnosticDiscriminator))
                collectors[name]["features"].append(local.cpu().numpy().astype(np.float32))
                collectors[name]["probabilities"].append(torch.softmax(logits, 1).cpu().numpy().astype(np.float32))
    return {
        name: {
            "features": normalize(np.concatenate(value["features"])),
            "probabilities": np.concatenate(value["probabilities"]),
        }
        for name, value in collectors.items()
    }


def teacher_statistics(data: Dict[str, Dict[str, np.ndarray]], labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_probabilities = []
    margins = []
    agreements = []
    row = np.arange(len(labels))
    for value in data.values():
        probabilities = value["probabilities"]
        target = probabilities[row, labels]
        other = probabilities.copy()
        other[row, labels] = -np.inf
        target_probabilities.append(target)
        margins.append(target - other.max(1))
        agreements.append(probabilities.argmax(1) == labels)
    return (
        np.mean(target_probabilities, axis=0).astype(np.float32),
        np.mean(margins, axis=0).astype(np.float32),
        np.mean(agreements, axis=0).astype(np.float32),
    )


def coverage_state(
    distance_by_teacher: Dict[str, np.ndarray], radius_by_teacher: Dict[str, np.ndarray]
) -> Tuple[Dict[str, float], float, float]:
    coverage = {}
    excess = []
    for name, matrix in distance_by_teacher.items():
        ratio = matrix.min(1) / np.maximum(radius_by_teacher[name], 1e-8)
        coverage[name] = float(np.mean(ratio <= 1.0))
        excess.append(float(np.mean(np.maximum(ratio - 1.0, 0.0))))
    return coverage, float(np.mean(list(coverage.values()))), float(np.mean(excess))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A4-v2 controller requires CUDA")
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    out = (ROOT / args.output_dir).resolve()
    if (out / "completed.json").is_file():
        print((out / "completed.json").read_text(encoding="utf-8"), flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    candidate_dir = out / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_summary = json.loads((ROOT / args.diagnostic_dir / "diagnostic_summary.json").read_text(encoding="utf-8"))
    if not diagnostic_summary.get("controller_authorized_by_diagnostic") or diagnostic_summary.get("test_accessed"):
        raise RuntimeError("Valid train-only A4-v2 diagnostic did not authorize controller")

    a3_root = (ROOT / args.a3_root).resolve()
    a3 = pd.read_csv(a3_root / "generated_manifest.csv").reset_index(drop=True)
    train = pd.read_csv(a3_root / "splits/train.csv")
    prior = pd.read_csv(ROOT / args.prior_index)
    prior_map = dict(zip(prior.image_id.astype(str), prior.prior_path.astype(str)))
    labels = a3.label.to_numpy(dtype=np.int64)
    label_names = a3.label_name.astype(str).to_numpy()
    source_ids_all = a3.source_image_id.astype(str).to_numpy()

    discriminator = A3DiagnosticDiscriminator().to(device)
    discriminator.load_state_dict(torch.load(ROOT / args.discriminator, map_location="cpu")["model_state"], strict=True)
    discriminator.freeze_all()
    vgg = build_vgg16_feature_extractor(device)
    base_scores, scored_vgg = score_frame(a3, discriminator, vgg, args, device)
    cache = np.load(ROOT / args.a3_quality_cache, allow_pickle=True)
    real_vgg = normalize(cache["real_vgg16"].astype(np.float32))
    a3_vgg = normalize(cache["generated_vgg16"].astype(np.float32))
    if scored_vgg.shape != a3_vgg.shape or float(np.max(np.abs(scored_vgg - a3_vgg))) > 5e-3:
        raise RuntimeError("A3 VGG cache reproduction failed")
    thresholds = quality_thresholds(
        base_scores, scored_vgg, real_vgg, cache["real_labels"].astype(str), cache["generated_labels"].astype(str), args
    )

    teachers = {name: load_teacher(name, ROOT / path, device) for name, path in DEFAULT_TEACHERS.items()}
    diagnostic_dir = ROOT / args.diagnostic_dir / "cache"
    # These trusted local caches contain pandas-derived object string arrays for IDs/names.
    medical_real = {name: np.load(diagnostic_dir / f"{name}_real.npz", allow_pickle=True) for name in teachers}
    medical_a3 = {name: np.load(diagnostic_dir / f"{name}_a3.npz", allow_pickle=True) for name in teachers}
    initial_medical = {
        name: {"features": normalize(data["local_features"]), "probabilities": data["probabilities"]}
        for name, data in medical_a3.items()
    }
    teacher_probability, teacher_margin, teacher_agreement = teacher_statistics(initial_medical, labels)
    teacher_thresholds: Dict[str, Dict[str, float]] = {}
    for class_name in MINORITY:
        mask = label_names == class_name
        teacher_thresholds[class_name] = {
            "mean_target_probability": float(np.quantile(teacher_probability[mask], args.teacher_quantile)),
            "mean_margin": float(np.quantile(teacher_margin[mask], args.teacher_quantile)),
            "agreement": float(np.quantile(teacher_agreement[mask], args.teacher_quantile)),
        }

    active_records: Dict[str, List[Dict[str, object]]] = {}
    active_vgg: Dict[str, np.ndarray] = {}
    active_quality: Dict[str, np.ndarray] = {}
    active_teacher_probability: Dict[str, np.ndarray] = {}
    active_features: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in teachers}
    real_features: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in teachers}
    radii_by_class: Dict[str, Dict[str, np.ndarray]] = {}
    distance_by_class: Dict[str, Dict[str, np.ndarray]] = {}
    real_ids_by_class: Dict[str, np.ndarray] = {}
    initial_state: Dict[str, object] = {}
    budgets: Dict[str, int] = {}
    accepted = {name: 0 for name in MINORITY}
    attempts = {name: 0 for name in MINORITY}
    for class_name in MINORITY:
        smask = label_names == class_name
        records = a3.loc[smask].to_dict("records")
        for local, record in enumerate(records):
            global_index = np.flatnonzero(smask)[local]
            record.update(base_scores.iloc[global_index].to_dict())
            record["teacher_probability"] = float(teacher_probability[global_index])
            record["teacher_margin"] = float(teacher_margin[global_index])
            record["teacher_agreement"] = float(teacher_agreement[global_index])
            record["origin"] = "A3"
        active_records[class_name] = records
        active_vgg[class_name] = scored_vgg[smask].copy()
        active_quality[class_name] = base_scores.loc[smask, "quality_score"].to_numpy(dtype=np.float32)
        active_teacher_probability[class_name] = teacher_probability[smask].copy()
        real_ids = train.loc[train.label_name.astype(str) == class_name, "image_id"].astype(str).to_numpy()
        real_ids_by_class[class_name] = real_ids
        class_distances: Dict[str, np.ndarray] = {}
        class_radii: Dict[str, np.ndarray] = {}
        for teacher_name in teachers:
            rdata = medical_real[teacher_name]
            rmask = rdata["label_names"].astype(str) == class_name
            sdata = medical_a3[teacher_name]
            teacher_smask = sdata["label_names"].astype(str) == class_name
            real_feature = normalize(rdata["local_features"][rmask])
            synthetic_feature = normalize(sdata["local_features"][teacher_smask])
            active_features[teacher_name][class_name] = synthetic_feature.copy()
            real_features[teacher_name][class_name] = real_feature
            radius = medical_radii(real_feature, args.knn_k)
            matrix = medical_distances(real_feature, synthetic_feature)
            matrix[real_ids[:, None] == source_ids_all[smask][None, :]] = np.inf
            class_distances[teacher_name] = matrix
            class_radii[teacher_name] = radius
        radii_by_class[class_name] = class_radii
        distance_by_class[class_name] = class_distances
        coverage, mean_coverage, excess = coverage_state(class_distances, class_radii)
        gap = 1.0 - mean_coverage
        if gap <= args.operational_gap_threshold:
            budget = 0
        else:
            fraction = np.clip(
                args.base_replacement_fraction + args.gap_gain * gap,
                args.base_replacement_fraction, args.max_replacement_fraction,
            )
            budget = int(math.ceil(len(records) * fraction))
        budgets[class_name] = budget
        initial_state[class_name] = {
            "coverage_by_teacher": coverage, "mean_coverage": mean_coverage,
            "mean_excess_distance_ratio": excess, "operational_gap": gap, "replacement_budget": budget,
        }
    active_classes = [name for name in MINORITY if budgets[name] > 0]
    if not active_classes:
        raise RuntimeError("Adaptive no-op: no operational medical local coverage gap exceeded train-only threshold")

    config = {
        **vars(args), "method": "A3 -> source-excluded lesion-aware medical coverage -> quality-first adaptive replacement",
        "test_accessed": False, "a3_frozen": True, "teachers": DEFAULT_TEACHERS,
        "teacher_thresholds": teacher_thresholds, "a3_quality_thresholds": thresholds,
        "initial_state": initial_state, "active_classes": active_classes, "replacement_budgets": budgets,
    }
    write_json(out / "run_config.json", config)

    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        (ROOT / args.model_dir).resolve(), variant="fp16", torch_dtype=torch.float16,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False, local_files_only=True,
    ).to(device)
    pipeline.load_lora_weights((ROOT / args.a3_lora).resolve())
    pipeline.fuse_lora(lora_scale=args.lora_scale)
    pipeline.set_progress_bar_config(disable=True)
    started = time.time()
    all_candidates: List[Dict[str, object]] = []
    audit: List[Dict[str, object]] = []
    candidate_index = 0
    round_index = 0
    stale_rounds = 0
    while True:
        if args.max_rounds and round_index >= args.max_rounds:
            break
        remaining = {
            name: min(budgets[name] - accepted[name], int(math.ceil(args.max_attempt_factor * budgets[name])) - attempts[name])
            for name in MINORITY
        }
        eligible = [name for name in active_classes if remaining[name] > 0]
        if not eligible or stale_rounds >= 3:
            break
        round_index += 1
        tasks: List[CandidateTask] = []
        allocations = {name: 0 for name in MINORITY}
        for slot in range(args.round_size):
            choices = [name for name in eligible if allocations[name] < remaining[name]]
            if not choices:
                break
            class_name = choices[slot % len(choices)]
            ratios = []
            for teacher_name in teachers:
                nearest = distance_by_class[class_name][teacher_name].min(1)
                ratios.append(nearest / np.maximum(radii_by_class[class_name][teacher_name], 1e-8))
            deficiency = np.median(np.stack(ratios), axis=0)
            weights = np.maximum(deficiency - 0.90, 0.05) ** 2
            source_local = int(rng.choice(len(weights), p=weights / weights.sum()))
            source_id = real_ids_by_class[class_name][source_local]
            source = train[train.image_id.astype(str) == source_id].iloc[0]
            class_dir = candidate_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            path = class_dir / f"{source_id}__a4v2__{candidate_index:05d}.png"
            tasks.append(CandidateTask(
                candidate_index=candidate_index, round_index=round_index, label=int(source.label),
                label_name=class_name, source_image_id=source_id, source_path=str(source.image_path),
                prior_path=prior_map[source_id], output_path=str(path.resolve()),
                seed=940000 + args.seed + candidate_index,
            ))
            allocations[class_name] += 1
            attempts[class_name] += 1
            candidate_index += 1
        write_json(out / "live_progress.json", {
            "status": "generating", "round": round_index, "attempts": attempts,
            "accepted": accepted, "budgets": budgets, "active_classes": active_classes,
        })
        generate_round(tasks, pipeline, args, out)
        frame = make_candidate_frame(tasks)
        frame["dataset_name"] = "aptos2019_a4v2_candidate_train_only"
        a3_candidate_scores, candidate_vgg = score_frame(frame, discriminator, vgg, args, device)
        medical_candidate = score_medical(frame, teachers, args, device)
        candidate_labels = frame.label.to_numpy(dtype=np.int64)
        candidate_teacher_p, candidate_margin, candidate_agreement = teacher_statistics(medical_candidate, candidate_labels)
        round_accepted = 0
        for local, task in enumerate(tasks):
            class_name = task.label_name
            row = frame.iloc[local].to_dict()
            row.update(a3_candidate_scores.iloc[local].to_dict())
            image = np.asarray(Image.open(task.output_path).convert("RGB"), dtype=np.uint8)
            pixel = compute_pixel_stats(image)
            row.update({
                "pixel_std": float(pixel["std"]), "entropy": float(pixel["entropy"]),
                "fundus_area_ratio": float(pixel["fundus_area_ratio"]),
                "valid_fundus": bool(pixel["is_valid_fundus"]), "near_constant": bool(pixel["is_near_constant"]),
                "teacher_probability": float(candidate_teacher_p[local]),
                "teacher_margin": float(candidate_margin[local]), "teacher_agreement": float(candidate_agreement[local]),
                "origin": "A4v2_candidate", "selected": False,
            })
            real_vgg_class = real_vgg[cache["real_labels"].astype(str) == class_name]
            nearest_vgg = float(pairwise_distance(candidate_vgg[local][None, :], real_vgg_class).min())
            row["nearest_real_distance"] = nearest_vgg
            qpass = candidate_passes(pd.Series(row), nearest_vgg, thresholds[class_name])
            t = teacher_thresholds[class_name]
            tpass = bool(
                row["teacher_probability"] >= t["mean_target_probability"]
                and row["teacher_margin"] >= t["mean_margin"]
                and row["teacher_agreement"] >= t["agreement"]
            )
            before_cov, before_mean, before_excess = coverage_state(
                distance_by_class[class_name], radii_by_class[class_name]
            )
            accepted_now = False
            if qpass and tpass and accepted[class_name] < budgets[class_name]:
                critical_sets = []
                for teacher_name in teachers:
                    critical_sets.append(set(distance_by_class[class_name][teacher_name].argmin(1).tolist()))
                noncritical = [
                    index for index in range(len(active_records[class_name]))
                    if sum(index in values for values in critical_sets) == 0
                ]
                if not noncritical:
                    noncritical = list(np.argsort(active_quality[class_name])[: max(12, len(active_quality[class_name]) // 5)])
                remove_index = min(
                    noncritical,
                    key=lambda index: float(0.65 * active_quality[class_name][index] + 0.35 * active_teacher_probability[class_name][index]),
                )
                proposed: Dict[str, np.ndarray] = {}
                for teacher_name in teachers:
                    feature = medical_candidate[teacher_name]["features"][local]
                    d = medical_distances(real_features[teacher_name][class_name], feature[None, :]).flatten()
                    d[real_ids_by_class[class_name] == task.source_image_id] = np.inf
                    matrix = distance_by_class[class_name][teacher_name].copy()
                    matrix[:, remove_index] = d
                    proposed[teacher_name] = matrix
                after_cov, after_mean, after_excess = coverage_state(proposed, radii_by_class[class_name])
                utility_improved = after_mean > before_mean + 1e-12 or (
                    abs(after_mean - before_mean) <= 1e-12 and after_excess < before_excess - 1e-5
                )
                quality_floor = float(row["quality_score"]) >= float(active_quality[class_name][remove_index]) - 0.02
                if utility_improved and quality_floor:
                    removed = active_records[class_name][remove_index]
                    row["selected"] = True
                    row["selection_rule"] = "medical local coverage/excess improved under A3 and teacher quality gates"
                    row["replaced_image_path"] = str(removed["image_path"])
                    active_records[class_name][remove_index] = dict(row)
                    active_vgg[class_name][remove_index] = candidate_vgg[local]
                    active_quality[class_name][remove_index] = float(row["quality_score"])
                    active_teacher_probability[class_name][remove_index] = float(row["teacher_probability"])
                    for teacher_name in teachers:
                        active_features[teacher_name][class_name][remove_index] = medical_candidate[teacher_name]["features"][local]
                        distance_by_class[class_name][teacher_name] = proposed[teacher_name]
                    accepted[class_name] += 1
                    round_accepted += 1
                    accepted_now = True
                else:
                    row["selection_rule"] = "rejected: no consensus medical coverage gain or quality floor failed"
            else:
                row["selection_rule"] = "rejected by A3 quality gate, medical teacher gate, or budget"
                after_cov, after_mean, after_excess = before_cov, before_mean, before_excess
            row.update({
                "a3_quality_gate_pass": qpass, "medical_teacher_gate_pass": tpass,
                "coverage_before": before_mean, "coverage_after": after_mean,
                "excess_before": before_excess, "excess_after": after_excess,
            })
            all_candidates.append(dict(row))
            audit.append({
                "round": round_index, "candidate_index": task.candidate_index, "class_name": class_name,
                "source_image_id": task.source_image_id, "a3_quality_gate_pass": qpass,
                "medical_teacher_gate_pass": tpass, "selected": accepted_now,
                "quality_score": float(row["quality_score"]), "teacher_probability": float(row["teacher_probability"]),
                "coverage_before": before_mean, "coverage_after": after_mean,
                "excess_before": before_excess, "excess_after": after_excess, "reason": row["selection_rule"],
            })
        stale_rounds = stale_rounds + 1 if round_accepted == 0 else 0
        pd.DataFrame(audit).to_csv(out / "adaptive_round_audit.csv", index=False)
        states = {
            name: coverage_state(distance_by_class[name], radii_by_class[name])[:2] for name in MINORITY
        }
        progress = {
            "status": "adaptive_update", "round": round_index, "round_candidates": len(tasks),
            "round_accepted": round_accepted, "attempts": attempts, "accepted": accepted,
            "budgets": budgets, "active_classes": active_classes, "coverage": states,
            "stale_rounds": stale_rounds, "elapsed_minutes": (time.time() - started) / 60,
        }
        write_json(out / "live_progress.json", progress)
        print(
            f"A4-v2 round {round_index}: candidates={len(tasks)} accepted={round_accepted} "
            f"total={sum(accepted.values())}; " + ", ".join(f"{name}:{states[name][1]:.3f}" for name in MINORITY),
            flush=True,
        )

    final_records = []
    for class_name in MINORITY:
        final_records.extend(active_records[class_name])
    final = pd.DataFrame(final_records).reset_index(drop=True)
    if len(final) != len(a3) or final.groupby("label_name").size().to_dict() != a3.groupby("label_name").size().to_dict():
        raise RuntimeError("A4-v2 changed fixed synthetic pool size or class counts")
    final.to_csv(out / "generated_manifest.csv", index=False, encoding="utf-8-sig")
    write_json(out / "generated_manifest.json", final.to_dict("records"))
    pd.DataFrame(all_candidates).to_csv(out / "all_candidates_manifest.csv", index=False)
    split_dir = out / "splits"
    split_dir.mkdir(exist_ok=True)
    for name in ["train.csv", "val.csv", "test.csv"]:
        shutil.copy2(a3_root / "splits" / name, split_dir / name)
    final_state = {
        name: {
            "coverage_by_teacher": coverage_state(distance_by_class[name], radii_by_class[name])[0],
            "mean_coverage": coverage_state(distance_by_class[name], radii_by_class[name])[1],
            "mean_excess_distance_ratio": coverage_state(distance_by_class[name], radii_by_class[name])[2],
        }
        for name in MINORITY
    }
    completed = {
        "status": "completed", "method": "A4-v2 diagnosis-aware adaptive controller",
        "test_accessed": False, "rounds": round_index, "candidates": len(all_candidates),
        "accepted": accepted, "budgets": budgets, "initial_state": initial_state,
        "final_state": final_state, "synthetic_pool_size": len(final),
        "elapsed_minutes": (time.time() - started) / 60,
    }
    write_json(out / "completed.json", completed)
    write_json(out / "live_progress.json", completed)
    print(json.dumps(completed, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
