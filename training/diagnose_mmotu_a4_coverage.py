from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.mmotu_dataset import MMOTU_CLASS_ORDER
from training.mmotu_a4_common import (
    calibrated_bootstraps, extract_teacher, jensen_shannon, load_teacher, normalize,
    resolve_teacher_paths, source_excluded_coverage, write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MMOTU A4 train-only local/prototype coverage diagnostic")
    parser.add_argument("--a3-manifest", default="outputs/mmotu2d/a3_adversarial/balanced_train/generated_manifest.csv")
    parser.add_argument("--train-split", default="outputs/mmotu2d/dataset_preparation/splits/train.csv")
    parser.add_argument("--teacher-root", default="outputs/mmotu2d/convnext_real_baseline/formal")
    parser.add_argument("--teacher-checkpoints", default="", help="Three comma-separated explicit checkpoints")
    parser.add_argument("--teacher-initialization", choices=("scratch", "imagenet"), default="scratch")
    parser.add_argument("--output-dir", default="outputs/mmotu2d/a4_adaptive/coverage_diagnostic")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--reference-fraction", type=float, default=0.70)
    parser.add_argument("--deficit-threshold", type=float, default=0.03)
    parser.add_argument("--same-sign-fraction", type=float, default=0.70)
    parser.add_argument("--prototype-js-threshold", type=float, default=0.08)
    parser.add_argument("--bootstrap-seeds", default=",".join(str(x) for x in range(2026, 2041)))
    return parser.parse_args()


def prototype_diagnostic(real: np.ndarray, synthetic: np.ndarray, seeds: List[int]):
    candidates = []
    max_k = min(5, max(2, len(real) // 10))
    for k in range(2, max_k + 1):
        assignments, silhouettes, minimums = [], [], []
        for seed in seeds[:5]:
            model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(real)
            assignments.append(model.labels_)
            silhouettes.append(float(silhouette_score(real, model.labels_, metric="euclidean")))
            minimums.append(int(np.bincount(model.labels_, minlength=k).min()))
        stability = float(np.mean([
            adjusted_rand_score(assignments[i], assignments[j])
            for i in range(len(assignments)) for j in range(i + 1, len(assignments))
        ]))
        candidates.append({
            "k": k, "silhouette_mean": float(np.mean(silhouettes)),
            "stability_ari_mean": stability, "minimum_cluster_size": int(min(minimums)),
        })
    valid = [row for row in candidates if row["silhouette_mean"] >= 0.15 and row["stability_ari_mean"] >= 0.60 and row["minimum_cluster_size"] >= max(4, int(0.05 * len(real)))]
    if not valid:
        return {"stable_multimode": False, "candidates": candidates, "selected_k": None, "js_divergence": 0.0}
    selected = max(valid, key=lambda row: (row["silhouette_mean"], row["stability_ari_mean"]))
    model = KMeans(n_clusters=int(selected["k"]), random_state=seeds[0], n_init=20).fit(real)
    real_assignment = model.labels_
    synthetic_assignment = ((synthetic[:, None, :] - model.cluster_centers_[None, :, :]) ** 2).sum(2).argmin(1)
    real_occupancy = np.bincount(real_assignment, minlength=int(selected["k"]))
    synthetic_occupancy = np.bincount(synthetic_assignment, minlength=int(selected["k"]))
    return {
        "stable_multimode": True, "candidates": candidates, "selected_k": int(selected["k"]),
        "silhouette": selected["silhouette_mean"], "stability_ari": selected["stability_ari_mean"],
        "real_occupancy": real_occupancy.tolist(), "synthetic_occupancy": synthetic_occupancy.tolist(),
        "js_divergence": jensen_shannon(real_occupancy, synthetic_occupancy),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MMOTU A4 diagnostic requires CUDA")
    device = torch.device("cuda")
    output = (ROOT / args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    if (output / "completed.json").is_file():
        print((output / "completed.json").read_text(encoding="utf-8"), flush=True); return
    real = pd.read_csv(ROOT / args.train_split).reset_index(drop=True)
    real["source_image_id"] = real.image_id.astype(str); real["prior_path"] = real.mask_path
    synthetic = pd.read_csv(ROOT / args.a3_manifest).reset_index(drop=True)
    expected_synthetic = int(sum(int(real.label.value_counts().max()) - int(value) for value in real.label.value_counts()))
    if real.empty or len(synthetic) != expected_synthetic:
        raise RuntimeError(
            f"Unexpected frozen medical train/A3 pool: real={len(real)} synthetic={len(synthetic)} expected={expected_synthetic}"
        )
    active_classes = [name for name in MMOTU_CLASS_ORDER if int((synthetic.label_name == name).sum()) > args.knn_k]
    seeds = [int(value) for value in args.bootstrap_seeds.split(",") if value.strip()]
    started = time.time(); data = {}
    teacher_paths = resolve_teacher_paths(ROOT, args.teacher_root, args.teacher_checkpoints)
    for seed, teacher_path in zip((2026, 2027, 2028), teacher_paths):
        name = f"seed_{seed}"
        model = load_teacher(teacher_path, device, args.teacher_initialization == "imagenet")
        r = extract_teacher(model, real, args.image_size, args.batch_size, args.num_workers, device, output / "cache" / f"{name}_real.npz", f"{name} real")
        s = extract_teacher(model, synthetic, args.image_size, args.batch_size, args.num_workers, device, output / "cache" / f"{name}_a3.npz", f"{name} A3")
        data[name] = (r, s); del model; torch.cuda.empty_cache()

    rows, bootstrap_rows, prototype_rows = [], [], []
    class_teacher_support: Dict[str, Dict[str, object]] = {}
    for class_name in active_classes:
        local_support = 0; prototype_support = 0; teacher_details = {}
        for teacher_name, (real_data, synthetic_data) in data.items():
            real_mask = real_data["label_names"].astype(str) == class_name
            synthetic_mask = synthetic_data["label_names"].astype(str) == class_name
            real_features = normalize(real_data["local_features"][real_mask])
            synthetic_features = normalize(synthetic_data["local_features"][synthetic_mask])
            real_ids = real_data["image_ids"][real_mask].astype(str)
            source_ids = synthetic_data["source_ids"][synthetic_mask].astype(str)
            coverage, ratio, missing, _matrix = source_excluded_coverage(real_features, synthetic_features, real_ids, source_ids, args.knn_k)
            bootstraps = calibrated_bootstraps(real_features, synthetic_features, real_ids, source_ids, args.knn_k, args.reference_fraction, seeds)
            deficits = np.asarray([row["coverage_deficit"] for row in bootstraps], dtype=float)
            local_supported = bool(len(deficits) and np.median(deficits) >= args.deficit_threshold and np.mean(deficits > 0) >= args.same_sign_fraction)
            prototype = prototype_diagnostic(real_features, synthetic_features, seeds)
            prototype_supported = bool(prototype["stable_multimode"] and prototype["js_divergence"] >= args.prototype_js_threshold)
            local_support += int(local_supported); prototype_support += int(prototype_supported)
            teacher_details[teacher_name] = {"local_supported": local_supported, "prototype_supported": prototype_supported}
            rows.append({
                "class_name": class_name, "teacher": teacher_name, "real_count": int(real_mask.sum()),
                "synthetic_count": int(synthetic_mask.sum()), "source_excluded_coverage": coverage,
                "uncovered_rate": float(missing.mean()), "nearest_radius_ratio_mean": float(ratio.mean()),
                "bootstrap_deficit_mean": float(deficits.mean()) if len(deficits) else None,
                "bootstrap_deficit_median": float(np.median(deficits)) if len(deficits) else None,
                "positive_deficit_fraction": float(np.mean(deficits > 0)) if len(deficits) else None,
                "local_supported": local_supported, "stable_multimode": prototype["stable_multimode"],
                "selected_k": prototype["selected_k"], "prototype_js_divergence": prototype["js_divergence"],
                "prototype_supported": prototype_supported,
            })
            for row in bootstraps:
                bootstrap_rows.append({"class_name": class_name, "teacher": teacher_name, **row})
            prototype_rows.append({"class_name": class_name, "teacher": teacher_name, **prototype})
        supported = local_support >= 2 or prototype_support >= 2
        class_teacher_support[class_name] = {
            "local_supporting_teachers": local_support, "prototype_supporting_teachers": prototype_support,
            "supported": supported, "teacher_details": teacher_details,
        }
    pd.DataFrame(rows).to_csv(output / "coverage_results.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output / "real_real_bootstraps.csv", index=False)
    write_json(output / "prototype_results.json", prototype_rows)
    supported_classes = [name for name, detail in class_teacher_support.items() if detail["supported"]]
    summary = {
        "status": "completed", "method": f"three {args.teacher_initialization} medical teachers; source-excluded ROI-local coverage plus stable K-Means occupancy diagnostic",
        "test_accessed": False, "active_generated_classes": active_classes,
        "thresholds": {"deficit": args.deficit_threshold, "same_sign_fraction": args.same_sign_fraction, "prototype_js": args.prototype_js_threshold, "teacher_consensus": 2},
        "class_decisions": class_teacher_support, "supported_classes": supported_classes,
        "controller_authorized_by_diagnostic": bool(supported_classes),
        "elapsed_minutes": (time.time() - started) / 60,
    }
    write_json(output / "diagnostic_summary.json", summary); write_json(output / "completed.json", summary)
    write_json(output / "live_progress.json", summary); write_json(output / "run_config.json", vars(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
