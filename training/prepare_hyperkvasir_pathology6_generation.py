"""Prepare the fixed six-class Raw generation/classification boundary.

The input-protocol audit already created a deterministic five-fold assignment.
For the expensive A0--A4 generation chain we use one leakage-safe outer fold:
fold 0 is the held-out test, fold 1 is validation, and folds 2--4 are the
real-train source pool.  This keeps the generator from seeing the final test
images while allowing the existing fold-0 Original-only scratch result to be
reused as the baseline reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hyperkvasir_pathology import HYPERKVASIR_PATHOLOGY6_CLASS_ORDER
from utils.endoscopy import build_mucosal_structure_prior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HyperKvasir pathology6 generation boundary")
    parser.add_argument(
        "--input-root",
        default="outputs/hyperkvasir_pathology6/input_protocol_20260829",
        help="Completed Raw/FOV/Sanitized input-protocol root",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hyperkvasir_pathology6/ablation_raw_20260829",
    )
    parser.add_argument("--generation-test-fold", type=int, default=0, choices=range(5))
    parser.add_argument("--prior-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    input_root = (ROOT / args.input_root).resolve()
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    completed = output / "completed.json"
    if completed.is_file():
        print(completed.read_text(encoding="utf-8"), flush=True)
        return

    frame = pd.read_csv(input_root / "all_samples.csv", encoding="utf-8-sig").reset_index(drop=True)
    required = {"image_id", "image_path", "label", "label_name", "outer_fold"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"all_samples.csv missing columns: {sorted(missing)}")
    if set(frame.label_name.astype(str)) != set(HYPERKVASIR_PATHOLOGY6_CLASS_ORDER):
        raise RuntimeError("unexpected six-class label set")

    test_fold = int(args.generation_test_fold)
    validation_fold = (test_fold + 1) % 5
    train = frame[~frame.outer_fold.isin([test_fold, validation_fold])].copy().reset_index(drop=True)
    validation = frame[frame.outer_fold.eq(validation_fold)].copy().reset_index(drop=True)
    test = frame[frame.outer_fold.eq(test_fold)].copy().reset_index(drop=True)
    if min(len(train), len(validation), len(test)) <= 0:
        raise RuntimeError("empty fixed generation split")

    prior_dir = output / "weak_structure_priors"
    prior_dir.mkdir(parents=True, exist_ok=True)
    prior_rows = []
    started = time.time()
    for index, row in train.iterrows():
        class_slug = str(row.label_slug)
        path = prior_dir / class_slug / f"{row.image_id}__mucosal_gradient.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            with Image.open(str(row.image_path)) as handle:
                prior = build_mucosal_structure_prior(handle.convert("RGB"), args.prior_size)
            prior.save(path)
        with Image.open(path) as handle:
            values = np.asarray(handle.convert("L"), dtype=np.float32) / 255.0
        prior_rows.append(
            {
                "image_id": str(row.image_id),
                "label": int(row.label),
                "label_name": str(row.label_name),
                "prior_path": str(path.resolve()),
                "prior_level": 3,
                "prior_source": "deterministic real-train multi-scale mucosal gradient; no mask/annotation",
                "prior_mean": float(values.mean()),
                "prior_active_ratio_gt_0p1": float((values > 0.1).mean()),
            }
        )
        done = index + 1
        if done == 1 or done % 100 == 0 or done == len(train):
            write_json(
                output / "live_progress.json",
                {
                    "status": "weak_prior",
                    "completed": done,
                    "total": len(train),
                    "elapsed_minutes": (time.time() - started) / 60,
                },
            )
            print(f"[HK pathology6 prep] weak priors {done}/{len(train)}", flush=True)

    prior_index = pd.DataFrame(prior_rows)
    prior_map = dict(zip(prior_index.image_id.astype(str), prior_index.prior_path.astype(str)))
    split_hashes = {}
    split_sizes = {}
    for name, value in (("train", train), ("validation", validation), ("test", test)):
        split = value.copy()
        split["source_outer_fold"] = split["outer_fold"].astype(int)
        split["prior_path"] = split.image_id.astype(str).map(prior_map).fillna("")
        # Shared A0/A1/A3/A4 modules use this compatibility column.  It is a
        # weak prior, never a segmentation mask.
        split["mask_path"] = split["prior_path"]
        path = output / "splits" / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(path, index=False, encoding="utf-8-sig")
        split_hashes[name] = sha256(path)
        split_sizes[name] = len(split)

    prior_index.to_csv(output / "weak_structure_priors" / "prior_index.csv", index=False, encoding="utf-8-sig")
    train_counts = train.label_name.value_counts().reindex(HYPERKVASIR_PATHOLOGY6_CLASS_ORDER, fill_value=0).astype(int)
    target = int(train_counts.max())
    targets = {name: target - int(train_counts[name]) for name in HYPERKVASIR_PATHOLOGY6_CLASS_ORDER}
    summary = {
        "status": "completed",
        "dataset": "HyperKvasir pathology-only six-class Raw A0-A4 chain",
        "source_input_protocol_root": str(input_root),
        "input_protocol_selection": "Raw",
        "class_names": HYPERKVASIR_PATHOLOGY6_CLASS_ORDER,
        "class_counts_all": frame.label_name.value_counts().reindex(HYPERKVASIR_PATHOLOGY6_CLASS_ORDER, fill_value=0).astype(int).to_dict(),
        "generation_test_fold": test_fold,
        "generation_validation_fold": validation_fold,
        "generation_train_folds": [fold for fold in range(5) if fold not in {test_fold, validation_fold}],
        "fixed_split_sizes": split_sizes,
        "train_class_counts": {name: int(train_counts[name]) for name in HYPERKVASIR_PATHOLOGY6_CLASS_ORDER},
        "train_max_count": target,
        "synthetic_targets_100pct": targets,
        "synthetic_budget_100pct": int(sum(targets.values())),
        "split_hashes": split_hashes,
        "source_all_samples_sha256": sha256(input_root / "all_samples.csv"),
        "prior_count": len(prior_index),
        "prior_source": "train-only deterministic multi-scale mucosal gradient; no GT mask",
        "test_accessed_by_preparation": False,
        "raw_source_unchanged": True,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    write_json(output / "dataset_summary.json", summary)
    config = {
        "input_root": str(input_root),
        "output_dir": str(output),
        "generation_test_fold": test_fold,
        "generation_validation_fold": validation_fold,
        "generation_train_folds": summary["generation_train_folds"],
        "seed": args.seed,
        "prior_size": args.prior_size,
        "test_accessed": False,
        "split_hashes": split_hashes,
    }
    write_json(output / "generation_boundary_config.json", config)
    report = [
        "# HyperKvasir pathology6 Raw A0-A4 固定生成边界",
        "",
        "本目录用于新的六类 pathology-only Raw 主线。fold 0 为最终 test，fold 1 为 validation，fold 2/3/4 为生成 train；生成模型不会读取 fold 0/1 图像。",
        "",
        f"总样本={len(frame)}；train/validation/test={len(train)}/{len(validation)}/{len(test)}；100% synthetic budget={sum(targets.values())}。",
        "",
        "| 类别 | Train | Validation | Test | 100% synthetic |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in HYPERKVASIR_PATHOLOGY6_CLASS_ORDER:
        report.append(
            f"| {name} | {int((train.label_name == name).sum())} | {int((validation.label_name == name).sum())} | "
            f"{int((test.label_name == name).sum())} | {targets[name]} |"
        )
    report += [
        "",
        "A1/A3/A4 使用 train-only mucosal-gradient weak prior；`mask_path` 只是兼容旧模块的列名，不代表人工 segmentation mask。Raw 图像源保持不变。",
        "",
        "复现：",
        "```bash",
        "python training/prepare_hyperkvasir_pathology6_generation.py",
        "```",
    ]
    (output / "固定生成边界报告.md").write_text("\n".join(report), encoding="utf-8")
    write_json(completed, summary)
    write_json(output / "live_progress.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
