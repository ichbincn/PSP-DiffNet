from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.mmotu_dataset import MMOTU_CLASS_ORDER
from metrics.generation_metrics import (
    DistributionMetricConfig,
    build_inception_feature_extractor,
    build_vgg16_feature_extractor,
    compute_distribution_metrics,
)
from utils.ultrasound import preprocess_ultrasound_pil

REAL_REFERENCE_DESCRIPTION = "fixed MMOTU real train, classes with generated samples"
PREPROCESSING_DESCRIPTION = "aspect-preserving resize-pad for real and generated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMOTU FID/KID/Improved P-R evaluation")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-split", default="outputs/mmotu2d/roi_rebuild_v1/dataset/splits/train.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--kid-subsets", type=int, default=50)
    parser.add_argument("--kid-subset-size", type=int, default=500)
    parser.add_argument("--ipr-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class UltrasoundPathDataset(Dataset):
    def __init__(self, paths: Sequence[str], size: int, mean: Sequence[float], std: Sequence[float]):
        self.paths = list(paths)
        self.size = size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as handle:
            image = preprocess_ultrasound_pil(handle.convert("RGB"), self.size)
        return self.transform(image)


def extract(paths, model, size, mean, std, device, config, stage, progress_path, started):
    loader = DataLoader(
        UltrasoundPathDataset(paths, size, mean, std),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        generator=torch.Generator().manual_seed(config.seed),
    )
    batches = []
    completed = 0
    with torch.inference_mode():
        for images in loader:
            images = images.to(device, non_blocking=True)
            features = model(images)
            if isinstance(features, tuple):
                features = features[0]
            batches.append(features.float().cpu().numpy())
            completed += len(images)
            write_json(
                progress_path,
                {
                    "status": "running",
                    "stage": stage,
                    "completed": completed,
                    "total": len(paths),
                    "elapsed_minutes": (time.time() - started) / 60,
                },
            )
    return np.concatenate(batches).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MMOTU quality evaluation requires CUDA")
    device = torch.device("cuda")
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "completed.json").is_file() and (output_dir / "summary_metrics.json").is_file():
        print(f"[MMOTU quality] completed: {output_dir}", flush=True)
        return
    started = time.time()
    progress_path = output_dir / "live_progress.json"
    manifest = pd.read_csv(ROOT / args.manifest).reset_index(drop=True)
    real_all = pd.read_csv(ROOT / args.train_split).reset_index(drop=True)
    generated_classes = [name for name in MMOTU_CLASS_ORDER if (manifest.label_name == name).any()]
    real = real_all[real_all.label_name.isin(generated_classes)].reset_index(drop=True)
    if len(manifest) < 4 or len(real) < 4:
        raise RuntimeError("Insufficient MMOTU quality samples")
    if not all(Path(path).is_file() for path in manifest.image_path.astype(str)):
        raise FileNotFoundError("Missing generated image")

    config = DistributionMetricConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        kid_subsets=args.kid_subsets,
        kid_subset_size=args.kid_subset_size,
        ipr_k=args.ipr_k,
        distance_chunk_size=256,
        seed=args.seed,
    )
    inception = build_inception_feature_extractor(device)
    real_inception = extract(real.image_path.tolist(), inception, 299, [0, 0, 0], [1, 1, 1], device, config, "real_inception", progress_path, started)
    generated_inception = extract(manifest.image_path.tolist(), inception, 299, [0, 0, 0], [1, 1, 1], device, config, "generated_inception", progress_path, started)
    del inception
    torch.cuda.empty_cache()
    vgg = build_vgg16_feature_extractor(device)
    real_vgg = extract(real.image_path.tolist(), vgg, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], device, config, "real_vgg", progress_path, started)
    generated_vgg = extract(manifest.image_path.tolist(), vgg, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], device, config, "generated_vgg", progress_path, started)
    del vgg
    torch.cuda.empty_cache()

    np.savez_compressed(
        output_dir / "feature_cache.npz",
        real_inception=real_inception,
        generated_inception=generated_inception,
        real_vgg16=real_vgg,
        generated_vgg16=generated_vgg,
        real_labels=real.label_name.to_numpy(),
        generated_labels=manifest.label_name.to_numpy(),
        real_paths=real.image_path.to_numpy(),
        generated_paths=manifest.image_path.to_numpy(),
        real_ids=real.image_id.astype(str).to_numpy(),
        source_ids=(
            manifest["source_image_id"].astype(str).to_numpy()
            if "source_image_id" in manifest.columns
            else np.full(len(manifest), "", dtype=str)
        ),
    )
    rows: List[dict] = []
    overall = compute_distribution_metrics(real_inception, generated_inception, real_vgg, generated_vgg, device, config)
    rows.append({"scope": "Overall_generated_classes", **overall})
    real_labels = real.label_name.to_numpy()
    generated_labels = manifest.label_name.to_numpy()
    eligible = []
    for index, class_name in enumerate(generated_classes):
        real_mask = real_labels == class_name
        generated_mask = generated_labels == class_name
        if real_mask.sum() <= args.ipr_k or generated_mask.sum() <= args.ipr_k:
            rows.append(
                {
                    "scope": class_name,
                    "real_count": int(real_mask.sum()),
                    "generated_count": int(generated_mask.sum()),
                    "fid": None,
                    "kid_mean": None,
                    "kid_std": None,
                    "kid_subset_size": int(min(real_mask.sum(), generated_mask.sum())),
                    "improved_precision": None,
                    "improved_recall": None,
                    "note": f"insufficient for IPR k={args.ipr_k}",
                }
            )
            continue
        metrics = compute_distribution_metrics(
            real_inception[real_mask],
            generated_inception[generated_mask],
            real_vgg[real_mask],
            generated_vgg[generated_mask],
            device,
            config,
            seed_offset=(index + 1) * 1000,
        )
        row = {"scope": class_name, **metrics}
        rows.append(row)
        eligible.append(row)
    macro = {
        "scope": "Macro_eligible_classes",
        "real_count": int(sum(row["real_count"] for row in eligible)),
        "generated_count": int(sum(row["generated_count"] for row in eligible)),
        "fid": float(np.mean([row["fid"] for row in eligible])),
        "kid_mean": float(np.mean([row["kid_mean"] for row in eligible])),
        "kid_std": float(np.mean([row["kid_std"] for row in eligible])),
        "kid_subset_size": "per_class",
        "improved_precision": float(np.mean([row["improved_precision"] for row in eligible])),
        "improved_recall": float(np.mean([row["improved_recall"] for row in eligible])),
        "eligible_classes": [row["scope"] for row in eligible],
    }
    rows.insert(1, macro)
    pd.DataFrame(rows).to_csv(output_dir / "metrics_by_scope.csv", index=False)
    summary = {
        "method": args.method_name,
        "protocol": {
            "real_reference": REAL_REFERENCE_DESCRIPTION,
            "preprocessing": PREPROCESSING_DESCRIPTION,
            "fid_kid": "PyTorch-FID Inception-v3 pool3 2048D",
            "ipr": "ImageNet VGG-16 second FC activation, k=3",
            "test_accessed": False,
            "small_class_policy": "per-class metric omitted when generated_count<=k; overall still includes it",
        },
        "metrics": {row["scope"]: {key: value for key, value in row.items() if key != "scope"} for row in rows},
        "elapsed_minutes": (time.time() - started) / 60,
    }
    write_json(output_dir / "summary_metrics.json", summary)
    write_json(output_dir / "run_config.json", vars(args))

    plot_rows = [row for row in rows if row["scope"] not in {"Macro_eligible_classes"} and row.get("fid") is not None]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    labels = [row["scope"] for row in plot_rows]
    axes[0].bar(labels, [row["fid"] for row in plot_rows]); axes[0].set_title("FID ↓")
    axes[1].bar(labels, [row["kid_mean"] for row in plot_rows]); axes[1].set_title("KID ↓")
    x = np.arange(len(labels)); width = 0.38
    axes[2].bar(x - width/2, [row["improved_precision"] for row in plot_rows], width, label="Precision")
    axes[2].bar(x + width/2, [row["improved_recall"] for row in plot_rows], width, label="Recall")
    axes[2].set_xticks(x, labels); axes[2].set_title("Improved P/R ↑"); axes[2].legend()
    for axis in axes:
        axis.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(output_dir / "distribution_metrics_by_class.png", dpi=220); plt.close(fig)

    completed = {"status": "completed", "method": args.method_name, "summary": summary["metrics"]["Overall_generated_classes"]}
    write_json(output_dir / "completed.json", completed)
    write_json(progress_path, completed)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
