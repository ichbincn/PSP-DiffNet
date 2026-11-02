"""Unified HyperKvasir Raw generation-quality evaluation and IPR sanity checks.

All real and synthetic paths are passed through the same EXIF/RGB -> 512px
square resize-pad adapter before the shared Inception/VGG feature extractors.
The script computes FID/KID/IPR for each generated method and the mandatory
real-vs-real, copy and flip controls plus k={3,5,10} sensitivity.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

import metrics.generation_metrics as shared
from utils.endoscopy import preprocess_endoscopy_pil


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "outputs/hyperkvasir_pathology6/ablation_raw_20260829/splits/train.csv"
DEFAULT_BASE = Path(
    os.environ.get(
        "HK_COMPARISON_ROOT",
        str(ROOT / "outputs/hyperkvasir_pathology6/comparisons/raw_20260829"),
    )
).resolve()
DEFAULT_OUT = DEFAULT_BASE / "quality"
METHOD_DIRS = {
    "A3 (ours; selected seed 20260831; final 100% A3 pool)": DEFAULT_BASE / "ours_a3_selected",
    "BAGAN-GP": DEFAULT_BASE / "bagan_gp",
    "LEGAN": DEFAULT_BASE / "legan",
    "IOIS": DEFAULT_BASE / "iois",
    "ECC-DM": DEFAULT_BASE / "ecc_dm",
}


def _endoscopy_result(image: Image.Image, crop: bool = True, mask_background: bool = True):
    del crop, mask_background
    return SimpleNamespace(image=preprocess_endoscopy_pil(image.convert("RGB"), 512))


# ImagePathDataset in the shared evaluator resolves this symbol at call time.
# Replacing only the preprocessing seam keeps its FID/KID/IPR implementation
# identical to the fundus/ultrasound comparison reports.
shared.preprocess_fundus_pil = _endoscopy_result


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_records(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).copy()
    if "image_path" not in frame.columns:
        raise RuntimeError(f"Missing image_path in {path}")
    missing = [p for p in frame.image_path.astype(str) if not Path(p).is_file()]
    if missing:
        raise RuntimeError(f"Missing image referenced by {path}: {missing[0]}")
    return frame


def manifest_for(directory: Path) -> Path | None:
    for name in ("generated_manifest.csv", "manifest.csv"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def records_for(directory: Path) -> pd.DataFrame | None:
    manifest = manifest_for(directory)
    return load_records(manifest) if manifest else None


def make_flip_control(real: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows = []
    root = output / "flip_images"
    root.mkdir(parents=True, exist_ok=True)
    for row in real.itertuples(index=False):
        target = root / f"{row.image_id}.png"
        if not target.is_file():
            with Image.open(row.image_path) as image:
                ImageOps.mirror(image.convert("RGB")).save(target)
        rows.append({"image_id": f"flip_{row.image_id}", "image_path": str(target.resolve()), "label": int(row.label), "label_name": str(row.label_name)})
    return pd.DataFrame(rows)


def feature_sets(real_paths, generated_paths, device, config, cache_path: Path, tag: str):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    key_real = f"{tag}_real_inception"
    key_gen = f"{tag}_gen_inception"
    key_real_vgg = f"{tag}_real_vgg"
    key_gen_vgg = f"{tag}_gen_vgg"
    if cache_path.is_file():
        payload = np.load(cache_path)
        if all(key in payload for key in (key_real, key_gen, key_real_vgg, key_gen_vgg)):
            return payload[key_real], payload[key_gen], payload[key_real_vgg], payload[key_gen_vgg]
    inception = shared.build_inception_feature_extractor(device)
    real_i = shared.extract_features(real_paths, inception, 299, [0.0] * 3, [1.0] * 3, device, config, f"{tag}_real_inception")
    gen_i = shared.extract_features(generated_paths, inception, 299, [0.0] * 3, [1.0] * 3, device, config, f"{tag}_gen_inception")
    del inception
    if device.type == "cuda":
        torch.cuda.empty_cache()
    vgg = shared.build_vgg16_feature_extractor(device)
    real_v = shared.extract_features(real_paths, vgg, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], device, config, f"{tag}_real_vgg")
    gen_v = shared.extract_features(generated_paths, vgg, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], device, config, f"{tag}_gen_vgg")
    del vgg
    if device.type == "cuda":
        torch.cuda.empty_cache()
    np.savez_compressed(cache_path, **{key_real: real_i, key_gen: gen_i, key_real_vgg: real_v, key_gen_vgg: gen_v})
    return real_i, gen_i, real_v, gen_v


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HyperKvasir pathology6 Raw generation quality")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--train-split", default=str(TRAIN))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--kid-subsets", type=int, default=50)
    parser.add_argument("--kid-subset-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_split = Path(args.train_split).resolve()
    real = load_records(train_split)
    real_paths = real.image_path.astype(str).tolist()
    config = shared.DistributionMetricConfig(batch_size=args.batch_size, num_workers=args.num_workers, kid_subsets=args.kid_subsets, kid_subset_size=args.kid_subset_size, ipr_k=3, distance_chunk_size=256, seed=args.seed)
    metrics_rows = []
    methods_payload = {}
    for method, directory in METHOD_DIRS.items():
        records = records_for(directory)
        if records is None:
            methods_payload[method] = {"status": "pending", "directory": str(directory)}
            continue
        generated_paths = records.image_path.astype(str).tolist()
        cache = output / "feature_cache" / f"{method.replace('/', '_')}.npz"
        real_i, gen_i, real_v, gen_v = feature_sets(real_paths, generated_paths, device, config, cache, method.replace("-", "_"))
        values = shared.compute_distribution_metrics(real_i, gen_i, real_v, gen_v, device, config)
        values.update({"method": method, "status": "completed", "native_resolution_note": "method-specific native size retained in manifest; metric input canonicalized to 512 square then feature resize"})
        methods_payload[method] = values
        metrics_rows.append(values)
    pd.DataFrame(metrics_rows).to_csv(output / "metrics_by_method.csv", index=False, encoding="utf-8-sig")

    # Four sanity controls: real-vs-real, exact copy, mild horizontal flip,
    # and k sensitivity for every generated method.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(real))
    midpoint = len(order) // 2
    real_a = real.iloc[order[:midpoint]].reset_index(drop=True)
    real_b = real.iloc[order[midpoint:]].reset_index(drop=True)
    flip = make_flip_control(real, output / "sanity")
    controls = {"real_vs_real": (real_a.image_path.astype(str).tolist(), real_b.image_path.astype(str).tolist()), "real_copy": (real_paths, real_paths), "real_flip": (real_paths, flip.image_path.astype(str).tolist())}
    sanity_rows = []
    for name, (left, right) in controls.items():
        cache = output / "feature_cache" / f"sanity_{name}.npz"
        li, ri, lv, rv = feature_sets(left, right, device, config, cache, f"sanity_{name}")
        values = shared.compute_distribution_metrics(li, ri, lv, rv, device, config)
        values.update({"control": name, "k": 3})
        sanity_rows.append(values)
    for method, directory in METHOD_DIRS.items():
        records = records_for(directory)
        if records is None:
            continue
        generated_paths = records.image_path.astype(str).tolist()
        cache = output / "feature_cache" / f"{method.replace('/', '_')}.npz"
        real_i, gen_i, real_v, gen_v = feature_sets(real_paths, generated_paths, device, config, cache, method.replace("-", "_"))
        for k in (3, 5, 10):
            precision, recall = shared.improved_precision_recall(real_v, gen_v, device, k=k, chunk_size=config.distance_chunk_size)
            sanity_rows.append({"control": method, "k": k, "real_count": len(real_i), "generated_count": len(gen_i), "improved_precision": precision, "improved_recall": recall})
    pd.DataFrame(sanity_rows).to_csv(output / "ipr_sanity_and_k_sensitivity.csv", index=False, encoding="utf-8-sig")
    write_json(output / "summary_metrics.json", {"methods": methods_payload, "sanity": sanity_rows, "protocol": {"dataset": "HyperKvasir pathology-only six-class Raw", "real_reference": str(train_split), "canonical_metric_preprocessing": "EXIF transpose + RGB + 512px aspect-preserving square resize-pad for real and synthetic, then identical feature transforms", "kid_subsets": args.kid_subsets, "kid_subset_size": args.kid_subset_size, "ipr_k_sensitivity": [3, 5, 10], "seed": args.seed, "test_accessed": False}})
    print(output / "summary_metrics.json", flush=True)


if __name__ == "__main__":
    main()
