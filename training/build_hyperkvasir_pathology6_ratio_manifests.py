"""Build nested Raw HyperKvasir pathology6 ratio manifests from frozen A4.

Ratios 20--100% are nested subsets of the same A4 pool; no diffusion sampling
is repeated.  The ranking is deterministic and uses only fields produced by
the frozen A4 train-only controller (quality/diagnostic/coverage provenance).
Ratio 0% is represented by an empty manifest and is handled as Original-only
by the downstream runner.  Ratio 120% is produced separately by the A4
extension stage because it requires new images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASS_ORDER = ["Barrett", "Esophagitis", "Polyps", "UC-mild", "UC-moderate", "UC-severe"]
RATIOS = [0, 20, 40, 60, 80, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build nested HyperKvasir pathology6 A4 ratio manifests")
    parser.add_argument("--a4-manifest", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def numeric_rank(frame: pd.DataFrame, column: str, descending: bool = True) -> np.ndarray:
    if column not in frame.columns:
        return np.zeros(len(frame), dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.notna().any():
        values = values.fillna(float(values.median()))
    else:
        values = pd.Series(np.zeros(len(frame)), index=frame.index)
    return values.rank(method="average", pct=True).to_numpy(float) if descending else (-values).rank(method="average", pct=True).to_numpy(float)


def image_hash(path: str) -> np.uint64:
    with Image.open(path) as handle:
        image = handle.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=np.float32).reshape(-1)
    bits = values >= values.mean()
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return np.uint64(result)


def rank_class(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    work = frame.copy().reset_index(drop=True)
    if work.empty:
        work["ratio_priority_rank"] = np.asarray([], dtype=np.int64)
        return work
    quality = (
        0.40 * numeric_rank(work, "quality_score")
        + 0.20 * numeric_rank(work, "teacher_probability")
        + 0.15 * numeric_rank(work, "teacher_agreement")
        + 0.10 * numeric_rank(work, "realism_probability")
        + 0.10 * numeric_rank(work, "structure_probability")
        + 0.05 * numeric_rank(work, "origin", descending=True)
    )
    if "quality_gate_pass" in work:
        quality += 0.40 * work["quality_gate_pass"].fillna(False).astype(bool).to_numpy(float)
    if "selected" in work:
        quality += 0.20 * work["selected"].fillna(False).astype(bool).to_numpy(float)
    hashes = np.asarray([image_hash(path) for path in work.image_path.astype(str)], dtype=np.uint64)
    source = work.get("source_image_id", work.image_id).astype(str).to_numpy()
    ids = work.image_id.astype(str).to_numpy()
    remaining = np.ones(len(work), dtype=bool)
    min_distance = np.full(len(work), 64.0, dtype=float)
    seen_sources: set[str] = set()
    order: list[int] = []
    rng = np.random.default_rng(seed)
    tie_noise = rng.uniform(0, 1e-10, len(work))
    for _ in range(len(work)):
        new_source = np.asarray([item not in seen_sources for item in source], dtype=float)
        score = quality + 0.08 * new_source + 0.08 * min_distance / 64.0 + tie_noise
        score[~remaining] = -np.inf
        max_score = float(score.max())
        tied = np.flatnonzero(np.isclose(score, max_score, atol=1e-12, rtol=0.0))
        chosen = int(tied[np.argmin(ids[tied])])
        order.append(chosen)
        remaining[chosen] = False
        seen_sources.add(source[chosen])
        xor = np.bitwise_xor(hashes, hashes[chosen]).view(np.uint8).reshape(-1, 8)
        distance = np.unpackbits(xor, axis=1).sum(axis=1).astype(float)
        min_distance = np.minimum(min_distance, distance)
    rank = np.empty(len(work), dtype=np.int64)
    rank[np.asarray(order)] = np.arange(1, len(work) + 1)
    work["ratio_priority_rank"] = rank
    work["ratio_priority_score"] = quality
    work["ratio_priority_policy"] = (
        "A4 quality/diagnostic score -> quality gate and selected provenance -> "
        "source diversity -> perceptual non-redundancy; deterministic nested subsets"
    )
    return work.sort_values(["ratio_priority_rank", "image_id"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    a4_path = (ROOT / args.a4_manifest).resolve()
    split_root = (ROOT / args.split_root).resolve()
    output = (ROOT / args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(a4_path).reset_index(drop=True)
    required = {"image_id", "image_path", "label", "label_name"}
    if not required.issubset(manifest.columns):
        raise RuntimeError(f"A4 manifest missing columns: {sorted(required - set(manifest.columns))}")
    if not set(manifest.label_name.astype(str)).issubset(set(CLASS_ORDER)):
        raise RuntimeError("Unexpected pathology6 class in A4 manifest")
    parts = [rank_class(manifest[manifest.label_name.eq(name)], args.seed + idx) for idx, name in enumerate(CLASS_ORDER)]
    ranked = pd.concat(parts, ignore_index=True) if parts else manifest.iloc[:0].copy()
    ranked.to_csv(output / "frozen_a4_priority_manifest.csv", index=False, encoding="utf-8-sig")
    summary = {
        "source_a4_manifest": str(a4_path),
        "source_pool_size": int(len(manifest)),
        "ratios": {},
        "selection": "nested deterministic quality/diagnostic/coverage-provenance priority",
        "generation_rerun": False,
        "test_accessed": False,
    }
    split_names = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
    source_hash = hashlib.sha256(a4_path.read_bytes()).hexdigest()
    for ratio in RATIOS:
        tag = f"ratio_{ratio:03d}"
        destination = output / tag
        (destination / "splits").mkdir(parents=True, exist_ok=True)
        for name, filename in split_names.items():
            source = split_root / filename
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, destination / "splits" / filename)
        selected_parts = []
        per_class = {}
        if ratio:
            fraction = ratio / 100.0
            for name in CLASS_ORDER:
                class_frame = ranked[ranked.label_name.eq(name)]
                count = int(round(len(class_frame) * fraction))
                per_class[name] = count
                selected_parts.append(class_frame.head(count))
        else:
            per_class = {name: 0 for name in CLASS_ORDER}
        if selected_parts:
            selected = pd.concat(selected_parts, ignore_index=True).sort_values(["label", "ratio_priority_rank", "image_id"]).reset_index(drop=True)
        else:
            selected = ranked.iloc[:0].copy()
        selected.to_csv(destination / "generated_manifest.csv", index=False, encoding="utf-8-sig")
        config = {
            "ratio_percent": ratio,
            "ratio": ratio / 100.0,
            "source_a4_manifest": str(a4_path),
            "source_a4_manifest_sha256": source_hash,
            "pool_size": int(len(selected)),
            "per_class": per_class,
            "nested": True,
            "generation_rerun": False,
            "test_accessed": False,
            "ratio0_is_original_only": ratio == 0,
        }
        write_json(destination / "run_config.json", config)
        summary["ratios"][tag] = {"pool_size": int(len(selected)), "per_class": per_class}
        print(f"[HK pathology6 ratios] {tag}: {len(selected)} images {per_class}", flush=True)
    write_json(output / "nested_subset_summary.json", summary)
    write_json(output / "completed.json", {"status": "completed", **summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
