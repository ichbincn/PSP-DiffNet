"""Prepare the six-class HyperKvasir Raw/FOV/FOV+Sanitized input audit.

The transformation is deterministic and label-blind: it does not use a mask,
class, validation score, or test label.  Raw keeps the original RGB image;
FOV removes only black outer scope margins; Sanitized additionally inpaints
high-confidence small overlay components restricted to the outer border band.
The source dataset is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hyperkvasir_pathology import (
    HYPERKVASIR_PATHOLOGY6_CLASS_ORDER,
    assign_stratified_folds,
    build_hyperkvasir_pathology6_dataframe,
)


DEFAULT_DATASET = ROOT / "datasets/HyperKvasir"
DEFAULT_OUTPUT = ROOT / "outputs/hyperkvasir_pathology6/input_protocol_20260829"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HyperKvasir six-class input protocol")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _largest_contiguous(values: np.ndarray) -> Tuple[int, int] | None:
    """Return the longest true interval in a one-dimensional boolean array."""
    values = np.asarray(values, dtype=bool)
    best = None
    start = None
    for index, value in enumerate(np.r_[values, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            candidate = (start, index)
            if best is None or candidate[1] - candidate[0] > best[1] - best[0]:
                best = candidate
            start = None
    return best


def detect_fov_bbox(array: np.ndarray) -> Tuple[int, int, int, int]:
    """Detect the connected non-black field of view with a conservative margin."""
    h, w = array.shape[:2]
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    # The threshold is intentionally low: dark mucosa remains foreground;
    # only near-black outer scope borders are removed.
    binary = gray > 12
    # Circular endoscopy FOVs often touch all four image edges, so a connected
    # component bbox would remain the full rectangle.  Row/column occupancy at
    # 65% removes black corners while retaining dark mucosa.
    row_interval = _largest_contiguous(binary.mean(axis=1) > 0.65)
    col_interval = _largest_contiguous(binary.mean(axis=0) > 0.65)
    if row_interval is None or col_interval is None:
        return (0, 0, w, h)
    y0, y1 = row_interval
    x0, x1 = col_interval
    bw, bh = x1 - x0, y1 - y0
    # Do not crop normal full-frame images.  A 3% margin protects tissue near
    # the detected FOV boundary and makes the transform conservative.
    if bw * bh >= 0.97 * w * h or bw < 0.55 * w or bh < 0.55 * h:
        return (0, 0, w, h)
    mx, my = max(2, int(round(0.03 * bw))), max(2, int(round(0.03 * bh)))
    return (max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my))


def detect_overlay_mask(array: np.ndarray) -> np.ndarray:
    """Find only small, high-contrast components near the FOV border.

    The central lesion/texture area is never touched.  The mask is a heuristic
    quality-control transform, not a segmentation mask and not class-aware.
    """
    h, w = array.shape[:2]
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
    # White/bright overlay text and measurement glyphs; coloured UI glyphs are
    # included when bright and saturated.  Border restriction avoids mucosal
    # highlights inside the lesion.
    bright = ((gray >= 175) | ((hsv[:, :, 1] >= 85) & (hsv[:, :, 2] >= 145))).astype(np.uint8)
    band = np.zeros((h, w), dtype=np.uint8)
    band[: int(0.20 * h), :] = 1
    band[int(0.80 * h) :, :] = 1
    band[:, : int(0.20 * w)] = 1
    band[:, int(0.80 * w) :] = 1
    candidate = bright * band
    # Remove isolated noise but preserve glyph strokes.
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidate, 8)
    mask = np.zeros((h, w), dtype=np.uint8)
    max_component = max(64, int(0.012 * h * w))
    for index in range(1, count):
        x, y, bw, bh, area = [int(value) for value in stats[index]]
        if area < 3 or area > max_component:
            continue
        touches_border_band = x < 0.22 * w or x + bw > 0.78 * w or y < 0.22 * h or y + bh > 0.78 * h
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        textlike = aspect >= 1.35 or area <= 0.0025 * h * w or (bw <= 0.18 * w and bh <= 0.18 * h)
        if touches_border_band and textlike:
            mask[labels == index] = 255
    # Large saturated rectangular UI panels (for example the green measurement
    # widget in the lower-left) are handled separately.  Restricting this to
    # border-touching components prevents whole-lesion colour regions from
    # being interpreted as overlays.
    # Cyan/green measurement widgets are especially distinctive in this
    # dataset; use a hue restriction so red mucosal tissue is not connected
    # into the candidate component.
    hue = hsv[:, :, 0]
    colored = (
        (hsv[:, :, 1] >= 105)
        & (hsv[:, :, 2] >= 65)
        & (hue >= 35)
        & (hue <= 100)
        & (band > 0)
    ).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(colored, 8)
    for index in range(1, count):
        x, y, bw, bh, area = [int(value) for value in stats[index]]
        if area < max(20, int(0.002 * h * w)) or area > int(0.10 * h * w):
            continue
        touches_border_band = x < 0.22 * w or x + bw > 0.78 * w or y < 0.22 * h or y + bh > 0.78 * h
        if touches_border_band and bw <= 0.40 * w and bh <= 0.40 * h:
            # Cover the complete rectangular widget, not only its saturated
            # pixels; otherwise a green panel leaves a black/green outline.
            margin = max(2, int(round(0.01 * min(h, w))))
            xa, ya = max(0, x - margin), max(0, y - margin)
            xb, yb = min(w, x + bw + margin), min(h, y + bh + margin)
            mask[ya:yb, xa:xb] = 255
    # Connect nearby text strokes/measurement lines, but keep the total
    # intervention small.  This bound is an explicit guard against erasing
    # diagnostically useful tissue.
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    if float((mask > 0).mean()) > 0.12:
        return np.zeros_like(mask)
    return mask


def make_fov_and_sanitized(image_path: Path, fov_path: Path, sanitized_path: Path) -> Dict[str, object]:
    with Image.open(image_path) as handle:
        image = ImageOps.exif_transpose(handle.convert("RGB"))
        array = np.asarray(image, dtype=np.uint8)
    x0, y0, x1, y1 = detect_fov_bbox(array)
    fov_array = array[y0:y1, x0:x1]
    fov = Image.fromarray(fov_array, mode="RGB")
    fov_path.parent.mkdir(parents=True, exist_ok=True)
    sanitized_path.parent.mkdir(parents=True, exist_ok=True)
    fov.save(fov_path, quality=95, subsampling=0)
    overlay_mask = detect_overlay_mask(fov_array)
    if np.any(overlay_mask):
        sanitized_array = cv2.inpaint(cv2.cvtColor(fov_array, cv2.COLOR_RGB2BGR), overlay_mask, 3, cv2.INPAINT_TELEA)
        sanitized_array = cv2.cvtColor(sanitized_array, cv2.COLOR_BGR2RGB)
    else:
        sanitized_array = fov_array.copy()
    Image.fromarray(sanitized_array, mode="RGB").save(sanitized_path, quality=95, subsampling=0)
    return {
        "original_width": int(array.shape[1]),
        "original_height": int(array.shape[0]),
        "fov_x0": x0,
        "fov_y0": y0,
        "fov_x1": x1,
        "fov_y1": y1,
        "fov_area_fraction": float((x1 - x0) * (y1 - y0) / max(array.shape[0] * array.shape[1], 1)),
        "sanitized_overlay_fraction": float((overlay_mask > 0).mean()),
        "sanitized_overlay_pixels": int((overlay_mask > 0).sum()),
    }


def make_contact_sheet(frame: pd.DataFrame, output: Path) -> None:
    chosen = []
    for class_name in HYPERKVASIR_PATHOLOGY6_CLASS_ORDER:
        subset = frame[frame.label_name.eq(class_name)].sort_values("image_id")
        if len(subset):
            chosen.append(subset.iloc[0])
    tiles = []
    for row in chosen:
        canv = Image.new("RGB", (768, 270), "white")
        paths = [Path(row.image_path), Path(row.fov_path), Path(row.sanitized_path)]
        for index, path in enumerate(paths):
            with Image.open(path) as handle:
                tile = ImageOps.contain(handle.convert("RGB"), (250, 230))
            canv.paste(tile, (index * 256 + (250 - tile.width) // 2, 5))
        draw = ImageDraw.Draw(canv)
        draw.text((5, 240), f"{row.label_name} | {row.image_id} | Raw / FOV / Sanitized", fill="black")
        tiles.append(canv)
    columns = 1
    rows = len(tiles)
    sheet = Image.new("RGB", (768, max(rows, 1) * 270), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (0, index * 270))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=95)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = output_dir / "completed.json"
    if completed.exists() and not args.force:
        print(completed.read_text(encoding="utf-8"), flush=True)
        return
    frame = build_hyperkvasir_pathology6_dataframe(dataset_dir)
    frame = assign_stratified_folds(frame, n_splits=args.n_splits, seed=args.seed)
    frame["fov_path"] = ""
    frame["sanitized_path"] = ""
    rows: List[Dict[str, object]] = []
    started = time.time()
    for index, row in frame.iterrows():
        image_id = str(row.image_id)
        fov_path = output_dir / "preprocessed" / "fov" / f"{image_id}.jpg"
        sanitized_path = output_dir / "preprocessed" / "fov_sanitized" / f"{image_id}.jpg"
        stats = make_fov_and_sanitized(Path(row.image_path), fov_path, sanitized_path)
        record = row.to_dict()
        record.update({"fov_path": str(fov_path.resolve()), "sanitized_path": str(sanitized_path.resolve()), **stats})
        rows.append(record)
        done = index + 1
        if done == 1 or done % 100 == 0 or done == len(frame):
            progress = {"status": "preprocessing", "completed": done, "total": len(frame), "elapsed_seconds": time.time() - started}
            write_json(output_dir / "live_progress.json", progress)
            print(f"[HyperKvasir pathology6] preprocess {done}/{len(frame)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "all_samples.csv", index=False, encoding="utf-8-sig")
    frame[["image_id", "label", "label_name", "raw_finding", "outer_fold", "validation_fold", "image_path", "fov_path", "sanitized_path"]].to_csv(
        output_dir / "fold_assignments.csv", index=False, encoding="utf-8-sig"
    )
    for fold in range(args.n_splits):
        fold_dir = output_dir / "splits" / f"fold_{fold}"
        val_fold = (fold + 1) % args.n_splits
        for split_name, split_frame in (
            ("train", frame[~frame.outer_fold.isin([fold, val_fold])]),
            ("validation", frame[frame.outer_fold.eq(val_fold)]),
            ("test", frame[frame.outer_fold.eq(fold)]),
        ):
            split_dir = fold_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            split_frame.to_csv(split_dir / "samples.csv", index=False, encoding="utf-8-sig")
    make_contact_sheet(frame, output_dir / "raw_fov_sanitized_examples.jpg")
    counts = frame.groupby(["label", "label_name"]).size().reset_index(name="count")
    split_counts = []
    for fold in range(args.n_splits):
        val_fold = (fold + 1) % args.n_splits
        for split_name, subset in (
            ("train", frame[~frame.outer_fold.isin([fold, val_fold])]),
            ("validation", frame[frame.outer_fold.eq(val_fold)]),
            ("test", frame[frame.outer_fold.eq(fold)]),
        ):
            values = subset.label_name.value_counts().reindex(HYPERKVASIR_PATHOLOGY6_CLASS_ORDER, fill_value=0)
            for class_name, count in values.items():
                split_counts.append({"fold": fold, "split": split_name, "label_name": class_name, "count": int(count)})
    pd.DataFrame(split_counts).to_csv(output_dir / "fold_class_counts.csv", index=False, encoding="utf-8-sig")
    summary = {
        "status": "completed",
        "dataset": "HyperKvasir pathology-only six-class",
        "dataset_dir": str(dataset_dir),
        "n_samples": int(len(frame)),
        "class_order": HYPERKVASIR_PATHOLOGY6_CLASS_ORDER,
        "raw_to_merged": {
            "Barrett": "barretts + barretts-short-segment",
            "Esophagitis": "esophagitis-a + esophagitis-b-d",
            "Polyps": "polyps",
            "UC-mild": "ulcerative-colitis-grade-0-1 + ulcerative-colitis-grade-1",
            "UC-moderate": "ulcerative-colitis-grade-1-2 + ulcerative-colitis-grade-2",
            "UC-severe": "ulcerative-colitis-grade-2-3 + ulcerative-colitis-grade-3",
        },
        "class_counts": {str(name): int(count) for name, count in frame.label_name.value_counts().reindex(HYPERKVASIR_PATHOLOGY6_CLASS_ORDER).items()},
        "folds": args.n_splits,
        "fold_seed": args.seed,
        "fold_rule": "outer test fold=f; validation fold=(f+1)%5; train=remaining three folds",
        "input_variants": {
            "raw": "EXIF-transposed RGB source image; square aspect-preserving resize-pad is applied only in classifier loader",
            "fov": "deterministic connected non-black FOV crop with 3% margin; no label/mask/test information",
            "fov_sanitized": "FOV plus border-band high-contrast small overlay inpainting; intervention capped at 12% pixels; no central tissue suppression",
        },
        "source_unchanged": True,
        "mask_usage": False,
        "preprocessing_artifact_stats": {
            "fov_area_fraction_median": float(frame.fov_area_fraction.median()),
            "fov_area_fraction_mean": float(frame.fov_area_fraction.mean()),
            "sanitized_overlay_fraction_median": float(frame.sanitized_overlay_fraction.median()),
            "sanitized_overlay_fraction_mean": float(frame.sanitized_overlay_fraction.mean()),
            "sanitized_nonzero_images": int((frame.sanitized_overlay_pixels > 0).sum()),
        },
        "files": {
            "all_samples": str((output_dir / "all_samples.csv").resolve()),
            "fold_assignments": str((output_dir / "fold_assignments.csv").resolve()),
            "examples": str((output_dir / "raw_fov_sanitized_examples.jpg").resolve()),
        },
    }
    write_json(output_dir / "dataset_summary.json", summary)
    report_lines = [
        "# HyperKvasir pathology-only 六类 Raw / FOV / FOV+Sanitized 输入协议资料",
        "",
        "## 1. 目的与任务边界",
        "",
        "本资料只为正式输入协议 sanity experiment 服务，不训练扩散模型、不使用 synthetic image、不使用 mask。主任务保留六个医学意义一致的 pathology 类别；landmark、quality、procedure 与仅 6 张的 Hemorrhoids 排除。",
        "",
        "## 2. 六类合并",
        "",
        "| 类别 | 原始 Finding | 总数 |",
        "|---|---|---:|",
    ]
    for class_name in HYPERKVASIR_PATHOLOGY6_CLASS_ORDER:
        subset = frame[frame.label_name.eq(class_name)]
        raw = "+".join(sorted(subset.raw_finding.unique()))
        report_lines.append(f"| {class_name} | {raw} | {len(subset)} |")
    report_lines += [
        "",
        "## 3. 三种输入定义",
        "",
        "- **Raw**：只做 EXIF 方向校正和 RGB 读取；分类器内统一进行 224×224 长宽比保持 resize + 黑色 padding。",
        "- **FOV**：以近黑色外圈为依据找最大连通视野，外扩 3% 保护边缘组织；没有类别、mask、validation 或 test 信息。",
        "- **FOV+Sanitized**：在 FOV 内仅对外侧 20% 边带内高对比、小连通覆盖物执行 Telea inpainting；干预面积超过 12% 时整张跳过，中央病灶纹理不处理。这是保守的 UI/文字敏感性变体，不是人工标注 mask。",
        "",
        "## 4. 划分协议",
        "",
        f"StratifiedKFold(n_splits={args.n_splits}, shuffle=True, random_state={args.seed})。对外层 fold f，test=f，validation=(f+1)%5，train 为其余三折；每个样本恰好作为 test 一次。所有三种输入读取完全相同的 image_id/fold 清单。",
        "",
        "## 5. 复现",
        "",
        "```bash",
        f"python -u training/prepare_hyperkvasir_pathology6_input.py --dataset-dir datasets/HyperKvasir --output-dir {output_dir.relative_to(ROOT)}",
        "```",
        "",
        f"原始样本清单：`{(output_dir / 'all_samples.csv').relative_to(ROOT)}`；折叠清单：`{(output_dir / 'fold_assignments.csv').relative_to(ROOT)}`；示例图：`{(output_dir / 'raw_fov_sanitized_examples.jpg').relative_to(ROOT)}`。",
        "",
        "## 6. 当前状态",
        "",
        "预处理是确定性的、源数据只读。后续分类脚本在同一 fold 上分别运行 scratch 与 ImageNet-pretrained ConvNeXt-Tiny，唯一实验变量为输入变体与初始化轨道。",
    ]
    (output_dir / "输入协议预处理报告.md").write_text("\n".join(report_lines), encoding="utf-8")
    write_json(completed, summary)
    write_json(output_dir / "live_progress.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
