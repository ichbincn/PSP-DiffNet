from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.mmotu_dataset import MMOTU_CLASS_ORDER, MMOTU_CLASS_SLUGS
from training.mmotu_roi_common import load_rgb_and_mask, square_lesion_context_crop_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the formal MMOTU lesion+25% context ROI coordinate system")
    parser.add_argument("--source-split-dir", default="outputs/mmotu2d/dataset_preparation/splits")
    parser.add_argument("--output-dir", default="outputs/mmotu2d/roi_rebuild_v1/dataset")
    parser.add_argument("--margin-fraction", type=float, default=0.25)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def montage_tile(image: np.ndarray, mask: np.ndarray, row: pd.Series, geometry: Dict[str, int]) -> Image.Image:
    raw = Image.fromarray(image).convert("RGB").resize((300, 300), Image.Resampling.BILINEAR)
    overlay = np.asarray(raw).copy()
    resized_mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((300, 300), Image.Resampling.NEAREST)) > 0
    green = np.zeros_like(overlay)
    green[:, :, 1] = 255
    overlay[resized_mask] = (0.65 * overlay[resized_mask] + 0.35 * green[resized_mask]).astype(np.uint8)
    canvas = Image.new("RGB", (600, 330), "white")
    canvas.paste(raw, (0, 0))
    canvas.paste(Image.fromarray(overlay), (300, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (6, 306),
        f"{row.label_name} | ID={row.image_id} | side={geometry['crop_side']} | raw ROI / mask overlay QA",
        fill="black",
    )
    return canvas


def main() -> None:
    args = parse_args()
    source_dir = (ROOT / args.source_split_dir).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    split_dir = output_dir / "splits"
    image_root = output_dir / "images"
    mask_root = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_path = output_dir / "completed.json"
    if completed_path.is_file():
        payload = json.loads(completed_path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed" and payload.get("count") == 1469:
            print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
            return

    split_frames: Dict[str, pd.DataFrame] = {}
    geometry_rows: List[Dict[str, object]] = []
    qa_candidates: Dict[int, tuple] = {}
    total = 0
    for split in ("train", "validation", "test"):
        source_path = source_dir / f"{split}.csv"
        frame = pd.read_csv(source_path)
        rows: List[Dict[str, object]] = []
        for _, row in frame.iterrows():
            image, mask = load_rgb_and_mask(row.image_path, row.mask_path)
            roi_image, roi_mask, geometry = square_lesion_context_crop_pair(
                image, mask, margin_fraction=args.margin_fraction
            )
            label = int(row.label)
            slug = MMOTU_CLASS_SLUGS[label]
            image_path = image_root / split / slug / f"{row.image_id}.png"
            mask_path = mask_root / split / slug / f"{row.image_id}_mask.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(roi_image).save(image_path)
            Image.fromarray(roi_mask.astype(np.uint8) * 255, mode="L").save(mask_path)
            if not roi_mask.any() or roi_image.shape[:2] != roi_mask.shape:
                raise RuntimeError(f"Invalid ROI pair: {row.image_id}")
            record = row.to_dict()
            record.update(
                {
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "original_image_path": str(Path(row.image_path).resolve()),
                    "original_mask_path": str(Path(row.mask_path).resolve()),
                    "roi_protocol": "GT lesion bbox + 25% max-side context on each side; square zero-pad",
                    "roi_margin_fraction_each_side": float(args.margin_fraction),
                    "roi_crop_side": int(geometry["crop_side"]),
                }
            )
            rows.append(record)
            geometry_rows.append({"split": split, "image_id": str(row.image_id), "label": label, **geometry})
            if split == "train" and label not in qa_candidates:
                qa_candidates[label] = (roi_image, roi_mask, row, geometry)
            total += 1
            if total == 1 or total % 100 == 0:
                progress = {"status": "running", "completed": total, "total": 1469, "stage": split}
                write_json(output_dir / "live_progress.json", progress)
                print(f"[MMOTU ROI] {total}/1469 split={split}", flush=True)
        split_frames[split] = pd.DataFrame(rows)

    split_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in split_frames.items():
        frame.to_csv(split_dir / f"{split}.csv", index=False)
    geometry_frame = pd.DataFrame(geometry_rows)
    geometry_frame.to_csv(output_dir / "roi_geometry.csv", index=False)

    if set(qa_candidates) != set(range(len(MMOTU_CLASS_ORDER))):
        raise RuntimeError("Missing QA class")
    montage = Image.new("RGB", (1200, 1320), "white")
    for label in range(len(MMOTU_CLASS_ORDER)):
        tile = montage_tile(*qa_candidates[label])
        montage.paste(tile, ((label % 2) * 600, (label // 2) * 330))
    qa_dir = ROOT / "实验分析/超声数据/ROI预处理"
    qa_dir.mkdir(parents=True, exist_ok=True)
    montage_path = qa_dir / "八类ROI图像与同坐标GTmask质检.png"
    montage.save(montage_path)

    split_hashes = {split: sha256(split_dir / f"{split}.csv") for split in split_frames}
    counts = {split: int(len(frame)) for split, frame in split_frames.items()}
    class_counts = {
        split: {name: int((frame.label_name == name).sum()) for name in MMOTU_CLASS_ORDER}
        for split, frame in split_frames.items()
    }
    completed = {
        "status": "completed",
        "count": int(total),
        "counts": counts,
        "class_counts": class_counts,
        "protocol": "GT mask union bbox; context margin=25% of max bbox side on every side; square crop; zero padding",
        "margin_fraction_each_side": float(args.margin_fraction),
        "mask_usage": "deterministic geometry and aligned binary structural prior; no colored overlay enters image",
        "split_reused": True,
        "test_used_for_parameter_selection": False,
        "all_masks_nonempty": True,
        "all_image_mask_shapes_aligned": True,
        "split_sha256": split_hashes,
        "qa_montage": str(montage_path.resolve()),
    }
    write_json(output_dir / "FROZEN_ROI_PROTOCOL.json", completed)
    write_json(completed_path, completed)
    write_json(output_dir / "live_progress.json", completed)
    print(json.dumps(completed, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
