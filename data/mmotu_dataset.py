from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from data.imbalance_utils import build_imbalance_plan


MMOTU_CLASS_ORDER = [
    "Chocolate cyst",
    "Serous cystadenoma",
    "Teratoma",
    "Theca cell tumor",
    "Simple cyst",
    "Normal ovary",
    "Mucinous cystadenoma",
    "High-grade serous carcinoma",
]

MMOTU_CLASS_SLUGS = [
    "chocolate_cyst",
    "serous_cystadenoma",
    "teratoma",
    "theca_cell_tumor",
    "simple_cyst",
    "normal_ovary",
    "mucinous_cystadenoma",
    "high_grade_serous_carcinoma",
]


def _read_class_files(dataset_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for split_name in ["train_cls.txt", "val_cls.txt"]:
        split_path = dataset_dir / split_name
        if not split_path.exists():
            continue
        with split_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                filename, label = parts[0], int(parts[1])
                if not 0 <= label < len(MMOTU_CLASS_ORDER):
                    raise ValueError(f"Invalid MMOTU label {label} in {split_path}")
                image_id = Path(filename).stem
                rows.append(
                    {
                        "image_id": image_id,
                        "filename": filename,
                        "image_path": str((dataset_dir / "images" / filename).resolve()),
                        "mask_path": str((dataset_dir / "annotations" / f"{image_id}_binary.PNG").resolve()),
                        "label": label,
                        "label_name": MMOTU_CLASS_ORDER[label],
                        "label_slug": MMOTU_CLASS_SLUGS[label],
                        "split_file": split_name,
                    }
                )
    return rows


def build_mmotu_official_dataframes(dataset_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return the official 1000-image development and 469-image held-out frames."""
    dataset_dir = Path(dataset_dir)
    rows = _read_class_files(dataset_dir)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"No MMOTU class files were found in {dataset_dir}")
    if frame.image_id.duplicated().any():
        duplicated = frame.loc[frame.image_id.duplicated(keep=False), "image_id"].tolist()
        raise RuntimeError(f"Duplicated MMOTU image IDs: {duplicated[:10]}")
    development = frame[frame.split_file == "train_cls.txt"].drop(columns="split_file").reset_index(drop=True)
    held_out = frame[frame.split_file == "val_cls.txt"].drop(columns="split_file").reset_index(drop=True)
    if len(development) != 1000 or len(held_out) != 469:
        raise RuntimeError(f"Unexpected MMOTU official split sizes: {len(development)}/{len(held_out)}")
    return development, held_out


def scan_mmotu_dataset(dataset_dir: str | Path) -> Dict[str, object]:
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "annotations"
    rows = _read_class_files(dataset_dir)
    dataframe = pd.DataFrame(rows)
    class_counts = dataframe["label"].value_counts().sort_index().to_dict() if not dataframe.empty else {}
    mask_files = [
        str(path.resolve()) for path in masks_dir.rglob("*")
        if path.is_file() and not path.name.startswith("._") and path.suffix.lower() == ".png"
    ] if masks_dir.exists() else []
    image_files = [
        str(path.resolve()) for path in images_dir.rglob("*")
        if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ] if images_dir.exists() else []
    imbalance = build_imbalance_plan({str(key): int(value) for key, value in class_counts.items()})
    return {
        "dataset_name": "mmotu2d",
        "dataset_dir": str(dataset_dir.resolve()),
        "status": "ok" if image_files else "warning",
        "total_images": int(len(image_files)),
        "num_classes": int(len(class_counts)),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "label_files": [str((dataset_dir / name).resolve()) for name in ["train_cls.txt", "val_cls.txt"] if (dataset_dir / name).exists()],
        "mask_candidates": mask_files[:200],
        "prior_level": 1 if mask_files else 3,
        "imbalance": imbalance,
        "class_names": MMOTU_CLASS_ORDER,
        "notes": [
            "OTU_2d was mapped to the MMOTU_2D slot for this project.",
            "AppleDouble files prefixed with ._ are excluded from all counts.",
            "The official train_cls/val_cls boundary is retained as development/held-out test.",
        ],
    }
