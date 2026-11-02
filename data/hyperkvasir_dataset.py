from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from data.imbalance_utils import build_imbalance_plan


HYPERKVASIR_UPPER_CLASS_ORDER = [
    "pylorus",
    "retroflex-stomach",
    "z-line",
    "barretts",
    "barretts-short-segment",
    "esophagitis-a",
    "esophagitis-b-d",
]

HYPERKVASIR_UPPER_CLASS_SLUGS = [name.replace("-", "_") for name in HYPERKVASIR_UPPER_CLASS_ORDER]

HYPERKVASIR_UPPER_PROMPTS = {
    "pylorus": "high quality clinical upper gastrointestinal endoscopy image of the pylorus, realistic gastric mucosa and illumination",
    "retroflex-stomach": "high quality clinical upper gastrointestinal endoscopy image in retroflex view of the stomach, realistic gastric folds and mucosa",
    "z-line": "high quality clinical upper gastrointestinal endoscopy image of a normal gastroesophageal z-line, realistic mucosa and illumination",
    "barretts": "high quality clinical upper gastrointestinal endoscopy image of Barrett esophagus, realistic mucosal morphology and illumination",
    "barretts-short-segment": "high quality clinical upper gastrointestinal endoscopy image of short segment Barrett esophagus, realistic mucosal morphology",
    "esophagitis-a": "high quality clinical upper gastrointestinal endoscopy image of Los Angeles grade A esophagitis, realistic mucosal lesion",
    "esophagitis-b-d": "high quality clinical upper gastrointestinal endoscopy image of Los Angeles grade B to D esophagitis, realistic mucosal lesion and severity",
}

# The four pathological categories are the train-set minority diagnostic classes.
HYPERKVASIR_UPPER_MINORITY = [
    "barretts",
    "barretts-short-segment",
    "esophagitis-a",
    "esophagitis-b-d",
]


def _valid_image_paths(dataset_dir: Path) -> List[Path]:
    suffixes = {".jpg", ".jpeg", ".png"}
    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in suffixes
    )


def build_hyperkvasir_upper_dataframe(dataset_dir: str | Path) -> pd.DataFrame:
    """Build the verified Upper-GI seven-class frame from CSV and local files."""
    dataset_dir = Path(dataset_dir).resolve()
    label_file = dataset_dir / "image-labels.csv"
    if not label_file.is_file():
        raise FileNotFoundError(label_file)
    labels = pd.read_csv(label_file, encoding="utf-8-sig")
    required = {"Video file", "Organ", "Finding", "Classification"}
    if not required.issubset(labels.columns):
        raise ValueError(f"HyperKvasir label columns missing: {required - set(labels.columns)}")
    labels = labels[labels["Organ"].eq("Upper GI")].copy()
    paths = _valid_image_paths(dataset_dir / "upper-gi-tract")
    by_stem = {path.stem: path for path in paths}
    if len(by_stem) != len(paths):
        raise RuntimeError("Duplicate valid Upper-GI image stems")
    rows: List[Dict[str, object]] = []
    class_to_label = {name: index for index, name in enumerate(HYPERKVASIR_UPPER_CLASS_ORDER)}
    for row in labels.itertuples(index=False):
        image_id = str(getattr(row, "_0"))  # CSV column `Video file` is first and contains the image UUID.
        finding = str(row.Finding)
        if finding not in class_to_label:
            raise RuntimeError(f"Unexpected Upper-GI finding: {finding}")
        image_path = by_stem.get(image_id)
        if image_path is None:
            raise FileNotFoundError(f"Missing HyperKvasir image for CSV ID {image_id}")
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path.resolve()),
                "label": class_to_label[finding],
                "label_name": finding,
                "label_slug": HYPERKVASIR_UPPER_CLASS_SLUGS[class_to_label[finding]],
                "organ": "Upper GI",
                "classification": str(row.Classification),
                "finding": finding,
            }
        )
    frame = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    expected_ids = set(labels["Video file"].astype(str))
    if set(frame.image_id) != expected_ids or len(frame) != len(labels):
        raise RuntimeError("Upper-GI CSV/image one-to-one audit failed")
    if set(frame.label_name) != set(HYPERKVASIR_UPPER_CLASS_ORDER):
        raise RuntimeError("Upper-GI seven-class protocol is incomplete")
    return frame


def scan_hyperkvasir_dataset(dataset_dir: str | Path) -> Dict[str, object]:
    dataset_dir = Path(dataset_dir)
    label_file = dataset_dir / "image-labels.csv"
    if not label_file.exists():
        return {
            "dataset_name": "hyperkvasir",
            "dataset_dir": str(dataset_dir.resolve()),
            "status": "error",
            "message": "HyperKvasir label file image-labels.csv was not found.",
        }
    dataframe = pd.read_csv(label_file, encoding="utf-8-sig")
    if "Finding" not in dataframe.columns:
        raise ValueError("HyperKvasir image-labels.csv must contain a Finding column.")
    image_files = _valid_image_paths(dataset_dir)
    upper = dataframe[dataframe["Organ"].eq("Upper GI")]
    class_counts = upper["Finding"].value_counts().reindex(HYPERKVASIR_UPPER_CLASS_ORDER).astype(int).to_dict()
    imbalance = build_imbalance_plan(class_counts)
    return {
        "dataset_name": "hyperkvasir_upper_gi_7class",
        "dataset_dir": str(dataset_dir.resolve()),
        "status": "ok",
        "total_images_all_labeled": int(len(image_files)),
        "total_images": int(len(upper)),
        "num_classes": len(HYPERKVASIR_UPPER_CLASS_ORDER),
        "class_counts": class_counts,
        "class_names": HYPERKVASIR_UPPER_CLASS_ORDER,
        "label_files": [str(label_file.resolve())],
        "mask_candidates": [],
        "prior_level": 3,
        "imbalance": imbalance,
        "notes": [
            "The formal protocol uses Upper GI only; Lower GI is excluded.",
            "AppleDouble files prefixed with ._ are excluded from all counts.",
            "No pixel-level mask is available; A1 uses a deterministic train-image-derived weak structure prior.",
        ],
    }
