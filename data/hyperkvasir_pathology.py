"""Six-class pathology-only HyperKvasir protocol.

This module is deliberately separate from the legacy Upper-GI seven-class
pipeline.  It defines the frozen six-class label merge and deterministic
stratified fold assignment used by the Raw/FOV/FOV+Sanitized input audit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from sklearn.model_selection import StratifiedKFold


HYPERKVASIR_PATHOLOGY6_CLASS_ORDER = [
    "Barrett",
    "Esophagitis",
    "Polyps",
    "UC-mild",
    "UC-moderate",
    "UC-severe",
]
HYPERKVASIR_PATHOLOGY6_CLASS_SLUGS = [name.lower().replace("-", "_") for name in HYPERKVASIR_PATHOLOGY6_CLASS_ORDER]

# Polyps is the largest class; the other five classes are the diagnostic
# minority group for the requested headline Minority Recall metric.
HYPERKVASIR_PATHOLOGY6_MINORITY = [
    "Barrett",
    "Esophagitis",
    "UC-mild",
    "UC-moderate",
    "UC-severe",
]

# Class-conditioned prompts used by the six-class pathology-only diffusion
# track.  These prompts describe the clinical finding rather than any UI,
# device, or annotation content in the source screenshot.
HYPERKVASIR_PATHOLOGY6_PROMPTS = {
    "Barrett": (
        "high quality clinical gastrointestinal endoscopy image of Barrett esophagus, "
        "realistic mucosal morphology, natural endoscopic illumination"
    ),
    "Esophagitis": (
        "high quality clinical gastrointestinal endoscopy image of esophagitis, "
        "realistic inflamed mucosa and natural endoscopic illumination"
    ),
    "Polyps": (
        "high quality clinical gastrointestinal endoscopy image of a colorectal polyp, "
        "realistic mucosal morphology and natural endoscopic illumination"
    ),
    "UC-mild": (
        "high quality clinical gastrointestinal endoscopy image of mild ulcerative colitis, "
        "realistic mucosal inflammation and natural endoscopic illumination"
    ),
    "UC-moderate": (
        "high quality clinical gastrointestinal endoscopy image of moderate ulcerative colitis, "
        "realistic mucosal inflammation and natural endoscopic illumination"
    ),
    "UC-severe": (
        "high quality clinical gastrointestinal endoscopy image of severe ulcerative colitis, "
        "realistic mucosal inflammation and natural endoscopic illumination"
    ),
}

HYPERKVASIR_PATHOLOGY6_RAW_TO_MERGED: Dict[str, str] = {
    "barretts": "Barrett",
    "barretts-short-segment": "Barrett",
    "esophagitis-a": "Esophagitis",
    "esophagitis-b-d": "Esophagitis",
    "polyps": "Polyps",
    "ulcerative-colitis-grade-0-1": "UC-mild",
    "ulcerative-colitis-grade-1": "UC-mild",
    "ulcerative-colitis-grade-1-2": "UC-moderate",
    "ulcerative-colitis-grade-2": "UC-moderate",
    "ulcerative-colitis-grade-2-3": "UC-severe",
    "ulcerative-colitis-grade-3": "UC-severe",
}


def valid_image_index(dataset_dir: Path) -> Dict[str, Path]:
    """Index one local non-AppleDouble image for every UUID stem."""
    roots = (dataset_dir / "upper-gi-tract", dataset_dir / "lower-gi-tract")
    paths = [
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    index: Dict[str, Path] = {}
    for path in paths:
        if path.stem in index:
            raise RuntimeError(f"duplicate local image stem: {path.stem}")
        index[path.stem] = path.resolve()
    return index


def build_hyperkvasir_pathology6_dataframe(dataset_dir: str | Path) -> pd.DataFrame:
    """Build the merged six-class pathology frame with exact local matching."""
    dataset_dir = Path(dataset_dir).resolve()
    labels_path = dataset_dir / "image-labels.csv"
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    required = {"Video file", "Organ", "Finding", "Classification"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"HyperKvasir labels missing columns: {sorted(missing)}")
    labels = labels[labels["Classification"].eq("pathological-findings")].copy()
    labels = labels[labels["Finding"].isin(HYPERKVASIR_PATHOLOGY6_RAW_TO_MERGED)].copy()
    image_index = valid_image_index(dataset_dir)
    class_to_label = {name: index for index, name in enumerate(HYPERKVASIR_PATHOLOGY6_CLASS_ORDER)}
    rows: List[Dict[str, object]] = []
    for row in labels.itertuples(index=False):
        image_id = str(getattr(row, "_0"))
        raw_finding = str(row.Finding)
        merged = HYPERKVASIR_PATHOLOGY6_RAW_TO_MERGED[raw_finding]
        image_path = image_index.get(image_id)
        if image_path is None:
            raise FileNotFoundError(f"missing local image for pathology label {image_id}")
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "label": class_to_label[merged],
                "label_name": merged,
                "label_slug": HYPERKVASIR_PATHOLOGY6_CLASS_SLUGS[class_to_label[merged]],
                "raw_finding": raw_finding,
                "organ": str(row.Organ),
                "classification": str(row.Classification),
                "is_synthetic": False,
            }
        )
    frame = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    if frame.image_id.duplicated().any():
        raise RuntimeError("duplicate pathology image IDs")
    if set(frame.label_name) != set(HYPERKVASIR_PATHOLOGY6_CLASS_ORDER):
        raise RuntimeError("six-class pathology protocol is incomplete")
    return frame


def assign_stratified_folds(frame: pd.DataFrame, n_splits: int = 5, seed: int = 20260829) -> pd.DataFrame:
    """Assign deterministic outer folds; each fold has an adjacent validation fold."""
    output = frame.copy().reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    output["outer_fold"] = -1
    labels = output["label"].to_numpy()
    for fold, (_train_idx, test_idx) in enumerate(splitter.split(output, labels)):
        output.loc[test_idx, "outer_fold"] = fold
    if (output.outer_fold < 0).any():
        raise RuntimeError("not every pathology sample received a fold")
    output["validation_fold"] = (output["outer_fold"] + 1) % n_splits
    return output


def fold_frames(frame: pd.DataFrame, fold: int, n_splits: int = 5) -> Dict[str, pd.DataFrame]:
    test_fold = int(fold)
    val_fold = (test_fold + 1) % n_splits
    train = frame[~frame.outer_fold.isin([test_fold, val_fold])].copy()
    validation = frame[frame.outer_fold.eq(val_fold)].copy()
    test = frame[frame.outer_fold.eq(test_fold)].copy()
    if min(len(train), len(validation), len(test)) <= 0:
        raise RuntimeError(f"empty fold {fold}")
    return {"train": train.reset_index(drop=True), "validation": validation.reset_index(drop=True), "test": test.reset_index(drop=True)}
