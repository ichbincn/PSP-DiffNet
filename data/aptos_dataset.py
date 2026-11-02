from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from data.split_utils import save_splits, stratified_train_val_test_split


APTOS_CLASS_ORDER = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]


def build_aptos_dataframe(dataset_dir: str | Path) -> pd.DataFrame:
    dataset_dir = Path(dataset_dir)
    rows: List[Dict[str, object]] = []
    class_names = [name for name in APTOS_CLASS_ORDER if (dataset_dir / name).exists()]
    for label_index, class_name in enumerate(class_names):
        class_dir = dataset_dir / class_name
        for image_path in sorted(class_dir.iterdir()):
            # Ignore macOS AppleDouble sidecar files such as `._image.png`.
            # They carry filesystem metadata, not image pixels, but retain the
            # original extension and would otherwise silently double counts.
            if image_path.name.startswith("._"):
                continue
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                continue
            rows.append(
                {
                    "image_id": image_path.stem,
                    "image_path": str(image_path.resolve()),
                    "label": label_index,
                    "label_name": class_name,
                    "dataset_name": "aptos2019",
                }
            )
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        raise FileNotFoundError(f"No images were found in APTOS dataset: {dataset_dir}")
    return dataframe


def build_or_load_aptos_splits(config: Dict[str, object], force_rebuild: bool = False) -> Dict[str, pd.DataFrame]:
    split_dir = Path(config["paths"]["output_root"]) / "dataset_summary" / "splits"
    split_files = {name: split_dir / f"{name}.csv" for name in ["train", "val", "test"]}
    if not force_rebuild and all(path.exists() for path in split_files.values()):
        return {name: pd.read_csv(path) for name, path in split_files.items()}

    dataframe = build_aptos_dataframe(config["paths"]["dataset_dir"])
    split_cfg = config["split"]
    splits = stratified_train_val_test_split(
        dataframe=dataframe,
        label_col="label",
        seed=int(config["seed"]),
        train_ratio=float(split_cfg["train"]),
        val_ratio=float(split_cfg["val"]),
        test_ratio=float(split_cfg["test"]),
    )
    save_splits(splits, split_dir)
    return splits


@dataclass
class AptosSample:
    image_path: str
    image_id: str
    label: int
    label_name: str
    prior_path: Optional[str] = None


class AptosDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None, prior_dir: str | Path | None = None):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        self.prior_dir = Path(prior_dir) if prior_dir else None

    def __len__(self) -> int:
        return len(self.dataframe)

    def _resolve_prior_path(self, row: pd.Series) -> Optional[str]:
        if self.prior_dir is None:
            return ""
        candidate = self.prior_dir / row["label_name"] / f"{row['image_id']}_heatmap.png"
        if candidate.exists():
            return str(candidate.resolve())
        return ""

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": int(row["label"]),
            "label_name": row["label_name"],
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "prior_path": self._resolve_prior_path(row),
        }
