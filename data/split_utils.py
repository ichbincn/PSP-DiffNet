from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.common import ensure_dir


def stratified_train_val_test_split(
    dataframe: pd.DataFrame,
    label_col: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, pd.DataFrame]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    train_df, holdout_df = train_test_split(
        dataframe,
        test_size=(1.0 - train_ratio),
        stratify=dataframe[label_col],
        random_state=seed,
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=relative_test,
        stratify=holdout_df[label_col],
        random_state=seed,
    )
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def save_splits(splits: Dict[str, pd.DataFrame], output_dir: str | Path) -> Dict[str, str]:
    output_dir = ensure_dir(output_dir)
    output_paths = {}
    for split_name, dataframe in splits.items():
        path = output_dir / f"{split_name}.csv"
        dataframe.to_csv(path, index=False)
        output_paths[split_name] = str(path.resolve())
    return output_paths
