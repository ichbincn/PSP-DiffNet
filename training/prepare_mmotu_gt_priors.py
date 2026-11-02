from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.mmotu_dataset import MMOTU_CLASS_SLUGS
from utils.ultrasound import preprocess_ultrasound_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MMOTU Level-1 GT mask priors")
    parser.add_argument("--train-split", default="outputs/mmotu2d/dataset_preparation/splits/train.csv")
    parser.add_argument("--output-dir", default="outputs/mmotu2d/a1_structure/gt_mask_priors")
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "prior_index.csv"
    if (output_dir / "completed.json").is_file() and index_path.is_file():
        print(f"[MMOTU priors] completed: {output_dir}", flush=True)
        return
    frame = pd.read_csv(ROOT / args.train_split)
    rows = []
    for index, row in frame.iterrows():
        class_dir = output_dir / MMOTU_CLASS_SLUGS[int(row.label)]
        class_dir.mkdir(parents=True, exist_ok=True)
        prior_path = class_dir / f"{row.image_id}_gt_mask.png"
        with Image.open(row.mask_path) as handle:
            prior = preprocess_ultrasound_mask(handle, args.image_size)
        prior.save(prior_path)
        array = np.asarray(prior, dtype=np.uint8) > 0
        rows.append(
            {
                "image_id": str(row.image_id),
                "label": int(row.label),
                "label_name": str(row.label_name),
                "source_mask_path": str(Path(row.mask_path).resolve()),
                "prior_path": str(prior_path.resolve()),
                "prior_level": 1,
                "prior_source": "expert GT binary ROI mask",
                "prior_area_ratio": float(array.mean()),
            }
        )
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(frame):
            progress = {"status": "running", "completed": index + 1, "total": len(frame)}
            write_json(output_dir / "live_progress.json", progress)
            print(f"[MMOTU priors] {index + 1}/{len(frame)}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(index_path, index=False)
    completed = {
        "status": "completed",
        "count": len(result),
        "prior_level": 1,
        "prior_source": "expert GT binary ROI mask; train only",
        "mean_area_ratio": float(result.prior_area_ratio.mean()),
        "test_accessed": False,
    }
    write_json(output_dir / "completed.json", completed)
    write_json(output_dir / "live_progress.json", completed)
    print(json.dumps(completed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
