from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.aptos_dataset import build_aptos_dataframe
from data.dataset_registry import discover_datasets
from data.hyperkvasir_dataset import scan_hyperkvasir_dataset
from data.imbalance_utils import build_imbalance_plan
from data.mmotu_dataset import scan_mmotu_dataset
from utils.common import ensure_dir, save_json, write_text


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_EXTENSIONS = {".csv", ".xlsx", ".txt", ".json"}
MASK_KEYWORDS = {"mask", "masks", "segmentation", "annotations", "groundtruth", "gt", "label"}


def _scan_generic_assets(dataset_dir: Path) -> Dict[str, List[str]]:
    image_files = []
    label_files = []
    mask_candidates = []
    for path in dataset_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            image_files.append(str(path.resolve()))
        if suffix in LABEL_EXTENSIONS:
            label_files.append(str(path.resolve()))
        if any(keyword in path.name.lower() for keyword in MASK_KEYWORDS):
            mask_candidates.append(str(path.resolve()))
    return {
        "image_files": image_files,
        "label_files": label_files,
        "mask_candidates": mask_candidates,
    }


def scan_aptos_dataset(dataset_dir: str | Path, debug_cap: int | None = None) -> Dict[str, object]:
    dataset_dir = Path(dataset_dir)
    dataframe = build_aptos_dataframe(dataset_dir)
    class_counts = dataframe["label_name"].value_counts().sort_index().to_dict()
    imbalance = build_imbalance_plan(class_counts, debug_cap=debug_cap)
    assets = _scan_generic_assets(dataset_dir)
    return {
        "dataset_name": "aptos2019",
        "dataset_dir": str(dataset_dir.resolve()),
        "status": "ok",
        "total_images": int(len(dataframe)),
        "num_classes": int(len(class_counts)),
        "class_counts": {key: int(value) for key, value in class_counts.items()},
        "label_files": assets["label_files"],
        "mask_candidates": assets["mask_candidates"],
        "prior_level": 3,
        "label_source": "folder_names",
        "label_column": "label_name",
        "notes": [
            "APTOS2019 labels were inferred from the class folder names.",
            "No lesion mask is expected; GradCAM prior (PSPH Level 3) will be used.",
        ],
        "imbalance": imbalance,
    }


def render_imbalance_markdown(summary: Dict[str, object]) -> str:
    imbalance = summary["imbalance"]
    lines = [
        f"# {summary['dataset_name']} imbalance report",
        "",
        f"- Dataset path: `{summary['dataset_dir']}`",
        f"- Total images: {summary.get('total_images', 0)}",
        f"- Num classes: {summary.get('num_classes', 0)}",
        f"- Prior level: {summary.get('prior_level', 'unknown')}",
        f"- Max count: {imbalance['max_count']}",
        f"- Min count: {imbalance['min_count']}",
        f"- Imbalance ratio: {imbalance['imbalance_ratio']:.4f}",
        "",
        "## Class counts",
    ]
    for class_name, count in summary["class_counts"].items():
        planned = imbalance["augmentation_targets"].get(class_name, 0)
        lines.append(f"- {class_name}: {count} samples, planned synthetic samples = {planned}")
    if summary.get("notes"):
        lines.append("")
        lines.append("## Notes")
        lines.extend(f"- {note}" for note in summary["notes"])
    return "\n".join(lines) + "\n"


def scan_all_datasets(root: str | Path, output_dir: str | Path, debug_cap: int | None = None) -> Dict[str, object]:
    root = Path(root)
    output_dir = ensure_dir(output_dir)
    discovered = discover_datasets(root)
    if not discovered:
        raise FileNotFoundError(f"No supported datasets were found in: {root}")

    results = {}
    for item in discovered:
        name, dataset_dir = item["name"], Path(item["path"])
        if name == "aptos2019":
            summary = scan_aptos_dataset(dataset_dir, debug_cap=debug_cap)
        elif name == "hyperkvasir":
            summary = scan_hyperkvasir_dataset(dataset_dir)
        elif name == "mmotu2d":
            summary = scan_mmotu_dataset(dataset_dir)
        else:
            continue

        dataset_output_dir = ensure_dir(output_dir / name)
        save_json(dataset_output_dir / "dataset_summary.json", summary)
        pd.DataFrame(
            [{"class_name": key, "count": value} for key, value in summary.get("class_counts", {}).items()]
        ).to_csv(dataset_output_dir / "dataset_summary.csv", index=False)
        write_text(dataset_output_dir / "imbalance_report.md", render_imbalance_markdown(summary))
        results[name] = summary

    save_json(output_dir / "dataset_scan_summary.json", results)
    overview_rows = []
    for dataset_name, summary in results.items():
        overview_rows.append(
            {
                "dataset_name": dataset_name,
                "dataset_dir": summary["dataset_dir"],
                "total_images": summary.get("total_images", 0),
                "num_classes": summary.get("num_classes", 0),
                "prior_level": summary.get("prior_level", "unknown"),
                "imbalance_ratio": summary.get("imbalance", {}).get("imbalance_ratio", 0.0),
                "status": summary.get("status", "unknown"),
            }
        )
    pd.DataFrame(overview_rows).to_csv(output_dir / "dataset_scan_summary.csv", index=False)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan all supported medical imaging datasets.")
    parser.add_argument("--root", required=True, help="Root directory containing dataset folders.")
    parser.add_argument("--output", required=True, help="Directory where summary files will be written.")
    parser.add_argument("--debug-cap", type=int, default=None, help="Optional cap for planned synthetic samples per class.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scan_all_datasets(root=args.root, output_dir=args.output, debug_cap=args.debug_cap)
