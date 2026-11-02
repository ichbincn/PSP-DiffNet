from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


DATASET_REGISTRY: Dict[str, Dict[str, object]] = {
    "aptos2019": {
        "aliases": ["aptos2019", "aptos", "aptos_2019"],
        "expected_dir_names": ["APTOS2019"],
        "default_prior_level": 3,
        "task": "classification",
    },
    "hyperkvasir": {
        "aliases": ["hyperkvasir", "hyper_kvasir"],
        "expected_dir_names": ["HyperKvasir"],
        "default_prior_level": 2,
        "task": "classification_or_detection",
    },
    "mmotu2d": {
        "aliases": ["mmotu2d", "mmotu_2d", "otu_2d", "otu2d"],
        "expected_dir_names": ["MMOTU_2D", "OTU_2d"],
        "default_prior_level": 1,
        "task": "classification_with_masks",
    },
}


def normalize_dataset_name(name: str) -> Optional[str]:
    lowered = name.lower()
    for key, metadata in DATASET_REGISTRY.items():
        aliases = set(metadata["aliases"]) | set(metadata["expected_dir_names"])
        if lowered in {alias.lower() for alias in aliases}:
            return key
    return None


def discover_datasets(root: str | Path) -> List[Dict[str, str]]:
    root = Path(root)
    discovered = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        normalized = normalize_dataset_name(child.name)
        if normalized is not None:
            discovered.append({"name": normalized, "path": str(child.resolve())})
    return discovered


def get_dataset_metadata(name: str) -> Dict[str, object]:
    normalized = normalize_dataset_name(name)
    if normalized is None:
        raise KeyError(f"Unknown dataset name: {name}")
    return DATASET_REGISTRY[normalized]
