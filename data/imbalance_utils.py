from __future__ import annotations

from typing import Dict, List


def compute_class_counts(records: List[Dict[str, object]], label_key: str = "label_name") -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        label_name = str(record[label_key])
        counts[label_name] = counts.get(label_name, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def build_imbalance_plan(
    class_counts: Dict[str, int],
    target_policy: str = "majority",
    target_count: int | None = None,
    target_ratio: float | None = None,
    debug_cap: int | None = None,
) -> Dict[str, object]:
    if not class_counts:
        return {
            "class_counts": {},
            "max_count": 0,
            "min_count": 0,
            "imbalance_ratio": 0.0,
            "minority_classes": [],
            "augmentation_targets": {},
        }

    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = float(max_count / max(min_count, 1))

    augmentation_targets = {}
    for class_name, count in class_counts.items():
        if target_policy == "fixed" and target_count is not None:
            desired = max(target_count, count)
        elif target_policy == "ratio" and target_ratio is not None:
            desired = max(int(round(max_count * target_ratio)), count)
        else:
            desired = max_count
        if debug_cap is not None:
            desired = min(desired, count + debug_cap)
        augmentation_targets[class_name] = max(desired - count, 0)

    minority_classes = [class_name for class_name, amount in augmentation_targets.items() if amount > 0]
    return {
        "class_counts": class_counts,
        "max_count": max_count,
        "min_count": min_count,
        "imbalance_ratio": imbalance_ratio,
        "minority_classes": minority_classes,
        "augmentation_targets": augmentation_targets,
    }
