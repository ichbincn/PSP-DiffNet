from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from utils.common import ensure_dir, save_json


def _safe_auc(binary_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(binary_true)) < 2:
        return None
    return float(roc_auc_score(binary_true, scores))


def _safe_auprc(binary_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(binary_true)) < 2:
        return None
    return float(average_precision_score(binary_true, scores))


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: np.ndarray,
    class_names: List[str],
    minority_classes: List[str],
    output_dir: str | Path,
    prefix: str,
    include_ordinal_agreement: bool = True,
) -> Dict[str, object]:
    output_dir = ensure_dir(output_dir)
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    y_prob_np = np.asarray(y_prob)
    binary_true = label_binarize(y_true_np, classes=np.arange(len(class_names)))

    per_class_rows = []
    per_class_auc = []
    per_class_auprc = []
    for index, class_name in enumerate(class_names):
        class_true = (y_true_np == index).astype(int)
        class_pred = (y_pred_np == index).astype(int)
        precision = float(precision_score(class_true, class_pred, zero_division=0))
        recall = float(recall_score(class_true, class_pred, zero_division=0))
        f1 = float(f1_score(class_true, class_pred, zero_division=0))
        auc = _safe_auc(class_true, y_prob_np[:, index])
        auprc = _safe_auprc(class_true, y_prob_np[:, index])
        per_class_auc.append(auc)
        per_class_auprc.append(auprc)
        per_class_rows.append(
            {
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auroc": auc,
                "auprc": auprc,
                "is_minority": class_name in minority_classes,
            }
        )

    macro_auroc_values = [value for value in per_class_auc if value is not None]
    macro_auprc_values = [value for value in per_class_auprc if value is not None]
    metrics = {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_np, y_pred_np)),
        "macro_f1": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "macro_auroc": float(np.mean(macro_auroc_values)) if macro_auroc_values else None,
        "macro_auprc": float(np.mean(macro_auprc_values)) if macro_auprc_values else None,
        "per_class": per_class_rows,
    }
    if include_ordinal_agreement:
        metrics["quadratic_weighted_kappa"] = float(
            cohen_kappa_score(y_true_np, y_pred_np, weights="quadratic")
        )

    minority_rows = [row for row in per_class_rows if row["class_name"] in minority_classes]
    metrics["minority_avg_recall"] = float(np.mean([row["recall"] for row in minority_rows])) if minority_rows else None
    metrics["minority_avg_f1"] = float(np.mean([row["f1"] for row in minority_rows])) if minority_rows else None

    cm = confusion_matrix(y_true_np, y_pred_np, labels=np.arange(len(class_names)))
    cm_path = output_dir / f"{prefix}_confusion_matrix.png"
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()

    roc_path = output_dir / f"{prefix}_roc_curves.png"
    plt.figure(figsize=(7, 6))
    for index, class_name in enumerate(class_names):
        class_true = (y_true_np == index).astype(int)
        if len(np.unique(class_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(class_true, y_prob_np[:, index])
        plt.plot(fpr, tpr, label=class_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    pr_path = output_dir / f"{prefix}_pr_curves.png"
    plt.figure(figsize=(7, 6))
    for index, class_name in enumerate(class_names):
        class_true = (y_true_np == index).astype(int)
        if len(np.unique(class_true)) < 2:
            continue
        precision_curve, recall_curve, _ = precision_recall_curve(class_true, y_prob_np[:, index])
        plt.plot(recall_curve, precision_curve, label=class_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=200)
    plt.close()

    pd.DataFrame(per_class_rows).to_csv(output_dir / f"{prefix}_per_class_metrics.csv", index=False)
    save_json(output_dir / f"{prefix}_metrics.json", metrics)
    return metrics
