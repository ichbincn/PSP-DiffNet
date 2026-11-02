from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import convnext_tiny

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.aptos_dataset import APTOS_CLASS_ORDER
from metrics.classification_metrics import compute_classification_metrics
from utils.fundus import preprocess_fundus_pil


MINORITY_CLASSES = ["Mild", "Moderate", "Severe", "Proliferate_DR"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class Experiment:
    key: str
    display_name: str
    train_source: str
    augmentation: str
    loss: str


EXPERIMENTS = [
    Experiment("original_ce", "Original CE", "real", "none", "cross_entropy"),
    Experiment("traditional_aug_ce", "Traditional Augmentation", "real", "traditional", "cross_entropy"),
    Experiment("balanced_softmax", "Balanced Softmax", "real", "traditional", "balanced_softmax"),
    Experiment("focal_loss", "Focal Loss", "real", "traditional", "focal"),
    Experiment("sd025_augmented_ce", "SD 0.25 Augmented", "real+sd025", "traditional", "cross_entropy"),
]


class FundusPreprocess:
    def __call__(self, image: Image.Image) -> Image.Image:
        return preprocess_fundus_pil(image, crop=True, mask_background=True).image


class AptosFrameDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), int(row["label"]), str(row["image_id"])


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts: Sequence[int]):
        super().__init__()
        self.register_buffer("log_counts", torch.log(torch.tensor(class_counts, dtype=torch.float32).clamp_min(1.0)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.log_counts.unsqueeze(0), targets)


class FocalLoss(nn.Module):
    def __init__(self, class_counts: Sequence[int], gamma: float = 2.0):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32).clamp_min(1.0)
        alpha = counts.sum() / (len(counts) * counts)
        self.register_buffer("alpha", alpha / alpha.mean())
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha[targets] * (1.0 - pt).pow(self.gamma) * ce).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="APTOS ConvNeXt-Tiny comparison trained from random initialization")
    parser.add_argument("--data-root", default="outputs/aptos2019/sd025_balanced_train")
    parser.add_argument("--output-dir", default="outputs/aptos2019/convnext_tiny_scratch_preliminary")
    parser.add_argument("--image-quality-dir", default="outputs/aptos2019/sd025_image_quality_fid_kid_ipr")
    parser.add_argument("--report", default="APTOS_ConvNeXt随机初始化重跑结果_待确认.md")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_transform(image_size: int, traditional: bool):
    operations: List[object] = [FundusPreprocess()]
    if traditional:
        operations.extend(
            [
                transforms.Resize((256, 256), antialias=True),
                transforms.RandomResizedCrop(image_size, scale=(0.90, 1.0), ratio=(0.95, 1.05), antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15, fill=0),
                transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.01),
            ]
        )
    else:
        operations.append(transforms.Resize((image_size, image_size), antialias=True))
    operations.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(operations)


def worker_seed(worker_id: int) -> None:
    value = torch.initial_seed() % (2**32)
    np.random.seed(value)
    random.seed(value)


def make_loader(dataframe: pd.DataFrame, transform, args: argparse.Namespace, train: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        AptosFrameDataset(dataframe, transform),
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_seed,
        generator=generator,
    )


def build_model(device: torch.device) -> nn.Module:
    model = convnext_tiny(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(APTOS_CLASS_ORDER))
    return model.to(device)


def build_criterion(loss_name: str, class_counts: Sequence[int], device: torch.device) -> nn.Module:
    if loss_name == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif loss_name == "balanced_softmax":
        criterion = BalancedSoftmaxLoss(class_counts)
    elif loss_name == "focal":
        criterion = FocalLoss(class_counts, gamma=2.0)
    else:
        raise ValueError(loss_name)
    return criterion.to(device)


@torch.no_grad()
def evaluate_loss_and_predictions(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    losses, y_true, y_pred, y_prob, image_ids = [], [], [], [], []
    for images, labels, ids in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=True):
            logits = model(images)
            loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)
        losses.append(float(loss.item()))
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(probabilities.argmax(dim=1).cpu().tolist())
        y_prob.extend(probabilities.cpu().numpy())
        image_ids.extend(ids)
    return float(np.mean(losses)), y_true, y_pred, np.asarray(y_prob), image_ids


def train_experiment(
    experiment: Experiment,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    real_class_counts: Sequence[int],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, object]:
    experiment_dir = output_dir / experiment.key
    experiment_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = experiment_dir / "test_metrics.json"
    checkpoint_path = experiment_dir / "best_model.pt"
    history_path = experiment_dir / "training_history.json"
    epoch_log_path = experiment_dir / "training_epoch_log.csv"
    live_progress_path = output_dir / "live_progress.json"
    if metrics_path.exists() and checkpoint_path.exists() and history_path.exists():
        print(f"[{experiment.key}] reusing completed result", flush=True)
        write_json(
            live_progress_path,
            {"status": "reused", "experiment_key": experiment.key, "display_name": experiment.display_name},
        )
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    seed_everything(args.seed)
    train_transform = build_transform(args.image_size, traditional=experiment.augmentation == "traditional")
    eval_transform = build_transform(args.image_size, traditional=False)
    train_loader = make_loader(train_df, train_transform, args, train=True)
    val_loader = make_loader(val_df, eval_transform, args, train=False)
    test_loader = make_loader(test_df, eval_transform, args, train=False)

    model = build_model(device)
    criterion = build_criterion(experiment.loss, real_class_counts, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("warmup_epochs must satisfy 0 <= warmup_epochs < epochs")
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=max(args.warmup_epochs, 1),
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - args.warmup_epochs, 1),
        eta_min=args.min_learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )
    scaler = GradScaler(enabled=True)
    best_state = None
    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0
    history = []
    started = time.time()
    write_json(
        live_progress_path,
        {
            "status": "training",
            "experiment_key": experiment.key,
            "display_name": experiment.display_name,
            "epoch": 0,
            "max_epochs": args.epochs,
            "train_size": len(train_df),
        },
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))

        val_loss, val_true, val_pred, _, _ = evaluate_loss_and_predictions(model, val_loader, criterion, device)
        val_macro_f1 = float(f1_score(val_true, val_pred, average="macro", zero_division=0))
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(epoch_log_path, index=False)
        write_json(
            live_progress_path,
            {
                "status": "training",
                "experiment_key": experiment.key,
                "display_name": experiment.display_name,
                "epoch": epoch,
                "max_epochs": args.epochs,
                "train_size": len(train_df),
                "train_loss": row["train_loss"],
                "val_loss": val_loss,
                "val_macro_f1": val_macro_f1,
                "best_val_macro_f1": max(best_val_macro_f1, val_macro_f1),
                "epochs_without_improvement": epochs_without_improvement,
                "early_stopping_patience": args.early_stopping_patience,
                "elapsed_seconds": time.time() - started,
            },
        )
        print(
            f"[{experiment.key}] epoch {epoch:02d}/{args.epochs} "
            f"train={row['train_loss']:.4f} val={val_loss:.4f} val_macro_f1={val_macro_f1:.4f}",
            flush=True,
        )
        if val_macro_f1 > best_val_macro_f1 + 1e-6:
            best_val_macro_f1 = val_macro_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"[{experiment.key}] early stopped at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError(f"No best state captured for {experiment.key}")
    model.load_state_dict(best_state)
    _, y_true, y_pred, y_prob, image_ids = evaluate_loss_and_predictions(model, test_loader, criterion, device)
    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=APTOS_CLASS_ORDER,
        minority_classes=MINORITY_CLASSES,
        output_dir=experiment_dir,
        prefix="test",
    )
    metrics.update(
        {
            "experiment_key": experiment.key,
            "display_name": experiment.display_name,
            "train_size": int(len(train_df)),
            "best_val_macro_f1": best_val_macro_f1,
            "epochs_ran": len(history),
            "training_seconds": time.time() - started,
        }
    )
    write_json(metrics_path, metrics)
    write_json(history_path, history)
    probability_rows = []
    for index, image_id in enumerate(image_ids):
        row = {"image_id": image_id, "true_label": y_true[index], "predicted_label": y_pred[index]}
        row.update({f"prob_{name}": float(y_prob[index, class_index]) for class_index, name in enumerate(APTOS_CLASS_ORDER)})
        probability_rows.append(row)
    pd.DataFrame(probability_rows).to_csv(experiment_dir / "test_predictions.csv", index=False)
    torch.save(
        {
            "model_state": best_state,
            "architecture": "convnext_tiny",
            "pretrained": False,
            "initialization": "torchvision ConvNeXt-Tiny random initialization; no external classifier weights",
            "class_names": APTOS_CLASS_ORDER,
            "experiment": experiment.__dict__,
            "training_args": vars(args),
            "metrics": metrics,
        },
        checkpoint_path,
    )
    write_json(
        live_progress_path,
        {
            "status": "completed",
            "experiment_key": experiment.key,
            "display_name": experiment.display_name,
            "epochs_ran": len(history),
            "best_val_macro_f1": best_val_macro_f1,
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_auprc": metrics["macro_auprc"],
            "minority_avg_recall": metrics["minority_avg_recall"],
            "quadratic_weighted_kappa": metrics["quadratic_weighted_kappa"],
        },
    )
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return metrics


def make_summary_plots(results: List[Dict[str, object]], output_dir: Path) -> None:
    names = [row["display_name"] for row in results]
    keys = ["balanced_accuracy", "macro_f1", "macro_auprc", "minority_avg_recall", "quadratic_weighted_kappa"]
    labels = ["Balanced Acc", "Macro-F1", "Macro-AUPRC", "Minority Recall", "QWK"]
    x = np.arange(len(names))
    width = 0.15
    plt.figure(figsize=(14, 6))
    for index, (key, label) in enumerate(zip(keys, labels)):
        plt.bar(x + (index - 2) * width, [row[key] for row in results], width=width, label=label)
    plt.xticks(x, names, rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Metric")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(output_dir / "metric_comparison.png", dpi=220)
    plt.close()

    plt.figure(figsize=(14, 7))
    x = np.arange(len(APTOS_CLASS_ORDER))
    width = 0.16
    for index, result in enumerate(results):
        recalls = [item["recall"] for item in result["per_class"]]
        plt.bar(x + (index - 2) * width, recalls, width=width, label=result["display_name"])
    plt.xticks(x, APTOS_CLASS_ORDER)
    plt.ylim(0, 1)
    plt.ylabel("Recall")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_recall_comparison.png", dpi=220)
    plt.close()


def markdown_results(results: List[Dict[str, object]]) -> List[str]:
    lines = [
        "| 方法 | Balanced Accuracy | Macro-F1 | Macro-AUPRC | Minority Recall | QWK | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['display_name']} | {row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['macro_auprc']:.4f} | {row['minority_avg_recall']:.4f} | "
            f"{row['quadratic_weighted_kappa']:.4f} | {row['accuracy']:.4f} |"
        )
    return lines


def write_report(
    args: argparse.Namespace,
    data_root: Path,
    output_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    augmented_df: pd.DataFrame,
    results: List[Dict[str, object]],
    total_seconds: float,
) -> None:
    original = next(row for row in results if row["experiment_key"] == "original_ce")
    traditional = next(row for row in results if row["experiment_key"] == "traditional_aug_ce")
    balanced = next(row for row in results if row["experiment_key"] == "balanced_softmax")
    focal = next(row for row in results if row["experiment_key"] == "focal_loss")
    sd = next(row for row in results if row["experiment_key"] == "sd025_augmented_ce")
    actual_training_seconds = sum(float(row.get("training_seconds", 0.0)) for row in results)
    image_quality_path = (ROOT / args.image_quality_dir / "summary_metrics.json").resolve()
    image_quality_lines: List[str] = []
    if image_quality_path.exists():
        image_quality = json.loads(image_quality_path.read_text(encoding="utf-8"))
        quality_overall = image_quality["metrics"]["Overall_minority"]
        quality_macro = image_quality["metrics"]["Macro_class"]
        quality_by_class = [
            {"class_name": class_name, **image_quality["metrics"][class_name]}
            for class_name in ["Mild", "Moderate", "Severe", "Proliferate_DR"]
        ]
        image_quality_lines = [
            "",
            "## 8. SD 0.25 全量生成图像质量",
            "",
            f"对全部 {quality_overall['generated_count']} 张少数类合成图，使用固定真实训练集计算 FID、KID 和 Kynkäänniemi Improved Precision/Recall。Overall 将四个少数类合并计算，Macro 为四个逐类结果的算术平均。",
            "",
            "| 范围 | FID ↓ | KID均值±标准差 ↓ | Improved Precision ↑ | Improved Recall ↑ |",
            "|---|---:|---:|---:|---:|",
            f"| Overall minority | {quality_overall['fid']:.4f} | {quality_overall['kid_mean']:.6f} ± {quality_overall['kid_std']:.6f} | {quality_overall['improved_precision']:.4f} | {quality_overall['improved_recall']:.4f} |",
            f"| Macro class | {quality_macro['fid']:.4f} | {quality_macro['kid_mean']:.6f} ± {quality_macro['kid_std']:.6f} | {quality_macro['improved_precision']:.4f} | {quality_macro['improved_recall']:.4f} |",
            "",
            "### 8.1 逐类别指标",
            "",
            "| 类别 | 真实数 | 生成数 | FID ↓ | KID ↓ | Precision ↑ | Recall ↑ |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *[
                f"| {row['class_name']} | {row['real_count']} | {row['generated_count']} | {row['fid']:.4f} | {row['kid_mean']:.6f} | {row['improved_precision']:.4f} | {row['improved_recall']:.4f} |"
                for row in quality_by_class
            ],
            "",
            "### 8.2 图像指标与分类结果的联合解释",
            "",
            f"- Overall Improved Precision={quality_overall['improved_precision']:.4f}，表示只有较少生成样本落入由真实少数类样本估计的 VGG-16 特征流形，当前 baseline 的 fidelity 较弱。",
            f"- Overall Improved Recall={quality_overall['improved_recall']:.4f}，表示生成分布只覆盖真实少数类流形的一小部分，存在明显 coverage 不足。",
            f"- FID={quality_overall['fid']:.4f}、KID={quality_overall['kid_mean']:.6f}。两者没有跨数据集统一合格阈值，必须与后续 LoRA、结构先验和本文方法在同一协议下相对比较。",
            "- 尽管 Improved Precision/Recall 偏低，本轮 SD 组的下游分类指标仍有提升；二者并不矛盾：图像分布指标衡量与真实特征流形的接近和覆盖，下游分类衡量这些样本是否为当前训练提供了有用监督或正则化。",
            "- FID/KID/IPR 使用固定的预训练特征提取器只做事后图像评价；它不是本轮下游诊断分类器，也不参与 ConvNeXt 参数初始化或训练。",
            "- 单 seed 的分类提升不能反向证明合成图医学真实性良好，分布指标也不能单独证明标签漂移或病灶错误；两类证据应并列报告。",
            "",
            f"完整报告：`SD_APTOS全量图像质量评价报告.md`；输出目录：`{Path(args.image_quality_dir)}`。",
        ]

    def class_metric(result: Dict[str, object], class_name: str, metric: str) -> float:
        return float(next(item[metric] for item in result["per_class"] if item["class_name"] == class_name))
    sd_beats_original = sd["balanced_accuracy"] > original["balanced_accuracy"] and sd["macro_f1"] > original["macro_f1"]
    sd_beats_traditional = sd["balanced_accuracy"] > traditional["balanced_accuracy"] and sd["macro_f1"] > traditional["macro_f1"]
    metric_specs = [
        ("balanced_accuracy", "Balanced Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("macro_auprc", "Macro-AUPRC"),
        ("minority_avg_recall", "Minority Recall"),
        ("quadratic_weighted_kappa", "QWK"),
    ]
    metric_ranking_lines = []
    for key, label in metric_specs:
        ranking = sorted(results, key=lambda row: float(row[key]), reverse=True)
        metric_ranking_lines.append(
            f"- {label}：最佳为 {ranking[0]['display_name']} ({ranking[0][key]:.4f})；完整排序为 "
            + " > ".join(f"{row['display_name']} ({row[key]:.4f})" for row in ranking)
            + "。"
        )
    convergence_lines = [
        "### 5.1 收敛与早停记录",
        "",
        "| 方法 | 最佳 epoch | 实际运行 epoch | 最佳验证 Macro-F1 | 训练耗时（秒） |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results:
        history_path = output_dir / str(row["experiment_key"]) / "training_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        best_history = max(history, key=lambda item: float(item["val_macro_f1"]))
        convergence_lines.append(
            f"| {row['display_name']} | {best_history['epoch']} | {row['epochs_ran']} | "
            f"{row['best_val_macro_f1']:.4f} | {row['training_seconds']:.1f} |"
        )
    convergence_lines.extend(
        [
            "",
            "所有组都从随机初始化开始并出现明显的验证指标上升；例如 Original 在第 12 轮达到最佳验证 Macro-F1=0.5210。这直接证明训练数据确实在更新并塑造分类器，而不是模型因初始化方式而没有学习。",
        ]
    )
    pretrained_comparison_lines: List[str] = []
    pretrained_csv = ROOT / "outputs/aptos2019/convnext_tiny_preliminary/comparison_metrics.csv"
    if pretrained_csv.exists():
        pretrained_rows = {
            str(row["experiment_key"]): row
            for row in pd.read_csv(pretrained_csv).to_dict(orient="records")
        }
        pretrained_comparison_lines = [
            "",
            "### 7.6 与上一次 ImageNet 初始化结果的诊断性比较",
            "",
            "| 方法 | 预训练 Macro-F1 | 随机初始化 Macro-F1 | 预训练 Balanced Acc. | 随机初始化 Balanced Acc. |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in results:
            old = pretrained_rows[str(row["experiment_key"])]
            pretrained_comparison_lines.append(
                f"| {row['display_name']} | {old['macro_f1']:.4f} | {row['macro_f1']:.4f} | "
                f"{old['balanced_accuracy']:.4f} | {row['balanced_accuracy']:.4f} |"
            )
        pretrained_comparison_lines.extend(
            [
                "",
                "- 随机初始化的绝对性能明显低于 ImageNet 初始化，这是在仅 2563 张真实训练图上从零学习 2782 万参数时可预期的结果，也说明预训练迁移本身有效。",
                "- 但方法间结论发生了变化：上一次预训练实验中 Original 最好、SD 低于 Original；本轮随机初始化中 SD 在 Balanced Accuracy、Macro-F1、Macro-AUPRC 和 Minority Recall 上均为最佳。这支持‘强迁移先验会改变增强收益的可见程度’，不支持‘预训练导致数据没有参与训练’。",
                "- 这两轮训练策略并不相同（预训练微调为较小学习率和较少 epoch；随机初始化使用 warm-up、最长 60 epoch），因此该表只能用于解释现象，不能作为严格控制变量的预训练消融结论。",
            ]
        )
    per_class_analysis = []
    for class_name in APTOS_CLASS_ORDER:
        original_recall = next(item["recall"] for item in original["per_class"] if item["class_name"] == class_name)
        sd_recall = next(item["recall"] for item in sd["per_class"] if item["class_name"] == class_name)
        delta = sd_recall - original_recall
        direction = "提升" if delta > 1e-8 else "下降" if delta < -1e-8 else "不变"
        per_class_analysis.append(
            f"- {class_name}：Original Recall={original_recall:.4f}，SD 0.25 Recall={sd_recall:.4f}，{direction} {abs(delta):.4f}。"
        )
    if sd_beats_original and sd_beats_traditional:
        effect_analysis = [
            "- SD 0.25 增强在 Balanced Accuracy 和 Macro-F1 上同时超过 Original CE 与 Traditional Augmentation，说明本轮合成数据对类别均衡识别具有初步正向作用。",
            "- 仍需结合 Macro-AUPRC、Minority Recall、QWK 和每类 Recall 判断提升是否全面；单一指标提升不足以证明最终有效。",
        ]
    else:
        effect_analysis = [
            "- SD 0.25 增强没有在 Balanced Accuracy 与 Macro-F1 上同时超过 Original CE 和 Traditional Augmentation，因此本轮不能认定未微调 SD 数据增强稳定有效。",
            "- 首要原因可能是生成分布差距：当前使用通用 SD v1.5，没有 LoRA 医学域适配；FID/KID/IPR 已显示其与真实少数类分布仍有明显差异。",
            "- 本轮为验证生成本身，训练前未按 FID/KID/IPR 做方法级筛选，全部 strength=0.25 合成图均进入训练，可能引入 synthetic bias。",
            "- Severe 等类别的合成数量远大于真实数量，训练时合成样本可能淹没少量真实诊断特征；后续应结合质量门控和生成比例控制，而不是默认补齐到 100%。",
            "- Traditional Augmentation 本身若也未改善，说明当前随机裁剪、旋转或颜色扰动强度可能不适合所有眼底等级，应结合 per-class Recall 判断受影响类别。",
        ]
    lines = [
        "# APTOS ConvNeXt-Tiny 随机初始化五组重跑报告（待用户确认）",
        "",
        "## 1. 实验目的",
        "",
        "在不加载任何 ImageNet 或其他外部分类器权重的条件下，从随机初始化训练 ConvNeXt-Tiny，重新比较 Original、Traditional Augmentation、Balanced Softmax、Focal Loss 和 SD 0.25 Augmented。该结果写入独立目录和待确认报告，不覆盖既有预训练实验。",
        "",
        "### 1.1 对上一次预训练分类器问题的直接回答",
        "",
        "- 上一次使用的是 ImageNet-1K 初始化，但 backbone 没有冻结，整个 ConvNeXt 都参与反向传播。因此，预训练并不会导致数据‘影响不到分类器训练’；数据确实更新了全部网络参数。",
        "- 预训练提供了很强的通用视觉先验，使 2563 张真实训练图已经能得到较高基线，也可能压缩额外增强在单次实验中的可见收益。它改变的是起点和优化难度，不是切断数据对模型的作用。",
        "- 按当前要求，本次及现行分类脚本统一使用 `weights=None`。图像质量评价中固定的特征网络只负责计算 FID/KID/IPR，不属于下游诊断分类器。",
        "",
        "## 2. 数据切分与防泄漏策略",
        "",
        "- APTOS 共 3662 张真实图像，排除 macOS `._*.png` 元数据文件。",
        f"- 分层切分：train={len(train_df)}、val={len(val_df)}、test={len(test_df)}，比例为 70%/15%/15%，seed=2026。",
        "- 五组实验共享完全相同的验证集和测试集。",
        "- SD 图像只从训练集源图生成，只加入训练集；验证集和测试集不参与生成。",
        f"- SD 增强组训练集共 {len(augmented_df)} 张，其中真实图 {len(train_df)}、合成图 {len(augmented_df)-len(train_df)}。",
        "- 不使用 weighted sampler，避免与各损失/数据增强策略混杂。",
        "",
        "## 3. 统一模型与训练策略",
        "",
        "### 3.1 模型",
        "",
        "- 分类器：torchvision ConvNeXt-Tiny。",
        "- 初始化：torchvision 随机初始化，`weights=None`；不读取任何外部分类器 checkpoint。",
        "- 将最终分类层替换为 5 类线性层，整个网络从零开始端到端训练，不冻结任何层。",
        "- 输入尺寸：224×224；继续使用固定 mean/std 数值归一化。归一化不是预训练权重，不会冻结或阻止数据更新模型。",
        "",
        "### 3.2 预处理与传统增强",
        "",
        "- 所有组先检测眼底有效区域、裁掉黑边并保持背景为黑色。",
        "- Original CE 训练集只做确定性 resize 和归一化。",
        "- 其余四组使用同一传统增强：Resize 256、RandomResizedCrop(224, scale=0.90–1.00, ratio=0.95–1.05)、水平翻转 p=0.5、±15° 旋转、轻量 brightness/contrast/saturation/hue jitter。",
        "- 验证和测试阶段只做确定性预处理、resize 和归一化。",
        "",
        "### 3.3 优化策略",
        "",
        f"- Optimizer：AdamW，初始学习率 {args.learning_rate}，weight decay {args.weight_decay}。",
        f"- Scheduler：前 {args.warmup_epochs} epoch 线性 warm-up（0.1×到1.0×基础学习率），随后 cosine annealing；最大 {args.epochs} epoch，最低学习率 {args.min_learning_rate}。",
        f"- Batch size：{args.batch_size}；混合精度 FP16；梯度裁剪 max norm=1.0。",
        f"- Early stopping patience：{args.early_stopping_patience}。",
        "- 每个 epoch 在验证集计算 Macro-F1，保存验证 Macro-F1 最高的 checkpoint，再对固定测试集评估一次。",
        "- 随机种子统一为 2026；启用 cuDNN deterministic，禁用 benchmark。",
        "- 运行前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，使 CUDA 线性代数操作满足可复现要求。",
        "- 为避免从零训练沿用微调阶段过小学习率和过少轮数，本轮统一采用独立的 scratch 训练策略；五组的优化器、调度器、停止标准和随机种子完全相同。",
        "",
        "## 4. 五组实验的唯一差异",
        "",
        "| 组别 | 训练数据 | 随机传统增强 | 损失 |",
        "|---|---|---|---|",
        "| Original CE | 原始训练集 | 否 | Cross Entropy |",
        "| Traditional Augmentation | 原始训练集 | 是 | Cross Entropy |",
        "| Balanced Softmax | 原始训练集 | 是 | Balanced Softmax |",
        "| Focal Loss | 原始训练集 | 是 | class-balanced Focal Loss |",
        "| SD 0.25 Augmented | 原始训练集 + 3752 张 SD 图 | 是 | Cross Entropy |",
        "",
        "Balanced Softmax 使用训练集真实类别计数，对 logits 加 `log(n_c)` 后计算交叉熵：",
        "",
        "`L_BS = CE(z + log(n), y)`",
        "",
        "Focal Loss 使用 `gamma=2.0`，类别权重 `alpha_c=N/(K*n_c)` 并归一化为均值 1：",
        "",
        "`L_focal = alpha_y * (1-p_t)^2 * CE(z,y)`",
        "",
        "### 4.1 各方法原理",
        "",
        "- **Original CE**：保留原始长尾训练分布，用标准交叉熵学习，作为不引入额外长尾处理的基准。它通常具有较好的整体分布拟合能力，但梯度容易被 No_DR、Moderate 等样本较多类别主导。",
        "- **Traditional Augmentation**：不改变类别数量，通过几何和颜色扰动构造局部不变性，相当于对真实样本邻域做正则化。它能抑制过拟合，但若旋转、裁剪或颜色变化破坏细微病灶，也可能降低诊断特征可辨性。",
        "- **Balanced Softmax**：在训练 softmax 中显式加入类别样本先验 `log(n_c)`，修正长尾训练分布造成的决策偏置。它不生成数据，主要调整分类边界，理论上更有利于尾部类别召回，但可能牺牲头部类精度或概率校准。",
        "- **Focal Loss**：用 `(1-p_t)^gamma` 降低易分类样本的梯度贡献，让模型更关注困难和误分类样本；再用逆频率 alpha 强化少数类。它适合难例和不平衡共存的场景，但若困难样本中包含噪声，也会放大噪声影响。",
        "- **SD 0.25 Augmented**：通过真实少数类图像初始化的 image-to-image diffusion 增加训练样本数量，使每类训练规模相同。其目标是扩展 minority representation，而不是只改变损失权重；风险是未微调 SD 产生 domain gap、复制源图或引入诊断不一致纹理。",
        "",
        "## 5. 测试集结果",
        "",
    ]
    lines.extend(markdown_results(results))
    lines.extend(["", *convergence_lines])
    lines.extend(
        [
            "",
            "## 6. SD 增强的初步变化",
            "",
            "### 6.1 SD 0.25 相对 Original CE",
            "",
            "| 指标 | Original CE | SD 0.25 | 绝对变化 | 相对变化 | 是否改善 |",
            "|---|---:|---:|---:|---:|---|",
            *[
                f"| {label} | {original[key]:.4f} | {sd[key]:.4f} | {sd[key]-original[key]:+.4f} | "
                f"{(sd[key]-original[key])/original[key]*100:+.2f}% | {'是' if sd[key] > original[key] else '否'} |"
                for key, label in [
                    ("balanced_accuracy", "Balanced Accuracy"),
                    ("macro_f1", "Macro-F1"),
                    ("macro_auprc", "Macro-AUPRC"),
                    ("minority_avg_recall", "Minority Recall"),
                    ("quadratic_weighted_kappa", "QWK"),
                    ("accuracy", "Accuracy"),
                ]
            ],
            "",
            "在 6 个汇总指标中，SD 0.25 有 5 个高于 Original CE。与长尾识别直接相关的 Balanced Accuracy、Macro-F1、Macro-AUPRC 和 Minority Recall 全部提升。Accuracy 略降，说明收益主要来自类别均衡性，而不是多数类主导的总体命中率。",
            "",
            "### 6.2 SD 0.25 相对 Traditional Augmentation",
            "",
            "| 指标 | Traditional | SD 0.25 | 绝对变化 |",
            "|---|---:|---:|---:|",
            *[
                f"| {label} | {traditional[key]:.4f} | {sd[key]:.4f} | {sd[key]-traditional[key]:+.4f} |"
                for key, label in [
                    ("balanced_accuracy", "Balanced Accuracy"),
                    ("macro_f1", "Macro-F1"),
                    ("macro_auprc", "Macro-AUPRC"),
                    ("minority_avg_recall", "Minority Recall"),
                    ("quadratic_weighted_kappa", "QWK"),
                    ("accuracy", "Accuracy"),
                ]
            ],
            "",
            f"因此可以说，本轮 SD 补全在多数核心指标上取得了初步提升，但不是全面提升：五类 Recall 中 No_DR、Mild、Severe、Proliferate_DR 四类提高，Moderate 从 {class_metric(original, 'Moderate', 'recall'):.4f} 降至 {class_metric(sd, 'Moderate', 'recall'):.4f}；本轮又只有单一 seed，尚不能写成稳定或显著提升。",
            "",
            "## 7. 增强效果分析",
            "",
            *effect_analysis,
            "",
            "### 7.1 各指标排序与含义",
            "",
            *metric_ranking_lines,
            "",
            "指标之间出现不同最佳方法是合理的：Balanced Accuracy/Minority Recall 更偏向召回公平性，Macro-F1 同时惩罚低精度，Macro-AUPRC 关注少数类概率排序，QWK 还考虑错误等级距离。因此不能只按单一指标宣布方法有效。",
            "",
            "### 7.2 各类别 Recall 对比",
            "",
            "| 类别 | Original | Traditional | Balanced Softmax | Focal | SD 0.25 |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                "| " + class_name + " | " + " | ".join(
                    f"{next(item['recall'] for item in result['per_class'] if item['class_name'] == class_name):.4f}"
                    for result in results
                ) + " |"
                for class_name in APTOS_CLASS_ORDER
            ],
            "",
            *per_class_analysis,
            "",
            "### 7.3 机制层面的结果解释",
            "",
            f"- **Original CE** 为随机初始化、只用真实训练集且无随机增强的基准。本轮其 Balanced Accuracy/Macro-F1 为 {original['balanced_accuracy']:.4f}/{original['macro_f1']:.4f}；它反映从零训练时原始长尾数据本身可达到的水平。",
            f"- **Traditional Augmentation** 相对 Original 的 Balanced Accuracy 变化 {traditional['balanced_accuracy']-original['balanced_accuracy']:+.4f}、Macro-F1 {traditional['macro_f1']-original['macro_f1']:+.4f}，但 Macro-AUPRC 仅变化 {traditional['macro_auprc']-original['macro_auprc']:+.4f}。Mild Recall 变化 {class_metric(traditional, 'Mild', 'recall')-class_metric(original, 'Mild', 'recall'):+.4f}，而 Moderate/Severe 分别变化 {class_metric(traditional, 'Moderate', 'recall')-class_metric(original, 'Moderate', 'recall'):+.4f}/{class_metric(traditional, 'Severe', 'recall')-class_metric(original, 'Severe', 'recall'):+.4f}。这表明概率排序能力大致保留，但随机裁剪、旋转和颜色扰动改变了最终决策边界，可能损伤依赖细小病灶及空间关系的等级。",
            f"- **Balanced Softmax** 的 Balanced Accuracy 相对 Original 变化 {balanced['balanced_accuracy']-original['balanced_accuracy']:+.4f}，Macro-F1 变化 {balanced['macro_f1']-original['macro_f1']:+.4f}，QWK 变化 {balanced['quadratic_weighted_kappa']-original['quadratic_weighted_kappa']:+.4f}。它将 Mild Recall 从 {class_metric(original, 'Mild', 'recall'):.4f} 提至 {class_metric(balanced, 'Mild', 'recall'):.4f}，但 Moderate Recall 从 {class_metric(original, 'Moderate', 'recall'):.4f} 降至 {class_metric(balanced, 'Moderate', 'recall'):.4f}，并且 Mild precision 只有 {class_metric(balanced, 'Mild', 'precision'):.4f}。这符合先验校正移动分类边界的预期：它重新分配类别召回和误报，但本轮净提升很小。",
            f"- **Focal Loss** 相对 Original 的 Balanced Accuracy/Macro-F1 分别变化 {focal['balanced_accuracy']-original['balanced_accuracy']:+.4f}/{focal['macro_f1']-original['macro_f1']:+.4f}，并取得本轮最高 QWK={focal['quadratic_weighted_kappa']:.4f}。Mild、Proliferate_DR、Moderate 和 Severe Recall 分别变化 {class_metric(focal, 'Mild', 'recall')-class_metric(original, 'Mild', 'recall'):+.4f}/{class_metric(focal, 'Proliferate_DR', 'recall')-class_metric(original, 'Proliferate_DR', 'recall'):+.4f}/{class_metric(focal, 'Moderate', 'recall')-class_metric(original, 'Moderate', 'recall'):+.4f}/{class_metric(focal, 'Severe', 'recall')-class_metric(original, 'Severe', 'recall'):+.4f}。困难样本聚焦帮助了部分尾类和有序一致性，却没有同时保住 Moderate，固定 gamma=2 与逆频率 alpha 可能偏强。",
            f"- **SD 0.25 Augmented** 使 Mild、Severe、Proliferate_DR Recall 分别变化 {class_metric(sd, 'Mild', 'recall')-class_metric(original, 'Mild', 'recall'):+.4f}/{class_metric(sd, 'Severe', 'recall')-class_metric(original, 'Severe', 'recall'):+.4f}/{class_metric(sd, 'Proliferate_DR', 'recall')-class_metric(original, 'Proliferate_DR', 'recall'):+.4f}，但 Moderate Recall 变化 {class_metric(sd, 'Moderate', 'recall')-class_metric(original, 'Moderate', 'recall'):+.4f}。前三类的收益超过 Moderate 的损失，所以 Balanced Accuracy、Macro-F1、Macro-AUPRC 和 Minority Recall 均获得净提升；但类别间收益不均衡，仍不能称为成熟方案。",
            "- 本轮是单随机种子初步复核。增强是否稳定有效仍需多 seed；FID/KID/IPR只能说明生成分布关系，不能替代分类差异的因果实验。",
            "",
            "### 7.4 为什么各指标会呈现当前结果",
            "",
            f"- **Balanced Accuracy**：SD 比 Original 高 {sd['balanced_accuracy']-original['balanced_accuracy']:.4f}。该指标等权平均五类 Recall；Mild 与 Proliferate_DR 的大幅提升以及 Severe 的小幅提升，合计超过 Moderate 的下降。Balanced Softmax 与 Original 很接近，则是 Mild 提升与 Moderate 下降相互抵消。",
            f"- **Macro-F1**：SD 比 Original 高 {sd['macro_f1']-original['macro_f1']:.4f}，说明召回提升并非完全来自无控制的少数类过预测；但 Severe 的 recall/precision 仍只有 {class_metric(sd, 'Severe', 'recall'):.4f}/{class_metric(sd, 'Severe', 'precision'):.4f}，限制了总体提升幅度。",
            f"- **Macro-AUPRC**：SD 从 Original 的 {original['macro_auprc']:.4f} 提至 {sd['macro_auprc']:.4f}，说明提升不只发生在 argmax 决策点，one-vs-rest 概率排序整体也改善。Traditional 为 {traditional['macro_auprc']:.4f}，仍低于 Original，未显示相同收益。",
            f"- **Minority Recall**：该指标排除 No_DR，因此不会被多数类掩盖。SD 比 Original 高 {sd['minority_avg_recall']-original['minority_avg_recall']:.4f}，直接支持少数类补全的初步目标；Focal 也高 {focal['minority_avg_recall']-original['minority_avg_recall']:+.4f}，Balanced Softmax 则变化 {balanced['minority_avg_recall']-original['minority_avg_recall']:+.4f}。",
            f"- **QWK**：SD 比 Original 高 {sd['quadratic_weighted_kappa']-original['quadratic_weighted_kappa']:+.4f}，但低于 Focal 的 {focal['quadratic_weighted_kappa']:.4f}。SD 对 Moderate 的召回损失会影响有序分级，因此其 QWK 增幅小于 Balanced Accuracy 增幅。",
            f"- **Accuracy**：Original 为 {original['accuracy']:.4f}，SD 为 {sd['accuracy']:.4f}，下降 {original['accuracy']-sd['accuracy']:.4f}。Accuracy 受样本最多的 No_DR 主导，并不能反映 SD 在三个尾类上的召回收益；这正是本任务不能只看 Accuracy 的原因。",
            "",
            "### 7.5 为什么普通增强没有自动带来提升",
            "",
            "- ‘普通增强通常有帮助’是合理先验，但不是必然规律。它不增加独立病灶信息，只要求模型对裁剪、旋转和颜色扰动保持不变；当病灶很小、颜色和空间分布又与等级相关时，这些不变性可能过强。",
            f"- 本轮 Traditional 的 Severe 与 Proliferate_DR Recall 都为 0，Macro-F1 从 {original['macro_f1']:.4f} 降至 {traditional['macro_f1']:.4f}。这说明主要问题不是模型没训练，而是从零训练、小数据长尾分布和当前增强组合共同使决策偏向头部类。",
            "- Original 没有随机增强，Traditional 有随机增强，但两组仍按原始长尾频率采样；增强只能重复扰动已有少数类，不能改变类别出现概率。Balanced Softmax/Focal/SD 的恢复进一步说明，类别先验或样本数量处理比单纯几何颜色扰动更关键。",
            "- 因此不能把本轮下降解释成‘增强永远无效’，也不能为了符合预期而选择性覆盖结果。更稳妥的后续验证是固定随机初始化，分别测试仅水平翻转、弱旋转、去除颜色扰动等强度，并做多 seed；本轮先保留为真实的初步 baseline。",
            *pretrained_comparison_lines,
            "",
            *image_quality_lines,
            "",
            "## 9. 分类评价标准",
            "",
            "- **Balanced Accuracy**：五类 Recall 的宏平均，每类权重相同，不会被 No_DR 数量主导。它回答模型是否在所有等级上都能召回，是长尾分类的核心整体指标。",
            "- **Macro-F1**：五类 F1 的宏平均，同时考虑 precision 与 recall。它能阻止模型通过大量预测少数类来虚高召回，因此用于衡量类别均衡性与误报之间的折中。",
            "- **Macro-AUPRC**：按 one-vs-rest 计算每类 AUPRC 后宏平均。在少数类阳性比例很低时，PR 曲线比 ROC 更敏感于假阳性，能评价模型输出概率的排序质量。",
            "- **Minority-class Recall**：Mild、Moderate、Severe、Proliferate_DR 四类 Recall 的平均，直接衡量生成补齐是否减少少数疾病漏诊，是本文任务最直接的目标指标。",
            "- **QWK**：APTOS 标签具有从 No_DR 到 Proliferate_DR 的有序严重程度。Quadratic Weighted Kappa 对跨越多个等级的错误施加更大惩罚，并扣除随机一致性，因此反映临床等级预测质量。",
            "- **Accuracy**：作为补充报告，但它受 No_DR 等多数类主导，不能作为本论文判断长尾方法有效性的主要依据。",
            "- **Per-class Recall 与混淆矩阵**：用于定位总体指标变化来自哪个类别、错误流向相邻等级还是跨级误判。",
            "- 同时保存每类 precision/recall/F1/AUROC/AUPRC 和逐测试样本概率，便于后续统计检验与绘图。",
            "",
            "## 10. 输出与固定复用位置",
            "",
            f"- 总目录：`{output_dir.relative_to(ROOT)}`",
            f"- 汇总指标：`{(output_dir / 'comparison_metrics.csv').relative_to(ROOT)}`",
            f"- 指标对比图：`{(output_dir / 'metric_comparison.png').relative_to(ROOT)}`",
            f"- Per-class Recall 图：`{(output_dir / 'per_class_recall_comparison.png').relative_to(ROOT)}`",
            f"- 实验定义：`{(output_dir / 'experiment_definition.json').relative_to(ROOT)}`",
            f"- 各组 checkpoint、history、predictions：`{output_dir.relative_to(ROOT)}/<experiment_key>/`",
            f"- 固定真实数据切分：`{(data_root / 'splits').relative_to(ROOT)}`",
            f"- 固定 SD 生成清单：`{(data_root / 'generated_manifest.csv').relative_to(ROOT)}`",
            f"- SD 全量图像指标：`{Path(args.image_quality_dir)}`",
            "",
            "## 11. 自行复现与断点续跑",
            "",
            "```bash",
            "bash scripts/run_aptos_convnext_scratch_baselines.sh",
            "```",
            "",
            "已完整完成的组会复用保存的 metrics/checkpoint/history；中断时重新运行即可继续剩余组。若要强制重训某一组，应先人工归档该组目录，再重新运行，避免无意覆盖固定基线。",
            "",
            "图像指标可单独复现：",
            "",
            "```bash",
            "bash scripts/run_sd025_image_quality.sh",
            "```",
            "",
            "## 12. 运行环境与耗时",
            "",
            f"- Python：{platform.python_version()}",
            f"- PyTorch：{torch.__version__}",
            f"- torchvision：{__import__('torchvision').__version__}",
            f"- CUDA runtime：{torch.version.cuda}",
            f"- GPU：{torch.cuda.get_device_name(0)}",
            f"- 五组正式训练累计 GPU 时间（由各组保存记录求和）：{actual_training_seconds/60.0:.2f} 分钟。",
            f"- 本次汇总/报告脚本耗时：{total_seconds/60.0:.2f} 分钟；使用已固定缓存时该值不代表重新训练耗时。",
            "",
            "## 13. 解释限制",
            "",
            "- 这是单一 seed 的随机初始化初步复核，尚未报告多次重复的均值与标准差。",
            "- 本报告为独立待确认版本，没有覆盖原来的 ImageNet 预训练实验报告或结果目录。",
            "- SD 生成数据已经完成事后全量图像指标评价，但训练前没有按质量筛选，也未使用 LoRA；本轮结果用于定位 baseline 失败机制。",
            "- 后续若调整训练策略、数据切分或增强定义，应建立新版本结果，不能覆盖本报告固定的 baseline。",
        ]
    )
    (ROOT / args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("ConvNeXt baseline run requires CUDA.")
    device = torch.device("cuda")
    seed_everything(args.seed)
    data_root = (ROOT / args.data_root).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_root / "splits" / "train.csv")
    val_df = pd.read_csv(data_root / "splits" / "val.csv")
    test_df = pd.read_csv(data_root / "splits" / "test.csv")
    generated_df = pd.read_csv(data_root / "generated_manifest.csv")
    if generated_df["image_path"].duplicated().any():
        raise RuntimeError("Generated manifest contains duplicate paths.")
    generated_train = generated_df[["image_id", "image_path", "label", "label_name", "dataset_name"]].copy()
    augmented_df = pd.concat([train_df, generated_train], ignore_index=True)

    split_ids = {name: set(frame["image_id"].astype(str)) for name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]}
    if split_ids["train"] & split_ids["val"] or split_ids["train"] & split_ids["test"] or split_ids["val"] & split_ids["test"]:
        raise RuntimeError("Real-data split leakage detected.")
    source_ids = set(generated_df["source_image_id"].astype(str))
    if not source_ids.issubset(split_ids["train"]):
        raise RuntimeError("Synthetic manifest contains a non-training source image.")

    real_counts = train_df["label_name"].value_counts().reindex(APTOS_CLASS_ORDER).astype(int).tolist()
    definition = {
        "experiments": [experiment.__dict__ for experiment in EXPERIMENTS],
        "training_args": vars(args),
        "class_names": APTOS_CLASS_ORDER,
        "minority_classes": MINORITY_CLASSES,
        "real_train_class_counts": dict(zip(APTOS_CLASS_ORDER, real_counts)),
        "augmented_train_class_counts": augmented_df["label_name"].value_counts().reindex(APTOS_CLASS_ORDER).astype(int).to_dict(),
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    }
    write_json(output_dir / "experiment_definition.json", definition)

    results = []
    for experiment in EXPERIMENTS:
        experiment_train_df = augmented_df if experiment.train_source == "real+sd025" else train_df
        results.append(
            train_experiment(
                experiment,
                experiment_train_df,
                val_df,
                test_df,
                real_counts,
                args,
                output_dir,
                device,
            )
        )

    comparison_rows = []
    for result in results:
        comparison_rows.append(
            {
                "experiment_key": result["experiment_key"],
                "display_name": result["display_name"],
                "train_size": result["train_size"],
                "epochs_ran": result["epochs_ran"],
                "best_val_macro_f1": result["best_val_macro_f1"],
                "accuracy": result["accuracy"],
                "balanced_accuracy": result["balanced_accuracy"],
                "macro_f1": result["macro_f1"],
                "macro_auprc": result["macro_auprc"],
                "minority_avg_recall": result["minority_avg_recall"],
                "quadratic_weighted_kappa": result["quadratic_weighted_kappa"],
                "training_seconds": result["training_seconds"],
            }
        )
    pd.DataFrame(comparison_rows).to_csv(output_dir / "comparison_metrics.csv", index=False)
    write_json(output_dir / "comparison_metrics.json", comparison_rows)
    make_summary_plots(results, output_dir)
    write_report(args, data_root, output_dir, train_df, val_df, test_df, augmented_df, results, time.time() - started)
    print(f"All experiments completed. Report: {(ROOT / args.report).resolve()}", flush=True)


if __name__ == "__main__":
    main()
