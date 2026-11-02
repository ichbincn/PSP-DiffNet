from __future__ import annotations

import argparse
import copy
import json
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import platform
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.mmotu_dataset import MMOTU_CLASS_ORDER
from metrics.classification_metrics import compute_classification_metrics
from utils.ultrasound import preprocess_ultrasound_pil


MMOTU_MINORITY = [MMOTU_CLASS_ORDER[index] for index in (3, 4, 6, 7)]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
AUGMENTATION_DESCRIPTION = "aspect-preserving pad; HFlip0.5; affine ±10deg/3%/0.95-1.05; brightness0.10 contrast0.12"
NO_AUGMENTATION_DESCRIPTION = "deterministic aspect-preserving square resize-pad; no stochastic augmentation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scratch ConvNeXt-Tiny for fixed MMOTU splits")
    parser.add_argument("--train-split", default="outputs/mmotu2d/dataset_preparation/splits/train.csv")
    parser.add_argument("--validation-split", default="outputs/mmotu2d/dataset_preparation/splits/validation.csv")
    parser.add_argument("--test-split", default="outputs/mmotu2d/dataset_preparation/splits/test.csv")
    parser.add_argument("--synthetic-manifest", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evaluation-stage", choices=["validation_only", "formal_test"], default="formal_test")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--training-augmentation", choices=["none", "traditional"], default="traditional",
        help="none is deterministic ROI preprocessing; traditional adds the frozen ultrasound transforms.",
    )
    parser.add_argument(
        "--loss-mode", choices=["cross_entropy", "balanced_softmax", "focal"], default="cross_entropy"
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--synthetic-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--sampling-power", type=float, default=-1.0,
        help="Class inverse-frequency sampling exponent: 0 natural, 0.5 sqrt, 1 fully balanced; -1 uses pilot-name mapping",
    )
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--config-name", default="")
    parser.add_argument("--initialization", choices=["scratch", "imagenet"], default="scratch")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


class UltrasoundSquare:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        return preprocess_ultrasound_pil(image, self.size)


def build_transform(image_size: int, train: bool, augmentation: str = "traditional"):
    operations: List[object] = [UltrasoundSquare(image_size)]
    if train and augmentation == "traditional":
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=0),
                transforms.ColorJitter(brightness=0.10, contrast=0.12),
            ]
        )
    operations.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(operations)


class MMOTUFrameDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as handle:
            image = handle.convert("RGB")
        return self.transform(image), int(row.label), str(row.image_id), bool(row.is_synthetic)


def worker_seed(worker_id: int) -> None:
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)


def make_loader(frame: pd.DataFrame, args: argparse.Namespace, train: bool) -> DataLoader:
    sampler = None
    shuffle = train
    if train and args.sampling_power > 0:
        counts = frame.label.value_counts().to_dict()
        weights = [float(counts[int(label)]) ** (-args.sampling_power) for label in frame.label]
        sampler = WeightedRandomSampler(
            weights, num_samples=len(frame), replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        shuffle = False
    return DataLoader(
        MMOTUFrameDataset(
            frame,
            build_transform(args.image_size, train=train, augmentation=args.training_augmentation),
        ),
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_seed,
        generator=torch.Generator().manual_seed(args.seed),
    )


def build_model(device: torch.device, initialization: str = "scratch") -> nn.Module:
    weights = None if initialization == "scratch" else ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = convnext_tiny(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(MMOTU_CLASS_ORDER))
    return model.to(device)


def loss_values(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mode: str,
    class_counts: torch.Tensor,
    label_smoothing: float,
    focal_gamma: float,
) -> torch.Tensor:
    if loss_mode == "cross_entropy":
        return F.cross_entropy(logits, labels, reduction="none", label_smoothing=label_smoothing)
    if loss_mode == "balanced_softmax":
        return F.cross_entropy(logits + class_counts.clamp_min(1).log().unsqueeze(0), labels, reduction="none")
    if loss_mode == "focal":
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        alpha = class_counts.sum() / (len(class_counts) * class_counts.clamp_min(1))
        alpha = alpha / alpha.mean()
        return alpha[labels] * (1.0 - pt).pow(focal_gamma) * ce
    raise ValueError(loss_mode)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    class_counts: torch.Tensor,
):
    model.eval()
    losses, y_true, y_pred, y_prob, image_ids = [], [], [], [], []
    with torch.inference_mode():
        for images, labels, ids, _synthetic in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(enabled=True):
                logits = model(images)
                loss = loss_values(
                    logits, labels, args.loss_mode, class_counts, args.label_smoothing, args.focal_gamma
                ).mean()
            probabilities = torch.softmax(logits, dim=1)
            losses.append(float(loss.item()))
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(probabilities.argmax(1).cpu().tolist())
            y_prob.extend(probabilities.cpu().numpy())
            image_ids.extend(ids)
    return float(np.mean(losses)), y_true, y_pred, np.asarray(y_prob), image_ids


def main() -> None:
    args = parse_args()
    if args.sampling_power < 0:
        args.sampling_power = {"c1": 0.0, "c2": 0.5, "c3": 1.0}.get(args.config_name, 0.0)
    if not 0 <= args.sampling_power <= 1:
        raise ValueError("sampling-power must be in [0,1]")
    if not torch.cuda.is_available():
        raise RuntimeError("MMOTU ConvNeXt training requires CUDA")
    if not 0 < args.synthetic_loss_weight <= 1:
        raise ValueError("synthetic-loss-weight must be in (0,1]")
    seed_everything(args.seed)
    device = torch.device("cuda")
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_path = output_dir / "completed.json"
    if completed_path.is_file() and (output_dir / "best_model.pt").is_file():
        print(f"[MMOTU classifier] completed result exists: {output_dir}", flush=True)
        return

    train = pd.read_csv(ROOT / args.train_split)
    validation = pd.read_csv(ROOT / args.validation_split)
    train["is_synthetic"] = False
    validation["is_synthetic"] = False
    real_class_counts = torch.tensor(
        [int((train.label == index).sum()) for index in range(len(MMOTU_CLASS_ORDER))],
        dtype=torch.float32,
        device=device,
    )
    synthetic_count = 0
    manifest_sha256 = None
    if args.synthetic_manifest:
        import hashlib

        manifest_path = (ROOT / args.synthetic_manifest).resolve()
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        synthetic = pd.read_csv(manifest_path)
        required = {"image_id", "image_path", "label", "label_name"}
        if not required.issubset(synthetic.columns):
            raise RuntimeError(f"Synthetic manifest missing columns: {required - set(synthetic.columns)}")
        synthetic = synthetic.copy()
        synthetic["is_synthetic"] = True
        synthetic_count = len(synthetic)
        synthetic = synthetic.reindex(columns=train.columns)
        train = pd.concat([train, synthetic], ignore_index=True)

    test_accessed = args.evaluation_stage == "formal_test"
    test = None
    if test_accessed:
        test = pd.read_csv(ROOT / args.test_split)
        test["is_synthetic"] = False
    if set(train.label_name.unique()) != set(MMOTU_CLASS_ORDER):
        raise RuntimeError("Unexpected MMOTU class set")
    if train.image_id.astype(str).duplicated().any():
        raise RuntimeError("Duplicate image IDs in training frame")

    config = vars(args).copy()
    config.update(
        {
            "pretrained": args.initialization == "imagenet",
            "initialization": (
                "torchvision ConvNeXt-Tiny weights=None"
                if args.initialization == "scratch"
                else "torchvision ConvNeXt-Tiny IMAGENET1K_V1"
            ),
            "train_size": len(train),
            "real_train_size": int((~train.is_synthetic).sum()),
            "synthetic_count": synthetic_count,
            "synthetic_manifest_sha256": manifest_sha256,
            "test_accessed": test_accessed,
            "class_names": MMOTU_CLASS_ORDER,
            "minority_classes": MMOTU_MINORITY,
            "augmentation": (
                AUGMENTATION_DESCRIPTION if args.training_augmentation == "traditional"
                else NO_AUGMENTATION_DESCRIPTION
            ),
            "loss_mode": args.loss_mode,
            "focal_gamma": args.focal_gamma,
            "balanced_softmax_class_counts": real_class_counts.detach().cpu().int().tolist(),
            "sampling_power": args.sampling_power,
            "sampling_definition": "per-sample weight = class_count^(-sampling_power), replacement, epoch length unchanged",
        }
    )
    write_json(output_dir / "run_config.json", config)

    train_loader = make_loader(train, args, train=True)
    validation_loader = make_loader(validation, args, train=False)
    test_loader = make_loader(test, args, train=False) if test is not None else None
    model = build_model(device, args.initialization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=args.min_learning_rate
    )
    scaler = GradScaler(enabled=True)
    best_score = -float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for images, labels, _ids, is_synthetic in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            is_synthetic = is_synthetic.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                logits = model(images)
                losses = loss_values(
                    logits,
                    labels,
                    args.loss_mode,
                    real_class_counts,
                    args.label_smoothing,
                    args.focal_gamma,
                )
                weights = torch.ones_like(losses)
                weights[is_synthetic] = args.synthetic_loss_weight
                loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.item()))

        val_loss, y_true, y_pred, y_prob, _ = evaluate(model, validation_loader, device, args, real_class_counts)
        metrics = compute_classification_metrics(
            y_true, y_pred, y_prob, MMOTU_CLASS_ORDER, MMOTU_MINORITY,
            output_dir / "_validation_current", "validation", include_ordinal_agreement=False,
        )
        score = float(metrics["macro_f1"])
        if epoch <= args.warmup_epochs:
            warmup.step()
        else:
            cosine.step()
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "validation_loss": val_loss,
            "validation_balanced_accuracy": metrics["balanced_accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "validation_macro_auprc": metrics["macro_auprc"],
            "validation_minority_recall": metrics["minority_avg_recall"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "training_epoch_log.csv", index=False)
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        progress = {
            "status": "training",
            "epoch": epoch,
            "max_epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_score,
            "stale_epochs": stale,
            **row,
        }
        write_json(output_dir / "live_progress.json", progress)
        print(
            f"[MMOTU {args.config_name or output_dir.name}] epoch={epoch}/{args.epochs} "
            f"loss={row['train_loss']:.4f} valF1={metrics['macro_f1']:.4f} "
            f"valBA={metrics['balanced_accuracy']:.4f} valAUPRC={metrics['macro_auprc']:.4f} stale={stale}",
            flush=True,
        )
        if stale >= args.early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    model.load_state_dict(best_state)
    checkpoint = {
        "model_state": best_state,
        "architecture": "convnext_tiny",
        "num_classes": len(MMOTU_CLASS_ORDER),
        "class_names": MMOTU_CLASS_ORDER,
        "pretrained": args.initialization == "imagenet",
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "seed": args.seed,
        "config": config,
    }
    torch.save(checkpoint, output_dir / "best_model.pt")
    write_json(output_dir / "training_history.json", history)

    val_loss, y_true, y_pred, y_prob, ids = evaluate(model, validation_loader, device, args, real_class_counts)
    validation_metrics = compute_classification_metrics(
        y_true, y_pred, y_prob, MMOTU_CLASS_ORDER, MMOTU_MINORITY, output_dir, "validation",
        include_ordinal_agreement=False,
    )
    validation_metrics.update(
        {
            "architecture": "ConvNeXt-Tiny",
            "seed": args.seed,
            "pretrained": args.initialization == "imagenet",
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_score,
            "test_accessed": False,
            "synthetic_count": synthetic_count,
            "synthetic_manifest_sha256": manifest_sha256,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "synthetic_loss_weight": args.synthetic_loss_weight,
            "sampling_power": args.sampling_power,
        }
    )
    write_json(output_dir / "validation_metrics.json", validation_metrics)

    final_metrics = validation_metrics
    if test_loader is not None:
        test_loss, y_true, y_pred, y_prob, ids = evaluate(model, test_loader, device, args, real_class_counts)
        test_metrics = compute_classification_metrics(
            y_true, y_pred, y_prob, MMOTU_CLASS_ORDER, MMOTU_MINORITY, output_dir, "test",
            include_ordinal_agreement=False,
        )
        test_metrics.update(
            {
                "architecture": "ConvNeXt-Tiny",
                "seed": args.seed,
                "pretrained": args.initialization == "imagenet",
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_score,
                "test_accessed": True,
                "evaluation_stage": "formal_test",
                "synthetic_count": synthetic_count,
                "synthetic_manifest_sha256": manifest_sha256,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "synthetic_loss_weight": args.synthetic_loss_weight,
                "sampling_power": args.sampling_power,
                "training_seconds": time.time() - started,
            }
        )
        write_json(output_dir / "test_metrics.json", test_metrics)
        pd.DataFrame(
            {
                "image_id": ids,
                "true_label": y_true,
                "predicted_label": y_pred,
                **{f"prob_{index}_{name}": y_prob[:, index] for index, name in enumerate(MMOTU_CLASS_ORDER)},
            }
        ).to_csv(output_dir / "test_predictions.csv", index=False)
        final_metrics = test_metrics

    completed = {
        "status": "completed",
        "evaluation_stage": args.evaluation_stage,
        "test_accessed": test_accessed,
        "pretrained": args.initialization == "imagenet",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "synthetic_count": synthetic_count,
        "synthetic_manifest_sha256": manifest_sha256,
        "metrics": {
            "balanced_accuracy": final_metrics["balanced_accuracy"],
            "macro_f1": final_metrics["macro_f1"],
            "macro_auprc": final_metrics["macro_auprc"],
            "minority_recall": final_metrics["minority_avg_recall"],
        },
    }
    write_json(completed_path, completed)
    write_json(output_dir / "live_progress.json", completed)
    print(json.dumps(completed, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
