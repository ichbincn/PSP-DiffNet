from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny, densenet121, resnet50, swin_t
from torchvision.models.vision_transformer import VisionTransformer


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache/torch"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.aptos_dataset import APTOS_CLASS_ORDER
from metrics.classification_metrics import compute_classification_metrics
from training.run_aptos_convnext_baselines import AptosFrameDataset, build_transform, seed_everything, worker_seed


ARCHITECTURES = ["resnet50", "vit_small", "densenet121", "swin_tiny", "convnext_tiny"]
DISPLAY = {
    "resnet50": "ResNet50",
    "vit_small": "ViT-Small/16",
    "densenet121": "DenseNet121",
    "swin_tiny": "Swin-Tiny",
    "convnext_tiny": "ConvNeXt-Tiny",
}
METRICS = {
    "balanced_accuracy": "Balanced Accuracy",
    "macro_f1": "Macro-F1",
    "macro_auprc": "Macro-AUPRC",
    "minority_avg_recall": "Minority Recall",
    "quadratic_weighted_kappa": "QWK",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="APTOS scratch-classifier architecture generalization")
    parser.add_argument("--architecture", required=True, choices=ARCHITECTURES)
    parser.add_argument("--condition", required=True, choices=["real_only", "final_a4"])
    parser.add_argument("--ratio-root", default="outputs/aptos2019/a4_ratio_ablation")
    parser.add_argument("--output-root", default="outputs/aptos2019/a4_classifier_generalization")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--evaluation-stage", choices=["validation_only", "formal_test"], default="formal_test")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--synthetic-loss-weight", type=float, default=1.0)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--initialization", choices=["scratch", "imagenet"], default="scratch")
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_model(name: str, device: torch.device, initialization: str) -> nn.Module:
    if name == "resnet50":
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(APTOS_CLASS_ORDER))
    elif name == "densenet121":
        model = densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, len(APTOS_CLASS_ORDER))
    elif name == "swin_tiny":
        model = swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, len(APTOS_CLASS_ORDER))
    elif name == "vit_small":
        # Canonical ViT-S/16 dimensions; constructed locally with random weights.
        model = VisionTransformer(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=1536,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=len(APTOS_CLASS_ORDER),
        )
    elif name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if initialization == "imagenet" else None
        model = convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(APTOS_CLASS_ORDER))
    else:
        raise ValueError(name)
    if initialization == "imagenet" and name != "convnext_tiny":
        raise ValueError("This runner currently exposes ImageNet initialization only for ConvNeXt-Tiny")
    return model.to(device)


def make_loader(frame: pd.DataFrame, transform, args: argparse.Namespace, train: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        AptosFrameDataset(frame, transform),
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_seed,
        generator=generator,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    losses, y_true, y_pred, y_prob, image_ids = [], [], [], [], []
    for images, labels, ids in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=True):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
        probabilities = torch.softmax(logits.float(), dim=1)
        losses.append(float(loss.item()))
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(probabilities.argmax(1).cpu().tolist())
        y_prob.extend(probabilities.cpu().numpy())
        image_ids.extend(ids)
    return float(np.mean(losses)), y_true, y_pred, np.asarray(y_prob), image_ids


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Classifier generalization requires CUDA")
    seed_everything(args.seed)
    ratio_summary = json.loads((ROOT / "实验分析/眼底数据/FINAL_SYNTHETIC_RATIO.json").read_text(encoding="utf-8"))
    final_ratio = int(ratio_summary["FINAL_SYNTHETIC_RATIO"])
    if not 0 < args.synthetic_loss_weight <= 1:
        raise ValueError("--synthetic-loss-weight must be in (0, 1]")
    data_root = (ROOT / args.ratio_root / f"ratio_{final_ratio:03d}/balanced_train").resolve()
    output_dir = (
        (ROOT / args.output_dir).resolve()
        if args.output_dir
        else (ROOT / args.output_root / args.architecture / args.condition / f"seed_{args.seed}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = output_dir / "completed.json"
    if completed.is_file():
        print(completed.read_text(encoding="utf-8"), flush=True)
        return
    split_names = ["train", "val"] + (["test"] if args.evaluation_stage == "formal_test" else [])
    splits = {name: pd.read_csv(data_root / "splits" / f"{name}.csv") for name in split_names}
    manifest_path = data_root / "generated_manifest.csv"
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    synthetic_ids = set()
    if args.condition == "final_a4":
        generated = pd.read_csv(manifest_path)
        if not generated.image_path.astype(str).map(lambda value: Path(value).is_file()).all():
            raise RuntimeError("The fixed final synthetic pool has missing images")
        synthetic = generated[["image_id", "image_path", "label", "label_name", "dataset_name"]]
        synthetic_ids = set(synthetic["image_id"].astype(str))
        train = pd.concat([splits["train"], synthetic], ignore_index=True)
    else:
        train = splits["train"].copy()
    write_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "architecture_display": DISPLAY[args.architecture],
            "pretrained": args.initialization == "imagenet",
            "initialization_detail": (
                "torchvision ConvNeXt-Tiny IMAGENET1K_V1; all layers fine-tuned"
                if args.initialization == "imagenet"
                else "random initialization; no external classifier weights"
            ),
            "final_synthetic_ratio": final_ratio,
            "real_train_size": int(len(splits["train"])),
            "synthetic_count": int(len(train) - len(splits["train"])),
            "train_size": int(len(train)),
            "validation_size": int(len(splits["val"])),
            "test_size": int(len(splits.get("test", []))),
            "final_a4_manifest_sha256": manifest_sha256,
            "synthetic_only_in_train": True,
            "test_accessed_during_training_or_selection": False,
        },
    )
    train_loader = make_loader(train, build_transform(args.image_size, traditional=True), args, True)
    val_loader = make_loader(splits["val"], build_transform(args.image_size, traditional=False), args, False)
    test_loader = (
        make_loader(splits["test"], build_transform(args.image_size, traditional=False), args, False)
        if args.evaluation_stage == "formal_test"
        else None
    )
    device = torch.device("cuda")
    model = build_model(args.architecture, device, args.initialization)
    parameter_count = int(sum(value.numel() for value in model.parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=args.min_learning_rate
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[args.warmup_epochs]
    )
    scaler = GradScaler(enabled=True)
    best_state = None
    best_val = -1.0
    patience = 0
    history: List[Dict[str, float]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, labels, batch_ids in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                logits = model(images)
                per_sample_loss = nn.functional.cross_entropy(
                    logits, labels, label_smoothing=args.label_smoothing, reduction="none"
                )
                if args.condition == "final_a4" and args.synthetic_loss_weight < 1.0:
                    sample_weights = torch.tensor(
                        [args.synthetic_loss_weight if str(image_id) in synthetic_ids else 1.0 for image_id in batch_ids],
                        dtype=per_sample_loss.dtype,
                        device=device,
                    )
                    loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
                else:
                    loss = per_sample_loss.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        val_loss, val_true, val_pred, _val_prob, _ids = evaluate(model, val_loader, device)
        val_f1 = float(f1_score(val_true, val_pred, average="macro", zero_division=0))
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
            }
        )
        pd.DataFrame(history).to_csv(output_dir / "training_epoch_log.csv", index=False)
        write_json(
            output_dir / "live_progress.json",
            {
                "status": "training",
                "architecture": args.architecture,
                "condition": args.condition,
                "epoch": epoch,
                "max_epochs": args.epochs,
                "train_size": len(train),
                "val_macro_f1": val_f1,
                "best_val_macro_f1": max(best_val, val_f1),
                "elapsed_seconds": time.time() - started,
                "evaluation_stage": args.evaluation_stage,
                "test_accessed": False,
            },
        )
        print(
            f"[{args.architecture}/{args.condition}] epoch {epoch:02d}/{args.epochs} "
            f"train={np.mean(losses):.4f} val={val_loss:.4f} val_macro_f1={val_f1:.4f}",
            flush=True,
        )
        if val_f1 > best_val + 1e-6:
            best_val = val_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        scheduler.step()
        if patience >= args.early_stopping_patience:
            print(f"[{args.architecture}/{args.condition}] early stopped at epoch {epoch}", flush=True)
            break
    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    _val_loss, val_true, val_pred, val_prob, val_image_ids = evaluate(model, val_loader, device)
    validation_metrics = compute_classification_metrics(
        val_true,
        val_pred,
        val_prob,
        APTOS_CLASS_ORDER,
        APTOS_CLASS_ORDER[1:],
        output_dir,
        "validation",
    )
    validation_metrics.update(
        {
            "architecture": args.architecture,
            "condition": args.condition,
            "seed": args.seed,
            "pretrained": args.initialization == "imagenet",
            "initialization": args.initialization,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "synthetic_loss_weight": args.synthetic_loss_weight,
            "best_val_macro_f1": best_val,
            "epochs_ran": len(history),
            "test_accessed": False,
            "final_a4_manifest_sha256": manifest_sha256,
        }
    )
    write_json(output_dir / "validation_metrics.json", validation_metrics)
    torch.save(
        {
            "model_state": best_state,
            "architecture": args.architecture,
            "pretrained": args.initialization == "imagenet",
            "class_names": APTOS_CLASS_ORDER,
            "training_args": vars(args),
            "validation_metrics": validation_metrics,
        },
        output_dir / "best_model.pt",
    )
    if args.evaluation_stage == "validation_only":
        completed_payload = {
            "status": "validation_only_completed",
            "evaluation_stage": "validation_only",
            "test_accessed": False,
            **validation_metrics,
        }
        write_json(output_dir / "completed.json", completed_payload)
        write_json(output_dir / "live_progress.json", completed_payload)
        print(json.dumps(completed_payload, ensure_ascii=False, indent=2), flush=True)
        return
    if test_loader is None:
        raise RuntimeError("Formal test loader was not constructed")
    _loss, y_true, y_pred, y_prob, image_ids = evaluate(model, test_loader, device)
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        y_prob,
        APTOS_CLASS_ORDER,
        APTOS_CLASS_ORDER[1:],
        output_dir,
        "test",
    )
    metrics.update(
        {
            "architecture": args.architecture,
            "architecture_display": DISPLAY[args.architecture],
            "condition": args.condition,
            "seed": args.seed,
            "pretrained": args.initialization == "imagenet",
            "initialization": (
                "torchvision ConvNeXt-Tiny IMAGENET1K_V1; all layers fine-tuned"
                if args.initialization == "imagenet"
                else "random initialization; no external classifier weights"
            ),
            "final_synthetic_ratio": final_ratio,
            "train_size": int(len(train)),
            "parameter_count": parameter_count,
            "label_smoothing": args.label_smoothing,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "synthetic_loss_weight": args.synthetic_loss_weight,
            "evaluation_stage": "formal_test",
            "test_accessed": True,
            "final_a4_manifest_sha256": manifest_sha256,
            "best_val_macro_f1": best_val,
            "epochs_ran": len(history),
            "training_seconds": time.time() - started,
        }
    )
    write_json(output_dir / "test_metrics.json", metrics)
    write_json(output_dir / "training_history.json", history)
    prediction_rows = []
    for index, image_id in enumerate(image_ids):
        row = {"image_id": image_id, "true_label": y_true[index], "predicted_label": y_pred[index]}
        row.update({f"prob_{name}": float(y_prob[index, class_index]) for class_index, name in enumerate(APTOS_CLASS_ORDER)})
        prediction_rows.append(row)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "test_predictions.csv", index=False)
    checkpoint = torch.load(output_dir / "best_model.pt", map_location="cpu")
    checkpoint["metrics"] = metrics
    torch.save(checkpoint, output_dir / "best_model.pt")
    write_json(output_dir / "completed.json", {"status": "completed", **metrics})
    write_json(output_dir / "live_progress.json", {"status": "completed", **metrics})
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
