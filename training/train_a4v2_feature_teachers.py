from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.run_aptos_convnext_baselines import build_model, build_transform, seed_everything, worker_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train independent real-only A4-v2 feature teachers without test access")
    parser.add_argument("--a3-root", default="outputs/aptos2019/a3_adversarial_candidate/source_holdout_d4_v4/balanced_train")
    parser.add_argument("--output-dir", default="outputs/aptos2019/a4_from_a3_adversarial_candidate/source_holdout_d4_v4/feature_teachers")
    parser.add_argument("--seeds", default="2027,2028")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class FrameDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        return self.transform(Image.open(str(row.image_path)).convert("RGB")), int(row.label)


def loader(frame: pd.DataFrame, transform, batch_size: int, workers: int, seed: int, shuffle: bool):
    return DataLoader(
        FrameDataset(frame, transform), batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0, worker_init_fn=worker_seed,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def evaluate(model: nn.Module, data: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    truth: List[int] = []
    pred: List[int] = []
    losses: List[float] = []
    for images, labels in data:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=True):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
        truth.extend(labels.cpu().tolist())
        pred.extend(logits.argmax(1).cpu().tolist())
        losses.append(float(loss.item()))
    return {
        "loss": float(np.mean(losses)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
    }


def train_seed(seed: int, train: pd.DataFrame, val: pd.DataFrame, args: argparse.Namespace, device: torch.device):
    out = (ROOT / args.output_dir / f"seed_{seed}").resolve()
    checkpoint = out / "best_model.pt"
    completed = out / "completed.json"
    if checkpoint.is_file() and completed.is_file():
        print(f"seed {seed}: reuse completed teacher", flush=True)
        return json.loads(completed.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    # Exact Original-CE representation protocol: no stochastic traditional augmentation.
    transform = build_transform(args.image_size, traditional=False)
    train_loader = loader(train, transform, args.batch_size, args.num_workers, seed, True)
    val_loader = loader(val, transform, args.batch_size, args.num_workers, seed, False)
    model = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=args.min_learning_rate
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[args.warmup_epochs]
    )
    scaler = GradScaler(enabled=True)
    best = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    history: List[Dict[str, float]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                loss = nn.functional.cross_entropy(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": optimizer.param_groups[0]["lr"], **metrics}
        history.append(row)
        if metrics["macro_f1"] > best + 1e-12:
            best = metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        scheduler.step()
        write_json(out / "live_progress.json", {
            "status": "training", "seed": seed, "epoch": epoch, "best_epoch": best_epoch,
            "best_val_macro_f1": best, "current": row, "stale_epochs": stale,
            "test_accessed": False, "elapsed_minutes": (time.time() - started) / 60,
        })
        print(
            f"teacher seed={seed} epoch={epoch:02d} loss={row['train_loss']:.4f} "
            f"val_f1={metrics['macro_f1']:.4f} val_ba={metrics['balanced_accuracy']:.4f} "
            f"best={best:.4f}@{best_epoch}", flush=True,
        )
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No valid teacher checkpoint")
    torch.save({
        "model_state": best_state, "architecture": "convnext_tiny", "pretrained": False,
        "initialization": "random", "training_source": "real_train_only",
        "selection_source": "real_validation_only", "test_accessed": False,
        "seed": seed, "best_epoch": best_epoch, "best_val_macro_f1": best, "training_args": vars(args),
    }, checkpoint)
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    result = {
        "status": "completed", "seed": seed, "epochs_ran": len(history), "best_epoch": best_epoch,
        "best_val_macro_f1": best, "test_accessed": False, "checkpoint": str(checkpoint),
        "elapsed_minutes": (time.time() - started) / 60,
    }
    write_json(completed, result)
    write_json(out / "live_progress.json", result)
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A4-v2 teacher training requires CUDA")
    root = (ROOT / args.a3_root).resolve()
    # Deliberately load train/val only. No test CSV path is constructed.
    train = pd.read_csv(root / "splits/train.csv")
    val = pd.read_csv(root / "splits/val.csv")
    device = torch.device("cuda")
    results = [train_seed(int(seed), train, val, args, device) for seed in args.seeds.split(",") if seed.strip()]
    write_json(ROOT / args.output_dir / "summary.json", {"results": results, "test_accessed": False})
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
