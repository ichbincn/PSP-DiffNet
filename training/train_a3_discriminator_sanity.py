from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.aptos_dataset import APTOS_CLASS_ORDER
from metrics.classification_metrics import compute_classification_metrics
from models.a3_diagnostic_discriminator import A3DiagnosticDiscriminator
from training.run_aptos_convnext_baselines import FundusPreprocess, build_model, build_transform

MINORITY = ["Mild", "Moderate", "Severe", "Proliferate_DR"]
TEACHERS = {
    "original_ce": "outputs/aptos2019/convnext_tiny_scratch_preliminary/original_ce/best_model.pt",
    "balanced_softmax": "outputs/aptos2019/convnext_tiny_scratch_preliminary/balanced_softmax/best_model.pt",
    "focal_loss": "outputs/aptos2019/convnext_tiny_scratch_preliminary/focal_loss/best_model.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A3 leakage-safe three-head discriminator sanity")
    parser.add_argument("--output-dir", default="outputs/aptos2019/reproductions/a3_discriminator_sanity")
    parser.add_argument("--real-root", default="outputs/aptos2019/structure_guidance_v3/balanced_train")
    parser.add_argument("--a1-manifest", default="outputs/aptos2019/structure_guidance_v3/balanced_train/generated_manifest.csv")
    parser.add_argument("--prior-index", default="outputs/aptos2019/structure_guidance_v3/weak_localization_priors/prior_index.csv")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--class-epochs", type=int, default=20)
    parser.add_argument("--aux-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--samples-per-class", type=int, default=512)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--class-head-lr", type=float, default=1e-4)
    parser.add_argument(
        "--class-mode",
        choices=["student", "teacher_ensemble"],
        default="teacher_ensemble",
        help="Use the validated frozen three-teacher ensemble or train an experimental distilled student.",
    )
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
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


def grouped_calibration(image_id: str, seed: int) -> bool:
    digest = hashlib.sha256(f"a3:{seed}:{image_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 5 == 0


def prior_transform(path: str, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("L")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class ClassDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform, teacher_probs: Dict[str, np.ndarray]):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.teacher_probs = teacher_probs

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = self.transform(Image.open(row.image_path).convert("RGB"))
        image_id = str(row.image_id)
        return image, int(row.label), torch.from_numpy(self.teacher_probs[image_id]).float(), image_id


class PriorDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int, fake: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size
        self.fake = fake
        self.transform = transforms.Compose(
            [
                FundusPreprocess(),
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.by_class: Dict[str, List[int]] = {}
        for index, class_name in enumerate(self.frame.label_name.astype(str)):
            self.by_class.setdefault(class_name, []).append(index)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = self.transform(Image.open(row.image_path).convert("RGB"))
        prior = prior_transform(str(row.prior_path), self.image_size)
        candidates = self.by_class[str(row.label_name)]
        position = candidates.index(index)
        mismatch_index = candidates[(position + 1) % len(candidates)]
        mismatch = prior_transform(str(self.frame.iloc[mismatch_index].prior_path), self.image_size)
        source_id = str(row.source_image_id if self.fake else row.image_id)
        return image, int(row.label), prior, mismatch, source_id


def balanced_sampler(frame: pd.DataFrame, total_samples: int, seed: int) -> WeightedRandomSampler:
    counts = frame.label.value_counts().to_dict()
    weights = torch.tensor([1.0 / counts[int(label)] for label in frame.label], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=total_samples, replacement=True, generator=generator)


@torch.no_grad()
def compute_teacher_cache(
    frame: pd.DataFrame, output_dir: Path, device: torch.device, image_size: int, batch_size: int, workers: int
) -> Dict[str, np.ndarray]:
    cache_path = output_dir / "teacher_ensemble_probs.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return {str(key): value for key, value in zip(cached["image_ids"].tolist(), cached["probs"])}

    deterministic = build_transform(image_size, traditional=False)
    bare = frame.reset_index(drop=True)

    class BareDataset(Dataset):
        def __len__(self):
            return len(bare)

        def __getitem__(self, index):
            row = bare.iloc[index]
            return deterministic(Image.open(row.image_path).convert("RGB")), str(row.image_id)

    loader = DataLoader(BareDataset(), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    all_teacher_probs = []
    for name, relative_path in TEACHERS.items():
        checkpoint = torch.load(ROOT / relative_path, map_location="cpu")
        if checkpoint.get("pretrained") is not False:
            raise RuntimeError(f"Teacher {name} is not a weights=None checkpoint")
        model = build_model(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        teacher_batches = []
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            teacher_batches.append(torch.softmax(model(images), dim=1).cpu().numpy())
        all_teacher_probs.append(np.concatenate(teacher_batches, axis=0))
        del model
        torch.cuda.empty_cache()
    mean_probs = np.mean(np.stack(all_teacher_probs, axis=0), axis=0).astype(np.float32)
    image_ids = bare.image_id.astype(str).to_numpy()
    np.savez_compressed(cache_path, image_ids=image_ids, probs=mean_probs)
    return {str(key): value for key, value in zip(image_ids.tolist(), mean_probs)}


def basic_metrics(y_true: List[int], probs: np.ndarray) -> Dict[str, object]:
    pred = probs.argmax(axis=1)
    recalls = recall_score(y_true, pred, labels=list(range(5)), average=None, zero_division=0)
    return {
        "accuracy": float(np.mean(pred == np.asarray(y_true))),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "per_class_recall": {name: float(recalls[i]) for i, name in enumerate(APTOS_CLASS_ORDER)},
    }


@torch.no_grad()
def evaluate_class(model, loader, device) -> Tuple[Dict[str, object], List[int], np.ndarray]:
    model.eval()
    truth: List[int] = []
    probabilities = []
    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        probabilities.append(torch.softmax(model.class_logits(images), dim=1).cpu().numpy())
        truth.extend(labels.tolist())
    probs = np.concatenate(probabilities, axis=0)
    return basic_metrics(truth, probs), truth, probs


def train_class_phase(model, train_frame, val_frame, teacher_probs, args, output_dir, device):
    model.set_class_warmup_trainable()
    checkpoint = torch.load(ROOT / TEACHERS["original_ce"], map_location="cpu")
    model.load_medical_teacher(checkpoint["model_state"])
    model.set_class_warmup_trainable()
    class_trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    classifier_params = list(model.backbone.classifier.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": classifier_params, "lr": args.class_head_lr}],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.class_epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=True)
    train_dataset = ClassDataset(train_frame, build_transform(args.image_size, traditional=True), teacher_probs)
    val_dataset = ClassDataset(val_frame, build_transform(args.image_size, traditional=False), teacher_probs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=balanced_sampler(train_frame, args.samples_per_class * 5, args.seed),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    teacher_val_probs = np.stack([teacher_probs[str(image_id)] for image_id in val_frame.image_id], axis=0)
    teacher_metrics = basic_metrics(val_frame.label.astype(int).tolist(), teacher_val_probs)
    write_json(output_dir / "teacher_ensemble_val_metrics.json", teacher_metrics)

    history = []
    best = -1.0
    stale = 0
    best_path = output_dir / "best_class_phase.pt"
    for epoch in range(args.class_epochs):
        model.train()
        losses = []
        for images, labels, soft_targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            soft_targets = soft_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                logits = model.class_logits(images)
                hard = F.cross_entropy(logits, labels)
                temperature = 2.0
                distill = F.kl_div(
                    F.log_softmax(logits / temperature, dim=1), soft_targets, reduction="batchmean"
                ) * temperature**2
                # The ensemble distribution is the primary target. A small
                # hard-label term keeps the student grounded without letting
                # class-balanced repetition overwrite the teacher boundary.
                loss = 0.25 * hard + distill
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(class_trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        scheduler.step()
        metrics, _, _ = evaluate_class(model, val_loader, device)
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), **metrics}
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "class_training_history.csv", index=False)
        print(f"[A3 D class] seed={args.seed} epoch={epoch+1}/{args.class_epochs} loss={row['loss']:.4f} val_macro_f1={row['macro_f1']:.4f}", flush=True)
        if row["macro_f1"] > best:
            best = float(row["macro_f1"])
            stale = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch + 1, "metrics": metrics}, best_path)
        else:
            stale += 1
            if stale >= args.patience:
                break
    selected = torch.load(best_path, map_location="cpu")
    model.load_state_dict(selected["model_state"], strict=True)
    metrics, y_true, probs = evaluate_class(model, val_loader, device)
    final_metrics = compute_classification_metrics(y_true, probs.argmax(1).tolist(), probs, APTOS_CLASS_ORDER, MINORITY, output_dir, "class_val")
    shuffled = np.asarray(y_true).copy()
    np.random.default_rng(args.seed).shuffle(shuffled)
    shuffle_metrics = basic_metrics(shuffled.tolist(), probs)
    write_json(output_dir / "class_label_shuffle_metrics.json", shuffle_metrics)
    return metrics, teacher_metrics, shuffle_metrics


@torch.no_grad()
def evaluate_aux(model, real_loader, fake_loader, device) -> Dict[str, object]:
    model.eval()
    realism_true, realism_score = [], []
    structure_true, structure_score = [], []
    matched_structure_score, mismatch_structure_score = [], []
    shortcut_brightness, shortcut_contrast, shortcut_black_fraction = [], [], []
    for real_batch, fake_batch in zip(real_loader, fake_loader):
        real_images, real_labels, prior, mismatch, _ = real_batch
        fake_images, fake_labels, fake_prior, _, _ = fake_batch
        real_images = real_images.to(device); real_labels = real_labels.to(device)
        prior = prior.to(device); mismatch = mismatch.to(device)
        fake_images = fake_images.to(device); fake_labels = fake_labels.to(device); fake_prior = fake_prior.to(device)
        real_mid, real_feature, _ = model.encode(real_images)
        fake_mid, fake_feature, _ = model.encode(fake_images)
        real_logit = model.realism_from_features(real_feature, real_labels)
        fake_logit = model.realism_from_features(fake_feature, fake_labels)
        matched_logit = model.structure_from_features(real_mid, prior)
        mismatch_logit = model.structure_from_features(real_mid, mismatch)
        realism_true.extend([1] * len(real_logit) + [0] * len(fake_logit))
        realism_score.extend(torch.sigmoid(torch.cat([real_logit, fake_logit])).cpu().flatten().tolist())
        combined_images = torch.cat([real_images, fake_images], dim=0)
        mean = combined_images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = combined_images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        rgb = (combined_images * std + mean).clamp(0.0, 1.0)
        gray = rgb.mean(dim=1)
        shortcut_brightness.extend(gray.mean(dim=(1, 2)).cpu().tolist())
        shortcut_contrast.extend(gray.std(dim=(1, 2), unbiased=False).cpu().tolist())
        shortcut_black_fraction.extend((gray < 0.02).float().mean(dim=(1, 2)).cpu().tolist())
        structure_true.extend([1] * len(matched_logit) + [0] * len(mismatch_logit))
        structure_score.extend(torch.sigmoid(torch.cat([matched_logit, mismatch_logit])).cpu().flatten().tolist())
        matched_structure_score.extend(torch.sigmoid(matched_logit).cpu().flatten().tolist())
        mismatch_structure_score.extend(torch.sigmoid(mismatch_logit).cpu().flatten().tolist())
    realism_pred = (np.asarray(realism_score) >= 0.5).astype(int)
    structure_pred = (np.asarray(structure_score) >= 0.5).astype(int)
    def absolute_correlation(values: List[float]) -> float:
        left = np.asarray(realism_score, dtype=np.float64)
        right = np.asarray(values, dtype=np.float64)
        if left.std() < 1e-12 or right.std() < 1e-12:
            return 0.0
        return float(abs(np.corrcoef(left, right)[0, 1]))

    shortcut_correlations = {
        "brightness": absolute_correlation(shortcut_brightness),
        "contrast": absolute_correlation(shortcut_contrast),
        "black_fraction": absolute_correlation(shortcut_black_fraction),
    }
    return {
        "realism_auroc": float(roc_auc_score(realism_true, realism_score)),
        "realism_balanced_accuracy": float(balanced_accuracy_score(realism_true, realism_pred)),
        "structure_auroc": float(roc_auc_score(structure_true, structure_score)),
        "structure_balanced_accuracy": float(balanced_accuracy_score(structure_true, structure_pred)),
        "prior_shuffle_positive_score_drop": float(
            np.mean(np.asarray(matched_structure_score)) - np.mean(np.asarray(mismatch_structure_score))
        ),
        "realism_shortcut_abs_correlation": shortcut_correlations,
        "realism_shortcut_max_abs_correlation": float(max(shortcut_correlations.values())),
    }


def train_aux_phase(model, real_train, real_cal, fake_train, fake_cal, args, output_dir, device):
    model.set_aux_heads_trainable()
    trainable = list(model.trainable_parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.head_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.aux_epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=True)
    samples = args.samples_per_class * 4

    def loaders(real_frame, fake_frame, train: bool, seed_offset: int):
        real_ds = PriorDataset(real_frame, args.image_size, fake=False)
        fake_ds = PriorDataset(fake_frame, args.image_size, fake=True)
        if train:
            real_sampler = balanced_sampler(real_frame, samples, args.seed + seed_offset)
            fake_sampler = balanced_sampler(fake_frame, samples, args.seed + seed_offset + 1)
            return (
                DataLoader(real_ds, batch_size=args.batch_size, sampler=real_sampler, num_workers=args.num_workers, pin_memory=True),
                DataLoader(fake_ds, batch_size=args.batch_size, sampler=fake_sampler, num_workers=args.num_workers, pin_memory=True),
            )
        return (
            DataLoader(real_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
            DataLoader(fake_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
        )

    real_cal_loader, fake_cal_loader = loaders(real_cal, fake_cal, False, 0)
    history = []
    best = -1.0
    stale = 0
    best_path = output_dir / "best_discriminator.pt"
    for epoch in range(args.aux_epochs):
        real_loader, fake_loader = loaders(real_train, fake_train, True, epoch * 10)
        model.train(); model.backbone.eval()
        losses = []
        for real_batch, fake_batch in zip(real_loader, fake_loader):
            real_images, real_labels, prior, mismatch, _ = real_batch
            fake_images, fake_labels, _, _, _ = fake_batch
            real_images = real_images.to(device); real_labels = real_labels.to(device)
            prior = prior.to(device); mismatch = mismatch.to(device)
            fake_images = fake_images.to(device); fake_labels = fake_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), autocast(enabled=True):
                real_mid, real_feature, _ = model.encode(real_images)
                _, fake_feature, _ = model.encode(fake_images)
            with autocast(enabled=True):
                real_logit = model.realism_from_features(real_feature, real_labels)
                fake_logit = model.realism_from_features(fake_feature, fake_labels)
                matched_logit = model.structure_from_features(real_mid, prior)
                mismatch_logit = model.structure_from_features(real_mid, mismatch)
                realism = F.binary_cross_entropy_with_logits(real_logit, torch.ones_like(real_logit))
                realism += F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit))
                structure = F.binary_cross_entropy_with_logits(matched_logit, torch.ones_like(matched_logit))
                structure += F.binary_cross_entropy_with_logits(mismatch_logit, torch.zeros_like(mismatch_logit))
                loss = realism + structure
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.item()))
        scheduler.step()
        metrics = evaluate_aux(model, real_cal_loader, fake_cal_loader, device)
        # Select the checkpoint by its weakest normalized sanity margin. This
        # prevents a surplus on the easier structure task from hiding a miss
        # on the pre-registered realism threshold.
        bottleneck = min(metrics["realism_auroc"] / 0.65, metrics["structure_auroc"] / 0.75)
        score = bottleneck + 1e-3 * (metrics["realism_auroc"] + metrics["structure_auroc"])
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "selection_score": score, **metrics}
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "aux_training_history.csv", index=False)
        print(f"[A3 D aux] seed={args.seed} epoch={epoch+1}/{args.aux_epochs} loss={row['loss']:.4f} realism_auc={metrics['realism_auroc']:.4f} structure_auc={metrics['structure_auroc']:.4f}", flush=True)
        if score > best:
            best = score; stale = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch + 1, "metrics": metrics, "pretrained": False}, best_path)
        else:
            stale += 1
            if stale >= args.patience:
                break
    selected = torch.load(best_path, map_location="cpu")
    model.load_state_dict(selected["model_state"], strict=True)
    return evaluate_aux(model, real_cal_loader, fake_cal_loader, device), int(selected["epoch"])


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A3 discriminator sanity requires CUDA")
    seed_everything(args.seed)
    device = torch.device("cuda")
    root = (ROOT / args.output_dir / f"seed_{args.seed}").resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config = vars(args).copy()
    config.update({"test_data_used": False, "pretrained_imagenet": False, "device": torch.cuda.get_device_name(0)})
    write_json(root / "run_config.json", config)

    real_root = (ROOT / args.real_root).resolve()
    train_frame = pd.read_csv(real_root / "splits/train.csv")
    val_frame = pd.read_csv(real_root / "splits/val.csv")
    prior_index = pd.read_csv(ROOT / args.prior_index)
    prior_map = dict(zip(prior_index.image_id.astype(str), prior_index.prior_path.astype(str)))
    real_minority = train_frame[train_frame.label_name.isin(MINORITY)].copy()
    real_minority["prior_path"] = real_minority.image_id.astype(str).map(prior_map)
    if real_minority.prior_path.isna().any():
        raise RuntimeError("Missing A1 prior for a real minority training image")
    fake = pd.read_csv(ROOT / args.a1_manifest, encoding="utf-8-sig").copy()
    fake["source_image_id"] = fake.source_image_id.astype(str)

    real_minority["is_cal"] = real_minority.image_id.astype(str).map(lambda value: grouped_calibration(value, args.seed))
    fake["is_cal"] = fake.source_image_id.map(lambda value: grouped_calibration(value, args.seed))
    real_train = real_minority[~real_minority.is_cal].copy()
    real_cal = real_minority[real_minority.is_cal].copy()
    fake_train = fake[~fake.is_cal].copy()
    fake_cal = fake[fake.is_cal].copy()
    split_summary = {
        "real_train": len(real_train), "real_cal": len(real_cal),
        "fake_train": len(fake_train), "fake_cal": len(fake_cal),
        "source_overlap": len(set(real_train.image_id.astype(str)) & set(real_cal.image_id.astype(str))),
    }
    write_json(root / "source_group_split.json", split_summary)
    if split_summary["source_overlap"]:
        raise RuntimeError("Source-group leakage in A3 discriminator split")

    all_real = pd.concat([train_frame, val_frame], ignore_index=True)
    teacher_probs = compute_teacher_cache(all_real, root, device, args.image_size, args.batch_size, args.num_workers)
    model = A3DiagnosticDiscriminator(num_classes=5).to(device)
    teacher_val_probs = np.stack([teacher_probs[str(image_id)] for image_id in val_frame.image_id], axis=0)
    teacher_metrics = basic_metrics(val_frame.label.astype(int).tolist(), teacher_val_probs)
    write_json(root / "teacher_ensemble_val_metrics.json", teacher_metrics)
    if args.class_mode == "student":
        class_metrics, teacher_metrics, shuffle_metrics = train_class_phase(
            model, train_frame, val_frame, teacher_probs, args, root, device
        )
    else:
        # The class-consistency branch is the mean of three independently
        # trained weights=None medical teachers. It reads image pixels only;
        # the target label enters the loss after prediction, never the branch.
        class_metrics = teacher_metrics
        shuffled = val_frame.label.astype(int).to_numpy().copy()
        np.random.default_rng(args.seed).shuffle(shuffled)
        shuffle_metrics = basic_metrics(shuffled.tolist(), teacher_val_probs)
        write_json(root / "class_label_shuffle_metrics.json", shuffle_metrics)
        checkpoint = torch.load(ROOT / TEACHERS["original_ce"], map_location="cpu")
        model.load_medical_teacher(checkpoint["model_state"])
        write_json(
            root / "class_branch_definition.json",
            {
                "mode": "frozen_three_teacher_ensemble",
                "teachers": TEACHERS,
                "aggregation": "arithmetic_mean_of_softmax_probabilities",
                "pretrained_imagenet": False,
                "target_label_is_model_input": False,
                "student_attempt_rejected": True,
            },
        )
    aux_metrics, aux_epoch = train_aux_phase(model, real_train, real_cal, fake_train, fake_cal, args, root, device)

    class_threshold = float(teacher_metrics["macro_f1"]) - 0.02
    class_pass = float(class_metrics["macro_f1"]) >= class_threshold
    teacher_tail = teacher_metrics["per_class_recall"]
    class_tail = class_metrics["per_class_recall"]
    tail_pass = not (
        class_tail["Severe"] < teacher_tail["Severe"] - 0.05
        and class_tail["Proliferate_DR"] < teacher_tail["Proliferate_DR"] - 0.05
    )
    leakage_pass = float(shuffle_metrics["balanced_accuracy"]) <= 0.30
    sanity = {
        "seed": args.seed,
        "class_mode": args.class_mode,
        "class_metrics": class_metrics,
        "teacher_ensemble_metrics": teacher_metrics,
        "label_shuffle_metrics": shuffle_metrics,
        "aux_metrics": aux_metrics,
        "selected_aux_epoch": aux_epoch,
        "thresholds": {
            "class_macro_f1_min": class_threshold,
            "realism_auroc_min": 0.65,
            "structure_auroc_min": 0.75,
            "label_shuffle_balanced_accuracy_max": 0.30,
        },
        "checks": {
            "class_pass": class_pass,
            "tail_pass": tail_pass,
            "realism_pass": aux_metrics["realism_auroc"] >= 0.65,
            "structure_pass": aux_metrics["structure_auroc"] >= 0.75,
            "label_leakage_pass": leakage_pass,
            "prior_dependency_pass": aux_metrics["prior_shuffle_positive_score_drop"] > 0.0,
        },
        "test_data_used": False,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    sanity["all_checks_pass"] = all(sanity["checks"].values())
    write_json(root / "sanity_summary.json", sanity)
    write_json(root / "live_progress.json", {"status": "completed", **sanity})
    print(json.dumps(sanity, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
