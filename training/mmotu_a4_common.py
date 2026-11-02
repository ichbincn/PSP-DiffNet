from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.a3_diagnostic_discriminator import A3DiagnosticDiscriminator
from training.run_mmotu_convnext import build_model
from utils.ultrasound import preprocess_ultrasound_mask, preprocess_ultrasound_pil


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize(array: np.ndarray) -> np.ndarray:
    return (array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.clip(left @ right.T, -1.0, 1.0))).astype(np.float32)


def radii(features: np.ndarray, k: int) -> np.ndarray:
    matrix = distances(features, features)
    np.fill_diagonal(matrix, np.inf)
    return np.partition(matrix, kth=k - 1, axis=1)[:, k - 1]


class FeatureDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int):
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(str(row.image_path)) as handle:
            image = preprocess_ultrasound_pil(handle.convert("RGB"), self.image_size)
        with Image.open(str(row.prior_path)) as handle:
            prior = preprocess_ultrasound_mask(handle, self.image_size)
        image_01 = transforms.functional.to_tensor(image)
        return self.normalize(image_01), transforms.functional.to_tensor(prior), int(row.label), int(index)


def resolve_teacher_paths(root: Path, teacher_root: str, teacher_checkpoints: str) -> List[Path]:
    if teacher_checkpoints.strip():
        paths = [(root / value.strip()).resolve() for value in teacher_checkpoints.split(",") if value.strip()]
        if len(paths) != 3:
            raise RuntimeError(f"Expected exactly three teacher checkpoints, got {len(paths)}")
        return paths
    return [(root / teacher_root / f"seed_{seed}" / "best_model.pt").resolve() for seed in (2026, 2027, 2028)]


def load_teacher(
    path: Path, device: torch.device, expected_pretrained: Optional[bool] = False,
) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu")
    actual_pretrained = bool(payload.get("pretrained", False))
    if expected_pretrained is not None and actual_pretrained != expected_pretrained:
        raise RuntimeError(
            f"A4 teacher initialization mismatch: expected_pretrained={expected_pretrained} "
            f"actual={actual_pretrained} path={path}"
        )
    model = build_model(torch.device("cpu"))
    model.load_state_dict(payload["model_state"], strict=True)
    model.requires_grad_(False); model.eval(); model.to(device)
    return model


def encode_teacher(model: torch.nn.Module, images: torch.Tensor, priors: torch.Tensor):
    x = images
    mid = None
    for index, layer in enumerate(model.features):
        x = layer(x)
        if index == 5:
            mid = x
    pooled = model.avgpool(x)
    final = torch.flatten(model.classifier[0](pooled), 1)
    logits = model.classifier[2](final)
    if mid is None:
        raise RuntimeError("ConvNeXt features[5] unavailable")
    mask = F.interpolate(priors, size=mid.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
    inverse = 1 - mask
    roi = (mid * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1e-6)
    context = (mid * inverse).sum((2, 3)) / inverse.sum((2, 3)).clamp_min(1e-6)
    local = torch.cat([
        F.normalize(final.float(), dim=1), F.normalize(roi.float(), dim=1),
        F.normalize(context.float(), dim=1), F.normalize((roi - context).float(), dim=1),
    ], dim=1)
    return F.normalize(final.float(), dim=1), F.normalize(local, dim=1), logits.float()


@torch.no_grad()
def extract_teacher(
    model: torch.nn.Module, frame: pd.DataFrame, image_size: int, batch_size: int,
    num_workers: int, device: torch.device, cache: Path, stage: str,
) -> Dict[str, np.ndarray]:
    expected = frame.image_id.astype(str).tolist()
    if cache.is_file():
        saved = np.load(cache, allow_pickle=False)
        if saved["image_ids"].astype(str).tolist() == expected:
            print(f"[MMOTU A4] reuse {cache.name}", flush=True)
            return {key: saved[key] for key in saved.files}
    loader = DataLoader(
        FeatureDataset(frame, image_size), batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
    )
    global_features: List[np.ndarray] = []
    local_features: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    seen = 0
    for images, priors, _labels, indices in loader:
        images = images.to(device, non_blocking=True); priors = priors.to(device, non_blocking=True)
        with autocast(enabled=True):
            global_feature, local_feature, logits = encode_teacher(model, images, priors)
        global_features.append(global_feature.cpu().numpy().astype(np.float32))
        local_features.append(local_feature.cpu().numpy().astype(np.float32))
        probabilities.append(torch.softmax(logits, 1).cpu().numpy().astype(np.float32))
        seen += len(indices)
        if seen == len(frame) or seen % (batch_size * 10) == 0:
            print(f"[MMOTU A4] {stage}: {seen}/{len(frame)}", flush=True)
    result = {
        "global_features": np.concatenate(global_features),
        "local_features": np.concatenate(local_features),
        "probabilities": np.concatenate(probabilities),
        "image_ids": frame.image_id.astype(str).to_numpy(dtype=str),
        "source_ids": frame.source_image_id.astype(str).to_numpy(dtype=str),
        "labels": frame.label.to_numpy(dtype=np.int64),
        "label_names": frame.label_name.astype(str).to_numpy(dtype=str),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **result)
    return result


def source_excluded_coverage(
    real: np.ndarray, synthetic: np.ndarray, real_ids: np.ndarray, source_ids: np.ndarray, k: int,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    radius = radii(real, k)
    matrix = distances(real, synthetic)
    matrix[real_ids[:, None] == source_ids[None, :]] = np.inf
    nearest = matrix.min(1)
    ratio = nearest / np.maximum(radius, 1e-8)
    return float(np.mean(ratio <= 1)), ratio, ratio > 1, matrix


def calibrated_bootstraps(
    real: np.ndarray, synthetic: np.ndarray, real_ids: np.ndarray, source_ids: np.ndarray,
    k: int, reference_fraction: float, seeds: Sequence[int],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(real))
        n_reference = min(len(real) - 2, max(k + 2, int(round(reference_fraction * len(real)))))
        reference_index, query_index = order[:n_reference], order[n_reference:]
        reference, query = real[reference_index], real[query_index]
        radius = radii(reference, min(k, len(reference) - 1))
        real_coverage = float(np.mean(distances(reference, query).min(1) <= radius))
        query_ids = set(real_ids[query_index].tolist())
        eligible = np.flatnonzero(np.asarray([item in query_ids for item in source_ids]))
        if not len(eligible):
            continue
        count = min(len(query_index), len(eligible))
        selected = rng.choice(eligible, size=count, replace=False)
        synthetic_coverage = float(np.mean(distances(reference, synthetic[selected]).min(1) <= radius))
        rows.append({
            "seed": seed, "real_real_coverage": real_coverage,
            "synthetic_real_coverage_size_matched": synthetic_coverage,
            "coverage_deficit": real_coverage - synthetic_coverage,
        })
    return rows


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64); right = np.asarray(right, dtype=np.float64)
    left = left / max(left.sum(), 1e-12); right = right / max(right.sum(), 1e-12)
    middle = 0.5 * (left + right)
    def kl(value, target):
        mask = value > 0
        return float(np.sum(value[mask] * np.log(value[mask] / np.maximum(target[mask], 1e-12))))
    return 0.5 * kl(left, middle) + 0.5 * kl(right, middle)


def coverage_state(matrices: Dict[str, np.ndarray], radius_map: Dict[str, np.ndarray]):
    by_teacher: Dict[str, float] = {}
    excess = []
    for name, matrix in matrices.items():
        ratio = matrix.min(1) / np.maximum(radius_map[name], 1e-8)
        by_teacher[name] = float(np.mean(ratio <= 1))
        excess.append(float(np.maximum(ratio - 1, 0).mean()))
    return by_teacher, float(np.mean(list(by_teacher.values()))), float(np.mean(excess))
