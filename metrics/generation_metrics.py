from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import VGG16_Weights, vgg16

from pytorch_fid.inception import InceptionV3 as FIDInceptionV3

from utils.common import save_json
from utils.fundus import preprocess_fundus_pil


ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class DistributionMetricConfig:
    batch_size: int = 32
    num_workers: int = 4
    kid_subsets: int = 100
    kid_subset_size: int = 1000
    ipr_k: int = 3
    distance_chunk_size: int = 256
    seed: int = 2026


class ImagePathDataset(Dataset):
    def __init__(self, paths: Sequence[str | Path], size: int, mean: Sequence[float], std: Sequence[float]):
        self.paths = [str(path) for path in paths]
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as handle:
            image = preprocess_fundus_pil(handle.convert("RGB"), crop=True, mask_background=True).image
        return self.transform(image)


class FIDFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        block_index = FIDInceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = FIDInceptionV3(
            output_blocks=(block_index,),
            resize_input=False,
            normalize_input=True,
            requires_grad=False,
            use_fid_inception=True,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images)[0]
        if features.shape[2:] != (1, 1):
            features = torch.nn.functional.adaptive_avg_pool2d(features, output_size=(1, 1))
        return features.flatten(1)


def build_inception_feature_extractor(device: torch.device) -> nn.Module:
    model = FIDFeatureExtractor()
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_vgg16_feature_extractor(device: torch.device) -> nn.Module:
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    # Kynkäänniemi et al. use the activation after VGG-16's second fully connected layer.
    model.classifier = nn.Sequential(*list(model.classifier.children())[:5])
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)


def extract_features(
    paths: Sequence[str | Path],
    model: nn.Module,
    size: int,
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
    config: DistributionMetricConfig,
    stage: str,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    dataset = ImagePathDataset(paths, size=size, mean=mean, std=std)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        worker_init_fn=_worker_seed,
        generator=torch.Generator().manual_seed(config.seed),
    )
    feature_batches: List[np.ndarray] = []
    completed = 0
    with torch.inference_mode():
        for images in loader:
            images = images.to(device, non_blocking=True)
            features = model(images)
            if isinstance(features, tuple):
                features = features[0]
            feature_batches.append(features.float().cpu().numpy())
            completed += len(images)
            if progress_callback is not None:
                progress_callback(stage, completed, len(dataset))
    if not feature_batches:
        raise ValueError(f"No images were available for feature extraction stage: {stage}")
    return np.concatenate(feature_batches, axis=0).astype(np.float32, copy=False)


def frechet_distance(real_features: np.ndarray, generated_features: np.ndarray, device: torch.device) -> float:
    """Exact empirical Gaussian FID using a low-rank covariance identity.

    For centered feature matrices X and Y, the covariance cross term equals the
    nuclear norm of X @ Y.T divided by sqrt((n-1)(m-1)). This avoids unstable
    2048 x 2048 matrix square roots when a medical class has fewer samples than
    the Inception feature dimension.
    """
    if len(real_features) < 2 or len(generated_features) < 2:
        return float("nan")
    real = torch.as_tensor(real_features, dtype=torch.float32, device=device)
    generated = torch.as_tensor(generated_features, dtype=torch.float32, device=device)
    real_centered = real - real.mean(dim=0, keepdim=True)
    generated_centered = generated - generated.mean(dim=0, keepdim=True)
    mean_term = torch.sum((real.mean(dim=0) - generated.mean(dim=0)) ** 2)
    real_trace = torch.sum(real_centered**2) / (len(real) - 1)
    generated_trace = torch.sum(generated_centered**2) / (len(generated) - 1)
    cross = real_centered @ generated_centered.T
    singular_values = torch.linalg.svdvals(cross)
    cross_trace = singular_values.sum() / math.sqrt((len(real) - 1) * (len(generated) - 1))
    value = mean_term + real_trace + generated_trace - 2.0 * cross_trace
    return float(torch.clamp(value, min=0.0).item())


def _polynomial_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x @ y.T / x.shape[1] + 1.0) ** 3


def _unbiased_mmd2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if len(x) < 2 or len(y) < 2:
        return torch.tensor(float("nan"), device=x.device)
    kernel_xx = _polynomial_kernel(x, x)
    kernel_yy = _polynomial_kernel(y, y)
    kernel_xy = _polynomial_kernel(x, y)
    term_x = (kernel_xx.sum() - kernel_xx.diagonal().sum()) / (len(x) * (len(x) - 1))
    term_y = (kernel_yy.sum() - kernel_yy.diagonal().sum()) / (len(y) * (len(y) - 1))
    return term_x + term_y - 2.0 * kernel_xy.mean()


def kernel_inception_distance(
    real_features: np.ndarray,
    generated_features: np.ndarray,
    device: torch.device,
    subsets: int,
    maximum_subset_size: int,
    seed: int,
) -> Tuple[float, float, int]:
    subset_size = min(maximum_subset_size, len(real_features), len(generated_features))
    if subset_size < 2:
        return float("nan"), float("nan"), subset_size
    rng = np.random.default_rng(seed)
    real = torch.as_tensor(real_features, dtype=torch.float32, device=device)
    generated = torch.as_tensor(generated_features, dtype=torch.float32, device=device)
    values = []
    for _ in range(subsets):
        real_indices = torch.as_tensor(rng.choice(len(real), subset_size, replace=False), device=device)
        generated_indices = torch.as_tensor(rng.choice(len(generated), subset_size, replace=False), device=device)
        values.append(float(_unbiased_mmd2(real[real_indices], generated[generated_indices]).item()))
    return float(np.mean(values)), float(np.std(values, ddof=1)), subset_size


def _pairwise_squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    distance = (x**2).sum(dim=1, keepdim=True) + (y**2).sum(dim=1).unsqueeze(0) - 2.0 * (x @ y.T)
    return distance.clamp_min_(0.0)


def _knn_radii(features: torch.Tensor, k: int, chunk_size: int) -> torch.Tensor:
    if len(features) <= k:
        raise ValueError(f"Improved Precision/Recall requires more than k={k} samples, got {len(features)}")
    radii = []
    for start in range(0, len(features), chunk_size):
        stop = min(start + chunk_size, len(features))
        distances = _pairwise_squared_distance(features[start:stop], features)
        row_indices = torch.arange(stop - start, device=features.device)
        column_indices = torch.arange(start, stop, device=features.device)
        distances[row_indices, column_indices] = float("inf")
        radii.append(torch.kthvalue(distances, k=k, dim=1).values)
    return torch.cat(radii)


def _manifold_membership(
    query: torch.Tensor,
    reference: torch.Tensor,
    reference_radii: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    membership = []
    for start in range(0, len(query), chunk_size):
        distances = _pairwise_squared_distance(query[start : start + chunk_size], reference)
        membership.append((distances <= reference_radii.unsqueeze(0)).any(dim=1))
    return torch.cat(membership)


def improved_precision_recall(
    real_features: np.ndarray,
    generated_features: np.ndarray,
    device: torch.device,
    k: int = 3,
    chunk_size: int = 256,
) -> Tuple[float, float]:
    real = torch.as_tensor(real_features, dtype=torch.float32, device=device)
    generated = torch.as_tensor(generated_features, dtype=torch.float32, device=device)
    real_radii = _knn_radii(real, k=k, chunk_size=chunk_size)
    generated_radii = _knn_radii(generated, k=k, chunk_size=chunk_size)
    precision = _manifold_membership(generated, real, real_radii, chunk_size).float().mean()
    recall = _manifold_membership(real, generated, generated_radii, chunk_size).float().mean()
    return float(precision.item()), float(recall.item())


def compute_distribution_metrics(
    real_inception: np.ndarray,
    generated_inception: np.ndarray,
    real_vgg: np.ndarray,
    generated_vgg: np.ndarray,
    device: torch.device,
    config: DistributionMetricConfig,
    seed_offset: int = 0,
) -> Dict[str, float | int]:
    kid_mean, kid_std, kid_subset_size = kernel_inception_distance(
        real_inception,
        generated_inception,
        device=device,
        subsets=config.kid_subsets,
        maximum_subset_size=config.kid_subset_size,
        seed=config.seed + seed_offset,
    )
    precision, recall = improved_precision_recall(
        real_vgg,
        generated_vgg,
        device=device,
        k=config.ipr_k,
        chunk_size=config.distance_chunk_size,
    )
    return {
        "real_count": int(len(real_inception)),
        "generated_count": int(len(generated_inception)),
        "fid": frechet_distance(real_inception, generated_inception, device=device),
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "kid_subset_size": int(kid_subset_size),
        "improved_precision": precision,
        "improved_recall": recall,
    }


def compute_generation_metrics(
    real_records: List[Dict[str, object]],
    synthetic_records: List[Dict[str, object]],
    feature_model: nn.Module | None,
    output_path: str | Path,
    image_size: int,
    device: torch.device,
) -> Dict[str, object]:
    """Compatibility entry point for the original pipeline, now restricted to FID/KID/IPR."""
    del feature_model, image_size
    output_path = Path(output_path)
    if not synthetic_records:
        metrics: Dict[str, object] = {"message": "No synthetic records were provided."}
        save_json(output_path, metrics)
        return metrics
    real_paths = [str(row["image_path"]) for row in real_records]
    generated_paths = [str(row.get("generated_path", row.get("image_path"))) for row in synthetic_records]
    config = DistributionMetricConfig()
    inception = build_inception_feature_extractor(device)
    real_inception = extract_features(
        real_paths, inception, 299, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], device, config, "real_inception"
    )
    generated_inception = extract_features(
        generated_paths, inception, 299, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], device, config, "generated_inception"
    )
    del inception
    if device.type == "cuda":
        torch.cuda.empty_cache()
    vgg = build_vgg16_feature_extractor(device)
    real_vgg = extract_features(
        real_paths, vgg, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], device, config, "real_vgg"
    )
    generated_vgg = extract_features(
        generated_paths, vgg, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], device, config, "generated_vgg"
    )
    del vgg
    metrics = compute_distribution_metrics(
        real_inception, generated_inception, real_vgg, generated_vgg, device=device, config=config
    )
    metrics["protocol"] = {
        "fid_kid_features": "PyTorch-FID TensorFlow-compatible Inception-v3 pool3 features (2048D)",
        "ipr_features": "torchvision VGG-16 ImageNet activation after second fully connected layer (4096D)",
        "ipr_k": config.ipr_k,
    }
    save_json(output_path, metrics)
    return metrics
