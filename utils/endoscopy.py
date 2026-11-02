from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps


@dataclass(frozen=True)
class EndoscopyPreprocessResult:
    image: Image.Image
    content_box: Tuple[int, int, int, int]
    scale: float


def resize_pad_endoscopy(image: Image.Image, size: int, *, resample=Image.Resampling.LANCZOS) -> EndoscopyPreprocessResult:
    """Aspect-preserving square conversion shared by every HyperKvasir stage."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {image.size}")
    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = image.resize((resized_width, resized_height), resample)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, (left, top))
    return EndoscopyPreprocessResult(canvas, (left, top, left + resized_width, top + resized_height), scale)


def preprocess_endoscopy_pil(image: Image.Image, size: int) -> Image.Image:
    return resize_pad_endoscopy(image, size).image


def build_mucosal_structure_prior(image: Image.Image, size: int) -> Image.Image:
    """Deterministic weak prior from multi-scale mucosal gradients; it is not a segmentation mask."""
    square = preprocess_endoscopy_pil(image, size)
    rgb = np.asarray(square, dtype=np.float32) / 255.0
    gray = np.asarray(square.convert("L"), dtype=np.float32) / 255.0
    # Suppress padded/true black borders while retaining dim mucosa.
    foreground = np.clip((rgb.max(axis=2) - 0.025) / 0.12, 0.0, 1.0)
    edge_small = np.asarray(square.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    blurred = np.asarray(square.convert("L").filter(ImageFilter.GaussianBlur(radius=max(2.0, size / 96))), dtype=np.float32) / 255.0
    local_contrast = np.abs(gray - blurred)
    signal = (0.60 * edge_small + 0.40 * local_contrast) * foreground
    positive = signal[foreground > 0.15]
    scale = float(np.quantile(positive, 0.90)) if positive.size else 1.0
    normalized = np.clip(signal / max(scale, 1e-6), 0.0, 1.0)
    # A soft gate is preferable to a binary pseudo-mask for latent anchoring.
    soft = Image.fromarray(np.uint8(np.round(normalized * 255)), mode="L").filter(
        ImageFilter.GaussianBlur(radius=max(1.0, size / 256))
    )
    return soft


def preprocess_endoscopy_prior(prior: Image.Image, size: int) -> Image.Image:
    prior = prior.convert("L")
    if prior.size == (size, size):
        return prior
    return prior.resize((size, size), Image.Resampling.BILINEAR)
