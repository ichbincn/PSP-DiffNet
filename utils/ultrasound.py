from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class UltrasoundPreprocessResult:
    image: Image.Image
    content_box: Tuple[int, int, int, int]
    scale: float


def resize_pad_pil(
    image: Image.Image,
    size: int,
    *,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
    fill: int | Tuple[int, int, int] = 0,
) -> UltrasoundPreprocessResult:
    """Aspect-preserving square resize used consistently across MMOTU stages."""
    image = image.copy()
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {image.size}")
    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = image.resize((resized_width, resized_height), resample)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    right = left + resized_width
    bottom = top + resized_height
    if image.mode == "RGB":
        canvas = Image.new("RGB", (size, size), fill if isinstance(fill, tuple) else (fill, fill, fill))
    else:
        canvas = Image.new(image.mode, (size, size), fill if isinstance(fill, int) else int(fill[0]))
    canvas.paste(resized, (left, top))
    return UltrasoundPreprocessResult(canvas, (left, top, right, bottom), scale)


def preprocess_ultrasound_pil(image: Image.Image, size: int) -> Image.Image:
    return resize_pad_pil(image.convert("RGB"), size=size, fill=0).image


def preprocess_ultrasound_mask(mask: Image.Image, size: int) -> Image.Image:
    array = np.asarray(mask.convert("L"))
    binary = Image.fromarray((array > 0).astype(np.uint8) * 255, mode="L")
    return resize_pad_pil(binary, size=size, resample=Image.Resampling.NEAREST, fill=0).image


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    foreground = np.argwhere(mask > 0)
    if len(foreground) == 0:
        return None
    y0, x0 = foreground.min(axis=0)
    y1, x1 = foreground.max(axis=0) + 1
    return int(x0), int(y0), int(x1), int(y1)


def masked_roi_image(image: Image.Image, mask: Image.Image, padding: float = 0.12) -> Image.Image:
    """Optional diagnostic crop; never used to overwrite the original data."""
    image = image.convert("RGB")
    mask_array = np.asarray(mask.convert("L")) > 0
    box = mask_bbox(mask_array)
    if box is None:
        return image
    x0, y0, x1, y1 = box
    width, height = image.size
    pad_x = int(round((x1 - x0) * padding))
    pad_y = int(round((y1 - y0) * padding))
    crop_box = (max(0, x0 - pad_x), max(0, y0 - pad_y), min(width, x1 + pad_x), min(height, y1 + pad_y))
    return ImageOps.exif_transpose(image).crop(crop_box)
