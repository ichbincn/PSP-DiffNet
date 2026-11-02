from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageOps


def load_rgb_and_mask(image_path: str | Path, mask_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as handle:
        image = np.asarray(ImageOps.exif_transpose(handle).convert("RGB"))
    with Image.open(mask_path) as handle:
        mask = np.asarray(ImageOps.exif_transpose(handle).convert("L")) > 0
    if image.shape[:2] != mask.shape:
        raise RuntimeError(f"Image/mask mismatch: {image_path} {image.shape[:2]} vs {mask.shape}")
    return image, mask


def square_lesion_context_crop_pair(
    image: np.ndarray,
    lesion_mask: np.ndarray,
    margin_fraction: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Apply one deterministic square crop/padding geometry to image and binary mask."""
    if image.shape[:2] != lesion_mask.shape:
        raise RuntimeError(f"Image/mask shape mismatch: {image.shape[:2]} vs {lesion_mask.shape}")
    foreground = np.argwhere(lesion_mask)
    if not len(foreground):
        raise RuntimeError("MMOTU ROI protocol requires a non-empty GT mask")
    y0, x0 = foreground.min(axis=0)
    y1, x1 = foreground.max(axis=0) + 1
    box_w, box_h = int(x1 - x0), int(y1 - y0)
    side = max(2, int(round(max(box_w, box_h) * (1.0 + 2.0 * margin_fraction))))
    cx = (float(x0) + float(x1)) / 2.0
    cy = (float(y0) + float(y1)) / 2.0
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = left + side
    bottom = top + side
    height, width = image.shape[:2]
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - width), max(0, bottom - height)
    padded_image = image
    padded_mask = lesion_mask.astype(bool)
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        padded_image = np.pad(
            padded_image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        padded_mask = np.pad(
            padded_mask,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=False,
        )
    shifted_left = left + pad_left
    shifted_right = right + pad_left
    shifted_top = top + pad_top
    shifted_bottom = bottom + pad_top
    crop_image = padded_image[shifted_top:shifted_bottom, shifted_left:shifted_right].copy()
    crop_mask = padded_mask[shifted_top:shifted_bottom, shifted_left:shifted_right].copy()
    if crop_image.shape[:2] != (side, side) or crop_mask.shape != (side, side):
        raise RuntimeError("Unexpected ROI crop geometry")
    geometry = {
        "original_width": int(width),
        "original_height": int(height),
        "lesion_x0": int(x0),
        "lesion_y0": int(y0),
        "lesion_x1": int(x1),
        "lesion_y1": int(y1),
        "crop_left": int(left),
        "crop_top": int(top),
        "crop_right": int(right),
        "crop_bottom": int(bottom),
        "crop_side": int(side),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "pad_right": int(pad_right),
        "pad_bottom": int(pad_bottom),
    }
    return crop_image, crop_mask, geometry
