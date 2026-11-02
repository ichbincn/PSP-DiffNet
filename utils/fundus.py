from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class FundusPreprocessResult:
    image: Image.Image
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]


def detect_fundus_mask(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("detect_fundus_mask expects an RGB image array.")

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    red = image_rgb[:, :, 0]
    green = image_rgb[:, :, 1]
    candidate = ((gray > 8) & (red > 12) & (green > 8)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if num_labels <= 1:
        return (candidate > 0).astype(np.uint8)

    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest_component).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, [largest_contour], contourIdx=-1, color=1, thickness=-1)
        mask = filled

    return mask.astype(np.uint8)


def compute_mask_bbox(mask: np.ndarray, padding_ratio: float = 0.04) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        height, width = mask.shape[:2]
        return 0, 0, width, height

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    side = int(max(width, height) * (1.0 + 2.0 * padding_ratio))
    cx = int((x_min + x_max) / 2)
    cy = int((y_min + y_max) / 2)

    x1 = max(cx - side // 2, 0)
    y1 = max(cy - side // 2, 0)
    x2 = min(x1 + side, mask.shape[1])
    y2 = min(y1 + side, mask.shape[0])
    x1 = max(x2 - side, 0)
    y1 = max(y2 - side, 0)
    return int(x1), int(y1), int(x2), int(y2)


def crop_to_bbox(image_rgb: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return image_rgb[y1:y2, x1:x2]


def preprocess_fundus_image(
    image_rgb: np.ndarray,
    crop: bool = True,
    mask_background: bool = True,
) -> Dict[str, object]:
    mask = detect_fundus_mask(image_rgb)
    processed = image_rgb.copy()
    if mask_background:
        processed[mask == 0] = 0
    bbox = compute_mask_bbox(mask)
    if crop:
        processed = crop_to_bbox(processed, bbox)
        mask = crop_to_bbox(mask, bbox)
    return {"image": processed, "mask": mask.astype(np.uint8), "bbox": bbox}


def preprocess_fundus_pil(image: Image.Image, crop: bool = True, mask_background: bool = True) -> FundusPreprocessResult:
    image_rgb = np.asarray(image.convert("RGB"))
    processed = preprocess_fundus_image(image_rgb, crop=crop, mask_background=mask_background)
    return FundusPreprocessResult(
        image=Image.fromarray(processed["image"]),
        mask=processed["mask"],
        bbox=processed["bbox"],
    )


def shannon_entropy(image_rgb: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        pixels = image_rgb[mask > 0]
    else:
        pixels = image_rgb.reshape(-1, image_rgb.shape[-1])
    if len(pixels) == 0:
        return 0.0
    gray = np.mean(pixels, axis=1).astype(np.uint8)
    hist = np.bincount(gray, minlength=256).astype(np.float64)
    prob = hist / max(hist.sum(), 1.0)
    valid = prob[prob > 0]
    return float(-(valid * np.log2(valid)).sum())


def fundus_area_ratio(mask: np.ndarray) -> float:
    return float(mask.mean())


def is_valid_fundus(mask: np.ndarray, min_area_ratio: float = 0.25) -> bool:
    return fundus_area_ratio(mask) >= min_area_ratio
