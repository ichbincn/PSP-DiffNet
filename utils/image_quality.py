from __future__ import annotations

from typing import Dict

import numpy as np

from utils.fundus import detect_fundus_mask, fundus_area_ratio, is_valid_fundus, shannon_entropy


def compute_pixel_stats(image_rgb: np.ndarray) -> Dict[str, float | bool]:
    mask = detect_fundus_mask(image_rgb)
    rgb_means = image_rgb.reshape(-1, 3).mean(axis=0)
    rgb_stds = image_rgb.reshape(-1, 3).std(axis=0)
    entropy = shannon_entropy(image_rgb, mask=mask if mask.any() else None)
    std_value = float(image_rgb.std())
    return {
        "mean": float(image_rgb.mean()),
        "std": std_value,
        "min": int(image_rgb.min()),
        "max": int(image_rgb.max()),
        "r_mean": float(rgb_means[0]),
        "g_mean": float(rgb_means[1]),
        "b_mean": float(rgb_means[2]),
        "r_std": float(rgb_stds[0]),
        "g_std": float(rgb_stds[1]),
        "b_std": float(rgb_stds[2]),
        "entropy": float(entropy),
        "fundus_area_ratio": float(fundus_area_ratio(mask)),
        "is_near_constant": bool(std_value < 5.0 or entropy < 1.0),
        "is_valid_fundus": bool(is_valid_fundus(mask)),
    }


def quality_gate(
    image_rgb: np.ndarray,
    entropy_threshold: float,
    std_threshold: float,
    min_fundus_area_ratio: float,
) -> Dict[str, object]:
    stats = compute_pixel_stats(image_rgb)
    checks = {
        "entropy_ok": stats["entropy"] >= entropy_threshold,
        "std_ok": stats["std"] >= std_threshold,
        "not_constant": not stats["is_near_constant"],
        "fundus_ok": stats["fundus_area_ratio"] >= min_fundus_area_ratio and stats["is_valid_fundus"],
    }
    checks["passed"] = all(checks.values())
    return {"stats": stats, "checks": checks}
