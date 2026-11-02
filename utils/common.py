from __future__ import annotations

import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


def configure_runtime_env(project_root: str | Path) -> None:
    project_root = Path(project_root)
    mpl_dir = project_root / ".cache" / "matplotlib"
    font_dir = project_root / ".cache" / "fontconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    font_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
    os.environ.setdefault("FONTCONFIG_PATH", str(font_dir))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device(prefer: str = "auto") -> torch.device:
    prefer = (prefer or "auto").lower()
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer in {"cpu", "debug"}:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(paths: Iterable[str | Path]) -> List[Path]:
    return [ensure_dir(path) for path in paths]


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: str | Path, content: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (parameter for parameter in params if parameter.requires_grad)
    return sum(parameter.numel() for parameter in params)


def format_seconds(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def get_runtime_info(device: torch.device) -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def normalize_array(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = array.astype(np.float32)
    array_min = float(array.min())
    array_max = float(array.max())
    return (array - array_min) / max(array_max - array_min, eps)
