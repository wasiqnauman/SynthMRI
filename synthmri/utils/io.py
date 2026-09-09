from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def git_commit_hash(repo_dir: str | Path | None = None) -> str | None:
    """Short git hash of the working tree (``None`` when git is unavailable)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir or Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir or Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return None


def to_uint8(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """[0, 1] float image(s) -> uint8 (values are clamped)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_png(x: torch.Tensor | np.ndarray, path: str | Path) -> None:
    """Save a (C,H,W) or (H,W) image in [0,1] as PNG (C=1 grey, C=3 RGB)."""
    from PIL import Image

    arr = to_uint8(x)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        else:
            raise ValueError(f"Cannot save image with {arr.shape[0]} channels as PNG")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def image_grid(images: torch.Tensor, nrow: int = 8, pad: int = 2) -> torch.Tensor:
    """Tile a (N,C,H,W) batch into one (C,H',W') image (no torchvision dependency)."""
    n, c, h, w = images.shape
    ncol = min(nrow, n)
    nrows = (n + ncol - 1) // ncol
    grid = torch.ones(c, nrows * (h + pad) + pad, ncol * (w + pad) + pad, dtype=images.dtype)
    for i in range(n):
        r, col = divmod(i, ncol)
        y0 = pad + r * (h + pad)
        x0 = pad + col * (w + pad)
        grid[:, y0 : y0 + h, x0 : x0 + w] = images[i]
    return grid
