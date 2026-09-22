"""Affine augmentation applied identically to images and masks.

Used when caching VAE latents: every training slice is encoded in several randomly transformed
versions (horizontal flip, small shift, rotation and isotropic scale) and ``LatentDataset`` draws
one of them per access, transforming the mask with the same stored parameters. Parameters are
rows ``[flip, dx, dy, angle_deg, scale]`` with shifts as a fraction of the image size.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

IDENTITY = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
HFLIP = np.array([1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def sample_affine_params(
    n: int, max_shift: float = 0.06, max_rotate: float = 10.0, scale_range: tuple[float, float] = (0.9, 1.1),
    hflip: bool = True, seed: int = 0,
) -> np.ndarray:
    """``(n, 5)`` random parameters drawn uniformly within the given ranges (reproducible via ``seed``)."""
    rng = np.random.RandomState(seed)
    p = np.empty((n, 5), np.float32)
    p[:, 0] = rng.randint(0, 2, n) if hflip else 0.0
    p[:, 1:3] = rng.uniform(-max_shift, max_shift, (n, 2))
    p[:, 3] = rng.uniform(-max_rotate, max_rotate, n)
    p[:, 4] = rng.uniform(scale_range[0], scale_range[1], n)
    return p


def is_identity(params) -> bool:
    return bool(np.allclose(np.asarray(params, dtype=np.float32), IDENTITY))


def is_hflip(params) -> bool:
    return bool(np.allclose(np.asarray(params, dtype=np.float32), HFLIP))


def affine_theta(params: torch.Tensor) -> torch.Tensor:
    """``(B, 5)`` parameters -> ``(B, 2, 3)`` matrices for ``F.affine_grid`` (output -> input coordinates)."""
    flip, dx, dy, ang, sc = params.float().unbind(1)
    a = ang * math.pi / 180.0
    cos, sin = torch.cos(a) / sc, torch.sin(a) / sc
    sx = torch.where(flip > 0.5, -torch.ones_like(cos), torch.ones_like(cos))
    row0 = torch.stack([cos * sx, -sin, -2.0 * dx], dim=1)
    row1 = torch.stack([sin * sx, cos, -2.0 * dy], dim=1)
    return torch.stack([row0, row1], dim=1)


def apply_affine(x: torch.Tensor, params: torch.Tensor | np.ndarray, mode: str = "bilinear") -> torch.Tensor:
    """Transform ``(B,C,H,W)`` images (``mode="bilinear"``, zero padding) or ``(B,H,W)`` integer masks
    (``mode="nearest"``). Values outside the source image become 0 (background)."""
    params = torch.as_tensor(np.asarray(params, dtype=np.float32) if not torch.is_tensor(params) else params)
    if params.ndim == 1:
        params = params[None]
    squeeze = x.ndim == 3
    xf = (x[:, None] if squeeze else x).float()
    theta = affine_theta(params.to(xf.device))
    grid = F.affine_grid(theta, list(xf.shape), align_corners=False)
    y = F.grid_sample(xf, grid, mode=mode, padding_mode="zeros", align_corners=False)
    if mode == "nearest":
        y = y.round()
    y = y.to(x.dtype)
    return y[:, 0] if squeeze else y


def apply_affine_image(x: torch.Tensor, params) -> torch.Tensor:
    """Images in [-1, 1]: transform in [0, 1] space so that padding is background (-1), not grey."""
    return apply_affine((x + 1.0) * 0.5, params, mode="bilinear") * 2.0 - 1.0


def transform_mask(mask: torch.Tensor, params) -> torch.Tensor:
    """Single ``(H, W)`` integer mask with one parameter row; cheap paths for identity and flip."""
    if is_identity(params):
        return mask
    if is_hflip(params):
        return mask.flip(-1)
    return apply_affine(mask[None], params, mode="nearest")[0]
