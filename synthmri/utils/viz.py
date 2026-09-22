"""Figure helpers: mask colouring, overlays and modality panels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from synthmri.utils.io import image_grid, save_png

# Canonical classes: 0 background, 1 NCR/NET, 2 oedema, 3 enhancing tumour (BraTS colours).
MASK_PALETTE = np.array(
    [[0.0, 0.0, 0.0], [0.90, 0.10, 0.10], [0.10, 0.75, 0.10], [0.95, 0.85, 0.10]], dtype=np.float32
)


def colorize_mask(mask: torch.Tensor | np.ndarray) -> torch.Tensor:
    """(H,W) int mask -> (3,H,W) float RGB in [0,1]."""
    m = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    rgb = MASK_PALETTE[np.clip(m, 0, len(MASK_PALETTE) - 1)]  # (H,W,3)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())


def overlay(image_gray: torch.Tensor, mask: torch.Tensor, alpha: float = 0.45) -> torch.Tensor:
    """Blend a coloured mask over a (H,W) or (1,H,W) grey image in [0,1] -> (3,H,W)."""
    g = image_gray.detach().float().cpu()
    if g.ndim == 2:
        g = g[None]
    base = g.repeat(3, 1, 1)
    col = colorize_mask(mask)
    fg = (mask.detach().cpu() > 0).float()[None]
    return base * (1 - alpha * fg) + col * alpha * fg


def modality_panels(images: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
    """(N,C,S,S) images in [0,1] -> (N*(C[+1]), 1|3, S, S) row-major panels: one row per sample.

    Each row shows the C modalities as grey panels, followed by the mask overlay on the first
    modality when ``masks`` is given. Feed the result to ``image_grid(..., nrow=C[+1])``.
    """
    n, c, h, w = images.shape
    panels = []
    for i in range(n):
        for k in range(c):
            panels.append(images[i, k : k + 1].repeat(3, 1, 1))
        if masks is not None:
            panels.append(overlay(images[i, 0], masks[i]))
    return torch.stack(panels)


def save_sample_sheet(images: torch.Tensor, path: str | Path, masks: torch.Tensor | None = None, max_rows: int = 8) -> None:
    """Save a per-modality contact sheet (rows = samples, columns = modalities [+ mask])."""
    images = images.detach().float().cpu()[:max_rows]
    masks = None if masks is None else masks.detach().cpu()[:max_rows]
    panels = modality_panels(images, masks)
    ncol = images.shape[1] + (1 if masks is not None else 0)
    save_png(image_grid(panels, nrow=ncol), path)


def save_rgb_grid(images: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    """Save a (N,3,S,S) batch as one RGB grid (channels = modalities, as the VAE sees them)."""
    save_png(image_grid(images.detach().float().cpu(), nrow=nrow), path)
