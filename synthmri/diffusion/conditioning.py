"""Mask conditioning for the latent U-Net.

The (B, S, S) integer mask becomes (B, K+1, S/8, S/8): K channels with the *fraction* of each
latent cell covered by each class (one-hot mask average-pooled to the latent grid, so soft
boundaries are preserved), plus one indicator channel that is 1 when the condition has been
dropped. Dropping the condition to an all-zero map alone would be ambiguous with "all
background", which is why the explicit null channel exists. Classifier-free guidance then
contrasts the conditional and null predictions at sampling time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mask_to_condition(
    mask: torch.Tensor, num_classes: int, latent_size: int, drop: torch.Tensor | None = None
) -> torch.Tensor:
    if mask.ndim != 3:
        raise ValueError(f"mask must be (B,S,S), got {tuple(mask.shape)}")
    onehot = F.one_hot(mask.long(), num_classes).permute(0, 3, 1, 2).float()
    cond = F.adaptive_avg_pool2d(onehot, latent_size)
    null = cond.new_zeros(cond.shape[0], 1, latent_size, latent_size)
    if drop is not None:
        drop = drop.to(cond.device).bool()
        cond[drop] = 0.0
        null[drop] = 1.0
    return torch.cat([cond, null], dim=1)


def null_condition(batch_size: int, num_classes: int, latent_size: int, device=None) -> torch.Tensor:
    cond = torch.zeros(batch_size, num_classes + 1, latent_size, latent_size, device=device)
    cond[:, -1] = 1.0
    return cond
