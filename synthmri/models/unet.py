"""Latent-space denoising U-Net (diffusers ``UNet2DModel``) built from the config.

For mask-conditioned models the one-hot tumour mask, average-pooled to the latent resolution,
plus a "null-condition" indicator channel are concatenated to the noisy latent, so
``in_channels = 4 + num_mask_classes + 1``. See ``synthmri.diffusion.conditioning``.
"""

from __future__ import annotations

from diffusers import UNet2DModel
from torch import nn

from synthmri.config import Config
from synthmri.models.vae import LATENT_CHANNELS


def condition_channels(cfg: Config) -> int:
    if cfg.model.conditioning == "none":
        return 0
    if cfg.model.conditioning == "mask":
        return cfg.model.num_mask_classes + 1
    raise ValueError(f"Unknown conditioning: {cfg.model.conditioning!r}")


def build_unet(cfg: Config) -> UNet2DModel:
    u = cfg.model.unet
    levels = len(u.block_out_channels)
    attn = set(u.attention_levels)
    bad = [a for a in attn if a < 0 or a >= levels]
    if bad:
        raise ValueError(f"attention_levels {bad} out of range for {levels} levels")
    down = tuple("AttnDownBlock2D" if i in attn else "DownBlock2D" for i in range(levels))
    up = tuple("AttnUpBlock2D" if (levels - 1 - i) in attn else "UpBlock2D" for i in range(levels))
    return UNet2DModel(
        sample_size=cfg.latent_size,
        in_channels=LATENT_CHANNELS + condition_channels(cfg),
        out_channels=LATENT_CHANNELS,
        layers_per_block=u.layers_per_block,
        block_out_channels=tuple(u.block_out_channels),
        down_block_types=down,
        up_block_types=up,
        attention_head_dim=u.attention_head_dim,
        dropout=u.dropout,
        norm_num_groups=u.norm_num_groups,
    )


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
