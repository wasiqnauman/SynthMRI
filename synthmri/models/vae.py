"""Frozen pretrained VAE (Stable Diffusion ``sd-vae-ft-mse``) behind a small uniform interface.

The three MRI modalities are mapped onto the VAE's RGB input channels. The latent space is
4 x (S/8) x (S/8); latents are multiplied by ``scaling_factor`` (0.18215 for the SD VAE) so
that they have roughly unit variance, as in Rombach et al. (2022).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

LATENT_CHANNELS = 4
DOWNSAMPLE = 8


class VAEWrapper(nn.Module):
    def __init__(self, vae: nn.Module, scaling_factor: float | None = None):
        super().__init__()
        self.vae = vae
        self.scaling_factor = float(scaling_factor or vae.config.scaling_factor)
        self.downsample = DOWNSAMPLE
        self.latent_channels = LATENT_CHANNELS

    @torch.no_grad()
    def encode_dist(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Unscaled posterior (mean, logvar) for images ``x`` in [-1, 1]."""
        dist = self.vae.encode(x).latent_dist
        return dist.mean, dist.logvar

    @torch.no_grad()
    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Scaled latent; ``sample=False`` returns the posterior mean."""
        dist = self.vae.encode(x).latent_dist
        z = dist.sample() if sample else dist.mode()
        return z * self.scaling_factor

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Scaled latent -> image in [-1, 1] (not clamped)."""
        return self.vae.decode(z / self.scaling_factor).sample

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        return self.decode(self.encode(x, sample=sample))


def load_vae(pretrained: str = "stabilityai/sd-vae-ft-mse", device=None, dtype=torch.float32, scaling_factor=None) -> VAEWrapper:
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(pretrained, torch_dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()
    if device is not None:
        vae.to(device)
    return VAEWrapper(vae, scaling_factor)


class _Dist:
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor):
        self.mean, self.logvar = mean, logvar
        self.std = torch.exp(0.5 * logvar)

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self) -> torch.Tensor:
        return self.mean


class DummyVAE(nn.Module):
    """Parameter-free stand-in with the same interface and 8x downsampling, for tests.

    Encoder: 8x8 average pooling of the 3 input channels plus a zero 4th channel; decoder:
    nearest-neighbour upsampling. Near-lossless for piecewise-constant test images, so tests
    can check the plumbing without downloading weights.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.config = SimpleNamespace(scaling_factor=1.0, latent_channels=LATENT_CHANNELS)

    def encode(self, x: torch.Tensor):
        pooled = F.avg_pool2d(x, DOWNSAMPLE)
        b, c, h, w = pooled.shape
        mean = torch.cat([pooled, pooled.new_zeros(b, LATENT_CHANNELS - c, h, w)], dim=1)
        logvar = torch.full_like(mean, -20.0)
        return SimpleNamespace(latent_dist=_Dist(mean, logvar))

    def decode(self, z: torch.Tensor):
        x = F.interpolate(z[:, : self.in_channels], scale_factor=DOWNSAMPLE, mode="nearest")
        return SimpleNamespace(sample=x)
