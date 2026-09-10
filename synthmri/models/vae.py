"""Frozen pretrained VAE (Stable Diffusion ``sd-vae-ft-mse``) behind a small uniform interface.

The three MRI modalities are mapped onto the VAE's RGB input channels. The latent space is
4 x (S/8) x (S/8); latents are multiplied by ``scaling_factor`` (0.18215 for the SD VAE) so
that they have roughly unit variance, as in Rombach et al. (2022).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

LATENT_CHANNELS = 4
DOWNSAMPLE = 8
DECODER_KEYS = ("decoder", "post_quant_conv")  # the parts a decoder fine-tune changes (synthmri.models.vae_finetune)


@contextmanager
def cudnn_heuristics():
    """Run with ``cudnn.benchmark`` off.

    Autotuning the VAE's full-resolution convolutions tries FFT-style algorithms whose workspaces
    peak at ~18 GB (128 px) / ~36 GB (256 px) for a 16-image decode, while the heuristic choice is
    within a few percent of the same speed. The VAE only ever sees a handful of shapes, so the
    U-Net keeps autotuning and the VAE does not.
    """
    prev = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = prev


def decoder_state_dict(vae: nn.Module) -> dict[str, torch.Tensor]:
    """CPU copy of the ``decoder`` + ``post_quant_conv`` tensors of an AutoencoderKL-like module."""
    return {k: v.detach().cpu().clone() for k, v in vae.state_dict().items() if k.split(".")[0] in DECODER_KEYS}


def load_decoder_weights(vae: nn.Module, path: str | Path) -> int:
    """Load a fine-tuned decoder saved by ``finetune_decoder`` (``decoder.pt``); returns the number of tensors."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    state = obj["state"] if isinstance(obj, dict) and "state" in obj else obj
    expected = {k for k in vae.state_dict() if k.split(".")[0] in DECODER_KEYS}
    missing, unexpected = expected - set(state), set(state) - expected
    if missing or unexpected:
        raise ValueError(f"{path} does not match this VAE's decoder: {len(missing)} missing, {len(unexpected)} unexpected tensors")
    vae.load_state_dict(state, strict=False)
    return len(state)


class VAEWrapper(nn.Module):
    def __init__(self, vae: nn.Module, scaling_factor: float | None = None, decoder_weights: str | Path | None = None):
        super().__init__()
        self.vae = vae
        self.scaling_factor = float(scaling_factor or vae.config.scaling_factor)
        self.downsample = DOWNSAMPLE
        self.latent_channels = LATENT_CHANNELS
        self.decoder_weights = str(decoder_weights) if decoder_weights else None
        if decoder_weights:
            load_decoder_weights(vae, decoder_weights)

    @torch.no_grad()
    def encode_dist(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Unscaled posterior (mean, logvar) for images ``x`` in [-1, 1]."""
        with cudnn_heuristics():
            dist = self.vae.encode(x).latent_dist
        return dist.mean, dist.logvar

    @torch.no_grad()
    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Scaled latent; ``sample=False`` returns the posterior mean."""
        with cudnn_heuristics():
            dist = self.vae.encode(x).latent_dist
        z = dist.sample() if sample else dist.mode()
        return z * self.scaling_factor

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Scaled latent -> image in [-1, 1] (not clamped)."""
        with cudnn_heuristics():
            return self.vae.decode(z / self.scaling_factor).sample

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        return self.decode(self.encode(x, sample=sample))


def load_vae(pretrained: str = "stabilityai/sd-vae-ft-mse", device=None, dtype=torch.float32, scaling_factor=None,
             decoder_weights: str | Path | None = None) -> VAEWrapper:
    """``decoder_weights``: optional ``decoder.pt`` from ``scripts/finetune_vae_decoder.py`` (encoder stays the pretrained one)."""
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(pretrained, torch_dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()
    wrapper = VAEWrapper(vae, scaling_factor, decoder_weights)
    if device is not None:
        wrapper.to(device)
    return wrapper


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

    def __init__(self, in_channels: int = 3, trainable_decoder: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.config = SimpleNamespace(scaling_factor=1.0, latent_channels=LATENT_CHANNELS)
        self.decoder = nn.Identity()
        if trainable_decoder:  # identity-initialised 1x1 conv so decoder fine-tuning has something to train
            self.post_quant_conv = nn.Conv2d(LATENT_CHANNELS, LATENT_CHANNELS, 1)
            with torch.no_grad():
                self.post_quant_conv.weight.copy_(torch.eye(LATENT_CHANNELS).view(LATENT_CHANNELS, LATENT_CHANNELS, 1, 1))
                self.post_quant_conv.bias.zero_()
        else:
            self.post_quant_conv = nn.Identity()

    def encode(self, x: torch.Tensor):
        pooled = F.avg_pool2d(x, DOWNSAMPLE)
        b, c, h, w = pooled.shape
        mean = torch.cat([pooled, pooled.new_zeros(b, LATENT_CHANNELS - c, h, w)], dim=1)
        logvar = torch.full_like(mean, -20.0)
        return SimpleNamespace(latent_dist=_Dist(mean, logvar))

    def decode(self, z: torch.Tensor):
        z = self.decoder(self.post_quant_conv(z))
        x = F.interpolate(z[:, : self.in_channels], scale_factor=DOWNSAMPLE, mode="nearest")
        return SimpleNamespace(sample=x)
