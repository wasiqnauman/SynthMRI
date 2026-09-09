"""Sampling from a trained latent diffusion model (DDIM or ancestral DDPM), with optional
mask conditioning and classifier-free guidance."""

from __future__ import annotations

import torch
from torch import nn

from synthmri.config import DiffusionConfig
from synthmri.diffusion.conditioning import null_condition
from synthmri.diffusion.schedulers import build_sampling_scheduler
from synthmri.models.vae import LATENT_CHANNELS, VAEWrapper


class LatentSampler:
    def __init__(
        self,
        unet: nn.Module,
        vae: VAEWrapper | None,
        diffusion_cfg: DiffusionConfig,
        latent_size: int,
        sampler: str = "ddim",
        steps: int = 50,
        guidance_scale: float = 1.0,
        eta: float = 0.0,
        num_mask_classes: int = 4,
        device: torch.device | str = "cuda",
        autocast_dtype: torch.dtype | None = torch.bfloat16,
    ):
        self.unet = unet.to(device).eval()
        self.vae = vae.to(device).eval() if vae is not None else None
        self.scheduler = build_sampling_scheduler(diffusion_cfg, sampler)
        self.sampler = sampler
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.eta = eta
        self.latent_size = latent_size
        self.num_mask_classes = num_mask_classes
        self.device = torch.device(device)
        self.autocast_dtype = autocast_dtype if self.device.type == "cuda" else None

    def _predict(self, latents: torch.Tensor, t, cond: torch.Tensor | None) -> torch.Tensor:
        if cond is None:
            model_in = latents
        elif self.guidance_scale != 1.0:
            null = null_condition(latents.shape[0], self.num_mask_classes, self.latent_size, latents.device)
            model_in = torch.cat([torch.cat([latents, cond], 1), torch.cat([latents, null], 1)], 0)
        else:
            model_in = torch.cat([latents, cond], 1)
        model_in = self.scheduler.scale_model_input(model_in, t)
        if self.autocast_dtype is not None:
            with torch.autocast("cuda", dtype=self.autocast_dtype):
                pred = self.unet(model_in, t).sample.float()
        else:
            pred = self.unet(model_in, t).sample.float()
        if cond is not None and self.guidance_scale != 1.0:
            pred_c, pred_u = pred.chunk(2)
            pred = pred_u + self.guidance_scale * (pred_c - pred_u)
        return pred

    @torch.no_grad()
    def sample_latents(self, n: int, cond: torch.Tensor | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
        shape = (n, LATENT_CHANNELS, self.latent_size, self.latent_size)
        latents = torch.randn(shape, generator=generator, device=self.device)
        self.scheduler.set_timesteps(self.steps, device=self.device)
        latents = latents * self.scheduler.init_noise_sigma
        cond = None if cond is None else cond.to(self.device)
        for t in self.scheduler.timesteps:
            pred = self._predict(latents, t, cond)
            if self.sampler == "ddim":
                latents = self.scheduler.step(pred, t, latents, eta=self.eta, generator=generator).prev_sample
            else:
                latents = self.scheduler.step(pred, t, latents, generator=generator).prev_sample
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Latents -> images in [0, 1]."""
        if self.vae is None:
            raise RuntimeError("No VAE attached to the sampler")
        x = self.vae.decode(latents.float())
        return (x / 2.0 + 0.5).clamp(0.0, 1.0)

    @torch.no_grad()
    def sample_images(self, n: int, cond: torch.Tensor | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
        return self.decode(self.sample_latents(n, cond=cond, generator=generator))
