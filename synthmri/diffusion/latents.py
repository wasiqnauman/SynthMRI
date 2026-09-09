"""Encode a preprocessed split once with the frozen VAE and cache the posterior parameters.

Because the VAE is frozen, its posterior for each slice never changes during diffusion
training, so encoding online every step is wasted compute. We store the posterior mean and
log-variance (float16) for the original and, optionally, the horizontally flipped slice, and
``LatentDataset`` draws a fresh sample from the stored posterior on every access.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from synthmri.data.dataset import SliceDataset
from synthmri.models.vae import VAEWrapper


def latents_filename(vae_name: str, modalities: tuple[str, ...], hflip: bool) -> str:
    tag = re.sub(r"[^A-Za-z0-9]+", "-", vae_name).strip("-")
    return f"latents_{tag}_{'-'.join(modalities)}_{'flip' if hflip else 'noflip'}.npz"


@torch.no_grad()
def cache_latents(
    vae: VAEWrapper,
    processed_dir: str | Path,
    split: str,
    modalities: tuple[str, ...],
    vae_name: str,
    hflip: bool = True,
    batch_size: int = 64,
    num_workers: int = 4,
    device: torch.device | str = "cuda",
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    out = Path(processed_dir) / split / latents_filename(vae_name, modalities, hflip)
    if out.exists() and not overwrite:
        return out
    ds = SliceDataset(processed_dir, split, modalities=modalities, hflip=False, return_mask=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    means, logvars = [], []
    vae.to(device)
    for batch in tqdm(loader, desc=f"encode[{split}]", disable=not show_progress):
        x = batch["image"].to(device, non_blocking=True)
        variants = [x, x.flip(-1)] if hflip else [x]
        m_list, lv_list = [], []
        for v in variants:
            m, lv = vae.encode_dist(v)
            m_list.append(m.float().cpu())
            lv_list.append(lv.float().cpu())
        means.append(torch.stack(m_list, dim=1).half().numpy())
        logvars.append(torch.stack(lv_list, dim=1).half().numpy())
    mean = np.concatenate(means) if means else np.zeros((0, len(variants), 4, 1, 1), np.float16)
    logvar = np.concatenate(logvars) if logvars else np.zeros_like(mean)
    np.savez(out, mean=mean, logvar=logvar, scaling_factor=np.float32(vae.scaling_factor), modalities=np.array(modalities))
    return out
