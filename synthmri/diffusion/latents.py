"""Encode a preprocessed split once with the frozen VAE and cache the posterior parameters.

Because the VAE is frozen, its posterior for each slice never changes during diffusion
training, so encoding online every step is wasted compute. We store the posterior mean and
log-variance (float16) for the original slice, optionally its horizontal flip and ``augment``
random affine variants (flip / shift / rotation / scale, see ``synthmri.data.augment``), and
``LatentDataset`` draws one variant and a fresh sample from its stored posterior on every access.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from synthmri.data.augment import HFLIP, IDENTITY, apply_affine_image, sample_affine_params
from synthmri.data.dataset import SliceDataset
from synthmri.models.vae import VAEWrapper


def latents_filename(vae_name: str, modalities: tuple[str, ...], hflip: bool, augment: int = 0) -> str:
    tag = re.sub(r"[^A-Za-z0-9]+", "-", vae_name).strip("-")
    aug = f"_aug{augment}" if augment > 0 else ""
    return f"latents_{tag}_{'-'.join(modalities)}_{'flip' if hflip else 'noflip'}{aug}.npz"


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
    augment: int = 0,
    aug_max_shift: float = 0.06,
    aug_max_rotate: float = 10.0,
    aug_scale: tuple[float, float] = (0.9, 1.1),
    aug_seed: int = 0,
) -> Path:
    """Variants per slice: original, horizontal flip (if ``hflip``) and ``augment`` random affine
    transforms; their parameters are stored as ``aug_params`` (N, F, 5)."""
    out = Path(processed_dir) / split / latents_filename(vae_name, modalities, hflip, augment)
    settings = f"augment={augment} shift={aug_max_shift} rotate={aug_max_rotate} scale={tuple(aug_scale)} seed={aug_seed}"
    if out.exists() and not overwrite:
        if augment > 0:
            stored = str(np.load(out)["aug_settings"])
            if stored != settings:
                raise RuntimeError(f"{out} was cached with different augmentation settings ({stored}); delete it or pass overwrite=True")
        return out
    ds = SliceDataset(processed_dir, split, modalities=modalities, hflip=False, return_mask=False)
    n = len(ds)
    fixed = [IDENTITY] + ([HFLIP] if hflip else [])
    params = np.tile(np.stack(fixed)[None], (n, 1, 1)).astype(np.float32)  # (N, F0, 5)
    if augment > 0:
        rnd = sample_affine_params(n * augment, aug_max_shift, aug_max_rotate, aug_scale, hflip=hflip, seed=aug_seed).reshape(n, augment, 5)
        params = np.concatenate([params, rnd], axis=1)
    n_var = params.shape[1]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    means, logvars = [], []
    vae.to(device)
    start = 0
    for batch in tqdm(loader, desc=f"encode[{split}]", disable=not show_progress):
        x = batch["image"].to(device, non_blocking=True)
        b = x.shape[0]
        m_list, lv_list = [], []
        for f in range(n_var):
            if f == 0:
                v = x
            elif f == 1 and hflip:
                v = x.flip(-1)
            else:
                v = apply_affine_image(x, torch.from_numpy(params[start : start + b, f]).to(device))
            m, lv = vae.encode_dist(v)
            m_list.append(m.float().cpu())
            lv_list.append(lv.float().cpu())
        means.append(torch.stack(m_list, dim=1).half().numpy())
        logvars.append(torch.stack(lv_list, dim=1).half().numpy())
        start += b
    mean = np.concatenate(means) if means else np.zeros((0, n_var, 4, 1, 1), np.float16)
    logvar = np.concatenate(logvars) if logvars else np.zeros_like(mean)
    np.savez(out, mean=mean, logvar=logvar, scaling_factor=np.float32(vae.scaling_factor), modalities=np.array(modalities),
             aug_params=params, aug_settings=np.array(settings))
    return out
