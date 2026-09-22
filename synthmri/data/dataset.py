"""PyTorch datasets over the preprocessed slice arrays and over cached VAE latents."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from synthmri.data.augment import HFLIP, IDENTITY, transform_mask
from synthmri.data.brats import MODALITIES


class SliceDataset(Dataset):
    """Yields ``{"image": (C,S,S) float32 in [-1,1], "mask": (S,S) int64, "index": int}``.

    ``modalities`` selects and orders the image channels (default FLAIR, T1ce, T2 — the three
    channels mapped onto the RGB inputs of the pretrained VAE). Arrays are memory-mapped, so
    construction is cheap and DataLoader workers share the page cache.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        modalities: tuple[str, ...] = ("flair", "t1ce", "t2"),
        hflip: bool = False,
        return_mask: bool = True,
    ):
        d = Path(processed_dir) / split
        if not (d / "images.npy").exists():
            raise FileNotFoundError(f"No preprocessed data at {d}; run scripts/preprocess.py first")
        self.images = np.load(d / "images.npy", mmap_mode="r")
        self.masks = np.load(d / "masks.npy", mmap_mode="r")
        self.meta = pd.read_csv(d / "meta.csv") if (d / "meta.csv").stat().st_size > 1 else pd.DataFrame()
        self.mod_idx = [MODALITIES.index(m) for m in modalities]
        self.modalities = tuple(modalities)
        self.hflip = hflip
        self.return_mask = return_mask
        self.split = split

    def __len__(self) -> int:
        return int(self.images.shape[0])

    @property
    def image_size(self) -> int:
        return int(self.images.shape[-1])

    @property
    def patient_ids(self) -> np.ndarray:
        return self.meta["patient_id"].to_numpy()

    def __getitem__(self, i: int) -> dict:
        img = np.ascontiguousarray(self.images[i][self.mod_idx]).astype(np.float32)
        image = torch.from_numpy(img) * 2.0 - 1.0
        out = {"image": image, "index": i}
        if self.return_mask:
            out["mask"] = torch.from_numpy(np.ascontiguousarray(self.masks[i]).astype(np.int64))
        if self.hflip and torch.rand(()) < 0.5:
            out["image"] = out["image"].flip(-1)
            if self.return_mask:
                out["mask"] = out["mask"].flip(-1)
        return out


class LatentDataset(Dataset):
    """Cached VAE posteriors: samples ``z = mean + std * eps`` on every access.

    ``latents.npz`` holds ``mean`` and ``logvar`` of shape (N, F, 4, h, w) float16, where F is 1
    (original orientation) or 2 (original + horizontally flipped). Sampling from the stored
    posterior each step is equivalent to encoding online with ``latent_dist.sample()``.
    """

    def __init__(self, latents_file: str | Path, processed_dir: str | Path, split: str, hflip: bool = True):
        """``hflip=True`` draws uniformly from all cached variants (original, flip, affine augmentations);
        ``hflip=False`` always returns the original orientation."""
        z = np.load(latents_file)
        self.mean = torch.from_numpy(z["mean"])  # float16 (N, F, 4, h, w); converted per item
        self.logvar = torch.from_numpy(z["logvar"])
        self.scaling_factor = float(z["scaling_factor"])
        self.aug_params = z["aug_params"] if "aug_params" in z.files else None
        self.masks = np.load(Path(processed_dir) / split / "masks.npy", mmap_mode="r")
        if self.mean.shape[0] != self.masks.shape[0]:
            raise ValueError("latents.npz and masks.npy have different numbers of slices")
        self.num_variants = int(self.mean.shape[1])
        self.num_flips = self.num_variants  # backwards-compatible name
        self.hflip = hflip and self.num_variants > 1

    def __len__(self) -> int:
        return int(self.mean.shape[0])

    @property
    def latent_size(self) -> int:
        return int(self.mean.shape[-1])

    def variant_params(self, i: int, f: int) -> np.ndarray:
        if self.aug_params is not None:
            return self.aug_params[i, f]
        return HFLIP if f == 1 else IDENTITY

    def __getitem__(self, i: int) -> dict:
        f = int(torch.randint(0, self.num_variants, ())) if self.hflip else 0
        mean, logvar = self.mean[i, f].float(), self.logvar[i, f].float()
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        mask = torch.from_numpy(np.ascontiguousarray(self.masks[i]).astype(np.int64))
        mask = transform_mask(mask, self.variant_params(i, f))
        return {"latent": z * self.scaling_factor, "mask": mask, "index": i, "variant": f}
