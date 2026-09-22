"""Distribution-level fidelity: FID and KID (torch-fidelity, InceptionV3 features).

Inception features were trained on natural images, so absolute FID values on MRI are not
comparable to natural-image benchmarks; they remain a valid *relative* measure between models
evaluated on the same real reference set. We report per-modality scores (each grey modality
replicated to RGB) and the composite RGB image the VAE sees.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from synthmri.utils.io import save_png


def export_pngs(images: np.ndarray | torch.Tensor, out_dir: str | Path, modalities: tuple[str, ...], overwrite: bool = False) -> dict[str, Path]:
    """Write ``images`` (N,C,S,S) in [0,1] as per-modality grey PNGs plus a composite RGB set."""
    out_dir = Path(out_dir)
    if isinstance(images, torch.Tensor):
        images = images.detach().float().cpu().numpy()
    images = np.asarray(images, dtype=np.float32)
    dirs = {m: out_dir / m for m in modalities}
    if images.shape[1] == 3:
        dirs["rgb"] = out_dir / "rgb"
    for d in dirs.values():
        if d.exists() and not overwrite and any(d.iterdir()):
            continue
        d.mkdir(parents=True, exist_ok=True)
    for i in range(images.shape[0]):
        for k, m in enumerate(modalities):
            p = dirs[m] / f"{i:06d}.png"
            if overwrite or not p.exists():
                save_png(images[i, k], p)
        if "rgb" in dirs:
            p = dirs["rgb"] / f"{i:06d}.png"
            if overwrite or not p.exists():
                save_png(images[i], p)
    return dirs


def compute_fid_kid(fake_dir: str | Path, real_dir: str | Path, device: str = "cuda", kid_subset_size: int | None = None) -> dict:
    import torch_fidelity

    n_fake = len(list(Path(fake_dir).glob("*.png")))
    n_real = len(list(Path(real_dir).glob("*.png")))
    subset = kid_subset_size or max(10, min(1000, n_fake, n_real))
    m = torch_fidelity.calculate_metrics(
        input1=str(fake_dir),
        input2=str(real_dir),
        cuda=str(device).startswith("cuda") and torch.cuda.is_available(),
        fid=True,
        kid=True,
        kid_subset_size=subset,
        kid_subsets=100 if min(n_fake, n_real) >= subset else 10,
        verbose=False,
        samples_find_deep=False,
    )
    return {
        "fid": float(m["frechet_inception_distance"]),
        "kid_mean": float(m["kernel_inception_distance_mean"]),
        "kid_std": float(m["kernel_inception_distance_std"]),
        "n_fake": n_fake,
        "n_real": n_real,
    }


def fidelity_report(fake_root: str | Path, real_root: str | Path, keys: tuple[str, ...], device: str = "cuda") -> dict:
    out = {}
    for k in keys:
        fd, rd = Path(fake_root) / k, Path(real_root) / k
        if fd.exists() and rd.exists():
            out[k] = compute_fid_kid(fd, rd, device=device)
    return out
