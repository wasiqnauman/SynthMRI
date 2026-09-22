"""VAE reconstruction quality on held-out slices (PSNR / SSIM / LPIPS per modality).

The frozen VAE bounds what the latent diffusion model can produce: a generated image can never
be sharper than the VAE's own reconstructions. Reporting this ceiling separately makes the
diffusion model's fidelity interpretable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

from synthmri.models.vae import VAEWrapper
from synthmri.utils.io import image_grid, save_png


@torch.no_grad()
def vae_roundtrip(images, vae: VAEWrapper, device: torch.device | str = "cuda", batch_size: int = 64) -> np.ndarray:
    """``decode(encode(x))`` (posterior mean) of ``(N,C,S,S)`` images in [0, 1]; returns float16 in [0, 1].
    Used as a control in the segmentation study: real slices that carry only the detail the frozen
    VAE can reproduce, i.e. the ceiling any latent-diffusion sample is subject to."""
    vae = vae.to(device).eval()
    out = []
    for s in range(0, images.shape[0], batch_size):
        x = torch.from_numpy(np.ascontiguousarray(images[s : s + batch_size]).astype(np.float32)).to(device) * 2 - 1
        xr = vae.reconstruct(x, sample=False).clamp(-1, 1) / 2 + 0.5
        out.append(xr.cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def _lpips_model(device):
    try:
        import lpips

        return lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception:  # pragma: no cover - lpips is optional at runtime
        return None


@torch.no_grad()
def evaluate_vae_reconstruction(
    vae: VAEWrapper,
    dataset: Dataset,
    modalities: tuple[str, ...],
    device: torch.device | str = "cuda",
    batch_size: int = 32,
    max_items: int | None = None,
    num_workers: int = 2,
    use_lpips: bool = True,
) -> dict:
    """Return per-modality mean/std PSNR, SSIM and LPIPS of ``decode(encode(x))`` vs ``x``."""
    vae = vae.to(device).eval()
    lp = _lpips_model(device) if use_lpips else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    per = {m: {"psnr": [], "ssim": [], "lpips": []} for m in modalities}
    per["rgb"] = {"psnr": [], "ssim": [], "lpips": []}
    n = 0
    for batch in loader:
        x = batch["image"].to(device)  # [-1,1]
        xr = vae.reconstruct(x, sample=False).clamp(-1, 1)
        x01, xr01 = x / 2 + 0.5, xr / 2 + 0.5
        for k, m in enumerate(modalities):
            a, b = x01[:, k : k + 1], xr01[:, k : k + 1]
            per[m]["psnr"] += peak_signal_noise_ratio(b, a, data_range=1.0, reduction="none", dim=(1, 2, 3)).tolist()
            per[m]["ssim"] += structural_similarity_index_measure(b, a, data_range=1.0, reduction="none").tolist()
            if lp is not None:
                per[m]["lpips"] += lp(xr[:, k : k + 1].repeat(1, 3, 1, 1), x[:, k : k + 1].repeat(1, 3, 1, 1)).flatten().tolist()
        per["rgb"]["psnr"] += peak_signal_noise_ratio(xr01, x01, data_range=1.0, reduction="none", dim=(1, 2, 3)).tolist()
        per["rgb"]["ssim"] += structural_similarity_index_measure(xr01, x01, data_range=1.0, reduction="none").tolist()
        if lp is not None:
            per["rgb"]["lpips"] += lp(xr, x).flatten().tolist()
        n += x.shape[0]
        if max_items is not None and n >= max_items:
            break
    out = {"n": n, "modalities": list(modalities), "metrics": {}}
    for key, d in per.items():
        out["metrics"][key] = {
            met: {"mean": float(np.mean(v)), "std": float(np.std(v))} for met, v in d.items() if len(v)
        }
    return out


@torch.no_grad()
def save_reconstruction_examples(vae: VAEWrapper, dataset: Dataset, path: str | Path, indices: list[int], device="cuda") -> None:
    """Rows = examples; columns = [orig m1, recon m1, orig m2, recon m2, ...]."""
    vae = vae.to(device).eval()
    x = torch.stack([dataset[i]["image"] for i in indices]).to(device)
    xr = vae.reconstruct(x, sample=False).clamp(-1, 1)
    x01, xr01 = (x / 2 + 0.5).cpu(), (xr / 2 + 0.5).cpu()
    panels = []
    c = x.shape[1]
    for i in range(x.shape[0]):
        for k in range(c):
            panels.append(x01[i, k : k + 1].repeat(3, 1, 1))
            panels.append(xr01[i, k : k + 1].repeat(3, 1, 1))
    save_png(image_grid(torch.stack(panels), nrow=2 * c), path)
