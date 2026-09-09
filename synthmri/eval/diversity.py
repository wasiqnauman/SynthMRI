"""Sample diversity and memorisation checks.

* ``pairwise_ssim_diversity``: mean SSIM over random pairs of generated images (lower = more
  diverse; a collapsed model gives values near 1).
* ``nearest_neighbour_distances``: L2 distance from each generated image to its closest
  training image. Compared against the same statistic for *real held-out* slices, this tells
  whether the generator merely copies training slices: if generated samples are, on average,
  no closer to the training set than unseen real slices are, memorisation is not indicated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure

from synthmri.utils.io import image_grid, save_png


def _to_tensor(x) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
    return x.float()


@torch.no_grad()
def pairwise_ssim_diversity(images, n_pairs: int = 2000, seed: int = 0, device="cuda", batch: int = 256) -> dict:
    x = _to_tensor(images)
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least two images")
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, n, (n_pairs,), generator=g)
    j = torch.randint(0, n - 1, (n_pairs,), generator=g)
    j = j + (j >= i).long()  # ensure j != i
    vals = []
    for s in range(0, n_pairs, batch):
        a = x[i[s : s + batch]].to(device)
        b = x[j[s : s + batch]].to(device)
        vals.append(structural_similarity_index_measure(a, b, data_range=1.0, reduction="none").cpu())
    v = torch.cat(vals)
    return {"pairwise_ssim_mean": float(v.mean()), "pairwise_ssim_std": float(v.std()), "n_pairs": n_pairs}


def _features(x: torch.Tensor, size: int) -> torch.Tensor:
    """Downsample to (size,size), flatten, L2-normalise per image (cosine-like distance)."""
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="area")
    f = x.flatten(1)
    return f


@torch.no_grad()
def nearest_neighbour_distances(query, reference, device="cuda", feat_size: int = 64, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """For each query image, the L2 distance to and index of its nearest reference image."""
    # float32 throughout: a half-precision matmul loses ~0.1 in |x|^2 terms of order 10^3, which is
    # enough to turn an exact match into a spurious non-zero distance.
    q = _features(_to_tensor(query), feat_size).to(device).float()
    r = _features(_to_tensor(reference), feat_size).to(device).float()
    r_sq = (r**2).sum(1)
    dists, idxs = [], []
    for s in range(0, q.shape[0], chunk):
        qq = q[s : s + chunk]
        d2 = (qq**2).sum(1, keepdim=True) + r_sq[None] - 2 * (qq @ r.T)
        ix = d2.argmin(1)
        # Re-evaluate the winning pair exactly: the expansion above cancels catastrophically
        # for near-identical vectors and would report a spurious non-zero distance.
        m = (qq - r[ix]).norm(dim=1)
        dists.append(m.cpu())
        idxs.append(ix.cpu())
    return torch.cat(dists).numpy(), torch.cat(idxs).numpy()


def memorisation_report(fake, train, real_test, device="cuda", feat_size: int = 64) -> dict:
    """NN-distance statistics for generated and real held-out images against the training set."""
    d_fake, i_fake = nearest_neighbour_distances(fake, train, device, feat_size)
    d_test, _ = nearest_neighbour_distances(real_test, train, device, feat_size)
    thr = float(np.percentile(d_test, 5))
    return {
        "fake_to_train_nn_dist": {"mean": float(d_fake.mean()), "std": float(d_fake.std()), "min": float(d_fake.min()), "p5": float(np.percentile(d_fake, 5))},
        "test_to_train_nn_dist": {"mean": float(d_test.mean()), "std": float(d_test.std()), "min": float(d_test.min()), "p5": thr},
        "frac_fake_closer_than_test_p5": float((d_fake < thr).mean()),
        "nn_index": i_fake.tolist()[:64],
    }


def save_nearest_neighbour_figure(fake, train, nn_idx: np.ndarray, path: str | Path, n: int = 8, dists: np.ndarray | None = None) -> None:
    """Rows: generated image (left) next to its nearest training image (right); closest first."""
    fake, train = _to_tensor(fake), _to_tensor(train)
    order = np.argsort(dists)[:n] if dists is not None else np.arange(min(n, fake.shape[0]))
    panels = []
    for k in order:
        panels.append(fake[k])
        panels.append(train[int(nn_idx[k])])
    save_png(image_grid(torch.stack(panels), nrow=2), path)
