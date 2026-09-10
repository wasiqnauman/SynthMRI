"""3-D BraTS volumes -> normalised 2-D axial slice arrays, one memory-mappable file per split.

Per patient:
  1. Load the four modalities and the segmentation.
  2. Brain mask = any modality > 0 (BraTS volumes are skull-stripped, background is exactly 0).
  3. Per modality, clip intensities inside the brain to the [p_lo, p_hi] percentiles of that
     *volume* and rescale to [0, 1]; background stays 0. (Per-volume rather than per-slice
     normalisation keeps intensity consistent across neighbouring slices of the same scan.)
  4. Keep axial slices whose tumour area is >= ``tumour_frac_min`` of the slice.
  5. Centre-crop (drops the empty border) and resize to ``image_size`` (area interpolation for
     images, nearest-neighbour for masks).

Outputs under ``<out_dir>/<split>/``: ``images.npy`` (N, 4, S, S) float16 in [0, 1] with channel
order ``MODALITIES``; ``masks.npy`` (N, S, S) uint8 with canonical labels; ``meta.csv`` with the
patient id, slice index and tumour/brain fractions of each row.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from synthmri.data.brats import MODALITIES, PatientCase, load_volume, remap_labels
from synthmri.utils.io import save_json


@dataclass(frozen=True)
class PreprocessParams:
    image_size: int = 128
    crop_size: int | None = 224
    tumour_frac_min: float = 0.001
    clip_percentiles: tuple[float, float] = (0.5, 99.5)


def brain_mask(volumes: dict[str, np.ndarray]) -> np.ndarray:
    mask = np.zeros(next(iter(volumes.values())).shape, dtype=bool)
    for v in volumes.values():
        mask |= v > 0
    return mask


def normalise_volume(vol: np.ndarray, mask: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    """Percentile-clip inside ``mask`` and rescale to [0, 1]; voxels outside ``mask`` are set to 0."""
    vals = vol[mask]
    if vals.size == 0:
        return np.zeros_like(vol, dtype=np.float32)
    lo, hi = np.percentile(vals, percentiles)
    out = (np.clip(vol, lo, hi) - lo) / max(hi - lo, 1e-6)
    out = out.astype(np.float32)
    out[~mask] = 0.0
    return out


def center_crop(a: np.ndarray, size: int | None) -> np.ndarray:
    if size is None:
        return a
    h, w = a.shape[:2]
    if size >= h and size >= w:
        return a
    y0 = max((h - size) // 2, 0)
    x0 = max((w - size) // 2, 0)
    return a[y0 : y0 + min(size, h), x0 : x0 + min(size, w)]


def resize_image(a: np.ndarray, size: int) -> np.ndarray:
    if a.shape[0] == size and a.shape[1] == size:
        return a.astype(np.float32, copy=False)
    interp = cv2.INTER_AREA if size < a.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(a.astype(np.float32), (size, size), interpolation=interp)


def resize_mask(a: np.ndarray, size: int) -> np.ndarray:
    if a.shape[0] == size and a.shape[1] == size:
        return a.astype(np.uint8, copy=False)
    return cv2.resize(a.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)


def process_patient(case: PatientCase, params: PreprocessParams) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Return (images (n,4,S,S) float16, masks (n,S,S) uint8, meta rows) for one patient."""
    vols = {m: load_volume(case.modalities[m]) for m in MODALITIES}
    seg = remap_labels(load_volume(case.seg))
    bm = brain_mask(vols)
    norm = {m: normalise_volume(vols[m], bm, params.clip_percentiles) for m in MODALITIES}
    s = params.image_size
    images, masks, meta = [], [], []
    for z in range(seg.shape[2]):
        mz = seg[:, :, z]
        tumour_frac = float((mz > 0).mean())
        if tumour_frac < params.tumour_frac_min:
            continue
        chans = [resize_image(center_crop(norm[m][:, :, z], params.crop_size), s) for m in MODALITIES]
        mask = resize_mask(center_crop(mz, params.crop_size), s)
        images.append(np.stack(chans, axis=0).astype(np.float16))
        masks.append(mask)
        meta.append(
            {
                "patient_id": case.patient_id,
                "slice_index": int(z),
                "tumour_frac": tumour_frac,
                "brain_frac": float(bm[:, :, z].mean()),
                "has_et": bool((mz == 3).any()),
                "has_tc": bool(((mz == 1) | (mz == 3)).any()),
            }
        )
    if not images:
        empty = (np.zeros((0, len(MODALITIES), s, s), np.float16), np.zeros((0, s, s), np.uint8), [])
        return empty
    return np.stack(images), np.stack(masks), meta


def _worker(args: tuple[PatientCase, PreprocessParams]):
    return process_patient(*args)


def brain_bounding_box(case: PatientCase) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1) extent of the brain in the axial plane, to sanity-check ``crop_size``."""
    vols = {m: load_volume(case.modalities[m]) for m in MODALITIES}
    bm = brain_mask(vols).any(axis=2)
    ys, xs = np.where(bm)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def preprocess_dataset(
    cases: list[PatientCase],
    splits: dict,
    params: PreprocessParams,
    out_dir: str | Path,
    num_workers: int = 8,
    show_progress: bool = True,
) -> dict:
    """Preprocess every split and write arrays to ``out_dir``; returns the stats dictionary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {c.patient_id: c for c in cases}
    stats: dict = {"params": asdict(params), "splits": {}}
    for split in ("train", "val", "test"):
        ids = [pid for pid in splits[split] if pid in by_id]
        missing = sorted(set(splits[split]) - set(by_id))
        if missing:
            print(f"[{split}] {len(missing)} patient(s) in the split file are not on disk: {missing[:5]}")
        jobs = [(by_id[pid], params) for pid in ids]
        results = []
        if num_workers > 1 and len(jobs) > 1:
            with Pool(num_workers) as pool:
                it = pool.imap(_worker, jobs, chunksize=1)
                for r in tqdm(it, total=len(jobs), desc=f"preprocess[{split}]", disable=not show_progress):
                    results.append(r)
        else:
            for job in tqdm(jobs, desc=f"preprocess[{split}]", disable=not show_progress):
                results.append(_worker(job))
        images = np.concatenate([r[0] for r in results]) if results else np.zeros((0, 4, params.image_size, params.image_size), np.float16)
        masks = np.concatenate([r[1] for r in results]) if results else np.zeros((0, params.image_size, params.image_size), np.uint8)
        meta = pd.DataFrame([row for r in results for row in r[2]])
        d = out_dir / split
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "images.npy", images)
        np.save(d / "masks.npy", masks)
        meta.to_csv(d / "meta.csv", index=False)
        stats["splits"][split] = {
            "n_patients": len(ids),
            "n_slices": int(images.shape[0]),
            "slices_per_patient": float(images.shape[0] / max(len(ids), 1)),
            "class_pixel_fraction": [float((masks == c).mean()) if masks.size else 0.0 for c in range(4)],
        }
    save_json(stats, out_dir / "stats.json")
    return stats
