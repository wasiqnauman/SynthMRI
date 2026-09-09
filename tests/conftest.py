"""Shared fixtures: a tiny synthetic BraTS-style dataset (no real data, no downloads)."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from synthmri.config import Config, config_from_dict
from synthmri.data import find_patients, make_patient_splits
from synthmri.data.preprocess import PreprocessParams, preprocess_dataset

VOL = (40, 40, 8)  # (x, y, axial slices)
IMG = 32  # preprocessed slice size -> 4x4 latent


def _write_patient(d: Path, pid: str, rng: np.random.RandomState, seg_name: str | None = None, missing: tuple[str, ...] = ()):
    d.mkdir(parents=True)
    x, y, z = np.meshgrid(np.arange(VOL[0]), np.arange(VOL[1]), np.arange(VOL[2]), indexing="ij")
    brain = ((x - 20) ** 2 + (y - 20) ** 2) < 15**2
    tumour = ((x - 24) ** 2 + (y - 18) ** 2) < 5**2
    core = ((x - 24) ** 2 + (y - 18) ** 2) < 3**2
    enh = ((x - 24) ** 2 + (y - 18) ** 2) < 1.5**2
    seg = np.zeros(VOL, np.float32)
    seg[tumour] = 2
    seg[core] = 1
    seg[enh] = 4  # BraTS code for enhancing tumour
    seg[:, :, 0] = 0  # first slice has no tumour -> must be filtered out
    affine = np.eye(4)
    for m in ("flair", "t1", "t1ce", "t2"):
        if m in missing:
            continue
        vol = np.zeros(VOL, np.float32)
        vol[brain] = 100 + 50 * rng.rand(int(brain.sum())) + {"flair": 0, "t1": 20, "t1ce": 40, "t2": 60}[m]
        vol[tumour] += 80
        nib.save(nib.Nifti1Image(vol, affine), str(d / f"{pid}_{m}.nii"))
    if "seg" not in missing:
        nib.save(nib.Nifti1Image(seg, affine), str(d / (seg_name or f"{pid}_seg.nii")))


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("brats_raw")
    rng = np.random.RandomState(0)
    for i in range(1, 6):
        pid = f"BraTS20_Training_{i:03d}"
        _write_patient(root / pid, pid, rng, seg_name="W39_1998.09.19_Segm.nii" if i == 5 else None)
    _write_patient(root / "BraTS20_Training_099", "BraTS20_Training_099", rng, missing=("t2",))  # incomplete
    return root


@pytest.fixture(scope="session")
def splits(raw_dir) -> dict:
    cases = find_patients(raw_dir)
    return make_patient_splits([c.patient_id for c in cases], val_frac=0.2, test_frac=0.2, seed=0)


@pytest.fixture(scope="session")
def processed_dir(tmp_path_factory, raw_dir, splits) -> Path:
    out = tmp_path_factory.mktemp("brats_processed")
    cases = find_patients(raw_dir)
    params = PreprocessParams(image_size=IMG, crop_size=None, tumour_frac_min=0.001, clip_percentiles=(0.5, 99.5))
    preprocess_dataset(cases, splits, params, out, num_workers=1, show_progress=False)
    return out


@pytest.fixture
def tiny_cfg(tmp_path, processed_dir) -> Config:
    """A CPU-friendly config pointing at the fixture data with a very small U-Net."""
    return config_from_dict(
        {
            "name": "test",
            "data": {"processed_dir": str(processed_dir), "image_size": IMG, "crop_size": None, "num_workers": 0},
            "model": {
                "unet": {"block_out_channels": [16, 32], "layers_per_block": 1, "attention_levels": [1], "norm_num_groups": 8, "attention_head_dim": 4},
                "conditioning": "none",
            },
            "train": {
                "output_dir": str(tmp_path / "run"), "epochs": 1, "max_steps": 3, "batch_size": 4, "warmup_steps": 1,
                "mixed_precision": "no", "num_workers": 0, "sample_every": 1, "save_every": 1, "num_sample_images": 2,
                "sample_steps": 2, "log_every": 1, "ema_decay": 0.9,
            },
            "sample": {"num_images": 2, "batch_size": 2, "steps": 2},
        }
    )
