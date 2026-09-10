"""BraTS 2020 on-disk layout, label conventions and volume loading.

Expected layout (the Kaggle ``awsaf49/brats20-dataset-training-validation`` release)::

    <raw_dir>/BraTS20_Training_001/BraTS20_Training_001_flair.nii
    <raw_dir>/BraTS20_Training_001/BraTS20_Training_001_t1.nii
    <raw_dir>/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii
    <raw_dir>/BraTS20_Training_001/BraTS20_Training_001_t2.nii
    <raw_dir>/BraTS20_Training_001/BraTS20_Training_001_seg.nii

One case (``BraTS20_Training_355``) ships its segmentation as ``W39_1998.09.19_Segm.nii``;
``find_patients`` handles that instead of silently dropping the patient.

Labels: BraTS encodes {0: background, 1: necrotic/non-enhancing core, 2: peritumoural oedema,
4: enhancing tumour}. We remap 4 -> 3 so masks are contiguous class indices 0..3.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

MODALITIES: tuple[str, ...] = ("flair", "t1", "t1ce", "t2")
CLASS_NAMES: tuple[str, ...] = (
    "background",
    "necrotic/non-enhancing tumour core",
    "peritumoural oedema",
    "enhancing tumour",
)
NUM_CLASSES = len(CLASS_NAMES)
_BRATS_TO_CANONICAL = {0: 0, 1: 1, 2: 2, 4: 3}

# Standard BraTS evaluation regions in canonical label space.
REGIONS: dict[str, tuple[int, ...]] = {
    "WT": (1, 2, 3),  # whole tumour
    "TC": (1, 3),  # tumour core
    "ET": (3,),  # enhancing tumour
}


@dataclass(frozen=True)
class PatientCase:
    patient_id: str
    modalities: dict[str, Path]
    seg: Path | None


def _nifti_files(d: Path) -> list[Path]:
    return sorted(list(d.glob("*.nii")) + list(d.glob("*.nii.gz")))


def _stem(p: Path) -> str:
    name = p.name.lower()
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def find_patients(raw_dir: str | Path, require_seg: bool = True) -> list[PatientCase]:
    """Discover patient folders under ``raw_dir``; warns about (and skips) incomplete cases."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"BraTS directory not found: {raw_dir}")
    cases: list[PatientCase] = []
    skipped: list[str] = []
    for d in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        files = _nifti_files(d)
        mods: dict[str, Path] = {}
        seg: Path | None = None
        for f in files:
            s = _stem(f)
            for m in MODALITIES:
                if s.endswith("_" + m):
                    mods[m] = f
            if s.endswith("_seg") or "segm" in s:
                seg = f
        missing = [m for m in MODALITIES if m not in mods]
        if missing or (require_seg and seg is None):
            skipped.append(f"{d.name} (missing: {', '.join(missing + (['seg'] if seg is None else []))})")
            continue
        cases.append(PatientCase(patient_id=d.name, modalities=mods, seg=seg))
    if skipped:
        warnings.warn(f"Skipped {len(skipped)} incomplete patient folder(s): {skipped[:5]}", stacklevel=2)
    if not cases:
        raise FileNotFoundError(f"No complete BraTS patients found in {raw_dir}")
    return cases


def load_volume(path: str | Path, dtype=np.float32) -> np.ndarray:
    """Load a NIfTI volume as a NumPy array in voxel (i, j, k) order; k indexes axial slices."""
    img = nib.load(str(path))
    return np.asarray(img.dataobj).astype(dtype, copy=False)


def remap_labels(seg: np.ndarray) -> np.ndarray:
    """BraTS label codes {0,1,2,4} -> canonical {0,1,2,3} (uint8). Raises on unexpected codes."""
    seg_int = np.rint(seg).astype(np.int64)
    present = set(np.unique(seg_int).tolist())
    unknown = present - set(_BRATS_TO_CANONICAL)
    if unknown:
        raise ValueError(f"Unexpected BraTS label values: {sorted(unknown)}")
    out = np.zeros(seg_int.shape, dtype=np.uint8)
    for src, dst in _BRATS_TO_CANONICAL.items():
        if src != 0:
            out[seg_int == src] = dst
    return out


def region_mask(mask: np.ndarray, region: str) -> np.ndarray:
    """Boolean mask for a BraTS region ("WT", "TC", "ET") given canonical labels."""
    return np.isin(mask, REGIONS[region])
