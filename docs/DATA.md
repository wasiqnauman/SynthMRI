# Data

## Source

BraTS 2020 training set (369 patients), as redistributed on Kaggle
(`awsaf49/brats20-dataset-training-validation`). Each patient folder holds five NIfTI volumes of
240 × 240 × 155 voxels at 1 mm³: `flair`, `t1`, `t1ce`, `t2` and `seg`. Volumes are skull-stripped
and co-registered; background voxels are exactly 0. Patient `BraTS20_Training_355` ships its
segmentation as `W39_1998.09.19_Segm.nii`; `find_patients` recognises it. The 125 "validation"
patients of the release have no labels and are not used.

Labels: BraTS codes {0, 1, 2, 4} are remapped to canonical classes {0 background, 1 NCR/NET,
2 oedema, 3 enhancing tumour}. Evaluation regions: WT = {1,2,3}, TC = {1,3}, ET = {3}.

## Splits

Patient-level, seed 0, 10 % validation / 20 % test (`splits/brats2020_patient_splits.json`):

| split | patients | tumour slices (128 & 256 px) |
|---|---|---|
| train | 258 | 15,895 |
| val | 37 | 2,159 |
| test | 74 | 4,623 |

## Preprocessing (`scripts/preprocess.py`)

1. Brain mask = any modality > 0.
2. Per modality and per volume: clip intensities inside the brain to the [0.5, 99.5] percentiles,
   rescale to [0, 1]; background stays 0.
3. Keep axial slices whose tumour covers ≥ 0.1 % of the slice (≥ 58 voxels at 240²).
4. Centre-crop to 224 × 224 (the brain never extends beyond x ∈ [29, 223], y ∈ [40, 197]) and resize
   to the target size (area interpolation for images, nearest for masks).

Output per split: `images.npy` (N, 4, S, S) float16 in [0, 1] with channel order
(flair, t1, t1ce, t2); `masks.npy` (N, S, S) uint8; `meta.csv` (patient id, slice index, tumour and
brain fractions, ET/TC presence); `stats.json`. Files are memory-mapped by the datasets, so many
DataLoader workers share them without copies. Sizes: 2.1 GB (128 px) and 8.3 GB (256 px) for train.

Cached VAE latents are written next to the split arrays as
`latents_<vae>_<modalities>_<flip|noflip>[_aug<K>].npz` (mean and log-variance, float16, both
orientations; with `train.augment: K` also K random affine variants per training slice and their
transform parameters, see `synthmri/data/augment.py`).

## Local layout on this machine

`data/` is git-ignored. On the office workstation `data/raw/BraTS2020` is a symlink to
`/home/syed/data/BraTS2020` (the 4.5 GB zip plus the extracted 43 GB). Regenerate everything with
`scripts/preprocess.py`; nothing under `data/` is needed to run the unit tests.
