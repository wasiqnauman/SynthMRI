# SynthMRI — latent diffusion for synthetic multi-modal brain-tumour MRI

Research code for generating synthetic 2-D brain-tumour MRI slices (FLAIR / T1ce / T2) from the
BraTS 2020 training set with a **latent diffusion model** (LDM): a frozen, pretrained Stable Diffusion
VAE compresses each slice 8× into a 4-channel latent, and a small U-Net learns to denoise in that
latent space. A **mask-conditioned** variant generates paired (image, tumour-mask) data, which is
evaluated by whether it improves a downstream tumour segmenter.

```
BraTS 2020 3-D volumes ──preprocess──▶ 2-D tumour slices (patient-level train/val/test)
        │                                          │
        │                       frozen SD-VAE ─encode─▶ cached latents (+ flips)
        │                                          │
        │                       latent U-Net ◀──train──┘   (optional mask conditioning, CFG, EMA)
        ▼                                          │
   real test slices ◀──── evaluate ────── sample ──┘  DDIM/DDPM  ──▶ synthetic (image, mask) pairs
   FID / KID / SSIM-diversity / memorisation / VAE ceiling                │
                                                                          ▼
                                            2-D U-Net segmenter: real vs real+synthetic (Dice WT/TC/ET)
```

## Setup

```bash
conda env create -f environment.yml   # or: pip install -r requirements.txt && pip install -e .
conda activate synthmri
pytest                                # CPU-only unit tests, ~1 min, no data or downloads needed
```

The exact package versions used for the reported runs are in `requirements-lock.txt`.

## Data

Download the BraTS 2020 training data (Kaggle `awsaf49/brats20-dataset-training-validation`) and
point `data.raw_dir` at the folder containing the `BraTS20_Training_XXX` patient directories
(default: `data/raw/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData`). Then

```bash
python scripts/preprocess.py --config configs/ldm128_maskcond.yaml   # 128 px  (~1.5 min, 20 workers)
python scripts/preprocess.py --config configs/ldm256_maskcond.yaml   # 256 px  (~2 min)
```

This writes per-split arrays under `data/processed/brats{128,256}/{train,val,test}/` and reuses the
committed patient split `splits/brats2020_patient_splits.json` (258 / 37 / 74 patients). See
[docs/DATA.md](docs/DATA.md) for the preprocessing details and dataset statistics.

## Train, sample, evaluate

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0

python scripts/train.py    --config configs/ldm128_maskcond.yaml          # or ldm128_uncond / ldm256_*
python scripts/sample.py   --run runs/ldm128_maskcond --num_images 5000 --guidance_scale 2.0
python scripts/evaluate.py --run runs/ldm128_maskcond --samples runs/ldm128_maskcond/samples_ddim50_cfg2_seed0
python scripts/train_seg.py --config configs/ldm128_maskcond.yaml --out runs/seg/real100_s0            # real only
python scripts/train_seg.py --config configs/ldm128_maskcond.yaml --real_fraction 0.1 \
    --synthetic runs/ldm128_maskcond/samples_ddim50_cfg2_seed0 --n_synth 2000 --out runs/seg/real10_synth2000_s0
```

Any config value can be overridden on the command line with dotted keys, e.g.
`python scripts/train.py --config configs/ldm128_uncond.yaml train.epochs=50 train.batch_size=32`.
`configs/smoke.yaml` runs the whole pipeline for 30 steps as a check. `scripts/run_experiments.sh`
reproduces the full set of experiments in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md); results are
collected in [docs/RESULTS.md](docs/RESULTS.md).

Each run directory contains `config.yaml`, `run_info.json` (git commit, versions), `metrics.csv`,
TensorBoard logs (`tb/`), periodic sample sheets (`samples/`), rolling checkpoints, `best/` (lowest
validation loss, EMA weights) and `final/`.

## Repository layout

```
synthmri/
  config.py             typed YAML config with _base_ inheritance and dotted CLI overrides
  data/                 BraTS discovery (brats.py), patient splits, preprocessing, datasets
  models/               frozen VAE wrapper (+ test double), U-Net factory
  diffusion/            latent caching, conditioning, schedulers, training loop, sampling, checkpoints
  eval/                 VAE reconstruction, FID/KID, diversity & memorisation, downstream segmentation
  utils/                seeding, I/O, logging, figures
scripts/                preprocess / train / sample / evaluate / train_seg / run_experiments.sh
configs/                base.yaml + experiment configs + smoke.yaml
tests/                  pytest suite on a synthetic mini-BraTS (no downloads)
splits/                 committed patient-level split
docs/                   ARCHITECTURE, DATA, EXPERIMENTS, RESULTS, CHANGES (work log)
notebooks/              legacy exploratory notebook (superseded by the package)
```

## Notes

* Only one compute GPU is needed; a full 128-px run trains in about an hour on an RTX A6000.
* The three modalities are mapped onto the VAE's RGB channels, so exactly three modalities are used.
  T1 is preprocessed and stored as well (`MODALITIES = (flair, t1, t1ce, t2)`) for future work.
* `docs/CHANGES.md` is the dated work log for this repository.
