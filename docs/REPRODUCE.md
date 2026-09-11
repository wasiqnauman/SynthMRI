# Reproducing the experiments

Everything below was run on one NVIDIA RTX A6000 (48 GB) with the package versions in
`requirements-lock.txt`. A GPU with 16 GB is enough for every configuration: training allocates about
5 GB (128 px, batch 64) and 5.5 GB (256 px, batch 32; ~11 GB reserved as reported by `nvidia-smi`);
sampling about 2 GB (128 px, batch 64) and 8 GB (256 px, batch 32). Times are wall-clock on that card.

## 1. Environment

```bash
conda env create -f environment.yml && conda activate synthmri     # Python 3.11
# or: python -m venv .venv && . .venv/bin/activate && pip install -r requirements-lock.txt && pip install -e .
pytest                                                              # 45 CPU-only tests, ~1 min
```

`pytest` needs no data or downloads; it runs on a generated mini-BraTS (`tests/conftest.py`).

## 2. Data (BraTS 2020 training set, 369 patients)

Either source gives the same files (`BraTS20_Training_XXX/BraTS20_Training_XXX_{flair,t1,t1ce,t2,seg}.nii`):

* Kaggle `awsaf49/brats20-dataset-training-validation` (needs a Kaggle account), or
* the Hugging Face mirror `Babai12345/BRATS-2020` (`BRATS-2020.zip`, 4.5 GB; no account needed):

```bash
mkdir -p /path/to/BraTS2020 && cd /path/to/BraTS2020
curl -L -o BRATS-2020.zip https://huggingface.co/datasets/Babai12345/BRATS-2020/resolve/main/BRATS-2020.zip
unzip -q BRATS-2020.zip          # ~43 GB extracted
cd /path/to/SynthMRI && ln -s /path/to/BraTS2020 data/raw/BraTS2020
```

The expected folder is `data/raw/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData`
(override with `data.raw_dir=...` on any command line). Patient 355 ships its segmentation as
`W39_1998.09.19_Segm.nii`; the loader accepts it.

## 3. Preprocess

```bash
python scripts/preprocess.py --config configs/ldm128_maskcond.yaml   # 128 px, ~1.5 min with 16 workers
python scripts/preprocess.py --config configs/ldm256_maskcond.yaml   # 256 px, ~2 min
```

Uses the committed split `splits/brats2020_patient_splits.json` (258 / 37 / 74 patients) and writes
`data/processed/brats{128,256}/{train,val,test}/{images.npy,masks.npy,meta.csv}` plus `stats.json`.
Check: train / val / test contain 15,895 / 2,159 / 4,623 tumour-bearing slices. Details in
[DATA.md](DATA.md).

## 4. The full experiment set in one command

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
nohup setsid bash scripts/queue_2026-09-09.sh >> runs/experiments.log 2>&1 &     # or inside tmux/screen
```

`scripts/queue_2026-09-09.sh` is the exact order behind the reported numbers: baseline evaluation →
train the `do01` and `reg` candidates → `scripts/select_model.py` (validation data only) →
segmentation study with the chosen model's samples → unconditional and 256 px models with the chosen
recipe → tables and figures. It is a thin wrapper around `scripts/run_experiments.sh`, whose stages,
models, sample counts and seeds can be restricted through environment variables (`STAGES`, `MODELS`,
`CKPT`, `N_SAMPLES`, `N_SAMPLES_SEG`, `N_CURVE`, `SEEDS`, `SEG_SOURCE`, `SEG_EPOCHS`, `PYTHON`); see
its header. The runner is sequential, stops at the first error and skips stages whose outputs
already exist, so it can simply be re-launched after a fix. Approximate cost:

| stage | what | time |
|---|---|---|
| train each 128 px model (`ldm128_maskcond`, `_do01`, `_reg`, `ldm128_uncond*`) | 200 epochs, 49.6k steps | ~1.3 h (23 s / epoch) |
| latent cache | once per (resolution, modalities, augmentation); `_reg` encodes 8 variants per training slice | ~1 min plain; with `augment: 6` ~7 min at 128 px, ~22 min at 256 px |
| train `ldm256_maskcond*` | 200 epochs, 99.2k steps | ~2.9 h (51 s / epoch) |
| sample + evaluate | per mask model: 20,000 samples (training masks, g = 2), 5,000 (g = 1), 5,000 (validation masks, g = 2); per unconditional model 5,000; FID/KID, diversity, memorisation | ~1–2 h total |
| checkpoint curve | 2,000 samples per saved checkpoint, FID vs val + memorisation | ~30 min per model |
| segmentation study | 21 U-Nets × 40 epochs (3 seeds × {10 %, 25 %, 100 %} × {real, real + synthetic} + synthetic-only) | ~3–4 h |
| secondary segmentation analyses (`seg2`) | 27 U-Nets (3 seeds × 3 fractions × {1:1 synthetic, synthetic pre-training, VAE-reconstructed real}) | ~4 h |
| VAE decoder fine-tune + re-decoded 128 px sets + segmentation study (`VAE_DECODER=runs/vae_dec_brats128/decoder.pt SEG_DIR=runs/seg_ftdec MODELS=ldm128_maskcond_reg SEG_CFG=configs/ldm128_maskcond_reg.yaml SEG_SOURCE=runs/ldm128_maskcond_reg/samples_best_ddim50_cfg2_seed0_ftdec STAGES="vaedec sample eval seg seg2 collect"`) | decoder 8 epochs ≈ 1 h, 30k samples ≈ 15 min, 3 evals ≈ 10 min, 48 U-Nets ≈ 4.8 h | ~6.2 h |
| pre-training control (`seg3`) | 9 U-Nets (3 seeds × 3 fractions, 20 + 40 epochs on real slices) | ~1.5 h |
| segmentation study at 256 px (`SEG_DIR=runs/seg256 SEG_CFG=configs/ldm256_maskcond_reg.yaml SEG_SOURCE=runs/ldm256_maskcond_reg/samples_best_ddim50_cfg2_seed0 STAGES="seg collect"`) | 21 U-Nets on 256 px slices | ~5 h |
| collect | tables + figures (`scripts/collect_results.py`, `scripts/make_figures.py`); `python scripts/make_report.py` then renders `docs/RESULTS.md` + the tables to `docs/SynthMRI_results.pdf` (needs Chrome/Chromium) | seconds |

The follow-up studies declared in [EXPERIMENTS.md](EXPERIMENTS.md) were run by the chained queue
scripts `scripts/queue_2026-09-10.sh` (secondary analyses), `queue_2026-09-10b.sh` (256 px study),
`queue_2026-09-10c.sh` (pre-training control) and `queue_2026-09-10d.sh` (decoder fine-tune study);
each waits for the previous queue's process to exit and then calls the runner with the environment
variables shown in the table.

## 5. The same thing step by step

```bash
python scripts/train.py    --config configs/ldm128_maskcond.yaml                       # -> runs/ldm128_maskcond
python scripts/sample.py   --run runs/ldm128_maskcond --num_images 5000 --guidance_scale 2.0   # --checkpoint best (default) | final | <epoch>
#                                                     -> runs/ldm128_maskcond/samples_best_ddim50_cfg2_seed0
python scripts/evaluate.py --run runs/ldm128_maskcond --samples runs/ldm128_maskcond/samples_best_ddim50_cfg2_seed0
#                                                     -> .../samples_best_ddim50_cfg2_seed0/eval/{results.json,results.md,*.png}
python scripts/sample.py   --run runs/ldm128_maskcond --num_images 5000 --guidance_scale 2.0 --mask_source val   # unseen masks -> ..._valmasks
python scripts/checkpoint_curve.py --run runs/ldm128_maskcond   # -> runs/ldm128_maskcond/checkpoint_curve.json
python scripts/select_model.py runs/ldm128_maskcond runs/ldm128_maskcond_do01 runs/ldm128_maskcond_reg   # -> results/model_selection.json
python scripts/train_seg.py --config configs/ldm128_maskcond.yaml --real_fraction 0.1 --seed 0 \
    --synthetic runs/ldm128_maskcond/samples_best_ddim50_cfg2_seed0 --out runs/seg/real010_synth_s0
python scripts/collect_results.py      # -> docs/results_tables.md, results/summary.json
python scripts/make_figures.py         # -> docs/figures/*.png
```

Every run directory records `config.yaml` (fully resolved) and `run_info.json` (git commit, package
versions, GPU), so any number in [RESULTS.md](RESULTS.md) can be traced to the code that produced it.

## 6. Determinism

Seeds fix data order, noise, mask sampling and the DDIM starting noise (`train.seed`, `sample.seed`,
`--seed`). Training uses bf16 autocast and cuDNN autotuning, so re-runs on different hardware or
driver versions reproduce the reported numbers only up to small differences (diffusion loss within
~1 %, Dice within the seed spread reported in the tables). FID/KID are computed with torch-fidelity's
InceptionV3 weights, which are downloaded on first use.

## 7. Smoke test (about two minutes)

```bash
python scripts/train.py    --config configs/smoke.yaml
python scripts/sample.py   --run runs/smoke --num_images 16 --steps 10
python scripts/evaluate.py --run runs/smoke --samples runs/smoke/samples_best_ddim10_cfg2_seed0 --max_real 64 --skip_fid
python scripts/train_seg.py --config configs/smoke.yaml --out runs/smoke/seg_real --real_fraction 0.05 --epochs 1 --max_steps 5 --batch_size 8
```
