# Work log

## 2026-09-09 -- Result tables: checkpoint-curve summary, model-selection table, memorised-fraction column

`scripts/collect_results.py` now writes, next to the generative-quality table (which gains the
fraction of samples closer to a training slice than 95 % of real test slices), a table with the
best-validation-loss vs last checkpoint of every run that has a checkpoint curve (epoch, validation
loss, FID/KID vs validation slices, memorised fraction) and the candidate table behind
`results/model_selection.json`. The `smoke` run is excluded from all tables. Also removed an empty
`runs/ldm128_maskcond/samples_ddim50_cfg2_seed0/eval/` folder left behind by the old runner.

Files: scripts/collect_results.py
Follow-ups: none

## 2026-09-09 -- Regularised recipe (cached affine augmentation + dropout), validation-only model selection, unseen-mask sample sets; `.gitignore` was hiding `synthmri/data/`

The baseline LDM-128-mask memorises training slices after epoch ~34 (previous entry), so two
regularised candidates are trained next to it: `_do01` (U-Net dropout 0.1) and `_reg` (dropout 0.1
plus `train.augment: 6`). With `augment: K`, `cache_latents` encodes every training slice K extra
times under random affine transforms (horizontal flip, shift up to 6 % of the image, rotation up to
±10°, isotropic scale 0.9–1.1; `synthmri/data/augment.py`), stores the parameters in the cache
(`..._aug6.npz`) and `LatentDataset` draws one of the 8 variants per access, transforming the
conditioning mask with the same parameters (nearest neighbour). Validation latents are never
augmented. The winner among baseline / do01 / reg is chosen by `scripts/select_model.py` with a
rule fixed before the runs finished: lowest FID against real *validation* slices at the
lowest-validation-loss checkpoint, excluding runs whose memorisation fraction exceeds 0.15; the
test set is not used. The chosen run supplies the pool for the segmentation study, and its recipe is
reused for the unconditional and 256 px models (`configs/*_do01.yaml`, `configs/*_reg.yaml`).
Mask-conditioned models also get a 5,000-sample set conditioned on masks of *validation* patients
(`sample.py --mask_source val`, directory suffix `_valmasks`) to measure generalisation to tumour
shapes never seen in training. `scripts/run_experiments.sh` runs all of this; the exact overnight
order is `scripts/queue_2026-09-09.sh` (log `runs/experiments.log`).

Also fixed: `.gitignore` listed `data/` unanchored, which matched `synthmri/data/` -- the whole
data package (BraTS loading, preprocessing, splits, datasets) was missing from every commit on this
branch. Patterns are now anchored to the repository root (`/data/`, `/runs/`, `/outputs/`) and the
package is committed.

Files: .gitignore, synthmri/data/augment.py, synthmri/data/dataset.py, synthmri/diffusion/latents.py,
synthmri/diffusion/train.py, synthmri/config.py, scripts/sample.py, scripts/select_model.py,
scripts/run_experiments.sh, scripts/queue_2026-09-09.sh, configs/ldm128_maskcond_reg.yaml,
configs/ldm128_uncond_reg.yaml, configs/ldm256_maskcond_reg.yaml, configs/ldm128_uncond_do01.yaml,
configs/ldm256_maskcond_do01.yaml, tests/test_data.py, tests/test_diffusion.py, docs/EXPERIMENTS.md,
docs/REPRODUCE.md, docs/DATA.md, docs/ARCHITECTURE.md, README.md
Follow-ups: fill docs/RESULTS.md from the finished queue (checkpoint curves, model selection table,
ablation, segmentation Dice); measure the 256 px augmented-cache time

## 2026-09-09 -- Early stopping on validation loss; checkpoint FID/memorisation curve; dropout ablation

While LDM-128-mask was training, the validation diffusion loss bottomed out at epoch 34 and then
rose steadily (0.075 → 0.131 by epoch 173) while the training loss kept falling. Scoring
intermediate checkpoints (2,000 samples each, FID against real validation slices, nearest-training-
slice distance) showed FID improving 29.9 → 25.0 → 21.8 → 20.9 (epochs 34 / 60 / 100 / 150) but the
fraction of samples closer to a training slice than 95 % of real held-out slices growing
0.06 → 0.26 → 0.71 → 0.90, i.e. later checkpoints improve FID by reproducing training slices.
Changes: (1) `scripts/sample.py` gets `--checkpoint` (default `best` = lowest validation loss, EMA)
and names sample directories `samples_<ckpt>_...`; `resolve_checkpoint()` added. (2) New
`scripts/checkpoint_curve.py` scores every saved checkpoint (FID/KID vs val, memorisation vs train)
into `<run>/checkpoint_curve.json`, plus a figure in `make_figures.py`. (3) `train.keep_checkpoints`
(0 = keep all; base.yaml keeps every 10th epoch). (4) `scripts/run_experiments.sh` rewritten to be
idempotent, sample from `best/`, and include the `curve` stage and the dropout-0.1 ablation
`configs/ldm128_maskcond_do01.yaml`. Interim epoch-60/100 rows were saved into
`runs/ldm128_maskcond/checkpoint_curve/` (the checkpoints were pruned before the script existed;
noted in the rows). The queue was restarted with the new runner after the first model finished.

Files: scripts/sample.py, scripts/checkpoint_curve.py, scripts/run_experiments.sh, scripts/make_figures.py,
synthmri/diffusion/checkpoint.py, synthmri/diffusion/train.py, synthmri/config.py, configs/base.yaml,
configs/ldm128_maskcond_do01.yaml, tests/test_diffusion.py, docs/EXPERIMENTS.md, docs/REPRODUCE.md, README.md
Follow-ups: fill docs/RESULTS.md; decide from the ablation whether the uncond/256 models should also use dropout

## 2026-09-09 -- Stop cuDNN autotuning inside the VAE (18–36 GB transient spikes)

With `cudnn.benchmark = True` (set by `seed_everything` for non-deterministic runs) the first VAE
decode at a new shape tried FFT-style convolution algorithms whose workspaces peaked at 18 GB for
16 images at 128 px and 36 GB at 256 px, while the heuristic algorithm runs at the same speed
(0.15 vs 0.19 s per 16-image decode). `VAEWrapper.encode/encode_dist/decode` now run under a
`cudnn_heuristics()` guard that turns autotuning off for the VAE only and restores the flag
afterwards. Training and sampling memory now match the numbers in `docs/REPRODUCE.md`. The
`ldm128_maskcond` run had already started on the previous commit; the change affects only which
convolution kernel cuDNN picks for the VAE, not the model or the data.

Files: synthmri/models/vae.py, tests/test_models.py, docs/REPRODUCE.md
Follow-ups: none

## 2026-09-09 -- Launch the full experiment set; fix seed aggregation of segmentation runs

Started `scripts/run_experiments.sh` (stages train → sample → eval → seg → collect; log in
`runs/experiments.log`). `collect_results.py` grouped segmentation runs by their exact synthetic
slice count, which differs per seed under patient matching, so the three seeds of one condition would
have been reported as three single-seed rows; runs are now grouped by (fraction, synthetic,
synthetic-only) and the count range is reported. The "synthetic only" reference line in the Dice
figure moved from an in-plot label (it overlapped the bars) to the legend. Added `docs/REPRODUCE.md`
and made the segmentation protocol in `docs/EXPERIMENTS.md` describe the 20,000-sample pool and
patient matching.

Files: scripts/collect_results.py, scripts/make_figures.py, docs/REPRODUCE.md, docs/EXPERIMENTS.md
Follow-ups: fill docs/RESULTS.md from the finished runs; confirm the ldm256 timing in REPRODUCE.md

## 2026-09-09 -- Rebuild the notebook as a tested, reproducible research package

Replaced the single Colab notebook with the `synthmri` package, CLI scripts, YAML configs, a pytest
suite and documentation, so the project can be trained, evaluated and reproduced end-to-end for a
paper. Main behavioural changes vs the notebook: patient-level train/val/test splits (previously
none), per-volume percentile normalisation (previously per-slice min/max), centre crop before
resizing, all 369 patients used (patient 355's oddly named segmentation was silently skipped before),
cached VAE latents with flip augmentation, EMA + warm-up/cosine schedule + grad clipping + bf16,
optional mask conditioning with classifier-free guidance, DDIM sampling, and a full evaluation
suite (VAE ceiling, FID/KID, diversity, memorisation, downstream segmentation). Removed the joint
VAE fine-tuning (MSE-only, encoder drift). Legacy notebook moved to `notebooks/`.

Files: synthmri/**, scripts/**, configs/**, tests/**, docs/**, splits/brats2020_patient_splits.json,
README.md, pyproject.toml, requirements.txt, environment.yml, .gitignore
Follow-ups: run the full experiment set (`scripts/run_experiments.sh`) and fill docs/RESULTS.md
