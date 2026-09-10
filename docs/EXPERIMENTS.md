# Experimental protocol

All experiments use the committed patient split (`splits/brats2020_patient_splits.json`; 258 train /
37 val / 74 test patients) and the preprocessing in [DATA.md](DATA.md). Every number reported in
[RESULTS.md](RESULTS.md) comes from a run directory under `runs/` whose `config.yaml`,
`run_info.json` (git commit) and `metrics.csv` are kept. `scripts/run_experiments.sh` reproduces the
whole set; `scripts/collect_results.py` regenerates the tables.

## Models

| name | config | resolution | latent | conditioning | batch | epochs | steps |
|---|---|---|---|---|---|---|---|
| LDM-128 | `configs/ldm128_uncond.yaml` | 128 | 16×16×4 | none | 64 | 200 | 49.6k |
| LDM-128-mask | `configs/ldm128_maskcond.yaml` | 128 | 16×16×4 | tumour mask, p_drop = 0.1 | 64 | 200 | 49.6k |
| LDM-256-mask | `configs/ldm256_maskcond.yaml` | 256 | 32×32×4 | tumour mask, p_drop = 0.1 | 32 | 200 | 99.2k |
| LDM-128-mask-do0.1 (ablation) | `configs/ldm128_maskcond_do01.yaml` | 128 | 16×16×4 | as LDM-128-mask, U-Net dropout 0.1 | 64 | 200 | 49.6k |
| LDM-128-mask-reg | `configs/ldm128_maskcond_reg.yaml` | 128 | 16×16×4 | as LDM-128-mask, dropout 0.1 + affine augmentation | 64 | 200 | 49.6k |
| LDM-128-reg | `configs/ldm128_uncond_reg.yaml` | 128 | 16×16×4 | none, dropout 0.1 + affine augmentation | 64 | 200 | 49.6k |
| LDM-256-mask-reg | `configs/ldm256_maskcond_reg.yaml` | 256 | 32×32×4 | tumour mask, dropout 0.1 + affine augmentation | 32 | 200 | 99.2k |

Shared: frozen `stabilityai/sd-vae-ft-mse`; U-Net channels (128, 256, 512, 512), 2 res-blocks per
level, self-attention at levels 1–2; ε-prediction, linear β schedule, T = 1000; AdamW lr 1e-4,
wd 0.01, 500 warm-up steps then cosine; grad-clip 1.0; EMA 0.9999 with warm-up; bf16; horizontal flips.
Every 10th epoch is checkpointed and kept (`train.keep_checkpoints: 0`). The unconditional and 256 px
models are trained with whichever regularisation the selection below picks for the 128 px
mask-conditioned model (`_reg`, `_do01` or none); the matching configs exist for each case.

## Checkpoint selection (early stopping on held-out patients)

Every reported sample set comes from the EMA weights of the epoch with the **lowest validation
diffusion loss** (`<run>/best/`, selected on the 37 validation patients, never on test), not from
the last epoch. The validation loss is computed with fixed noise and timesteps spread evenly over
[0, T) so it is comparable across epochs. Reason: on LDM-128-mask the validation loss reaches its
minimum early (epoch 34) and then rises while the training loss keeps falling; later checkpoints
obtain a *lower* FID, but `scripts/checkpoint_curve.py` shows they do so by reproducing training
slices. The curve scores every saved checkpoint with the same 2,000 conditioning masks and seeds:
FID/KID against the real validation slices and the nearest-training-slice distance of the samples,
summarised as the fraction of samples that lie closer to a training slice than 95 % of real
held-out slices do (≈ 0.05 for a model that generalises). `docs/figures/<run>_checkpoint_curve.png`
plots loss, FID and that fraction against the epoch; the numbers are in `<run>/checkpoint_curve.json`.
The dropout-0.1 ablation tests whether regularisation postpones the memorisation and improves the
early-stopped model.

## Regularised recipe and model selection (validation data only)

Three candidates for the main 128 px mask-conditioned model differ only in regularisation:

* **baseline** -- horizontal flips (both orientations cached);
* **do01** -- plus U-Net dropout 0.1;
* **reg** -- dropout 0.1 plus cached affine augmentation (`train.augment: 6`): every training slice
  is encoded by the frozen VAE in 6 extra randomly transformed versions (horizontal flip with
  probability 0.5, shift up to 6 % of the image, rotation up to ±10°, isotropic scale 0.9–1.1;
  zero padding = background), on top of the original and its flip. Each access draws one of the 8
  variants uniformly and applies the identical transform to the conditioning mask (nearest
  neighbour). Transform parameters are drawn with a fixed seed and stored in the latent cache
  (`latents_..._aug6.npz`, `synthmri/data/augment.py`). Validation latents are never augmented, so
  validation losses stay comparable across candidates.

**Selection rule** (`scripts/select_model.py`, fixed before the candidate runs finished): for each
candidate take the `best` row of its checkpoint curve (lowest validation loss, 2,000 samples,
training masks); exclude candidates whose memorisation fraction exceeds 0.15; among the rest choose
the lowest FID against real validation slices. The test set plays no part. The choice is recorded in
`results/model_selection.json`; the chosen run provides the pool for the segmentation study and its
recipe is reused for the unconditional and 256 px models. All three candidates are still reported
in [RESULTS.md](RESULTS.md) as an ablation.

**Unseen masks.** Each mask-conditioned model also generates 5,000 samples conditioned on masks of
the *validation* patients (`scripts/sample.py --mask_source val`, directory suffix `_valmasks`),
tumour shapes the model never saw during training. FID/KID of that set against the test slices,
next to the training-mask set, shows whether the model generalises to new tumour geometry rather
than only redrawing training tumours.

## Generation quality (`scripts/sample.py` + `scripts/evaluate.py`)

5,000 samples per model and setting from the `best/` weights with DDIM, 50 steps, η = 0
(mask-conditioned: guidance scale 2.0 and 1.0, masks drawn uniformly from the training split with
random horizontal flips, plus the guidance-2.0 validation-mask set described above); sample
directories are named `samples_best_ddim50_cfg<g>_seed0[_valmasks]`. The guidance-2.0 /
training-mask set of every mask-conditioned model is generated with 20,000 samples; its first 5,000
are scored here and the whole set of the selected model is the pool for the segmentation study
below. Against all 4,623 real test slices:

* **FID / KID** per modality (grey → RGB) and for the composite RGB image, InceptionV3 features
  (torch-fidelity). Reference floor: FID/KID between real *val* and real *test* slices.
* **VAE ceiling**: PSNR / SSIM / LPIPS of `decode(encode(x))` on real test slices.
* **Diversity**: mean SSIM over 2,000 random sample pairs (real test slices as reference).
* **Memorisation**: L2 distance (64×64 grey features) from each sample to its nearest training
  slice, compared with the same statistic for real test slices; plus a figure of the closest pairs.

## Downstream segmentation (`scripts/train_seg.py`)

2-D MONAI U-Net (channels 32–512, 2 residual units, instance norm), Dice + CE loss, AdamW 3e-4 with
one-cycle schedule, 40 epochs, batch 32, horizontal flips; model selection on real validation
patients; evaluation on real test patients with per-patient Dice for WT / TC / ET.

| condition | real training patients | real slices | synthetic slices | seeds |
|---|---|---|---|---|
| real 10 % | 26 | ≈ 1,600 | 0 / ≈ 2,000 | 0, 1, 2 |
| real 25 % | 65 | ≈ 4,000 | 0 / ≈ 5,000 | 0, 1, 2 |
| real 100 % | 258 | 15,895 | 0 / 20,000 | 0, 1, 2 |
| synthetic only | 0 | 0 | 20,000 | 0, 1, 2 |

The synthetic pool is the 20,000-sample training-mask set of the selected 128 px mask-conditioned model
(`results/model_selection.json`; `best/` weights, guidance 2.0), each sample paired with the training
mask it was conditioned on. The patient subset for a fraction is drawn with the run's seed
(`subsample_patients`). **Patient matching:** a real-x % segmenter only receives synthetic slices
whose conditioning mask belongs to one of its own x % patients, so the low-data conditions never see
tumour shapes from patients they do not have (otherwise the masks alone would leak information from
the held-out 90 %). The synthetic count therefore scales with the fraction and varies slightly
across seeds; `collect_results.py` reports the range. The synthetic-only condition uses the full pool.

The real-only segmenters are additionally scored on all 20,000 synthetic pairs ("mask
consistency": per-slice Dice between the segmenter's prediction on the generated image and the mask
the image was conditioned on).

## What would strengthen the paper further

* A radiology-specific feature extractor (e.g. RadImageNet) for FID, alongside Inception.
* 3-D or 2.5-D generation; the present models are per-slice and ignore inter-slice consistency.
* Fine-tuning the VAE *decoder* (encoder frozen) with an L1 + LPIPS loss to raise the ceiling.
* External validation of the segmenter on BraTS 2021 patients not in BraTS 2020.
