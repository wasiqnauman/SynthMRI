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
FID/KID against the real validation slices and the nearest-training-slice distance of the samples
(training slices in both orientations), summarised as the fraction of samples that lie closer to a
training slice than 95 % of real held-out slices do (≈ 0.05 for a model that generalises). `docs/figures/<run>_checkpoint_curve.png`
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
* **Memorisation**: L2 distance (64×64 features) from each sample to its nearest training slice,
  where the training set is taken in both orientations (a copy of a mirrored training slice is a
  copy; the models train with flips), compared with the same statistic for real test slices;
  "memorised" = fraction of samples closer to a training slice than 95 % of real held-out slices
  are; plus a figure of the closest pairs.

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

### Secondary analyses (declared 2026-09-10 01:35, after the seed-0 results of the 10 % and 25 % conditions and before the remaining seeds finished)

The first two real + synthetic runs lost Dice on TC and ET relative to real only, and the mask-
consistency scores show that a real-trained segmenter finds ET in the synthetic images less often
than in real ones. Three follow-up conditions, run with the same 3 seeds after the primary study,
are meant to say *why*; the primary protocol above is unchanged and stays the headline result:

| condition | flag | question |
|---|---|---|
| real + synthetic, 1:1 | `--synth_ratio 1.0` | does the harm come from synthetic slices outnumbering real ones (≈ 1.25:1 in the primary protocol)? |
| synthetic pre-training, then real | `--pretrain_synthetic DIR --pretrain_epochs 20` | does the usual pre-train / fine-tune use of synthetic data help where mixing does not? |
| VAE-reconstructed real only | `--real_through_vae` | how much of the gap is the frozen VAE's own loss of detail (PSNR 26 dB, SSIM 0.83), independent of the generator? Real slices are replaced by `decode(encode(x))`. |

All three use the selected model's 20,000-sample pool with the same patient matching. Output
directories `runs/seg/real<pct>_{synth1x,pre,vae}_s<seed>`; `collect_results.py` lists them after
the primary rows and `make_figures.py` draws `segmentation_dice_secondary.png`.

#### Control for the pre-training condition (declared 2026-09-10 11:10, after the seed-0 results of the 10 % secondary conditions and before the 25 % / 100 % ones)

At 10 % real data (seed 0) synthetic pre-training raised mean Dice from 0.652 to 0.684 while mixing
lowered it. The pre-trained segmenter has seen 20 extra epochs, so the gain could come from the
longer schedule rather than from the synthetic slices. Control: `--pretrain_real` pre-trains for the
same 20 epochs on the real slices themselves, then fine-tunes exactly as the synthetic-pre-training
runs do (same seeds and fractions; `runs/seg/real<pct>_prereal_s<seed>`, stage `seg3`). The
synthetic pre-training condition counts as a real effect only if it also beats this control.

### Segmentation study at 256 px (declared 2026-09-10 09:45, after the 256 px model's generation scores and before any 256 px segmenter was trained)

LDM-256-mask-reg reaches FID 10.45 against the real test slices, next to a real val-vs-test floor of
9.55 (128 px: 20.85 vs 10.49), and the VAE ceiling rises from PSNR 26.4 / SSIM 0.83 to 29.0 / 0.88.
Because the 128 px study attributes the TC / ET loss to missing fine detail, the primary protocol is
repeated unchanged at 256 px: same seeds, fractions, patient matching, 40 epochs, batch 32, with the
segmenter reading `data/processed/brats256` and the pool
`runs/ldm256_maskcond_reg/samples_best_ddim50_cfg2_seed0` (20,000 samples). Output `runs/seg256/`;
`collect_results.py` writes it as a second segmentation block and `make_figures.py` as
`segmentation_dice_seg256.png`. Both resolutions are reported.

### VAE decoder fine-tuning (declared 2026-09-10 11:50, after the seed-0 secondary results at 10 % and 25 % real and before any decoder was trained)

The VAE-reconstructed-real control reproduces the loss on its own (seed 0: 10 % real 0.652 → 0.604,
25 % real 0.708 → 0.635, the latter seen after this declaration), i.e. the frozen natural-image decoder discards enhancing-tumour detail
before the diffusion model is even involved. `scripts/finetune_vae_decoder.py` trains only
`decoder` + `post_quant_conv` of `sd-vae-ft-mse` on the 128 px training slices (L1 + 0.5·LPIPS-VGG
per channel, AdamW 2e-5, batch 16, 8 epochs, horizontal flips; the epoch with the lowest validation
loss is kept). The encoder is untouched, so the cached latents and LDM-128-mask-reg stay valid: the
same latents (same seed) are decoded again with the new decoder into
`samples_best_ddim50_cfg2_seed0_ftdec` (20,000), the cfg-1 and validation-mask sets, and scored as
before (`--vae_decoder`). The primary and secondary segmentation protocols are then repeated
unchanged with that pool into `runs/seg_ftdec/`, with the VAE-reconstructed-real control also using
the fine-tuned decoder. Pre-declared readings: (a) the test ceiling (PSNR / SSIM / LPIPS-Alex on the
same test slices, before vs after) must improve, otherwise the study stops there; (b) the study
supports the hypothesis "the loss is the decoder's" only if the VAE-reconstructed-real control moves
towards real-only *and* real + synthetic moves in the same direction; a better ceiling with an
unchanged segmentation result would attribute the loss to the diffusion model instead. The real
pre-training control of `seg3` is decoder-independent and is not repeated.

## What would strengthen the paper further

* A radiology-specific feature extractor (e.g. RadImageNet) for FID, alongside Inception.
* 3-D or 2.5-D generation; the present models are per-slice and ignore inter-slice consistency.
* A medical-image autoencoder, or fine-tuning the encoder as well (which invalidates the cached
  latents and needs the diffusion models retrained): the decoder fine-tune above recovered only
  about a third of the autoencoder's Dice loss (RESULTS.md, section 7).
* External validation of the segmenter on BraTS 2021 patients not in BraTS 2020.
