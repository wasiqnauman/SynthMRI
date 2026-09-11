# Work log

## 2026-09-10 -- VAE decoder fine-tuned; ceiling gate passed; samples re-decoded and scored

`scripts/finetune_vae_decoder.py` ran as the first stage of the decoder study (queue7, 19:20–20:18):
the test-slice reconstruction ceiling of `sd-vae-ft-mse` at 128 px rises from PSNR 26.37 / SSIM
0.833 / LPIPS 0.041 to 27.79 / 0.883 / 0.032 after training only the decoder for 8 epochs, so the
pre-declared "ceiling must improve" gate is passed and the queue went on to re-decode the sample
sets (then 48 segmentation U-Nets into `runs/seg_ftdec`). `scripts/make_figures.py` gains
`vae_decoder_<data>.png` (real test slices through the frozen vs the fine-tuned decoder, CPU) and
and `<run>_<samples>_ftdec_frozen_vs_finetuned.png` (the same latents through both decoders; for
`_ftdec` sample dirs this replaces the real-vs-synthetic figure). Re-decoding the three 128 px
sample sets (20:18–20:41) leaves the composite FID unchanged (20.85 → 20.89) but cuts the
per-modality FIDs by 35–75 % (T2 69.6 → 17.6, T1ce 44.0 → 25.6, FLAIR 44.3 → 28.6); memorised
fraction 0.043 → 0.051 with identical latents. RESULTS.md sections 7.1–7.2 written, 7.3
(segmentation) stays *pending*; PDF rebuilt. `decoder.pt` (189 MB) lives under the git-ignored `runs/`.

Files: scripts/make_figures.py, docs/figures/vae_decoder_brats128.png, docs/figures/*_ftdec_frozen_vs_finetuned.png, docs/RESULTS.md, docs/results_tables.md, results/summary.json, README.md, docs/REPRODUCE.md, docs/SynthMRI_results.pdf
Follow-ups: section 7.3 when `runs/seg_ftdec` finishes (~2026-09-11 05:00)

## 2026-09-10 -- Pre-training control complete (3 seeds)

`seg3` finished 19:18 (9 U-Nets). A segmenter given the same 20 extra epochs on real data reaches
0.653 / 0.712 / 0.782 mean Dice (10 / 25 / 100 % real) vs 0.674 / 0.726 / 0.779 for synthetic
pre-training, so the synthetic-specific gain is +0.021 / +0.014 / −0.003 (raw gains over real only
were +0.044 / +0.015 / +0.001). RESULTS.md section 8, finding 6 and the README updated; PDF rebuilt.

Files: docs/RESULTS.md, README.md, docs/results_tables.md, results/summary.json, docs/figures/segmentation_dice_secondary.png, docs/SynthMRI_results.pdf
Follow-ups: section 7 of RESULTS.md when the decoder study finishes (~2026-09-11 06:00)

## 2026-09-10 -- Results PDF, README results-at-a-glance, figures embedded in RESULTS.md

`scripts/make_report.py` renders `docs/RESULTS.md` (+ `docs/results_tables.md` as an appendix) to
`docs/SynthMRI_results.pdf` with all referenced figures embedded (markdown → HTML → headless Chrome;
10 pages, 1.7 MB). RESULTS.md now embeds the key figures inline so GitHub renders them; README gains
a "Results at a glance" section, a status line and a documentation index. The user asked that the
branch stay current for the collaborator after every milestone (code, small results, figures, a
PDF), never large files.

Files: scripts/make_report.py, docs/SynthMRI_results.pdf, docs/RESULTS.md, README.md, docs/REPRODUCE.md
Follow-ups: rebuild the PDF after each results update (part of the milestone routine)

## 2026-09-10 -- 256 px segmentation study complete (3 seeds)

`runs/seg256/` finished 18:26 (21 U-Nets, ~4.8 h instead of the estimated 12). Mean Dice vs real
only: +0.027 at 10 % real (all regions up, 3/3 seeds), +0.003 at 25 %, −0.003 at 100 % (ET −0.015);
synthetic only 0.760 (= 25 %-real level). At 128 px the same protocol lost 0.030 / 0.036 / 0.021.
RESULTS.md section 6 written. `make_figures.py` no longer draws a secondary-analysis figure for a
study that has no secondary conditions (the seg256 one held a single bar series).

Files: docs/RESULTS.md, docs/results_tables.md, results/summary.json, docs/figures/segmentation_dice_seg256.png, scripts/make_figures.py, docs/REPRODUCE.md
Follow-ups: sections 7–8 of RESULTS.md as the decoder study and the pre-training control finish

## 2026-09-10 -- Secondary segmentation analyses complete (3 seeds)

`seg2` finished 13:38 (27 U-Nets). Relative to real only, mean Dice: 1:1 mixing −0.020 / −0.024 /
−0.018 (10 / 25 / 100 % real), VAE-reconstructed real only −0.057 / −0.072 / −0.072, synthetic
pre-training then fine-tuning +0.044 / +0.015 / +0.001. The frozen VAE alone reproduces the loss of
the primary protocol; pre-training helps in the low-data regime pending the `seg3` control.
RESULTS.md section 5 written from `docs/results_tables.md`; the segmentation table header now
names every synthetic pool used (was `?` when the first group had no synthetic slices).

Files: docs/RESULTS.md, docs/results_tables.md, results/summary.json, docs/figures/segmentation_dice_secondary.png, scripts/collect_results.py
Follow-ups: sections 6–8 of RESULTS.md as seg256, seg3 and the decoder study finish

## 2026-09-10 -- VAE decoder fine-tuning (queued) and pre-training control

The VAE-reconstructed-real control alone reproduces the segmentation loss at 10 % real (seed 0:
0.652 → 0.604), so the frozen natural-image decoder is the suspected bottleneck. New
`synthmri/models/vae_finetune.py` + `scripts/finetune_vae_decoder.py` train only `decoder` +
`post_quant_conv` (L1 + LPIPS-VGG, validation-selected epoch) and report the test ceiling before /
after; `load_vae(decoder_weights=...)`, `model.vae.decoder_weights` in the config and `--vae_decoder`
on `sample.py` (output suffix `_ftdec`), `evaluate.py` and `train_seg.py` decode with it; the
runner gets `VAE_DECODER` and a `vaedec` stage; `collect_results.py` keys the ceiling table by
decoder and adds a fine-tuning table. Declared in EXPERIMENTS.md with its reading rules before any
decoder was trained; `scripts/queue_2026-09-10d.sh` runs it after the pre-training control
(decoder → re-decoded sample sets → eval → `runs/seg_ftdec/` primary + secondary, ~10 h).
Test: `tests/test_models.py::test_decoder_finetune_and_reload` (46 CPU tests).

Files: synthmri/models/vae.py, synthmri/models/vae_finetune.py, synthmri/models/__init__.py, synthmri/config.py, configs/base.yaml,
scripts/finetune_vae_decoder.py, scripts/sample.py, scripts/evaluate.py, scripts/train_seg.py, scripts/checkpoint_curve.py,
synthmri/diffusion/train.py, scripts/run_experiments.sh, scripts/collect_results.py, scripts/queue_2026-09-10d.sh, tests/test_models.py,
docs/EXPERIMENTS.md, docs/REPRODUCE.md, README.md
Follow-ups: RESULTS.md sections for seg2, seg256, seg3 and the decoder study as each finishes

## 2026-09-10 -- 256 px model at the FID floor; segmentation study repeated at 256 px (queued)

`ldm256_maskcond_reg` (dropout 0.1 + augment 6, best epoch ≈ 100): FID 10.45 / KID 6.4×10⁻³ vs the
real test slices with a real val-vs-test floor of 9.55; 10.83 with unseen validation-patient masks;
memorised 0.030; VAE ceiling PSNR 29.0 dB / SSIM 0.88 (128 px: 26.4 / 0.83). Since the 128 px
segmentation loss sits in the fine regions, the primary segmentation protocol is repeated at 256 px
(declared in EXPERIMENTS.md before any 256 px segmenter ran): `scripts/queue_2026-09-10b.sh` waits
for the seg2 queue and then runs `SEG_DIR=runs/seg256 ... STAGES="seg collect"` (~12 h).
Tooling: `checkpoint_curve.py` now defaults its batch size to the run's `sample.batch_size` (the
256 px curve ran at 128 and hit a recoverable allocator OOM warning at 42 GB); `train_seg.py`
memory-maps the synthetic pool (7.4 GB at 256 px) and reads only the selected rows;
`run_experiments.sh` gets `SEG_DIR`; `collect_results.py` / `make_figures.py` produce one
segmentation block / figure per `runs/seg*` directory.

The main queue finished 10:59 (256 px checkpoint curve: FID(val) 15.10 / memorised 0.033 at the best
epoch 71; 12.94 / 0.291 at epoch 200 — the regularised model also starts copying late). First
draft of `docs/RESULTS.md` written from the finished runs, with the secondary analyses and the
256 px study marked pending.

Seed-0 secondary results at 10 % real (seen 11:08): mixing 1:1 = 0.616, VAE-reconstructed real only = 0.604
(vs 0.652 real only) — the frozen VAE's blur alone reproduces the loss; synthetic pre-training then
fine-tuning = 0.684. Because pre-training adds 20 epochs, a compute-matched control (`--pretrain_real`,
stage `seg3`, 9 U-Nets) was declared in EXPERIMENTS.md and queued after the 256 px study
(`scripts/queue_2026-09-10c.sh`); `collect_results.py` / `make_figures.py` label it
"real pre-training (control)".

Files: scripts/checkpoint_curve.py, scripts/train_seg.py, scripts/run_experiments.sh, scripts/collect_results.py,
scripts/make_figures.py, scripts/queue_2026-09-10b.sh, scripts/queue_2026-09-10c.sh, docs/EXPERIMENTS.md, docs/REPRODUCE.md, docs/RESULTS.md
Follow-ups: fill RESULTS.md sections 5 and 6 once seg2, seg256 and seg3 finish

## 2026-09-10 -- Primary segmentation study complete (3 seeds): synthetic slices help WT at 10 % real, hurt TC / ET everywhere

21 segmenters finished (`runs/seg/`, selected model `ldm128_maskcond_reg`, patient-matched synthetic
pool). Mean per-patient test Dice (WT / TC / ET, 3 seeds): 10 % real 0.801 / 0.580 / 0.509 → with
synthetic 0.809 / 0.559 / 0.430; 25 % real 0.847 / 0.676 / 0.611 → 0.844 / 0.660 / 0.521; 100 % real
0.886 / 0.775 / 0.673 → 0.879 / 0.742 / 0.650; synthetic only 0.829 / 0.669 / 0.481 (≈ the 10 %-real
level). Mask consistency of the synthetic pool under a 100 %-real segmenter: WT 0.789, TC 0.584,
ET 0.494 per slice. Reported as is; the pre-declared secondary analyses (previous entry) run next.
Tables in `docs/results_tables.md`, `results/summary.json`; figure `docs/figures/segmentation_dice.png`.

Files: docs/results_tables.md, results/summary.json
Follow-ups: docs/RESULTS.md after the uncond / 256 px models and the seg2 queue

## 2026-09-10 -- Secondary segmentation analyses declared and queued (1:1 synthetic, synthetic pre-training, VAE-reconstructed real)

The first real + synthetic segmenters (seed 0, 10 % and 25 % real) score *below* real only on TC and
ET (mean Dice 0.652 → 0.619 and 0.708 → 0.662), and the mask-consistency check shows a real-trained
segmenter recovers ET from synthetic images less often than from real ones (per-slice ET Dice
0.47–0.50 vs 0.55–0.64). Before the remaining seeds finished, three diagnostic conditions were
declared in EXPERIMENTS.md and added to `scripts/train_seg.py`: `--synth_ratio` (cap synthetic at
a multiple of the real slice count), `--pretrain_synthetic DIR --pretrain_epochs N` (train on
synthetic alone, then fine-tune on real; `train_segmenter` gained `init_state`) and
`--real_through_vae` (real training images replaced by their frozen-VAE reconstructions,
`synthmri.eval.recon.vae_roundtrip`). `run_experiments.sh` gets a `seg2` stage for them and now
defaults `SEG_SOURCE` to the run named in `results/model_selection.json`; `collect_results.py`
groups the new runs as separate rows; `make_figures.py` draws `segmentation_dice_secondary.png`.
The primary protocol is unchanged and remains the headline result. `scripts/queue_2026-09-10.sh`
waits for the running overnight queue and then runs `seg2` + `collect` (~4 h).

Files: scripts/train_seg.py, synthmri/eval/segmentation.py, synthmri/eval/recon.py, scripts/run_experiments.sh,
scripts/queue_2026-09-10.sh, scripts/collect_results.py, scripts/make_figures.py, tests/test_eval.py,
docs/EXPERIMENTS.md, docs/REPRODUCE.md
Follow-ups: docs/RESULTS.md after both queues finish

## 2026-09-10 -- Candidate results and model selection: the regularised recipe wins

All three 128 px mask-conditioned candidates finished (`runs/ldm128_maskcond{,_do01,_reg}`).
`scripts/select_model.py` (validation data only, rule in EXPERIMENTS.md) chose
`ldm128_maskcond_reg`: FID vs validation slices at the best-val-loss checkpoint 24.65 (epoch 101)
against 28.82 (do01, epoch 41) and 29.89 (baseline, epoch 34), with the lowest memorised fraction
(0.049 vs 0.093 / 0.086). Against the real test slices its 5,000-sample set scores FID 20.85 (baseline
26.22), 22.85 with unseen validation-patient masks (27.94), memorised 0.043 (0.069). Its checkpoint
curve stays flat (0.115 memorised at epoch 200 vs 0.966 for the baseline), so augmentation, not
dropout, is what postpones the memorisation; dropout alone behaves like the baseline. The
segmentation study therefore uses `runs/ldm128_maskcond_reg/samples_best_ddim50_cfg2_seed0`, and
the unconditional and 256 px models train with the `_reg` configs. Tables regenerated in
`docs/results_tables.md`, selection in `results/model_selection.json`. Small tooling fixes on the
way: the VAE-ceiling table is now one block per dataset instead of one per run, and the loss
panels push the train/val end labels apart when the curves end close together.

Files: scripts/collect_results.py, scripts/make_figures.py, docs/results_tables.md, results/summary.json,
results/model_selection.json
Follow-ups: docs/RESULTS.md once the segmentation study and the uncond / 256 px models finish

## 2026-09-09 -- Memorisation check now looks at training slices in both orientations

The first in-repo checkpoint curve of LDM-128-mask disagreed with the interim rows: with
conditioning masks flipped at random (as in every sample set), the fraction of memorised samples at
epochs 150–200 came out at ≈ 0.47, whereas the interim rows with unflipped masks gave 0.71 at
epoch 100. A sample that reproduces a *mirrored* training slice was not being caught because the
nearest-neighbour search only saw unflipped training slices. `nearest_neighbour_distances` /
`memorisation_report` now append the mirrored training set to the reference by default
(`include_flips`, index `>= N` = flipped image `i - N`; the nearest-neighbour figure shows the
mirrored slice), rows record `reference: train+hflip`, and `scripts/checkpoint_curve.py`
recomputes cached rows that used another reference. The baseline curve and its three evaluations
were rerun with the corrected check (interim epoch-60/100 rows keep a note). Memorised fractions
can only go up under the new reference, never down.

Files: synthmri/eval/diversity.py, scripts/checkpoint_curve.py, tests/test_eval.py, docs/EXPERIMENTS.md
Follow-ups: none

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
