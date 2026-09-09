# Work log

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
