# Architecture and design decisions

## Overview

The pipeline is a 2-D latent diffusion model (Rombach et al., 2022) over axial BraTS 2020 slices.

| Stage | Component | Where |
|---|---|---|
| Preprocess | 3-D NIfTI → per-volume normalised 2-D tumour slices, patient-level splits | `synthmri/data/` |
| Compress | frozen Stable-Diffusion VAE (`stabilityai/sd-vae-ft-mse`), 8× spatial, 4 latent channels | `synthmri/models/vae.py` |
| Cache | posterior mean/log-variance per slice (and its horizontal flip), stored once | `synthmri/diffusion/latents.py` |
| Denoise | `UNet2DModel` (diffusers), ε-prediction, 1000-step linear DDPM schedule | `synthmri/models/unet.py`, `synthmri/diffusion/train.py` |
| Condition | one-hot tumour mask average-pooled to the latent grid + null channel, 10 % condition dropout | `synthmri/diffusion/conditioning.py` |
| Sample | DDIM (default 50 steps) or ancestral DDPM, classifier-free guidance | `synthmri/diffusion/sample.py` |
| Evaluate | VAE ceiling, FID/KID, SSIM-diversity, nearest-neighbour memorisation, downstream Dice | `synthmri/eval/` |

## Decisions and rationale

**Patient-level splits.** Neighbouring axial slices of one patient are near-duplicates; a slice-level
split would leak test anatomy into training and inflate every metric. `splits/brats2020_patient_splits.json`
is committed (seed 0; 258 / 37 / 74 patients) so every experiment uses the same held-out patients.

**Per-volume percentile normalisation.** Each modality of each 3-D scan is clipped to its
[0.5, 99.5] percentiles over brain voxels and rescaled to [0, 1]; background stays exactly 0. The
original notebook normalised each 2-D slice by its own min/max, which makes the same tissue change
brightness from slice to slice and lets a single hot voxel rescale a slice.

**Centre crop 224 px before resizing.** The measured brain extent over all 369 patients is
x ∈ [29, 223], y ∈ [40, 197] of the 240×240 grid, so a centred 224-px crop drops only empty border
and keeps 1.75× more pixels on brain at 128 px than resizing the full frame.

**Three modalities → RGB.** The pretrained VAE has three input channels; FLAIR, T1ce and T2 are
mapped to them (T1 is stored for completeness). The VAE was trained on natural images, so its
reconstruction quality on MRI is measured explicitly (`scripts/evaluate.py`) and reported as the
ceiling for the generator.

**Frozen VAE + cached latents.** With a frozen VAE the posterior of a slice never changes, so encoding
it every step is wasted work. Caching the posterior parameters (not a single sample) and sampling
`z = μ + σ·ε` on each access is exactly equivalent to online `latent_dist.sample()`. Both orientations
are cached so horizontal-flip augmentation stays free. The notebook's joint VAE/U-Net fine-tuning was
removed: its VAE loss was a plain MSE (no KL/perceptual term), and moving the encoder shifts the latent
distribution under the diffusion model while training.

**Mask conditioning by concatenation.** The 4-class mask is one-hot encoded and average-pooled to the
latent resolution, giving soft per-cell class fractions, and concatenated to the noisy latent
(`in_channels = 4 + 4 + 1`). A separate *null* channel marks dropped conditions so "no condition" is
not confused with "all background". Conditions are dropped with p = 0.1 during training, enabling
classifier-free guidance at sampling time.

**EMA, warm-up + cosine LR, bf16, gradient clipping.** Standard diffusion-training stabilisers. All
validation, sampling and reported results use the EMA weights. bf16 autocast (Ampere) avoids the loss
scaling that fp16 needs.

**Validation loss with fixed noise.** The diffusion loss is very noisy in the timestep draw; validation
uses a fixed generator and timesteps spread evenly over [0, T) so epochs are comparable, and the
lowest-validation-loss EMA model is kept in `best/`.

**Evaluation protocol.** Inception-based FID/KID are relative measures on MRI; we therefore also
report the FID between two *real* sets (val vs test) as a floor, pairwise-SSIM diversity, and a
memorisation check (nearest-training-neighbour distance of samples vs of unseen real slices). The
downstream test trains a 2-D MONAI U-Net on real / real+synthetic / synthetic-only data and reports
per-patient Dice for WT / TC / ET on the held-out test patients.

## Reproducibility

Every run writes `config.yaml`, `run_info.json` (git commit, library versions, GPU), `metrics.csv`
and TensorBoard logs. Seeds control data order, initialisation, noise and sampling; cuDNN autotuning is
left on, so bit-exact repeats are not guaranteed but results are statistically reproducible.
