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

Shared: frozen `stabilityai/sd-vae-ft-mse`; U-Net channels (128, 256, 512, 512), 2 res-blocks per
level, self-attention at levels 1–2; ε-prediction, linear β schedule, T = 1000; AdamW lr 1e-4,
wd 0.01, 500 warm-up steps then cosine; grad-clip 1.0; EMA 0.9999 with warm-up; bf16; horizontal flips.

## Generation quality (`scripts/sample.py` + `scripts/evaluate.py`)

5,000 samples per model with DDIM, 50 steps, η = 0 (mask-conditioned: guidance scale 2.0 and 1.0,
masks drawn from the training split with random flips). Against all 4,623 real test slices:

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

| condition | real training patients | synthetic slices | seeds |
|---|---|---|---|
| real 10 % | 26 | 0 / 5,000 | 0, 1, 2 |
| real 25 % | 65 | 0 / 5,000 | 0, 1, 2 |
| real 100 % | 258 | 0 / 5,000 | 0, 1, 2 |
| synthetic only | 0 | 5,000 | 0, 1, 2 |

Synthetic pairs come from LDM-128-mask (guidance 2.0). The real-only segmenters are additionally
scored on the synthetic pairs ("mask consistency": how well the generated image matches the mask it
was conditioned on).

## What would strengthen the paper further

* A radiology-specific feature extractor (e.g. RadImageNet) for FID, alongside Inception.
* 3-D or 2.5-D generation; the present models are per-slice and ignore inter-slice consistency.
* Fine-tuning the VAE *decoder* (encoder frozen) with an L1 + LPIPS loss to raise the ceiling.
* External validation of the segmenter on BraTS 2021 patients not in BraTS 2020.
