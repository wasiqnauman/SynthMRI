#!/usr/bin/env python
"""Evaluate a trained run and a set of generated samples against the held-out test patients.

    python scripts/evaluate.py --run runs/ldm128_maskcond --samples runs/ldm128_maskcond/samples_ddim50_cfg2_seed0

Reports (written to ``<samples>/eval/results.json`` and ``results.md``):
  * VAE reconstruction ceiling on real test slices (PSNR / SSIM / LPIPS per modality)
  * FID / KID of samples vs real test slices, per modality and composite RGB, plus the
    real-val-vs-real-test reference (the "floor" two real sets of this size achieve)
  * pairwise-SSIM diversity of samples (and of real test slices for reference)
  * nearest-training-neighbour memorisation check (samples vs real held-out slices)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from synthmri.config import load_config
from synthmri.data.dataset import SliceDataset
from synthmri.diffusion.checkpoint import find_run_dir
from synthmri.eval.diversity import (
    memorisation_report,
    nearest_neighbour_distances,
    pairwise_ssim_diversity,
    save_nearest_neighbour_figure,
)
from synthmri.eval.fidelity import export_pngs, fidelity_report
from synthmri.eval.recon import evaluate_vae_reconstruction, save_reconstruction_examples
from synthmri.models.vae import load_vae
from synthmri.utils.io import save_json


def real_array(ds: SliceDataset, max_items: int | None = None) -> np.ndarray:
    n = len(ds) if max_items is None else min(len(ds), max_items)
    return np.asarray(ds.images[:n][:, ds.mod_idx], dtype=np.float32)


def results_markdown(r: dict) -> str:
    lines = ["# Evaluation", ""]
    if "vae_reconstruction" in r:
        lines += ["## VAE reconstruction ceiling (real test slices)", "", "| channel | PSNR (dB) | SSIM | LPIPS |", "|---|---|---|---|"]
        for k, m in r["vae_reconstruction"]["metrics"].items():
            lp = m.get("lpips", {}).get("mean", float("nan"))
            lines.append(f"| {k} | {m['psnr']['mean']:.2f} ± {m['psnr']['std']:.2f} | {m['ssim']['mean']:.3f} ± {m['ssim']['std']:.3f} | {lp:.3f} |")
        lines.append("")
    if "fidelity" in r:
        lines += ["## Fidelity (samples vs real test)", "", "| channel | FID ↓ | KID ×10³ ↓ | ref FID (val vs test) | ref KID ×10³ |", "|---|---|---|---|---|"]
        ref = r.get("fidelity_reference_val_vs_test", {})
        for k, m in r["fidelity"].items():
            rf = ref.get(k, {})
            lines.append(f"| {k} | {m['fid']:.2f} | {1000 * m['kid_mean']:.2f} ± {1000 * m['kid_std']:.2f} | "
                         f"{rf.get('fid', float('nan')):.2f} | {1000 * rf.get('kid_mean', float('nan')):.2f} |")
        lines.append("")
    if "diversity" in r:
        d = r["diversity"]
        lines += ["## Diversity (mean pairwise SSIM, lower = more diverse)", "",
                  f"- samples: {d['samples']['pairwise_ssim_mean']:.3f} ± {d['samples']['pairwise_ssim_std']:.3f}",
                  f"- real test: {d['real_test']['pairwise_ssim_mean']:.3f} ± {d['real_test']['pairwise_ssim_std']:.3f}", ""]
    if "memorisation" in r:
        m = r["memorisation"]
        lines += ["## Memorisation (L2 distance to nearest training slice, 64×64 grey features)", "",
                  f"- samples → train: mean {m['fake_to_train_nn_dist']['mean']:.3f}, min {m['fake_to_train_nn_dist']['min']:.3f}",
                  f"- real test → train: mean {m['test_to_train_nn_dist']['mean']:.3f}, min {m['test_to_train_nn_dist']['min']:.3f}",
                  f"- fraction of samples closer to train than the 5th percentile of real test slices: {m['frac_fake_closer_than_test_p5']:.3f}", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--samples", required=True, help="directory containing images.npy from scripts/sample.py")
    p.add_argument("--max_real", type=int, default=None, help="cap on real test slices used (default: all)")
    p.add_argument("--max_fake", type=int, default=5000, help="use the first N samples (same N for every model)")
    p.add_argument("--skip_fid", action="store_true")
    p.add_argument("--skip_recon", action="store_true")
    p.add_argument("--no_lpips", action="store_true")
    p.add_argument("--vae_decoder", default=None, help="decoder.pt from scripts/finetune_vae_decoder.py for the reconstruction ceiling")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = find_run_dir(args.run)
    cfg = load_config(run_dir / "config.yaml")
    mods = tuple(cfg.data.modalities)
    samples_dir = Path(args.samples)
    out = samples_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    vae_decoder = args.vae_decoder or cfg.model.vae.decoder_weights
    results: dict = {"run": str(run_dir), "samples": str(samples_dir), "modalities": list(mods), "vae_decoder": vae_decoder}

    test_ds = SliceDataset(cfg.data.processed_dir, "test", mods, hflip=False)
    val_ds = SliceDataset(cfg.data.processed_dir, "val", mods, hflip=False)
    train_ds = SliceDataset(cfg.data.processed_dir, "train", mods, hflip=False)
    fake = np.load(samples_dir / "images.npy")[: args.max_fake].astype(np.float32)
    real_test = real_array(test_ds, args.max_real)
    print(f"samples {fake.shape} | real test {real_test.shape}")

    if not args.skip_recon:
        vae = load_vae(cfg.model.vae.pretrained, device=device, scaling_factor=cfg.model.vae.scaling_factor, decoder_weights=vae_decoder)
        results["vae_reconstruction"] = evaluate_vae_reconstruction(vae, test_ds, mods, device, max_items=args.max_real, use_lpips=not args.no_lpips)
        save_reconstruction_examples(vae, test_ds, out / "vae_reconstruction_examples.png", list(range(0, len(test_ds), max(1, len(test_ds) // 6)))[:6], device)
        print("VAE reconstruction:", json.dumps(results["vae_reconstruction"]["metrics"]["rgb"]))

    if not args.skip_fid:
        tag = "-".join(mods)
        real_png = Path(cfg.data.processed_dir) / "test" / f"png_{tag}"
        val_png = Path(cfg.data.processed_dir) / "val" / f"png_{tag}"
        export_pngs(real_test, real_png, mods)
        export_pngs(real_array(val_ds, args.max_real), val_png, mods)
        fake_png = samples_dir / "png"
        export_pngs(fake, fake_png, mods)
        keys = (*mods, "rgb")
        results["fidelity"] = fidelity_report(fake_png, real_png, keys, device=str(device))
        results["fidelity_reference_val_vs_test"] = fidelity_report(val_png, real_png, keys, device=str(device))
        print("fidelity:", json.dumps(results["fidelity"]))

    results["diversity"] = {
        "samples": pairwise_ssim_diversity(fake, device=device),
        "real_test": pairwise_ssim_diversity(real_test, device=device),
    }
    train_arr = real_array(train_ds)
    results["memorisation"] = memorisation_report(fake, train_arr, real_test, device=device)
    d_fake, i_fake = nearest_neighbour_distances(fake, train_arr, device=device)
    save_nearest_neighbour_figure(fake, train_arr, i_fake, out / "nearest_training_neighbours.png", n=8, dists=d_fake)
    print("diversity:", json.dumps(results["diversity"]))
    print("memorisation:", json.dumps({k: v for k, v in results["memorisation"].items() if k != "nn_index"}))

    save_json(results, out / "results.json")
    (out / "results.md").write_text(results_markdown(results))
    print(f"wrote {out / 'results.json'} and results.md")


if __name__ == "__main__":
    main()
