#!/usr/bin/env python
"""Generate a synthetic dataset from a trained run.

    python scripts/sample.py --run runs/ldm128_maskcond --num_images 2000 --steps 50 --guidance_scale 2.0

By default the weights with the lowest validation loss (``<run>/best``, EMA) are used; ``--checkpoint final``
or ``--checkpoint 150`` select the last epoch or a specific one.

Outputs (default ``<run>/samples_<ckpt>_<sampler><steps>_cfg<g>_seed<s>[_<split>masks]/``): ``images.npy`` (N,3,S,S) float16
in [0,1] with the run's modality order, ``masks.npy`` for mask-conditioned models, ``meta.csv``,
``preview.png`` and per-modality PNGs under ``png/`` for FID.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from synthmri.config import load_config
from synthmri.data.dataset import SliceDataset
from synthmri.diffusion.checkpoint import find_run_dir, load_run, resolve_checkpoint
from synthmri.diffusion.conditioning import mask_to_condition
from synthmri.diffusion.sample import LatentSampler
from synthmri.eval.fidelity import export_pngs
from synthmri.models.vae import load_vae
from synthmri.utils.io import git_commit_hash, save_json
from synthmri.utils.viz import save_sample_sheet


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run directory (or checkpoint directory)")
    p.add_argument("--checkpoint", default=None, help="'best' (default), 'final', an epoch number or a checkpoint dir")
    p.add_argument("--num_images", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--sampler", choices=["ddim", "ddpm"], default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--mask_source", default=None, help="split whose masks condition the samples")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--no_png", action="store_true", help="skip PNG export")
    p.add_argument("--vae_decoder", default=None, help="decoder.pt from scripts/finetune_vae_decoder.py; output dir gets the suffix _ftdec")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = find_run_dir(args.run)
    ckpt_name = args.checkpoint or load_config(run_dir / "config.yaml").sample.checkpoint or "best"
    ckpt_dir, ckpt_tag = resolve_checkpoint(args.run, ckpt_name)
    cfg, unet, unet_dir = load_run(ckpt_dir, use_ema=not args.no_ema, device=device)
    s = cfg.sample
    num_images = args.num_images or s.num_images
    batch_size = args.batch_size or s.batch_size
    sampler_name = args.sampler or s.sampler
    steps = args.steps or s.steps
    guidance = s.guidance_scale if args.guidance_scale is None else args.guidance_scale
    eta = s.eta if args.eta is None else args.eta
    seed = s.seed if args.seed is None else args.seed
    mask_source = args.mask_source or s.mask_source
    conditional = cfg.model.conditioning == "mask"
    if not conditional:
        guidance = 1.0
    suffix = f"_{mask_source}masks" if conditional and mask_source != "train" else ""
    vae_decoder = args.vae_decoder or cfg.model.vae.decoder_weights
    if vae_decoder:
        suffix += "_ftdec"
    out = Path(args.output_dir) if args.output_dir else run_dir / f"samples_{ckpt_tag}_{sampler_name}{steps}_cfg{guidance:g}_seed{seed}{suffix}"
    out.mkdir(parents=True, exist_ok=True)

    vae = load_vae(cfg.model.vae.pretrained, device=device, scaling_factor=cfg.model.vae.scaling_factor, decoder_weights=vae_decoder)
    sampler = LatentSampler(unet, vae, cfg.diffusion, cfg.latent_size, sampler_name, steps, guidance, eta,
                            cfg.model.num_mask_classes, device)
    gen = torch.Generator(device=device).manual_seed(seed)
    cpu_gen = torch.Generator().manual_seed(seed)

    mask_ds = None
    if conditional:
        mask_ds = SliceDataset(cfg.data.processed_dir, mask_source, cfg.data.modalities, hflip=False, return_mask=True)
        src_idx = torch.randint(0, len(mask_ds), (num_images,), generator=cpu_gen).numpy()
        src_flip = (torch.rand(num_images, generator=cpu_gen) < 0.5).numpy()

    images, masks, rows = [], [], []
    t0 = time.time()
    for start in range(0, num_images, batch_size):
        n = min(batch_size, num_images - start)
        cond = None
        if conditional:
            m = torch.stack([mask_ds[int(i)]["mask"] for i in src_idx[start : start + n]])
            flip = torch.from_numpy(src_flip[start : start + n])
            m[flip] = m[flip].flip(-1)
            cond = mask_to_condition(m, cfg.model.num_mask_classes, cfg.latent_size)
            masks.append(m.numpy().astype(np.uint8))
            for k in range(n):
                i = int(src_idx[start + k])
                rows.append({"index": start + k, "source_mask_index": i, "source_patient": mask_ds.meta.iloc[i]["patient_id"],
                             "source_split": mask_source, "flipped": bool(src_flip[start + k])})
        else:
            rows += [{"index": start + k} for k in range(n)]
        imgs = sampler.sample_images(n, cond=cond, generator=gen)
        images.append(imgs.cpu().half().numpy())
        done = start + n
        print(f"{done}/{num_images} images  ({(time.time() - t0) / done:.2f} s/img)", flush=True)

    images = np.concatenate(images)
    np.save(out / "images.npy", images)
    if masks:
        np.save(out / "masks.npy", np.concatenate(masks))
    pd.DataFrame(rows).to_csv(out / "meta.csv", index=False)
    save_json(
        {"run": str(run_dir), "checkpoint": ckpt_tag, "unet_dir": str(unet_dir), "num_images": num_images, "sampler": sampler_name, "steps": steps,
         "guidance_scale": guidance, "eta": eta, "seed": seed, "use_ema": not args.no_ema, "mask_source": mask_source if conditional else None,
         "modalities": list(cfg.data.modalities), "vae_decoder": vae_decoder, "seconds": time.time() - t0, "git_commit": git_commit_hash(), "argv": sys.argv},
        out / "sample_info.json",
    )
    preview_masks = torch.from_numpy(np.concatenate(masks)[:8].astype(np.int64)) if masks else None
    save_sample_sheet(torch.from_numpy(images[:8].astype(np.float32)), out / "preview.png", masks=preview_masks)
    if not args.no_png:
        export_pngs(images, out / "png", cfg.data.modalities)
    print(f"wrote {num_images} samples to {out}")


if __name__ == "__main__":
    main()
