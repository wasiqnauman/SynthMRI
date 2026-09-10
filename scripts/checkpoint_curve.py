#!/usr/bin/env python
"""Score every saved checkpoint of a run: FID/KID against real *validation* slices and a memorisation
check against the training set. Motivates the checkpoint used for the reported results.

    python scripts/checkpoint_curve.py --run runs/ldm128_maskcond [--num_images 2000] [--mask_source train]

For each of ``best/``, ``final/``, ``checkpoints/epoch_*`` and ``checkpoints_keep/epoch_*``:
  * sample N images (same seed, same conditioning masks for every checkpoint; DDIM 50, guidance 2 for
    mask-conditioned models, masks drawn from ``--mask_source`` with random flips),
  * FID / KID of the composite RGB image against all real validation slices,
  * nearest-training-slice L2 distance (64x64 grey) of the samples, compared with the same statistic
    for real validation slices (training slices in both orientations): ``frac_closer_than_val_p5`` is the fraction of samples that lie closer
    to a training slice than 95 % of real held-out slices do (~0.05 for a model that generalises).
Rows are cached in ``<run>/checkpoint_curve/<tag>.json`` (delete to recompute) and merged with the
per-epoch losses from ``metrics.csv`` into ``<run>/checkpoint_curve.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from synthmri.data.dataset import SliceDataset
from synthmri.diffusion.checkpoint import find_run_dir, load_run
from synthmri.diffusion.conditioning import mask_to_condition
from synthmri.diffusion.sample import LatentSampler
from synthmri.eval.diversity import memorisation_report
from synthmri.eval.fidelity import compute_fid_kid, export_pngs
from synthmri.models.vae import load_vae
from synthmri.utils.io import git_commit_hash, save_json

NN_REFERENCE = "train+hflip"  # cached rows computed against another reference are recomputed


def checkpoint_epoch(ckpt_dir: Path) -> int | None:
    st = ckpt_dir / "training_state.pt"
    if st.exists():
        try:
            return int(torch.load(st, map_location="cpu", weights_only=False)["epoch"]) + 1
        except Exception:  # noqa: BLE001
            pass
    name = ckpt_dir.name
    return int(name.split("_")[1]) if name.startswith("epoch_") else None


def list_checkpoints(run: Path) -> list[tuple[str, Path]]:
    found: dict[str, Path] = {}
    for sub in ("checkpoints", "checkpoints_keep"):
        for d in sorted((run / sub).glob("epoch_*")) if (run / sub).exists() else []:
            found.setdefault(f"ep{int(d.name.split('_')[1]):04d}", d)
    for name in ("best", "final"):
        if (run / name).exists():
            found[name] = run / name
    return sorted(found.items(), key=lambda kv: (kv[0] not in ("best", "final"), kv[0]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--num_images", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=None, help="default: the run's sample.batch_size (VAE decoding at 256 px needs a small batch)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=2.0)
    p.add_argument("--mask_source", default="train", help="split whose masks condition the samples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", nargs="*", default=None, help="tags to score (default: all)")
    p.add_argument("--skip_fid", action="store_true")
    p.add_argument("--keep_pngs", action="store_true")
    args = p.parse_args()

    run = find_run_dir(args.run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = run / "checkpoint_curve"
    cache.mkdir(exist_ok=True)
    cfg = load_run(run, device="cpu")[0]
    if args.batch_size is None:
        args.batch_size = int(cfg.sample.batch_size)
    mods = cfg.data.modalities
    conditional = cfg.model.conditioning == "mask"
    guidance = args.guidance_scale if conditional else 1.0

    def real(split, masks=False):
        return SliceDataset(cfg.data.processed_dir, split, mods, hflip=False, return_mask=masks)

    train_ds, val_ds = real("train"), real("val")
    train_arr = np.asarray(train_ds.images[:][:, train_ds.mod_idx], dtype=np.float32)
    val_arr = np.asarray(val_ds.images[:][:, val_ds.mod_idx], dtype=np.float32)
    real_png = cache / "real_val_png"
    if not args.skip_fid:
        export_pngs(val_arr, real_png, mods)
    cond_batches = None
    if conditional:
        mask_ds = real(args.mask_source, masks=True)
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randint(0, len(mask_ds), (args.num_images,), generator=g)
        flip = torch.rand(args.num_images, generator=g) < 0.5
        cond_batches = []
        for s in range(0, args.num_images, args.batch_size):
            m = torch.stack([mask_ds[int(i)]["mask"] for i in idx[s : s + args.batch_size]])
            f = flip[s : s + args.batch_size]
            m[f] = m[f].flip(-1)
            cond_batches.append(mask_to_condition(m, cfg.model.num_mask_classes, cfg.latent_size))
    vae = load_vae(cfg.model.vae.pretrained, device=device, scaling_factor=cfg.model.vae.scaling_factor)

    for tag, ckpt in list_checkpoints(run):
        if args.only and tag not in args.only:
            continue
        row_path = cache / f"{tag}.json"
        if row_path.exists() and json.loads(row_path.read_text()).get("memorisation", {}).get("reference") == NN_REFERENCE:
            continue
        t0 = time.time()
        _, unet, unet_dir = load_run(ckpt, use_ema=True, device=device)
        sampler = LatentSampler(unet, vae, cfg.diffusion, cfg.latent_size, "ddim", args.steps, guidance, 0.0, cfg.model.num_mask_classes, device)
        gen = torch.Generator(device=device).manual_seed(args.seed)
        imgs = []
        for b, s in enumerate(range(0, args.num_images, args.batch_size)):
            n = min(args.batch_size, args.num_images - s)
            imgs.append(sampler.sample_images(n, cond=None if cond_batches is None else cond_batches[b], generator=gen).cpu().numpy())
        imgs = np.concatenate(imgs)
        epoch = checkpoint_epoch(ckpt)
        if epoch is None and tag == "final" and (run / "train_summary.json").exists():
            epoch = int(json.loads((run / "train_summary.json").read_text())["epochs"])
        row = {"tag": tag, "checkpoint": str(ckpt), "unet_dir": str(unet_dir), "epoch": epoch, "num_images": int(imgs.shape[0]),
               "mask_source": args.mask_source if conditional else None, "guidance_scale": guidance, "steps": args.steps, "seed": args.seed}
        if not args.skip_fid:
            fake_png = cache / f"fake_{tag}_png"
            export_pngs(imgs, fake_png, mods, overwrite=True)
            row["fid_val"] = compute_fid_kid(fake_png / "rgb", real_png / "rgb", device=str(device))
            if not args.keep_pngs:
                shutil.rmtree(fake_png, ignore_errors=True)
        mem = memorisation_report(imgs, train_arr, val_arr, device=str(device))
        mem.pop("nn_index", None)
        row["memorisation"] = {"reference": mem["reference"], "fake_to_train_nn_dist": mem["fake_to_train_nn_dist"],
                               "val_to_train_nn_dist": mem["test_to_train_nn_dist"], "frac_closer_than_val_p5": mem["frac_fake_closer_than_test_p5"]}
        row["seconds"] = time.time() - t0
        row["git_commit"] = git_commit_hash()
        save_json(row, row_path)
        fid = row.get("fid_val", {}).get("fid", float("nan"))
        print(f"{tag:8s} epoch {row['epoch']}: FID(val) {fid:6.2f} | NN mean {mem['fake_to_train_nn_dist']['mean']:.3f} | frac<p5 {mem['frac_fake_closer_than_test_p5']:.3f} ({row['seconds']:.0f}s)", flush=True)
        del unet, sampler
        torch.cuda.empty_cache()

    rows = [json.loads(p_.read_text()) for p_ in sorted(cache.glob("*.json"))]
    rows.sort(key=lambda r: (r["epoch"] is None, r["epoch"] or 0, r["tag"]))
    losses = []
    if (run / "metrics.csv").exists():
        df = pd.read_csv(run / "metrics.csv")
        ep = df[df["val_loss"].notna() & (df["val_loss"].astype(str) != "")]
        losses = [{"epoch": int(e), "train_loss": float(t), "val_loss": float(v)} for e, t, v in zip(ep["epoch"], ep["train_loss"], ep["val_loss"].astype(float))]
    best_epoch = None
    if (run / "best" / "training_state.pt").exists():
        best_epoch = checkpoint_epoch(run / "best")
    out = {"run": str(run), "best_epoch": best_epoch, "checkpoints": rows, "losses": losses}
    save_json(out, run / "checkpoint_curve.json")
    if not args.skip_fid:
        shutil.rmtree(real_png, ignore_errors=True) if not args.keep_pngs else None
    print(f"wrote {run / 'checkpoint_curve.json'} ({len(rows)} checkpoints)")


if __name__ == "__main__":
    main()
