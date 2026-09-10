#!/usr/bin/env python
"""Downstream segmentation experiment: real vs real+synthetic vs synthetic-only training data.

    # real only, all training patients
    python scripts/train_seg.py --config configs/ldm128_maskcond.yaml --out runs/seg/real100_s0 --seed 0
    # 10% of training patients + 2000 synthetic slices
    python scripts/train_seg.py --config configs/ldm128_maskcond.yaml --real_fraction 0.1 \
        --synthetic runs/ldm128_maskcond/samples_ddim50_cfg2_seed0 --n_synth 2000 --out runs/seg/real10_synth2000_s0
    # synthetic only
    python scripts/train_seg.py ... --synthetic <dir> --synthetic_only --out runs/seg/synth_only_s0

``--eval_synthetic DIR`` additionally scores the trained model on synthetic (image, mask) pairs,
which measures how faithfully generated images follow their conditioning masks.

Secondary analyses (docs/EXPERIMENTS.md): ``--synth_ratio 1.0`` caps the synthetic slices at the
number of real slices; ``--real_through_vae`` replaces the real training images by their frozen-VAE
reconstructions (isolates the VAE ceiling from the generator); ``--pretrain_synthetic DIR`` first
trains on the synthetic pairs alone for ``--pretrain_epochs`` and then fine-tunes on the real slices.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from synthmri.config import load_config
from synthmri.data.dataset import SliceDataset
from synthmri.eval.recon import vae_roundtrip
from synthmri.eval.segmentation import SegConfig, evaluate_segmenter, subsample_patients, train_segmenter
from synthmri.models.vae import load_vae
from synthmri.utils.io import git_commit_hash, save_json


def load_split(cfg, split: str):
    ds = SliceDataset(cfg.data.processed_dir, split, cfg.data.modalities, hflip=False)
    images = np.asarray(ds.images[:, ds.mod_idx])  # (N,3,S,S) float16 in [0,1]
    masks = np.asarray(ds.masks)
    return images, masks, ds.patient_ids


def load_synthetic(path: Path, n: int | None, seed: int, allowed_patients: set[str] | None = None):
    """Load synthetic (image, mask) pairs; optionally keep only those conditioned on masks of
    ``allowed_patients`` (so a low-data experiment never sees masks from patients it does not own)."""
    images = np.load(path / "images.npy", mmap_mode="r")  # only the selected rows are read into memory
    masks_path = path / "masks.npy"
    if not masks_path.exists():
        raise FileNotFoundError(f"{path} has no masks.npy: synthetic segmentation data needs a mask-conditioned model")
    masks = np.load(masks_path, mmap_mode="r")
    keep = np.ones(images.shape[0], dtype=bool)
    if allowed_patients is not None:
        meta = pd.read_csv(path / "meta.csv")
        if "source_patient" not in meta.columns:
            raise ValueError(f"{path}/meta.csv has no source_patient column; cannot match patients")
        keep = meta["source_patient"].isin(allowed_patients).to_numpy()
    idx = np.flatnonzero(keep)
    if n is not None and n < idx.size:
        idx = np.sort(np.random.RandomState(seed).choice(idx, n, replace=False))
    return np.ascontiguousarray(images[idx]), np.ascontiguousarray(masks[idx])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="config providing data paths/modalities")
    p.add_argument("--out", required=True)
    p.add_argument("--real_fraction", type=float, default=1.0, help="fraction of training *patients* to use")
    p.add_argument("--synthetic", default=None, help="sample directory with images.npy + masks.npy")
    p.add_argument("--n_synth", type=int, default=None, help="number of synthetic slices to add (default: all)")
    p.add_argument("--synth_ratio", type=float, default=None, help="synthetic slices as a multiple of the real slice count (overrides --n_synth)")
    p.add_argument("--synthetic_only", action="store_true")
    p.add_argument("--real_through_vae", action="store_true", help="control: real training images replaced by their frozen-VAE reconstructions")
    p.add_argument("--pretrain_synthetic", default=None, help="sample directory: train on it alone first, then fine-tune on the real slices")
    p.add_argument("--pretrain_epochs", type=int, default=20)
    p.add_argument("--no_match_patients", action="store_true",
                   help="by default synthetic slices are restricted to those conditioned on masks of the kept real patients")
    p.add_argument("--eval_synthetic", default=None, help="score the model on this synthetic set (mask consistency)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--eval_split", default="test")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tr_img, tr_msk, tr_pid = load_split(cfg, "train")
    keep = subsample_patients(tr_pid, args.real_fraction, args.seed)
    tr_img, tr_msk, tr_pid = tr_img[keep], tr_msk[keep], tr_pid[keep]
    n_real, n_real_patients = int(tr_img.shape[0]), int(len(np.unique(tr_pid)))
    if args.real_through_vae:
        vae = load_vae(cfg.model.vae.pretrained, device=device, scaling_factor=cfg.model.vae.scaling_factor)
        tr_img = vae_roundtrip(tr_img, vae, device)
        del vae
        torch.cuda.empty_cache()
    n_synth, n_pretrain = 0, 0
    allowed = None if (args.no_match_patients or args.synthetic_only) else set(np.unique(tr_pid).tolist())
    init_state = None
    if args.pretrain_synthetic:
        pre_img, pre_msk = load_synthetic(Path(args.pretrain_synthetic), args.n_synth, args.seed, allowed)
        n_pretrain = int(pre_img.shape[0])
        va_img, va_msk, va_pid = load_split(cfg, "val")
        pre_cfg = SegConfig(epochs=args.pretrain_epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed, in_channels=len(cfg.data.modalities), max_steps=args.max_steps)
        print(f"pre-training on {n_pretrain} synthetic slices for {args.pretrain_epochs} epochs")
        pre_model, _ = train_segmenter(pre_img.astype(np.float16), pre_msk, va_img, va_msk, va_pid, pre_cfg, out / "pretrain", device)
        init_state = {k: v.detach().cpu().clone() for k, v in pre_model.state_dict().items()}
        del pre_model
    if args.synthetic:
        n_synth_req = int(round(args.synth_ratio * n_real)) if args.synth_ratio is not None else args.n_synth
        sy_img, sy_msk = load_synthetic(Path(args.synthetic), n_synth_req, args.seed, allowed)
        n_synth = int(sy_img.shape[0])
        if args.synthetic_only:
            tr_img, tr_msk = sy_img.astype(np.float16), sy_msk
            n_real, n_real_patients = 0, 0
        else:
            tr_img = np.concatenate([tr_img, sy_img.astype(np.float16)])
            tr_msk = np.concatenate([tr_msk, sy_msk])
    va_img, va_msk, va_pid = load_split(cfg, "val")
    te_img, te_msk, te_pid = load_split(cfg, args.eval_split)
    print(f"train: {n_real} real slices from {n_real_patients} patients + {n_synth} synthetic | val {va_img.shape[0]} | {args.eval_split} {te_img.shape[0]}")

    seg_cfg = SegConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed, in_channels=len(cfg.data.modalities), max_steps=args.max_steps)
    model, history = train_segmenter(tr_img, tr_msk, va_img, va_msk, va_pid, seg_cfg, out, device, init_state=init_state)
    test = evaluate_segmenter(model, te_img, te_msk, te_pid, device)
    results = {
        "args": vars(args), "seg_config": asdict(seg_cfg), "n_real_slices": n_real, "n_real_patients": n_real_patients,
        "n_synthetic": n_synth, "synthetic_only": bool(args.synthetic_only), "eval_split": args.eval_split,
        "synth_ratio": args.synth_ratio, "real_through_vae": bool(args.real_through_vae),
        "n_pretrain_synthetic": n_pretrain, "pretrain_epochs": args.pretrain_epochs if args.pretrain_synthetic else 0,
        "synthetic_patient_matched": bool(args.synthetic and not args.no_match_patients and not args.synthetic_only),
        "test": {"patient": {r: {k: v for k, v in m.items() if k != "values"} for r, m in test["patient"].items() if isinstance(m, dict)},
                 "mean_WT_TC_ET": test["patient"]["mean_WT_TC_ET"], "slice": test["slice"], "n_patients": test["n_patients"]},
        "test_patient_values": {r: test["patient"][r]["values"] for r in ("WT", "TC", "ET")},
        "history": history, "git_commit": git_commit_hash(),
    }
    if args.eval_synthetic:
        sy_img, sy_msk = load_synthetic(Path(args.eval_synthetic), None, args.seed)
        pid = np.arange(sy_img.shape[0])  # each synthetic slice scored independently
        cons = evaluate_segmenter(model, sy_img, sy_msk, pid, device)
        results["synthetic_consistency"] = {"n": int(sy_img.shape[0]), "slice": cons["slice"]}
    save_json(results, out / "results.json")
    print(json.dumps({"test_patient_dice": {r: round(test["patient"][r]["mean"], 4) for r in ("WT", "TC", "ET")},
                      "mean": round(test["patient"]["mean_WT_TC_ET"], 4)}, indent=2))
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
