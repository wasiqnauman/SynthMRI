#!/usr/bin/env python
"""Fine-tune the pretrained VAE's decoder on the training slices and measure the new ceiling.

    python scripts/finetune_vae_decoder.py --config configs/ldm128_maskcond_reg.yaml --out runs/vae_dec_brats128

Only ``decoder`` + ``post_quant_conv`` are trained (L1 + LPIPS-VGG on the training split, model
selection on the validation split), so cached latents and every trained diffusion model stay valid;
samples are decoded again with ``scripts/sample.py --vae_decoder <out>/decoder.pt``. The script
reports the reconstruction ceiling on the test slices (PSNR / SSIM / LPIPS-Alex, as in
``scripts/evaluate.py``) before and after fine-tuning on the same slices.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from synthmri.config import load_config
from synthmri.data.dataset import SliceDataset
from synthmri.eval.recon import evaluate_vae_reconstruction, save_reconstruction_examples
from synthmri.models.vae import load_vae
from synthmri.models.vae_finetune import DecoderFinetuneConfig, finetune_decoder
from synthmri.utils.io import git_commit_hash, save_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="config providing data paths / modalities / VAE name")
    p.add_argument("--out", default=None, help="output dir (default runs/vae_dec_<dataset>)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lpips_weight", type=float, default=0.5)
    p.add_argument("--lpips_net", default="vgg")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=None, help="smoke tests")
    p.add_argument("--max_val", type=int, default=None, help="cap on validation slices per epoch")
    p.add_argument("--max_test", type=int, default=None, help="cap on test slices for the ceiling")
    p.add_argument("--no_lpips_eval", action="store_true")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mods = tuple(cfg.data.modalities)
    out = Path(args.out) if args.out else Path("runs") / f"vae_dec_{Path(cfg.data.processed_dir).name}"
    out.mkdir(parents=True, exist_ok=True)
    train_ds = SliceDataset(cfg.data.processed_dir, "train", mods, hflip=True, return_mask=False)
    val_ds = SliceDataset(cfg.data.processed_dir, "val", mods, hflip=False, return_mask=False)
    test_ds = SliceDataset(cfg.data.processed_dir, "test", mods, hflip=False, return_mask=False)

    vae = load_vae(cfg.model.vae.pretrained, device=device, scaling_factor=cfg.model.vae.scaling_factor)
    t0 = time.time()
    before = evaluate_vae_reconstruction(vae, test_ds, mods, device, max_items=args.max_test, use_lpips=not args.no_lpips_eval)
    print("test ceiling before:", json.dumps(before["metrics"]["rgb"]))

    ft_cfg = DecoderFinetuneConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, lpips_weight=args.lpips_weight,
                                   lpips_net=args.lpips_net, seed=args.seed, num_workers=args.num_workers, max_steps=args.max_steps,
                                   max_val_items=args.max_val)
    info = finetune_decoder(vae, train_ds, val_ds, ft_cfg, out, device)

    after = evaluate_vae_reconstruction(vae, test_ds, mods, device, max_items=args.max_test, use_lpips=not args.no_lpips_eval)
    print("test ceiling after: ", json.dumps(after["metrics"]["rgb"]))
    idx = list(range(0, len(test_ds), max(1, len(test_ds) // 6)))[:6]
    save_reconstruction_examples(vae, test_ds, out / "reconstruction_examples_after.png", idx, device)
    frozen = load_vae(cfg.model.vae.pretrained, device=device, scaling_factor=cfg.model.vae.scaling_factor)
    save_reconstruction_examples(frozen, test_ds, out / "reconstruction_examples_before.png", idx, device)
    results = {
        "config": args.config, "dataset": Path(cfg.data.processed_dir).name, "pretrained": cfg.model.vae.pretrained, "modalities": list(mods),
        "decoder_weights": str(out / "decoder.pt"), "finetune": {k: v for k, v in info.items() if k != "history"},
        "test_ceiling_before": before, "test_ceiling_after": after, "seconds": time.time() - t0, "git_commit": git_commit_hash(), "args": vars(args),
    }
    save_json(results, out / "results.json")
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
