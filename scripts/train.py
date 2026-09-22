#!/usr/bin/env python
"""Train a latent diffusion model.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/ldm128_maskcond.yaml
    python scripts/train.py --config configs/smoke.yaml train.max_steps=10
"""

from __future__ import annotations

import argparse

from synthmri.config import load_config
from synthmri.diffusion.train import train


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*", help="dotted config overrides, e.g. train.epochs=50")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    run_dir = train(cfg)
    print(f"run directory: {run_dir}")


if __name__ == "__main__":
    main()
