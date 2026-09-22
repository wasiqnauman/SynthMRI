#!/usr/bin/env python
"""Discover BraTS patients, build (or reuse) patient-level splits and write 2-D slice arrays.

    python scripts/preprocess.py --config configs/base.yaml data.image_size=256 data.processed_dir=data/processed/brats256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthmri.config import load_config
from synthmri.data import find_patients, load_splits, make_patient_splits, save_splits
from synthmri.data.preprocess import PreprocessParams, preprocess_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--workers", type=int, default=None, help="override data.num_workers")
    p.add_argument("overrides", nargs="*", help="dotted config overrides, e.g. data.image_size=256")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    d = cfg.data

    cases = find_patients(d.raw_dir)
    print(f"found {len(cases)} complete patients in {d.raw_dir}")
    split_path = Path(d.splits_file)
    if split_path.exists():
        splits = load_splits(split_path)
        print(f"using existing splits {split_path}")
    else:
        splits = make_patient_splits([c.patient_id for c in cases], d.val_frac, d.test_frac, d.split_seed)
        save_splits(splits, split_path)
        print(f"wrote new splits to {split_path}")
    print({k: len(v) for k, v in splits.items() if k != "meta"})

    params = PreprocessParams(
        image_size=d.image_size, crop_size=d.crop_size, tumour_frac_min=d.tumour_frac_min,
        clip_percentiles=tuple(d.clip_percentiles),
    )
    stats = preprocess_dataset(cases, splits, params, d.processed_dir, num_workers=args.workers or d.num_workers)
    print(json.dumps(stats["splits"], indent=2))
    print(f"wrote {d.processed_dir}")


if __name__ == "__main__":
    main()
