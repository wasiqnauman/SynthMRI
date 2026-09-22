#!/usr/bin/env python
"""Choose the main generative recipe using validation data only (rule fixed in docs/EXPERIMENTS.md).

    python scripts/select_model.py runs/ldm128_maskcond runs/ldm128_maskcond_do01 runs/ldm128_maskcond_reg

For every candidate run, take the ``best`` row (lowest validation loss checkpoint) of
``<run>/checkpoint_curve.json``; drop runs whose fraction of samples closer to a training slice than
95 % of held-out slices exceeds ``--max_memorised``; among the rest pick the lowest FID against real
validation slices. Prints the chosen run name on stdout and writes ``results/model_selection.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--max_memorised", type=float, default=0.15)
    p.add_argument("--out", default="results/model_selection.json")
    args = p.parse_args()
    rows = []
    for r in args.runs:
        f = Path(r) / "checkpoint_curve.json"
        if not f.exists():
            print(f"skip {r}: no checkpoint_curve.json", file=sys.stderr)
            continue
        d = json.loads(f.read_text())
        best = [c for c in d["checkpoints"] if c["tag"] == "best"]
        if not best:
            print(f"skip {r}: no 'best' row", file=sys.stderr)
            continue
        b = best[0]
        rows.append({"run": Path(r).name, "best_epoch": b.get("epoch"), "fid_val": b["fid_val"]["fid"], "kid_val": b["fid_val"]["kid_mean"],
                     "frac_memorised": b["memorisation"]["frac_closer_than_val_p5"], "nn_dist_mean": b["memorisation"]["fake_to_train_nn_dist"]["mean"]})
    eligible = [x for x in rows if x["frac_memorised"] <= args.max_memorised]
    if not eligible:
        raise SystemExit(f"no candidate has frac_memorised <= {args.max_memorised}: {rows}")
    chosen = min(eligible, key=lambda x: x["fid_val"])
    out = {"rule": f"lowest FID vs validation slices at the best-val-loss checkpoint among runs with frac_memorised <= {args.max_memorised}",
           "candidates": rows, "chosen": chosen["run"]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    for x in rows:
        print(f"{x['run']:26s} best epoch {x['best_epoch']!s:>4}  FID(val) {x['fid_val']:6.2f}  memorised {x['frac_memorised']:.3f}"
              f"{'' if x['frac_memorised'] <= args.max_memorised else '  (excluded)'}", file=sys.stderr)
    print(f"chosen: {chosen['run']}", file=sys.stderr)
    print(chosen["run"])


if __name__ == "__main__":
    main()
