#!/usr/bin/env python
"""Aggregate every results.json under runs/ into Markdown tables + one JSON summary.

    python scripts/collect_results.py            # writes docs/results_tables.md and results/summary.json

Segmentation runs are grouped by (real fraction, synthetic used, synthetic only) and averaged over
seeds (mean ± std of the per-patient test Dice).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def fmt(m: float, s: float | None = None, nd: int = 3) -> str:
    return f"{m:.{nd}f}" if s is None else f"{m:.{nd}f} ± {s:.{nd}f}"


def generation_tables(runs_dir: Path) -> tuple[list[str], dict]:
    lines, summary = [], {}
    rows = sorted(runs_dir.glob("*/samples_*/eval/results.json"))
    if not rows:
        return lines, summary
    lines += ["## Generative quality (samples vs real test slices)", "",
              "| run | samples | FID rgb ↓ | KID rgb ×10³ ↓ | FID flair | FID t1ce | FID t2 | ref FID rgb (val vs test) | pair-SSIM (samples / real) ↓ | NN dist (samples / real test) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for p in rows:
        r = json.loads(p.read_text())
        run, tag = p.parents[2].name, p.parents[1].name
        f = r.get("fidelity", {})
        ref = r.get("fidelity_reference_val_vs_test", {})
        d = r.get("diversity", {})
        m = r.get("memorisation", {})

        def g(k: str, key: str = "fid", _f=f) -> float:
            return _f.get(k, {}).get(key, float("nan"))

        lines.append(
            f"| {run} | {tag} | {g('rgb'):.2f} | {1000 * g('rgb', 'kid_mean'):.2f} | {g('flair'):.2f} | {g('t1ce'):.2f} | {g('t2'):.2f} | "
            f"{ref.get('rgb', {}).get('fid', float('nan')):.2f} | "
            f"{d.get('samples', {}).get('pairwise_ssim_mean', float('nan')):.3f} / {d.get('real_test', {}).get('pairwise_ssim_mean', float('nan')):.3f} | "
            f"{m.get('fake_to_train_nn_dist', {}).get('mean', float('nan')):.2f} / {m.get('test_to_train_nn_dist', {}).get('mean', float('nan')):.2f} |"
        )
        summary[f"{run}/{tag}"] = {k: v for k, v in r.items() if k not in ("memorisation",)} | {"memorisation": {k: v for k, v in m.items() if k != "nn_index"}}
    lines.append("")
    # VAE ceiling (one per run; identical for runs sharing data)
    seen = set()
    vae_lines = ["## VAE reconstruction ceiling (real test slices)", "", "| run | channel | PSNR (dB) | SSIM | LPIPS |", "|---|---|---|---|---|"]
    for p in rows:
        r = json.loads(p.read_text())
        if "vae_reconstruction" not in r or p.parents[2].name in seen:
            continue
        seen.add(p.parents[2].name)
        for k, mm in r["vae_reconstruction"]["metrics"].items():
            lp = mm.get("lpips", {}).get("mean", float("nan"))
            vae_lines.append(f"| {p.parents[2].name} | {k} | {fmt(mm['psnr']['mean'], mm['psnr']['std'], 2)} | {fmt(mm['ssim']['mean'], mm['ssim']['std'])} | {lp:.3f} |")
    if len(vae_lines) > 4:
        lines += vae_lines + [""]
    return lines, summary


def segmentation_tables(runs_dir: Path) -> tuple[list[str], dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for p in sorted(runs_dir.glob("seg/*/results.json")):
        r = json.loads(p.read_text())
        a = r["args"]
        # Patient-matched filtering makes the synthetic count seed-dependent, so it is reported, not grouped on.
        key = (float(a.get("real_fraction", 1.0)), bool(a.get("synthetic")), bool(a.get("synthetic_only")))
        groups[key].append(r)
    if not groups:
        return [], {}
    lines = ["## Downstream segmentation (per-patient Dice on held-out test patients, mean ± std over seeds)", "",
             "| training data | real patients | real slices | synthetic slices | seeds | WT | TC | ET | mean |", "|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for key in sorted(groups, key=lambda k: (k[2], k[0], k[1])):
        frac, synth, only = key
        rs = groups[key]
        name = "synthetic only" if only else (f"{frac:.0%} real" + (" + synthetic" if synth else ""))
        vals = {r_: [r["test"]["patient"][r_]["mean"] for r in rs] for r_ in ("WT", "TC", "ET")}
        vals["mean"] = [r["test"]["mean_WT_TC_ET"] for r in rs]
        cells = " | ".join(fmt(float(np.mean(v)), float(np.std(v))) for v in vals.values())
        n_synth = [int(r.get("n_synthetic", 0)) for r in rs]
        n_synth_cell = str(n_synth[0]) if min(n_synth) == max(n_synth) else f"{min(n_synth)}–{max(n_synth)}"
        n_real = [int(r["n_real_slices"]) for r in rs]
        n_real_cell = str(n_real[0]) if min(n_real) == max(n_real) else f"{min(n_real)}–{max(n_real)}"
        lines.append(f"| {name} | {rs[0]['n_real_patients']} | {n_real_cell} | {n_synth_cell} | {len(rs)} | {cells} |")
        summary[name] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v} for k, v in vals.items()} | {
            "n_seeds": len(rs), "n_real_patients": int(rs[0]["n_real_patients"]), "n_real_slices": n_real, "n_synthetic": n_synth}
    lines.append("")
    cons = {k: [r["synthetic_consistency"] for r in rs if "synthetic_consistency" in r] for k, rs in groups.items()}
    cons = {k: v for k, v in cons.items() if v}
    if cons:
        lines += ["## Mask consistency of synthetic samples (per-slice Dice of a real-trained segmenter vs the conditioning mask, mean ± std over seeds)", "",
                  "| segmenter | n synthetic | WT | TC | ET |", "|---|---|---|---|---|"]
        summary["mask_consistency"] = {}
        for key in sorted(cons, key=lambda k: (k[2], k[0], k[1])):
            frac, synth, only = key
            name = "synthetic only" if only else (f"{frac:.0%} real" + (" + synthetic" if synth else ""))
            vals = {r_: [c["slice"][r_]["mean"] for c in cons[key]] for r_ in ("WT", "TC", "ET")}
            cells = " | ".join(fmt(float(np.mean(v)), float(np.std(v))) for v in vals.values())
            lines.append(f"| {name} | {cons[key][0]['n']} | {cells} |")
            summary["mask_consistency"][name] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v} for k, v in vals.items()} | {"n_seeds": len(cons[key])}
        lines.append("")
    return lines, summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="runs")
    p.add_argument("--out_md", default="docs/results_tables.md")
    p.add_argument("--out_json", default="results/summary.json")
    args = p.parse_args()
    runs = Path(args.runs)
    g_lines, g_sum = generation_tables(runs)
    s_lines, s_sum = segmentation_tables(runs)
    md = ["# Result tables (auto-generated by scripts/collect_results.py)", ""] + g_lines + s_lines
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(md))
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"generation": g_sum, "segmentation": s_sum}, indent=2))
    print("\n".join(md))
    print(f"\nwrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
