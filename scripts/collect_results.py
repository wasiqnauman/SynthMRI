#!/usr/bin/env python
"""Aggregate every results.json under runs/ into Markdown tables + one JSON summary.

    python scripts/collect_results.py            # writes docs/results_tables.md and results/summary.json

Tables: generative quality per sample set, checkpoint curve (best-val-loss vs last checkpoint) and the
model selection, then the segmentation study, whose runs are grouped by (real fraction, synthetic used,
synthetic only) and averaged over seeds (mean ± std of the per-patient test Dice).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

EXCLUDE = {"smoke"}  # pipeline checks, never results


def fmt(m: float, s: float | None = None, nd: int = 3) -> str:
    return f"{m:.{nd}f}" if s is None else f"{m:.{nd}f} ± {s:.{nd}f}"


def generation_tables(runs_dir: Path) -> tuple[list[str], dict]:
    lines, summary = [], {}
    rows = sorted(p for p in runs_dir.glob("*/samples_*/eval/results.json") if p.parents[2].name not in EXCLUDE)
    if not rows:
        return lines, summary
    lines += ["## Generative quality (samples vs real test slices)", "",
              "| run | samples | FID rgb ↓ | KID rgb ×10³ ↓ | FID flair | FID t1ce | FID t2 | ref FID rgb (val vs test) | pair-SSIM (samples / real) ↓ | NN dist (samples / real test) | memorised ↓ |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
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
            f"{m.get('fake_to_train_nn_dist', {}).get('mean', float('nan')):.2f} / {m.get('test_to_train_nn_dist', {}).get('mean', float('nan')):.2f} | "
            f"{m.get('frac_fake_closer_than_test_p5', float('nan')):.3f} |"
        )
        summary[f"{run}/{tag}"] = {k: v for k, v in r.items() if k not in ("memorisation",)} | {"memorisation": {k: v for k, v in m.items() if k != "nn_index"}}
    lines.append("")
    # VAE ceiling: depends only on the data (resolution), so one block per processed dataset
    seen = set()
    vae_lines = ["## VAE reconstruction ceiling (real test slices; frozen VAE, so it depends only on the data)", "",
                 "| data | channel | PSNR (dB) | SSIM | LPIPS |", "|---|---|---|---|---|"]
    for p in rows:
        r = json.loads(p.read_text())
        if "vae_reconstruction" not in r:
            continue
        cfg_file = p.parents[2] / "config.yaml"
        data = Path(yaml.safe_load(cfg_file.read_text())["data"]["processed_dir"]).name if cfg_file.exists() else p.parents[2].name
        if data in seen:
            continue
        seen.add(data)
        for k, mm in r["vae_reconstruction"]["metrics"].items():
            lp = mm.get("lpips", {}).get("mean", float("nan"))
            vae_lines.append(f"| {data} ({p.parents[2].name}) | {k} | {fmt(mm['psnr']['mean'], mm['psnr']['std'], 2)} | {fmt(mm['ssim']['mean'], mm['ssim']['std'])} | {lp:.3f} |")
    if len(vae_lines) > 4:
        lines += vae_lines + [""]
    return lines, summary


def checkpoint_tables(runs_dir: Path, selection_json: Path) -> tuple[list[str], dict]:
    """Best-validation-loss vs last checkpoint of every run with a checkpoint curve, plus the model selection."""
    files = sorted(f for f in runs_dir.glob("*/checkpoint_curve.json") if f.parent.name not in EXCLUDE)
    lines, summary = [], {}
    if files:
        lines += ["## Checkpoint curve: best-validation-loss vs last checkpoint (2,000 samples each; FID/KID vs real validation slices; "
                  "memorised = fraction of samples closer to a training slice than 95 % of real held-out slices)", "",
                  "| run | best epoch | val loss | FID val ↓ | KID val ×10³ ↓ | memorised ↓ | last epoch | FID val | memorised |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for f in files:
            d = json.loads(f.read_text())
            run = f.parent.name
            rows = [r for r in d["checkpoints"] if r.get("epoch") is not None and "fid_val" in r]
            if not rows:
                continue
            best = next((r for r in rows if r["tag"] == "best"), None)
            last = max(rows, key=lambda r: r["epoch"])
            losses = {l_["epoch"]: l_ for l_ in d.get("losses", [])}

            def cell(r) -> str:
                return f"{r['fid_val']['fid']:.2f} | {1000 * r['fid_val']['kid_mean']:.2f} | {r['memorisation']['frac_closer_than_val_p5']:.3f}"

            if best is not None:
                vl = losses.get(best["epoch"], {}).get("val_loss", float("nan"))
                lines.append(f"| {run} | {best['epoch']} | {vl:.4f} | {cell(best)} | {last['epoch']} | {last['fid_val']['fid']:.2f} | "
                             f"{last['memorisation']['frac_closer_than_val_p5']:.3f} |")
            summary[run] = {"best_epoch": d.get("best_epoch"), "checkpoints": [
                {"tag": r["tag"], "epoch": r["epoch"], "fid_val": r["fid_val"]["fid"], "kid_val": r["fid_val"]["kid_mean"],
                 "frac_memorised": r["memorisation"]["frac_closer_than_val_p5"],
                 "nn_dist_mean": r["memorisation"]["fake_to_train_nn_dist"]["mean"]} for r in sorted(rows, key=lambda r: r["epoch"])]}
        lines.append("")
    if selection_json.exists():
        sel = json.loads(selection_json.read_text())
        lines += [f"## Model selection (validation data only): **{sel['chosen']}**", "", f"Rule: {sel['rule']}.", "",
                  "| candidate | best epoch | FID val ↓ | KID val ×10³ ↓ | memorised ↓ |", "|---|---|---|---|---|"]
        for c in sel["candidates"]:
            mark = " **(chosen)**" if c["run"] == sel["chosen"] else ""
            lines.append(f"| {c['run']}{mark} | {c['best_epoch']} | {c['fid_val']:.2f} | {1000 * c['kid_val']:.2f} | {c['frac_memorised']:.3f} |")
        lines.append("")
        summary["model_selection"] = sel
    return lines, summary


def seg_variant(r: dict) -> str:
    """Secondary-analysis label of a segmentation run ("" for the primary protocol)."""
    a = r["args"]
    if r.get("real_through_vae") or a.get("real_through_vae"):
        return "VAE-reconstructed real"
    if a.get("pretrain_synthetic"):
        return "synthetic pre-training"
    if a.get("synth_ratio") is not None:
        return f"synthetic {a['synth_ratio']:g}:1"
    return ""


def seg_name(key: tuple) -> str:
    frac, synth, only, variant = key
    if only:
        return "synthetic only"
    name = f"{frac:.0%} real"
    if variant == "VAE-reconstructed real":
        return f"{name} (VAE-reconstructed)"
    if variant == "synthetic pre-training":
        return f"{name}, synthetic pre-training"
    if variant:
        return f"{name} + {variant}"
    return name + (" + synthetic" if synth else "")


def segmentation_tables(runs_dir: Path) -> tuple[list[str], dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for p in sorted(runs_dir.glob("seg/*/results.json")):
        r = json.loads(p.read_text())
        a = r["args"]
        # Patient-matched filtering makes the synthetic count seed-dependent, so it is reported, not grouped on.
        key = (float(a.get("real_fraction", 1.0)), bool(a.get("synthetic")), bool(a.get("synthetic_only")), seg_variant(r))
        groups[key].append(r)
    if not groups:
        return [], {}
    lines = ["## Downstream segmentation (per-patient Dice on held-out test patients, mean ± std over seeds; "
             "rows after the primary protocol are the secondary analyses of EXPERIMENTS.md)", "",
             "| training data | real patients | real slices | synthetic slices | seeds | WT | TC | ET | mean |", "|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for key in sorted(groups, key=lambda k: (k[2], k[0], k[3] != "", k[1], k[3])):
        rs = groups[key]
        name = seg_name(key)
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
        for key in sorted(cons, key=lambda k: (k[2], k[0], k[3] != "", k[1], k[3])):
            name = seg_name(key)
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
    p.add_argument("--selection", default="results/model_selection.json")
    args = p.parse_args()
    runs = Path(args.runs)
    g_lines, g_sum = generation_tables(runs)
    c_lines, c_sum = checkpoint_tables(runs, Path(args.selection))
    s_lines, s_sum = segmentation_tables(runs)
    md = ["# Result tables (auto-generated by scripts/collect_results.py)", ""] + g_lines + c_lines + s_lines
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(md))
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"generation": g_sum, "checkpoints": c_sum, "segmentation": s_sum}, indent=2))
    print("\n".join(md))
    print(f"\nwrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
