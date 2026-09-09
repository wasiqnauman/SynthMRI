#!/usr/bin/env python
"""Paper figures from run directories -> docs/figures/ (PNG, 200 dpi).

    python scripts/make_figures.py [--runs runs] [--out docs/figures]

* <run>_loss.png                      training / validation diffusion loss per epoch
* <run>_<samples>_real_vs_synth.png   real test slices vs generated slices, one column per modality
* segmentation_dice.png               downstream Dice by real-data fraction, real vs real+synthetic
Missing inputs are skipped, so the script can be run at any point of the experiment set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from synthmri.config import load_config  # noqa: E402
from synthmri.data.dataset import SliceDataset  # noqa: E402
from synthmri.utils.viz import colorize_mask  # noqa: E402

# Fixed categorical order (colour-blind-validated): blue, orange, aqua, yellow.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def loss_curves(run: Path, out: Path) -> None:
    csv = run / "metrics.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    ep = df[df["val_loss"].notna() & (df["val_loss"] != "")].copy()
    if ep.empty:
        return
    ep["val_loss"] = ep["val_loss"].astype(float)
    fig, ax = plt.subplots(figsize=(4.2, 2.8), dpi=200)
    ax.plot(ep["epoch"], ep["train_loss"], color=SERIES[0], linewidth=1.6)
    ax.plot(ep["epoch"], ep["val_loss"], color=SERIES[1], linewidth=1.6)
    ax.text(ep["epoch"].iloc[-1], ep["train_loss"].iloc[-1], "  train", color=SERIES[0], fontsize=8, va="center")
    ax.text(ep["epoch"].iloc[-1], ep["val_loss"].iloc[-1], "  val (EMA)", color=SERIES[1], fontsize=8, va="center")
    ax.set_xlabel("epoch", color=INK2, fontsize=8)
    ax.set_ylabel("diffusion MSE", color=INK2, fontsize=8)
    ax.set_title(run.name, color=INK, fontsize=9, loc="left")
    _style(ax)
    ax.set_xlim(0, ep["epoch"].iloc[-1] * 1.18)
    fig.tight_layout()
    fig.savefig(out / f"{run.name}_loss.png")
    plt.close(fig)


def real_vs_synth(run: Path, samples: Path, out: Path, n: int = 4) -> None:
    if not (samples / "images.npy").exists():
        return
    cfg = load_config(run / "config.yaml")
    mods = list(cfg.data.modalities)
    ds = SliceDataset(cfg.data.processed_dir, "test", tuple(mods), hflip=False)
    rng = np.random.RandomState(0)
    ridx = rng.choice(len(ds), n, replace=False)
    real = np.stack([ds.images[i][ds.mod_idx] for i in ridx]).astype(np.float32)
    real_m = np.stack([ds.masks[i] for i in ridx])
    fake = np.load(samples / "images.npy")[:n].astype(np.float32)
    fake_m = np.load(samples / "masks.npy")[:n] if (samples / "masks.npy").exists() else None
    ncol = len(mods) + 1
    fig, axes = plt.subplots(2 * n, ncol, figsize=(1.35 * ncol, 1.35 * 2 * n), dpi=200)
    for r in range(2 * n):
        img = real[r] if r < n else fake[r - n]
        msk = real_m[r] if r < n else (fake_m[r - n] if fake_m is not None else None)
        for c in range(len(mods)):
            axes[r, c].imshow(img[c], cmap="gray", vmin=0, vmax=1)
        if msk is not None:
            axes[r, ncol - 1].imshow(colorize_mask(msk).permute(1, 2, 0).numpy())
        for c in range(ncol):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for sp in axes[r, c].spines.values():
                sp.set_visible(False)
        axes[r, 0].set_ylabel("real" if r < n else "synthetic", color=INK2, fontsize=7)
    for c, m in enumerate(mods + ["mask"]):
        axes[0, c].set_title(m.upper() if c < len(mods) else ("mask" if fake_m is not None else "real mask"), color=INK, fontsize=8)
    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.06, right=0.99, top=0.95, bottom=0.01)
    fig.savefig(out / f"{run.name}_{samples.name}_real_vs_synth.png")
    plt.close(fig)


def segmentation_figure(summary_json: Path, out: Path) -> None:
    if not summary_json.exists():
        return
    seg = json.loads(summary_json.read_text()).get("segmentation", {})
    fracs = [k for k in ("10% real", "25% real", "100% real") if k in seg]
    if not fracs:
        return
    regions = ["WT", "TC", "ET"]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.7), dpi=200, sharey=True)
    x = np.arange(len(fracs))
    w = 0.36
    for ax, r in zip(axes, regions):
        for j, (label, suffix, col) in enumerate((("real only", "", SERIES[0]), ("real + synthetic", " + synthetic", SERIES[1]))):
            means = [seg.get(f + suffix, {}).get(r, {}).get("mean", np.nan) for f in fracs]
            stds = [seg.get(f + suffix, {}).get(r, {}).get("std", 0.0) for f in fracs]
            bars = ax.bar(x + (j - 0.5) * w, means, w - 0.04, color=col, yerr=stds, error_kw={"elinewidth": 0.8, "ecolor": INK2, "capsize": 2}, label=label)
            for b, m_ in zip(bars, means):
                if np.isfinite(m_):
                    ax.text(b.get_x() + b.get_width() / 2, 0.02, f"{m_:.3f}", ha="center", va="bottom", fontsize=6, color="white", rotation=90)
        if "synthetic only" in seg:
            v = seg["synthetic only"].get(r, {}).get("mean", np.nan)
            ax.axhline(v, color=SERIES[2], linewidth=1.2, linestyle=(0, (4, 3)))
            ax.text(len(fracs) - 0.5, v, " synthetic only", color=SERIES[2], fontsize=7, va="bottom", ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace(" real", "") for f in fracs])
        ax.set_title(f"{r} Dice", color=INK, fontsize=9, loc="left")
        ax.set_ylim(0, 1)
        _style(ax)
    axes[0].set_ylabel("per-patient Dice (test)", color=INK2, fontsize=8)
    axes[0].set_xlabel("real training patients", color=INK2, fontsize=8)
    axes[-1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "segmentation_dice.png")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="docs/figures")
    p.add_argument("--summary", default="results/summary.json")
    args = p.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for run in sorted(p_ for p_ in runs.iterdir() if (p_ / "config.yaml").exists()):
        if run.name == "smoke":
            continue
        loss_curves(run, out)
        for samples in sorted(run.glob("samples_*")):
            real_vs_synth(run, samples, out)
    segmentation_figure(Path(args.summary), out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
