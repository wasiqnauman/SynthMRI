"""Checkpoint layout and loading helpers.

A run directory looks like::

    runs/<name>/
      config.yaml, run_info.json, train.log, metrics.csv, tb/
      samples/epoch_0010.png ...
      checkpoints/epoch_0010/{unet, unet_ema, training_state.pt}   (rolling, last k kept)
      best/unet_ema                                                 (lowest validation loss)
      final/{unet, unet_ema}

``resolve_unet_dir`` accepts any of: a run dir (-> final, else best, else latest checkpoint),
a checkpoint dir, or a directory that itself contains ``config.json``.
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import torch
from diffusers import UNet2DModel
from diffusers.training_utils import EMAModel

from synthmri.config import Config, load_config


def save_checkpoint(
    ckpt_dir: str | Path,
    unet: UNet2DModel,
    ema: EMAModel | None,
    optimizer=None,
    lr_scheduler=None,
    step: int = 0,
    epoch: int = 0,
    extra: dict | None = None,
) -> Path:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(ckpt_dir / "unet")
    if ema is not None:
        ema_unet = copy.deepcopy(unet)
        ema.copy_to(ema_unet.parameters())
        ema_unet.save_pretrained(ckpt_dir / "unet_ema")
        del ema_unet
    state = {"step": step, "epoch": epoch, "extra": extra or {}}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if lr_scheduler is not None:
        state["lr_scheduler"] = lr_scheduler.state_dict()
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, ckpt_dir / "training_state.pt")
    return ckpt_dir


def load_training_state(ckpt_dir: str | Path, unet: UNet2DModel, ema=None, optimizer=None, lr_scheduler=None) -> dict:
    ckpt_dir = Path(ckpt_dir)
    weights = UNet2DModel.from_pretrained(ckpt_dir / "unet")
    unet.load_state_dict(weights.state_dict())
    state = torch.load(ckpt_dir / "training_state.pt", map_location="cpu", weights_only=False)
    if ema is not None and "ema" in state:
        ema.load_state_dict(state["ema"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if lr_scheduler is not None and "lr_scheduler" in state:
        lr_scheduler.load_state_dict(state["lr_scheduler"])
    return state


def prune_checkpoints(checkpoints_dir: str | Path, keep_last: int) -> None:
    d = Path(checkpoints_dir)
    if not d.exists() or keep_last <= 0:
        return
    ckpts = sorted(p for p in d.iterdir() if p.is_dir())
    for p in ckpts[:-keep_last]:
        shutil.rmtree(p, ignore_errors=True)


def resolve_unet_dir(path: str | Path, use_ema: bool = True) -> Path:
    p = Path(path)
    sub = "unet_ema" if use_ema else "unet"
    if (p / "config.json").exists():
        return p
    if (p / sub).exists():
        return p / sub
    if (p / "unet").exists():  # checkpoint without EMA
        return p / "unet"
    for cand in ("final", "best"):
        if (p / cand / sub).exists():
            return p / cand / sub
        if (p / cand / "unet").exists():
            return p / cand / "unet"
    ckpts = sorted(q for q in (p / "checkpoints").glob("*") if q.is_dir()) if (p / "checkpoints").exists() else []
    if ckpts:
        return resolve_unet_dir(ckpts[-1], use_ema)
    raise FileNotFoundError(f"No U-Net weights found under {p}")


def resolve_checkpoint(run: str | Path, name: str = "best") -> tuple[Path, str]:
    """Map a run directory and a checkpoint name to ``(checkpoint_dir, tag)``.

    ``name`` is ``"best"`` (lowest validation loss; the default used for every reported sample set),
    ``"final"`` (last epoch), an epoch number such as ``"150"`` (looked up under ``checkpoints/`` and
    ``checkpoints_keep/``) or an explicit directory. If ``run`` already is a checkpoint directory it
    is returned unchanged with its own name as the tag.
    """
    run = Path(run)
    name = name or "best"
    if (run / "unet").exists() or (run / "unet_ema").exists() or (run / "config.json").exists():
        return run, run.name
    if name in ("best", "final"):
        d, tag = run / name, name
    elif name.isdigit():
        tag = f"ep{int(name):04d}"
        d = run / "checkpoints" / f"epoch_{int(name):04d}"
        if not d.exists():
            d = run / "checkpoints_keep" / f"epoch_{int(name):04d}"
    else:
        d, tag = Path(name), Path(name).name
    if not d.exists():
        raise FileNotFoundError(f"checkpoint {name!r} not found under {run}")
    return d, tag


def find_run_dir(path: str | Path) -> Path:
    """Walk up from ``path`` until a directory containing ``config.yaml`` is found."""
    p = Path(path).resolve()
    for cand in (p, *p.parents):
        if (cand / "config.yaml").exists():
            return cand
    raise FileNotFoundError(f"No config.yaml found above {path}")


def load_run(path: str | Path, use_ema: bool = True, device=None) -> tuple[Config, UNet2DModel, Path]:
    run_dir = find_run_dir(path)
    cfg = load_config(run_dir / "config.yaml")
    unet_dir = resolve_unet_dir(path, use_ema=use_ema)
    unet = UNet2DModel.from_pretrained(unet_dir)
    unet.eval()
    if device is not None:
        unet.to(device)
    return cfg, unet, unet_dir
