"""Fine-tune the *decoder* of the pretrained VAE on the target slices (encoder untouched).

The Stable Diffusion VAE was trained on natural images, and its decoder bounds the sharpness of
every latent-diffusion sample (the "VAE ceiling" reported by ``synthmri.eval.recon``). Training
only ``decoder`` + ``post_quant_conv`` with an L1 + LPIPS loss on the training slices raises that
ceiling while leaving the latent space, and therefore every cached latent and every trained
diffusion model, unchanged: existing samples are simply decoded again with the new decoder
(``scripts/sample.py --vae_decoder``).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from synthmri.models.vae import DECODER_KEYS, VAEWrapper, cudnn_heuristics, decoder_state_dict
from synthmri.utils.io import save_json
from synthmri.utils.logging import CSVLogger, get_logger
from synthmri.utils.seed import seed_everything


@dataclass
class DecoderFinetuneConfig:
    epochs: int = 8
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.0
    lpips_weight: float = 0.5  # 0 -> plain L1
    lpips_net: str = "vgg"  # training perceptual loss; the ceiling metric in eval uses AlexNet-LPIPS
    warmup_steps: int = 100
    grad_clip: float = 1.0
    seed: int = 0
    num_workers: int = 4
    max_steps: int | None = None  # for smoke tests
    max_val_items: int | None = None


def trainable_decoder_parameters(vae_module: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for name in DECODER_KEYS:
        mod = getattr(vae_module, name, None)
        if mod is not None:
            params += list(mod.parameters())
    return params


def _lpips(net: str, device):
    try:
        import lpips
    except Exception:  # pragma: no cover - optional
        return None
    m = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    m.requires_grad_(False)
    return m


def _perceptual(lp, xr: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mean LPIPS over the image channels, each grey channel replicated to RGB (as in the ceiling metric)."""
    vals = [lp(xr[:, k : k + 1].repeat(1, 3, 1, 1), x[:, k : k + 1].repeat(1, 3, 1, 1)).mean() for k in range(x.shape[1])]
    return torch.stack(vals).mean()


def _recon_loss(model: nn.Module, x: torch.Tensor, lp, lpips_weight: float) -> tuple[torch.Tensor, dict]:
    with torch.no_grad():
        z = model.encode(x).latent_dist.mode()  # unscaled posterior mean; the encoder is frozen
    xr = model.decode(z).sample
    l1 = (xr - x).abs().mean()
    loss = l1
    parts = {"l1": float(l1)}
    if lp is not None and lpips_weight > 0:
        pl = _perceptual(lp, xr.clamp(-1, 1), x)
        loss = loss + lpips_weight * pl
        parts["lpips"] = float(pl)
    mse = ((xr.clamp(-1, 1) - x) / 2).pow(2).mean(dim=(1, 2, 3))  # in [0,1] units
    parts["psnr"] = float((-10 * torch.log10(mse.clamp_min(1e-10))).mean())
    return loss, parts


@torch.no_grad()
def validate_decoder(model: nn.Module, loader: DataLoader, lp, lpips_weight: float, device, max_items: int | None = None) -> dict:
    sums: dict[str, float] = {}
    n = 0
    for batch in loader:
        x = batch["image"].to(device)
        loss, parts = _recon_loss(model, x, lp, lpips_weight)
        parts["loss"] = float(loss)
        b = x.shape[0]
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + v * b
        n += b
        if max_items is not None and n >= max_items:
            break
    return {k: v / max(n, 1) for k, v in sums.items()} | {"n": n}


def finetune_decoder(vae: VAEWrapper, train_ds: Dataset, val_ds: Dataset, cfg: DecoderFinetuneConfig, out_dir: str | Path, device=None) -> dict:
    """Train ``decoder`` + ``post_quant_conv`` of ``vae.vae``; keep the epoch with the lowest validation
    loss (L1 + w·LPIPS, computed with the training perceptual net) and leave those weights loaded.
    Writes ``decoder.pt`` (state of the fine-tuned parts + info), ``metrics.csv`` and ``history.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)
    logger = get_logger(f"synthmri.vae_finetune.{out_dir.name}", out_dir / "train.log")
    save_json(asdict(cfg), out_dir / "finetune_config.json")

    model = vae.to(device).vae
    model.requires_grad_(False)
    params = trainable_decoder_parameters(model)
    if not params:
        raise ValueError("the VAE has no trainable decoder parameters (decoder / post_quant_conv)")
    for p in params:
        p.requires_grad_(True)
    n_params = sum(p.numel() for p in params)
    lp = _lpips(cfg.lpips_net, device) if cfg.lpips_weight > 0 else None
    if cfg.lpips_weight > 0 and lp is None:
        logger.warning("lpips is not installed; training with L1 only")

    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                        drop_last=True, persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=min(cfg.num_workers, 2))
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(loader) if cfg.max_steps is None else min(cfg.max_steps, cfg.epochs * len(loader))
    warm = max(1, min(cfg.warmup_steps, total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / max(total, 1))))
    csv = CSVLogger(out_dir / "metrics.csv")

    history = []
    with cudnn_heuristics():
        val0 = validate_decoder(model, val_loader, lp, cfg.lpips_weight, device, cfg.max_val_items)
    history.append({"epoch": 0, "step": 0, "train_loss": float("nan"), **{f"val_{k}": v for k, v in val0.items()}})
    csv.log(history[-1])
    logger.info(f"fine-tuning {n_params / 1e6:.1f}M decoder parameters on {len(train_ds)} slices; before: val {val0}")
    best, best_epoch, best_state = val0["loss"], 0, decoder_state_dict(model)
    step, t0 = 0, time.time()
    done = False
    with cudnn_heuristics():
        for epoch in range(1, cfg.epochs + 1):
            run, n = 0.0, 0
            for batch in loader:
                x = batch["image"].to(device, non_blocking=True)
                loss, _ = _recon_loss(model, x, lp, cfg.lpips_weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                sched.step()
                step += 1
                run += float(loss)
                n += 1
                if cfg.max_steps is not None and step >= cfg.max_steps:
                    done = True
                    break
            val = validate_decoder(model, val_loader, lp, cfg.lpips_weight, device, cfg.max_val_items)
            row = {"epoch": epoch, "step": step, "train_loss": run / max(n, 1), **{f"val_{k}": v for k, v in val.items()}}
            history.append(row)
            csv.log(row)
            logger.info(f"epoch {epoch}/{cfg.epochs} train {row['train_loss']:.4f} | val loss {val['loss']:.4f} l1 {val['l1']:.4f} "
                        f"psnr {val['psnr']:.2f}" + (f" lpips {val['lpips']:.4f}" if "lpips" in val else "") + f" ({(time.time() - t0) / 60:.1f} min)")
            if val["loss"] < best:
                best, best_epoch, best_state = val["loss"], epoch, decoder_state_dict(model)
            if done:
                break
    model.load_state_dict(best_state, strict=False)
    for p in params:
        p.requires_grad_(False)
    info = {"best_epoch": best_epoch, "best_val_loss": best, "epochs_run": history[-1]["epoch"], "steps": step, "config": asdict(cfg),
            "n_trainable_params": n_params, "minutes": (time.time() - t0) / 60}
    torch.save({"state": best_state, "info": info}, out_dir / "decoder.pt")
    save_json({"history": history, **info}, out_dir / "history.json")
    logger.info(f"best epoch {best_epoch} (val loss {best:.4f}); wrote {out_dir / 'decoder.pt'}")
    return {"history": history, **info}
