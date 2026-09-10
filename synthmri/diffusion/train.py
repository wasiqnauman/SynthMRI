"""Latent diffusion training loop.

Design choices (all configurable in ``TrainConfig`` / ``DiffusionConfig``):

* frozen pretrained VAE; latents cached once (``use_cached_latents``) with horizontal-flip
  augmentation, sampled from the stored posterior every step;
* epsilon- or v-prediction objective with a DDPM noise schedule;
* AdamW + warmup/cosine LR, gradient clipping, bf16 autocast, EMA weights (used for all
  evaluation and sampling);
* mask conditioning with condition dropout for classifier-free guidance;
* validation loss on held-out patients with fixed noise and stratified timesteps, so it is
  comparable across epochs; the lowest-validation-loss EMA model is kept in ``best/``.
"""

from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader

from synthmri import __version__
from synthmri.config import Config
from synthmri.data.dataset import LatentDataset, SliceDataset
from synthmri.diffusion.checkpoint import load_training_state, prune_checkpoints, save_checkpoint
from synthmri.diffusion.conditioning import mask_to_condition
from synthmri.diffusion.latents import cache_latents
from synthmri.diffusion.sample import LatentSampler
from synthmri.diffusion.schedulers import build_noise_scheduler
from synthmri.models.unet import build_unet, count_parameters
from synthmri.models.vae import VAEWrapper, load_vae
from synthmri.utils.io import ensure_dir, git_commit_hash, save_json
from synthmri.utils.logging import CSVLogger, TensorBoardLogger, get_logger
from synthmri.utils.seed import seed_everything
from synthmri.utils.viz import save_sample_sheet


def _get_latents(batch: dict, vae: VAEWrapper | None, device) -> torch.Tensor:
    if "latent" in batch:
        return batch["latent"].to(device, non_blocking=True).float()
    if vae is None:
        raise RuntimeError("Batch has images but no VAE was provided")
    return vae.encode(batch["image"].to(device, non_blocking=True), sample=True)


def _diffusion_loss(unet, noise_scheduler, latents, mask, cfg: Config, accelerator, generator=None, timesteps=None):
    b = latents.shape[0]
    noise = torch.randn(latents.shape, generator=generator, device=latents.device)
    if timesteps is None:
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), generator=generator, device=latents.device)
    noisy = noise_scheduler.add_noise(latents, noise, timesteps)
    if cfg.diffusion.prediction_type == "epsilon":
        target = noise
    elif cfg.diffusion.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(cfg.diffusion.prediction_type)
    model_in = noisy
    if cfg.model.conditioning == "mask":
        drop = torch.rand(b, generator=generator, device=latents.device) < cfg.model.cond_dropout
        cond = mask_to_condition(mask.to(latents.device), cfg.model.num_mask_classes, cfg.latent_size, drop)
        model_in = torch.cat([noisy, cond], dim=1)
    with accelerator.autocast():
        pred = unet(model_in, timesteps).sample
    return F.mse_loss(pred.float(), target.float())


@torch.no_grad()
def validate(unet, noise_scheduler, loader, vae, cfg: Config, accelerator, max_batches: int = 50) -> float:
    """Validation MSE with a fixed generator and timesteps spread evenly over [0, T)."""
    unet.eval()
    device = accelerator.device
    gen = torch.Generator(device=device).manual_seed(12345)
    T = noise_scheduler.config.num_train_timesteps
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        latents = _get_latents(batch, vae, device)
        b = latents.shape[0]
        timesteps = torch.linspace(0, T - 1, b, device=device).round().long()
        loss = _diffusion_loss(unet, noise_scheduler, latents, batch.get("mask"), cfg, accelerator, gen, timesteps)
        losses.append(loss.item())
    unet.train()
    return float(sum(losses) / max(len(losses), 1))


def _build_datasets(cfg: Config, vae: VAEWrapper | None, device, logger):
    d, t = cfg.data, cfg.train
    if t.use_cached_latents:
        if vae is None:
            raise RuntimeError("use_cached_latents requires a VAE")
        files = {}
        for split in ("train", "val"):
            files[split] = cache_latents(
                vae, d.processed_dir, split, d.modalities, cfg.model.vae.pretrained, hflip=t.hflip,
                batch_size=t.batch_size, num_workers=min(t.num_workers, 4), device=device,
            )
            logger.info(f"cached latents [{split}]: {files[split]}")
        train_ds = LatentDataset(files["train"], d.processed_dir, "train", hflip=t.hflip)
        val_ds = LatentDataset(files["val"], d.processed_dir, "val", hflip=False)
        return train_ds, val_ds
    train_ds = SliceDataset(d.processed_dir, "train", d.modalities, hflip=t.hflip)
    val_ds = SliceDataset(d.processed_dir, "val", d.modalities, hflip=False)
    return train_ds, val_ds


def train(cfg: Config, vae: VAEWrapper | None = None) -> Path:
    """Train according to ``cfg``; returns the run directory. ``vae`` may be injected (tests)."""
    out = ensure_dir(cfg.train.output_dir)
    logger = get_logger("synthmri.train", out / "train.log")
    seed_everything(cfg.train.seed)
    accelerator = Accelerator(mixed_precision=cfg.train.mixed_precision, project_dir=str(out))
    device = accelerator.device
    cfg.save(out / "config.yaml")
    save_json(
        {"git_commit": git_commit_hash(), "argv": sys.argv, "synthmri": __version__, "torch": torch.__version__,
         "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        out / "run_info.json",
    )

    if vae is None:
        vae = load_vae(cfg.model.vae.pretrained, device=device, scaling_factor=cfg.model.vae.scaling_factor)
    else:
        vae = vae.to(device)
    train_ds, val_ds = _build_datasets(cfg, vae, device, logger)
    logger.info(f"train slices: {len(train_ds)} | val slices: {len(val_ds)} | latent {cfg.latent_size}x{cfg.latent_size}")
    nw = cfg.train.num_workers
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=nw, pin_memory=True,
                              drop_last=True, persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=min(nw, 2), pin_memory=True)

    unet = build_unet(cfg)
    noise_scheduler = build_noise_scheduler(cfg.diffusion)
    logger.info(f"U-Net parameters: {count_parameters(unet) / 1e6:.1f}M | in_channels={unet.config.in_channels}")
    ema = EMAModel(unet.parameters(), decay=cfg.train.ema_decay, use_ema_warmup=cfg.train.ema_warmup, inv_gamma=1.0, power=0.75)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=cfg.train.lr, betas=(0.9, 0.999), weight_decay=cfg.train.weight_decay, eps=1e-8)
    steps_per_epoch = len(train_loader)
    total_steps = cfg.train.epochs * steps_per_epoch
    if cfg.train.max_steps is not None:
        total_steps = min(total_steps, cfg.train.max_steps)
    lr_scheduler = get_scheduler(cfg.train.lr_scheduler, optimizer, num_warmup_steps=min(cfg.train.warmup_steps, total_steps), num_training_steps=total_steps)

    start_epoch, step = 0, 0
    best_val = math.inf
    if cfg.train.resume:
        state = load_training_state(cfg.train.resume, unet, ema, optimizer, lr_scheduler)
        start_epoch, step = int(state["epoch"]) + 1, int(state["step"])
        best_val = float(state.get("extra", {}).get("best_val", math.inf))
        logger.info(f"resumed from {cfg.train.resume} at epoch {start_epoch}, step {step}")

    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(unet, optimizer, train_loader, lr_scheduler)
    ema.to(device)
    csv_log = CSVLogger(out / "metrics.csv")
    tb = TensorBoardLogger(out / "tb")

    # Fixed conditioning masks for the periodic sample sheet (first val slices).
    n_show = min(cfg.train.num_sample_images, len(val_ds))
    show_masks = torch.stack([val_ds[i]["mask"] for i in range(n_show)]) if n_show else None

    def sample_sheet(epoch: int) -> None:
        raw = accelerator.unwrap_model(unet)
        ema_unet = copy.deepcopy(raw)
        ema.copy_to(ema_unet.parameters())
        sampler = LatentSampler(ema_unet, vae, cfg.diffusion, cfg.latent_size, "ddim", cfg.train.sample_steps,
                                num_mask_classes=cfg.model.num_mask_classes, device=device)
        gen = torch.Generator(device=device).manual_seed(cfg.train.seed)
        cond = None
        if cfg.model.conditioning == "mask":
            cond = mask_to_condition(show_masks, cfg.model.num_mask_classes, cfg.latent_size)
        imgs = sampler.sample_images(n_show, cond=cond, generator=gen)
        save_sample_sheet(imgs, out / "samples" / f"epoch_{epoch:04d}.png", masks=show_masks if cond is not None else None)
        tb.image("samples", imgs[0], epoch)
        del ema_unet, sampler

    logger.info(f"training for {cfg.train.epochs} epochs ({total_steps} steps), batch {cfg.train.batch_size}, lr {cfg.train.lr}")
    done = False
    t_start = time.time()
    for epoch in range(start_epoch, cfg.train.epochs):
        unet.train()
        run_loss, run_n, t_epoch = 0.0, 0, time.time()
        for batch in train_loader:
            latents = _get_latents(batch, vae, device)
            loss = _diffusion_loss(unet, noise_scheduler, latents, batch.get("mask"), cfg, accelerator)
            accelerator.backward(loss)
            if accelerator.sync_gradients and cfg.train.grad_clip > 0:
                accelerator.clip_grad_norm_(unet.parameters(), cfg.train.grad_clip)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ema.step(unet.parameters())
            step += 1
            run_loss += loss.item()
            run_n += 1
            if step % cfg.train.log_every == 0:
                lr = lr_scheduler.get_last_lr()[0]
                mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
                logger.info(f"epoch {epoch + 1} step {step}/{total_steps} loss {run_loss / run_n:.4f} lr {lr:.2e} mem {mem:.1f}GB")
                csv_log.log({"step": step, "epoch": epoch + 1, "train_loss": run_loss / run_n, "lr": lr, "val_loss": ""})
                tb.scalar("train/loss", run_loss / run_n, step)
                tb.scalar("train/lr", lr, step)
            if cfg.train.max_steps is not None and step >= cfg.train.max_steps:
                done = True
                break
        train_loss = run_loss / max(run_n, 1)
        last = done or (epoch + 1 == cfg.train.epochs)

        # Validation with EMA weights.
        val_loss = float("nan")
        if (epoch + 1) % cfg.train.val_every == 0 or last:
            ema.store(unet.parameters())
            ema.copy_to(unet.parameters())
            val_loss = validate(unet, noise_scheduler, val_loader, vae, cfg, accelerator)
            ema.restore(unet.parameters())
            tb.scalar("val/loss", val_loss, step)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(out / "best", accelerator.unwrap_model(unet), ema, step=step, epoch=epoch)
        csv_log.log({"step": step, "epoch": epoch + 1, "train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0], "val_loss": val_loss})
        logger.info(f"=== epoch {epoch + 1} done in {time.time() - t_epoch:.0f}s | train {train_loss:.4f} | val {val_loss:.4f} (best {best_val:.4f})")

        if (epoch + 1) % cfg.train.sample_every == 0 or last:
            sample_sheet(epoch + 1)
        if (epoch + 1) % cfg.train.save_every == 0 or last:
            save_checkpoint(out / "checkpoints" / f"epoch_{epoch + 1:04d}", accelerator.unwrap_model(unet), ema, optimizer,
                            lr_scheduler, step=step, epoch=epoch, extra={"best_val": best_val})
            prune_checkpoints(out / "checkpoints", keep_last=cfg.train.keep_checkpoints)
        if done:
            break

    save_checkpoint(out / "final", accelerator.unwrap_model(unet), ema, step=step, epoch=epoch)
    save_json({"steps": step, "epochs": epoch + 1, "best_val_loss": best_val, "hours": (time.time() - t_start) / 3600}, out / "train_summary.json")
    tb.close()
    logger.info(f"done: {step} steps, best val {best_val:.4f}, weights in {out / 'final'}")
    return out
