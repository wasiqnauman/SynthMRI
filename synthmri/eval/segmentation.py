"""Downstream utility: does synthetic (image, mask) data help a tumour segmenter?

A 2-D U-Net (MONAI) is trained on real slices only, on real + synthetic, or on synthetic
only, and evaluated on held-out *patients* with the standard BraTS regions (WT / TC / ET).
Dice is aggregated per patient over that patient's slices before averaging across patients,
which is closer to the usual volumetric protocol than a per-slice mean.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from synthmri.data.brats import REGIONS
from synthmri.utils.io import save_json
from synthmri.utils.logging import CSVLogger, get_logger
from synthmri.utils.seed import seed_everything


@dataclass
class SegConfig:
    epochs: int = 40
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_classes: int = 4
    in_channels: int = 3
    channels: tuple[int, ...] = (32, 64, 128, 256, 512)
    seed: int = 0
    amp: bool = True
    hflip: bool = True
    num_workers: int = 4
    max_steps: int | None = None


class ArraySliceDataset(Dataset):
    """(N,C,S,S) images in [0,1] + (N,S,S) integer masks (NumPy, possibly memory-mapped)."""

    def __init__(self, images, masks, hflip: bool = False):
        if images.shape[0] != masks.shape[0]:
            raise ValueError("images and masks must have the same length")
        self.images, self.masks, self.hflip = images, masks, hflip

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, i: int) -> dict:
        x = torch.from_numpy(np.ascontiguousarray(self.images[i]).astype(np.float32))
        y = torch.from_numpy(np.ascontiguousarray(self.masks[i]).astype(np.int64))
        if self.hflip and torch.rand(()) < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        return {"image": x, "mask": y}


def build_seg_model(cfg: SegConfig):
    from monai.networks.nets import UNet

    return UNet(
        spatial_dims=2,
        in_channels=cfg.in_channels,
        out_channels=cfg.num_classes,
        channels=tuple(cfg.channels),
        strides=(2,) * (len(cfg.channels) - 1),
        num_res_units=2,
        norm="instance",
    )


def region_counts(pred: np.ndarray, true: np.ndarray) -> dict[str, tuple[float, float]]:
    """Per region: (2 * |P ∩ T|, |P| + |T|) so Dice can be aggregated over slices."""
    out = {}
    for name, labels in REGIONS.items():
        p = np.isin(pred, labels)
        t = np.isin(true, labels)
        out[name] = (2.0 * float((p & t).sum()), float(p.sum() + t.sum()))
    return out


def dice_from_counts(num: float, den: float) -> float:
    return 1.0 if den == 0 else num / den


@torch.no_grad()
def predict_labels(model, images, device, batch_size: int = 64, amp: bool = True) -> np.ndarray:
    model.eval()
    preds = []
    for s in range(0, images.shape[0], batch_size):
        x = torch.from_numpy(np.ascontiguousarray(images[s : s + batch_size]).astype(np.float32)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            logits = model(x)
        preds.append(logits.argmax(1).cpu().numpy().astype(np.uint8))
    return np.concatenate(preds)


@torch.no_grad()
def evaluate_segmenter(model, images, masks, patient_ids: np.ndarray, device, batch_size: int = 64) -> dict:
    """Per-patient region Dice (mean/std across patients) plus per-slice means."""
    pred = predict_labels(model, images, device, batch_size)
    per_patient: dict[str, dict[str, list[float]]] = {}
    slice_dice = {r: [] for r in REGIONS}
    for i, pid in enumerate(patient_ids):
        c = region_counts(pred[i], np.asarray(masks[i]))
        d = per_patient.setdefault(str(pid), {r: [0.0, 0.0] for r in REGIONS})
        for r, (num, den) in c.items():
            d[r][0] += num
            d[r][1] += den
            slice_dice[r].append(dice_from_counts(num, den))
    result = {"n_patients": len(per_patient), "n_slices": int(images.shape[0]), "patient": {}, "slice": {}}
    for r in REGIONS:
        vals = [dice_from_counts(*per_patient[p][r]) for p in per_patient]
        result["patient"][r] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
        result["slice"][r] = {"mean": float(np.mean(slice_dice[r])), "std": float(np.std(slice_dice[r]))}
    result["patient"]["mean_WT_TC_ET"] = float(np.mean([result["patient"][r]["mean"] for r in REGIONS]))
    return result


def train_segmenter(
    train_images, train_masks, val_images, val_masks, val_patient_ids, cfg: SegConfig, out_dir: str | Path, device=None,
    init_state: dict | None = None,
):
    """Train and return (model, history); the best-val-Dice weights are restored at the end.
    ``init_state`` (a state dict) starts from previously trained weights, e.g. after synthetic pre-training."""
    from monai.losses import DiceCELoss

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)
    logger = get_logger(f"synthmri.seg.{out_dir.name}", out_dir / "train.log")
    save_json(asdict(cfg), out_dir / "seg_config.json")
    model = build_seg_model(cfg).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    ds = ArraySliceDataset(train_images, train_masks, hflip=cfg.hflip)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=cfg.num_workers > 0)
    loss_fn = DiceCELoss(softmax=True, to_onehot_y=True, include_background=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(loader) if cfg.max_steps is None else min(cfg.max_steps, cfg.epochs * len(loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(total, 1), pct_start=0.1)
    csv = CSVLogger(out_dir / "metrics.csv")
    best, best_state, step, history = -math.inf, None, 0, []
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        run, n = 0.0, 0
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True).unsqueeze(1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda"):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step < total:
                sched.step()
            step += 1
            run += loss.item()
            n += 1
            if cfg.max_steps is not None and step >= cfg.max_steps:
                break
        val = evaluate_segmenter(model, val_images, val_masks, val_patient_ids, device)
        score = val["patient"]["mean_WT_TC_ET"]
        row = {"epoch": epoch + 1, "step": step, "train_loss": run / max(n, 1), "val_mean_dice": score,
               **{f"val_{r}": val["patient"][r]["mean"] for r in REGIONS}}
        history.append(row)
        csv.log(row)
        logger.info(f"epoch {epoch + 1}/{cfg.epochs} loss {row['train_loss']:.4f} val mean Dice {score:.4f} "
                    f"(WT {row['val_WT']:.3f} TC {row['val_TC']:.3f} ET {row['val_ET']:.3f})")
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if cfg.max_steps is not None and step >= cfg.max_steps:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "segmenter.pt")
    logger.info(f"best val mean Dice {best:.4f} in {(time.time() - t0) / 60:.1f} min")
    return model, history


def subsample_patients(patient_ids: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Boolean row mask keeping a random ``fraction`` of the patients (all of their slices)."""
    uniq = np.unique(patient_ids)
    rng = np.random.RandomState(seed)
    keep = set(rng.choice(uniq, size=max(1, int(round(fraction * len(uniq)))), replace=False).tolist())
    return np.isin(patient_ids, list(keep))


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:  # small helper kept for tests
    return F.softmax(logits, dim=1)
