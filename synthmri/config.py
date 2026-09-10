"""Typed, YAML-backed experiment configuration.

A config file may contain a ``_base_`` key pointing at another YAML file (relative to
the file itself); the child is deep-merged on top of the base. Command-line overrides use
dotted keys, e.g. ``train.epochs=5 data.image_size=64``; values are parsed as YAML.
Unknown keys raise immediately so typos cannot silently be ignored.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml


@dataclass
class DataConfig:
    raw_dir: str = "data/raw/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    processed_dir: str = "data/processed/brats128"
    splits_file: str = "splits/brats2020_patient_splits.json"
    image_size: int = 128
    crop_size: int | None = 224  # centre crop applied before resizing (None = no crop)
    modalities: tuple[str, ...] = ("flair", "t1ce", "t2")  # channels fed to the VAE (must be 3)
    tumour_frac_min: float = 0.001  # keep slices whose tumour area >= this fraction of the slice
    val_frac: float = 0.1
    test_frac: float = 0.2
    split_seed: int = 0
    clip_percentiles: tuple[float, float] = (0.5, 99.5)  # per-volume intensity clipping (brain voxels)
    num_workers: int = 8


@dataclass
class VAEConfig:
    pretrained: str = "stabilityai/sd-vae-ft-mse"
    scaling_factor: float | None = None  # None -> vae.config.scaling_factor


@dataclass
class UNetConfig:
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    attention_levels: tuple[int, ...] = (2,)  # 0-based resolution levels that get self-attention
    attention_head_dim: int = 8
    dropout: float = 0.0
    norm_num_groups: int = 32


@dataclass
class ModelConfig:
    vae: VAEConfig = field(default_factory=VAEConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    conditioning: str = "none"  # "none" | "mask"
    num_mask_classes: int = 4  # background, NCR/NET, ED, ET
    cond_dropout: float = 0.1  # probability of dropping the condition (classifier-free guidance)


@dataclass
class DiffusionConfig:
    num_train_timesteps: int = 1000
    beta_schedule: str = "linear"  # "linear" | "scaled_linear" | "squaredcos_cap_v2"
    prediction_type: str = "epsilon"  # "epsilon" | "v_prediction"


@dataclass
class TrainConfig:
    output_dir: str = "runs/ldm128_uncond"
    seed: int = 0
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine"
    warmup_steps: int = 500
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    ema_warmup: bool = True
    mixed_precision: str = "bf16"  # "no" | "fp16" | "bf16"
    use_cached_latents: bool = True  # encode the dataset once with the frozen VAE
    hflip: bool = True  # random horizontal flip (cached for both orientations)
    val_every: int = 1  # epochs
    sample_every: int = 10  # epochs
    save_every: int = 10  # epochs
    keep_checkpoints: int = 3  # rolling epoch checkpoints kept under checkpoints/; 0 = keep all
    num_sample_images: int = 16
    sample_steps: int = 50
    log_every: int = 50  # steps
    num_workers: int = 8
    resume: str | None = None  # path to a checkpoint directory
    max_steps: int | None = None  # stop early (smoke tests)


@dataclass
class SampleConfig:
    checkpoint: str = "best"  # "best" (lowest validation loss), "final", an epoch number or a checkpoint dir
    output_dir: str = ""  # default: <run>/samples
    num_images: int = 1000
    batch_size: int = 64
    sampler: str = "ddim"  # "ddim" | "ddpm"
    steps: int = 50
    guidance_scale: float = 1.0  # >1 enables classifier-free guidance (mask-conditioned only)
    eta: float = 0.0
    seed: int = 0
    use_ema: bool = True
    mask_source: str = "train"  # split whose masks condition the samples (mask-conditioned only)


@dataclass
class Config:
    name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)

    @property
    def latent_size(self) -> int:
        return self.data.image_size // 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


# ----------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml_with_base(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base_ref = raw.pop("_base_", None)
    if base_ref is None:
        return raw
    base_path = (path.parent / base_ref).resolve()
    return _deep_merge(_read_yaml_with_base(base_path), raw)


def _coerce(value: Any, annotation: Any) -> Any:
    """Coerce YAML values to the dataclass field annotation (tuples, optionals, nested)."""
    origin = get_origin(annotation)
    if is_dataclass(annotation):
        return _from_dict(annotation, value or {})
    if origin is tuple:
        if value is None:
            return None
        return tuple(value)
    if origin is not None and str(origin) in ("types.UnionType", "typing.Union"):
        if value is None:
            return None
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            return _coerce(value, arg)
    # YAML 1.1 reads "2e-5" (no dot) as a string; coerce numeric strings to the declared type.
    if annotation is float and isinstance(value, (str, int)) and not isinstance(value, bool):
        return float(value)
    if annotation is int and isinstance(value, str):
        return int(value)
    if annotation is bool and isinstance(value, str):
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        raise ValueError(f"Cannot interpret {value!r} as a boolean")
    return value


def _from_dict(cls: type, d: dict) -> Any:
    if not isinstance(d, dict):
        raise TypeError(f"Expected a mapping for {cls.__name__}, got {type(d).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise KeyError(f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    # Resolve string annotations (``from __future__ import annotations``).
    hints = _type_hints(cls)
    for name in known:
        if name in d:
            kwargs[name] = _coerce(d[name], hints[name])
    return cls(**kwargs)


def _type_hints(cls: type) -> dict[str, Any]:
    import typing

    return typing.get_type_hints(cls)


def _apply_overrides(d: dict, overrides: list[str] | None) -> dict:
    out = copy.deepcopy(d)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, val = item.split("=", 1)
        parsed = yaml.safe_load(val)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise KeyError(f"Cannot set {key}: {p} is not a mapping")
        node[parts[-1]] = parsed
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a YAML config (with ``_base_`` inheritance) and apply dotted overrides."""
    d = _read_yaml_with_base(Path(path)) if path is not None else {}
    d = _apply_overrides(d, overrides)
    return _from_dict(Config, d)


def config_from_dict(d: dict) -> Config:
    return _from_dict(Config, d)


def replace(cfg: Config, **changes: Any) -> Config:
    return dataclasses.replace(cfg, **changes)
