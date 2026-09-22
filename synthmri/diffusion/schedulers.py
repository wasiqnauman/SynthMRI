from __future__ import annotations

from diffusers import DDIMScheduler, DDPMScheduler

from synthmri.config import DiffusionConfig


def build_noise_scheduler(cfg: DiffusionConfig) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=cfg.prediction_type,
        clip_sample=False,
    )


def build_sampling_scheduler(cfg: DiffusionConfig, sampler: str = "ddim"):
    if sampler == "ddpm":
        return build_noise_scheduler(cfg)
    if sampler == "ddim":
        return DDIMScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            beta_schedule=cfg.beta_schedule,
            prediction_type=cfg.prediction_type,
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
    raise ValueError(f"Unknown sampler {sampler!r} (expected 'ddim' or 'ddpm')")
