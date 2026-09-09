import dataclasses

import pytest
import torch

from synthmri.diffusion.conditioning import mask_to_condition, null_condition
from synthmri.diffusion.schedulers import build_noise_scheduler, build_sampling_scheduler
from synthmri.models import DummyVAE, VAEWrapper, build_unet, count_parameters
from synthmri.models.unet import condition_channels


def test_unet_shapes_unconditional(tiny_cfg):
    unet = build_unet(tiny_cfg)
    x = torch.randn(2, 4, tiny_cfg.latent_size, tiny_cfg.latent_size)
    out = unet(x, torch.tensor([1, 999])).sample
    assert out.shape == x.shape
    assert unet.config.in_channels == 4
    assert count_parameters(unet) > 0
    assert unet.config.down_block_types == ("DownBlock2D", "AttnDownBlock2D")
    assert unet.config.up_block_types == ("AttnUpBlock2D", "UpBlock2D")


def test_unet_shapes_mask_conditioned(tiny_cfg):
    cfg = dataclasses.replace(tiny_cfg, model=dataclasses.replace(tiny_cfg.model, conditioning="mask"))
    unet = build_unet(cfg)
    assert condition_channels(cfg) == 5 and unet.config.in_channels == 9
    x = torch.randn(2, 9, cfg.latent_size, cfg.latent_size)
    assert unet(x, torch.tensor([5, 5])).sample.shape == (2, 4, cfg.latent_size, cfg.latent_size)


def test_unet_bad_attention_level(tiny_cfg):
    cfg = dataclasses.replace(tiny_cfg, model=dataclasses.replace(tiny_cfg.model, unet=dataclasses.replace(tiny_cfg.model.unet, attention_levels=(5,))))
    with pytest.raises(ValueError):
        build_unet(cfg)


def test_dummy_vae_roundtrip():
    vae = VAEWrapper(DummyVAE(), scaling_factor=None)
    x = torch.zeros(1, 3, 32, 32)
    x[:, :, 8:16, 8:24] = 0.5  # aligned to the 8x8 latent grid -> lossless
    z = vae.encode(x, sample=False)
    assert z.shape == (1, 4, 4, 4) and vae.scaling_factor == 1.0
    assert torch.allclose(vae.decode(z), x)
    mean, logvar = vae.encode_dist(x)
    assert mean.shape == logvar.shape == (1, 4, 4, 4)


def test_mask_conditioning():
    mask = torch.zeros(2, 16, 16, dtype=torch.long)
    mask[0, :8, :] = 2
    mask[1, 4:8, 4:8] = 3
    cond = mask_to_condition(mask, num_classes=4, latent_size=2)
    assert cond.shape == (2, 5, 2, 2)
    assert torch.allclose(cond[:, :4].sum(1), torch.ones(2, 2, 2))  # class fractions sum to 1
    assert cond[:, 4].sum() == 0  # nothing dropped
    assert cond[0, 2, 0, :].tolist() == [1.0, 1.0] and cond[0, 2, 1, :].tolist() == [0.0, 0.0]
    dropped = mask_to_condition(mask, 4, 2, drop=torch.tensor([True, False]))
    assert dropped[0, :4].sum() == 0 and dropped[0, 4].min() == 1
    assert torch.equal(dropped[1], cond[1])
    null = null_condition(3, 4, 2)
    assert null.shape == (3, 5, 2, 2) and null[:, 4].min() == 1 and null[:, :4].abs().sum() == 0
    with pytest.raises(ValueError):
        mask_to_condition(mask[0], 4, 2)


def test_schedulers(tiny_cfg):
    ddpm = build_noise_scheduler(tiny_cfg.diffusion)
    assert ddpm.config.num_train_timesteps == 1000 and ddpm.config.prediction_type == "epsilon"
    ddim = build_sampling_scheduler(tiny_cfg.diffusion, "ddim")
    ddim.set_timesteps(5)
    assert len(ddim.timesteps) == 5
    with pytest.raises(ValueError):
        build_sampling_scheduler(tiny_cfg.diffusion, "euler")
