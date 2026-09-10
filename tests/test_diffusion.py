import dataclasses

import numpy as np
import pytest
import torch

from synthmri.data.dataset import LatentDataset
from synthmri.diffusion.checkpoint import find_run_dir, load_run, resolve_checkpoint, resolve_unet_dir
from synthmri.diffusion.conditioning import mask_to_condition
from synthmri.diffusion.latents import cache_latents
from synthmri.diffusion.sample import LatentSampler
from synthmri.diffusion.train import train
from synthmri.models import DummyVAE, VAEWrapper, build_unet


def test_cache_latents_and_latent_dataset(processed_dir, tiny_cfg):
    vae = VAEWrapper(DummyVAE())
    path = cache_latents(vae, processed_dir, "train", tiny_cfg.data.modalities, "dummy", hflip=True, batch_size=4, num_workers=0, device="cpu", show_progress=False)
    z = np.load(path)
    n = np.load(processed_dir / "train" / "masks.npy").shape[0]
    assert z["mean"].shape == (n, 2, 4, tiny_cfg.latent_size, tiny_cfg.latent_size) and z["mean"].dtype == np.float16
    # the flipped variant's mean is the horizontal flip of the original's
    assert np.allclose(z["mean"][:, 1], z["mean"][:, 0][..., ::-1], atol=1e-3)
    ds = LatentDataset(path, processed_dir, "train", hflip=False)
    item = ds[0]
    assert item["latent"].shape == (4, tiny_cfg.latent_size, tiny_cfg.latent_size)
    assert item["mask"].shape == (tiny_cfg.data.image_size,) * 2
    assert torch.allclose(item["latent"], torch.from_numpy(z["mean"][0, 0].astype(np.float32)), atol=1e-3)
    # cached -> second call is a no-op returning the same file
    assert cache_latents(vae, processed_dir, "train", tiny_cfg.data.modalities, "dummy", hflip=True, device="cpu", show_progress=False) == path


@pytest.mark.parametrize("sampler", ["ddim", "ddpm"])
def test_sampler_runs(tiny_cfg, sampler):
    unet = build_unet(tiny_cfg)
    s = LatentSampler(unet, VAEWrapper(DummyVAE()), tiny_cfg.diffusion, tiny_cfg.latent_size, sampler, steps=3, device="cpu")
    gen = torch.Generator().manual_seed(0)
    imgs = s.sample_images(2, generator=gen)
    assert imgs.shape == (2, 3, tiny_cfg.data.image_size, tiny_cfg.data.image_size)
    assert imgs.min() >= 0 and imgs.max() <= 1
    again = s.sample_images(2, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(imgs, again), "same seed -> same samples"


def test_sampler_guidance(tiny_cfg):
    cfg = dataclasses.replace(tiny_cfg, model=dataclasses.replace(tiny_cfg.model, conditioning="mask"))
    unet = build_unet(cfg)
    mask = torch.zeros(2, cfg.data.image_size, cfg.data.image_size, dtype=torch.long)
    cond = mask_to_condition(mask, 4, cfg.latent_size)
    s = LatentSampler(unet, VAEWrapper(DummyVAE()), cfg.diffusion, cfg.latent_size, "ddim", steps=2, guidance_scale=3.0, device="cpu")
    assert s.sample_latents(2, cond=cond).shape == (2, 4, cfg.latent_size, cfg.latent_size)


@pytest.mark.slow
@pytest.mark.parametrize("conditioning", ["none", "mask"])
def test_train_end_to_end_cpu(tiny_cfg, conditioning):
    cfg = dataclasses.replace(tiny_cfg, model=dataclasses.replace(tiny_cfg.model, conditioning=conditioning))
    run = train(cfg, vae=VAEWrapper(DummyVAE()))
    assert (run / "config.yaml").exists() and (run / "metrics.csv").exists() and (run / "train_summary.json").exists()
    assert (run / "final" / "unet_ema" / "config.json").exists() and (run / "best" / "unet_ema").exists()
    assert list((run / "samples").glob("epoch_*.png"))
    assert find_run_dir(run / "final" / "unet_ema") == run.resolve()
    assert resolve_unet_dir(run).name == "unet_ema"
    cfg2, unet, unet_dir = load_run(run, use_ema=True)
    assert cfg2.model.conditioning == conditioning and unet.config.in_channels == (4 if conditioning == "none" else 9)
    assert resolve_checkpoint(run) == (run / "best", "best")
    assert resolve_checkpoint(run, "") == (run / "best", "best")
    assert resolve_checkpoint(run, "final") == (run / "final", "final")
    assert resolve_checkpoint(run / "final", "best") == (run / "final", "final")  # explicit dir wins
    ep = sorted((run / "checkpoints").iterdir())[-1]
    assert resolve_checkpoint(run, str(int(ep.name.split("_")[1]))) == (ep, f"ep{int(ep.name.split('_')[1]):04d}")
    assert resolve_checkpoint(run, str(ep)) == (ep, ep.name)
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(run, "999")
    # resume from the epoch checkpoint for one more step
    ckpt = sorted((run / "checkpoints").iterdir())[-1]
    cfg3 = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, resume=str(ckpt), output_dir=str(run) + "_resumed", epochs=2, max_steps=None))
    run2 = train(cfg3, vae=VAEWrapper(DummyVAE()))
    assert (run2 / "final" / "unet").exists()
