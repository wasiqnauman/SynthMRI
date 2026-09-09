from pathlib import Path

import pytest
import yaml

from synthmri.config import Config, config_from_dict, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_roundtrip(tmp_path):
    cfg = Config()
    cfg.save(tmp_path / "c.yaml")
    back = load_config(tmp_path / "c.yaml")
    assert back == cfg
    assert cfg.latent_size == cfg.data.image_size // 8


def test_overrides_and_types():
    cfg = load_config(None, ["train.epochs=5", "data.modalities=[t1,t2,flair]", "data.crop_size=null", "train.lr=2e-5"])
    assert cfg.train.epochs == 5
    assert cfg.data.modalities == ("t1", "t2", "flair")
    assert cfg.data.crop_size is None
    assert cfg.train.lr == pytest.approx(2e-5)


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        config_from_dict({"train": {"epoch": 3}})
    with pytest.raises(KeyError):
        load_config(None, ["train.nope=1"])


def test_base_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"train": {"epochs": 7, "lr": 0.5}, "name": "b"}))
    (tmp_path / "child.yaml").write_text(yaml.safe_dump({"_base_": "base.yaml", "train": {"lr": 0.1}}))
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.name == "b"
    assert cfg.train.epochs == 7 and cfg.train.lr == 0.1


@pytest.mark.parametrize("name", sorted(p.name for p in CONFIGS.glob("*.yaml")))
def test_repo_configs_load(name):
    cfg = load_config(CONFIGS / name)
    assert cfg.data.image_size % 8 == 0
    assert len(cfg.data.modalities) == 3, "the pretrained VAE takes exactly three channels"
    assert cfg.model.conditioning in ("none", "mask")
