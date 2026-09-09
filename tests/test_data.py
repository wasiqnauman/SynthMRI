import numpy as np
import pytest
import torch

from synthmri.data import (
    MODALITIES,
    SliceDataset,
    find_patients,
    load_splits,
    make_patient_splits,
    remap_labels,
    save_splits,
)
from synthmri.data.preprocess import center_crop, normalise_volume, resize_image, resize_mask
from synthmri.data.splits import assert_disjoint
from tests.conftest import IMG


def test_find_patients_handles_quirks(raw_dir):
    with pytest.warns(UserWarning, match="incomplete"):
        cases = find_patients(raw_dir)
    ids = [c.patient_id for c in cases]
    assert len(cases) == 5 and "BraTS20_Training_099" not in ids
    p5 = next(c for c in cases if c.patient_id.endswith("005"))
    assert p5.seg.name == "W39_1998.09.19_Segm.nii"
    assert set(p5.modalities) == set(MODALITIES)


def test_remap_labels():
    seg = np.array([[0, 1], [2, 4]], dtype=np.float32)
    out = remap_labels(seg)
    assert out.dtype == np.uint8 and out.tolist() == [[0, 1], [2, 3]]
    with pytest.raises(ValueError):
        remap_labels(np.array([0, 3]))


def test_splits_disjoint_and_deterministic(tmp_path):
    ids = [f"p{i}" for i in range(20)]
    a = make_patient_splits(ids, 0.2, 0.2, seed=1)
    b = make_patient_splits(ids, 0.2, 0.2, seed=1)
    c = make_patient_splits(ids, 0.2, 0.2, seed=2)
    assert a["train"] == b["train"] and a["test"] != c["test"]
    assert len(a["train"]) + len(a["val"]) + len(a["test"]) == 20
    assert_disjoint(a)
    save_splits(a, tmp_path / "s.json")
    assert load_splits(tmp_path / "s.json")["test"] == a["test"]
    with pytest.raises(ValueError):
        assert_disjoint({"train": ["x"], "val": ["x"], "test": []})


def test_normalise_and_resize():
    vol = np.zeros((10, 10, 3), np.float32)
    mask = np.zeros_like(vol, dtype=bool)
    mask[2:8, 2:8] = True
    vol[mask] = np.linspace(0, 100, mask.sum())
    out = normalise_volume(vol, mask, (0, 100))
    assert out.min() == 0 and out.max() == pytest.approx(1.0) and out[~mask].max() == 0
    assert center_crop(np.ones((10, 10)), 6).shape == (6, 6)
    assert center_crop(np.ones((10, 10)), None).shape == (10, 10)
    assert resize_image(np.ones((10, 10)), 5).shape == (5, 5)
    m = resize_mask(np.array([[0, 3], [1, 2]], np.uint8), 4)
    assert set(np.unique(m)) <= {0, 1, 2, 3}


def test_preprocessed_arrays(processed_dir, splits):
    for split in ("train", "val", "test"):
        images = np.load(processed_dir / split / "images.npy")
        masks = np.load(processed_dir / split / "masks.npy")
        assert images.dtype == np.float16 and masks.dtype == np.uint8
        assert images.shape[1:] == (4, IMG, IMG) and masks.shape[1:] == (IMG, IMG)
        assert images.shape[0] == masks.shape[0] == len(splits[split]) * 7, "7 tumour slices per fixture patient"
        assert 0 <= float(images.min()) and float(images.max()) <= 1
        assert set(np.unique(masks).tolist()) == {0, 1, 2, 3}


def test_slice_dataset(processed_dir):
    ds = SliceDataset(processed_dir, "train", modalities=("flair", "t1ce", "t2"), hflip=False)
    item = ds[0]
    assert item["image"].shape == (3, IMG, IMG) and item["mask"].shape == (IMG, IMG)
    assert item["image"].min() >= -1 and item["image"].max() <= 1
    assert item["mask"].dtype == torch.int64
    raw = torch.from_numpy(ds.images[0][[0, 2, 3]].astype(np.float32)) * 2 - 1
    assert torch.allclose(item["image"], raw)
    ds_flip = SliceDataset(processed_dir, "train", hflip=True)
    torch.manual_seed(0)
    flips = 0
    for _ in range(20):
        it = ds_flip[0]
        if torch.equal(it["image"], raw.flip(-1)):
            assert torch.equal(it["mask"], item["mask"].flip(-1))
            flips += 1
        else:
            assert torch.equal(it["image"], raw) and torch.equal(it["mask"], item["mask"])
    assert 0 < flips < 20
    assert len(ds.patient_ids) == len(ds)
