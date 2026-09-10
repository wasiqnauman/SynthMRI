import numpy as np
import pytest
import torch
from torch import nn

from synthmri.data.dataset import SliceDataset
from synthmri.eval.diversity import (
    memorisation_report,
    nearest_neighbour_distances,
    pairwise_ssim_diversity,
    save_nearest_neighbour_figure,
)
from synthmri.eval.fidelity import export_pngs
from synthmri.eval.recon import evaluate_vae_reconstruction, save_reconstruction_examples
from synthmri.eval.segmentation import (
    ArraySliceDataset,
    SegConfig,
    dice_from_counts,
    evaluate_segmenter,
    region_counts,
    subsample_patients,
    train_segmenter,
)
from synthmri.models import DummyVAE, VAEWrapper


def test_region_counts_and_dice():
    pred = np.array([[0, 1], [2, 3]])
    true = np.array([[0, 1], [2, 0]])
    c = region_counts(pred, true)
    assert c["WT"] == (2 * 2, 3 + 2) and c["ET"] == (0.0, 1.0) and c["TC"] == (2 * 1, 2 + 1)
    assert dice_from_counts(0, 0) == 1.0 and dice_from_counts(4, 5) == 0.8


def test_evaluate_segmenter_perfect_model():
    masks = np.random.RandomState(0).randint(0, 4, size=(6, 16, 16)).astype(np.uint8)
    onehot = np.eye(4, dtype=np.float32)[masks].transpose(0, 3, 1, 2)  # identity model -> argmax = mask
    res = evaluate_segmenter(nn.Identity(), onehot, masks, np.array(["a", "a", "b", "b", "c", "c"]), torch.device("cpu"))
    assert res["n_patients"] == 3
    for r in ("WT", "TC", "ET"):
        assert res["patient"][r]["mean"] == pytest.approx(1.0) and res["slice"][r]["mean"] == pytest.approx(1.0)
    assert res["patient"]["mean_WT_TC_ET"] == pytest.approx(1.0)


def test_subsample_patients():
    pids = np.array(["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3)
    keep = subsample_patients(pids, 0.5, seed=0)
    assert keep.sum() == 6 and len(np.unique(pids[keep])) == 2
    assert np.array_equal(keep, subsample_patients(pids, 0.5, seed=0))


@pytest.mark.slow
def test_train_segmenter_smoke(tmp_path):
    rng = np.random.RandomState(0)
    n, s = 8, 32
    masks = np.zeros((n, s, s), np.uint8)
    masks[:, 8:20, 8:20] = 2
    masks[:, 12:16, 12:16] = 3
    images = (rng.rand(n, 3, s, s) * 0.2 + 0.5 * (masks > 0)[:, None]).astype(np.float16)
    pids = np.array(["p1"] * 4 + ["p2"] * 4)
    cfg = SegConfig(epochs=1, batch_size=4, channels=(8, 16, 32), num_workers=0, amp=False, max_steps=2)
    model, hist = train_segmenter(images, masks, images, masks, pids, cfg, tmp_path / "seg", torch.device("cpu"))
    assert len(hist) == 1 and (tmp_path / "seg" / "segmenter.pt").exists()
    ds = ArraySliceDataset(images, masks, hflip=True)
    assert ds[0]["image"].shape == (3, s, s) and ds[0]["mask"].dtype == torch.int64


def test_diversity_and_memorisation(tmp_path):
    rng = np.random.RandomState(0)
    ref = rng.rand(10, 3, 32, 32).astype(np.float32)
    same = pairwise_ssim_diversity(np.repeat(ref[:1], 4, 0), n_pairs=6, device="cpu")
    assert same["pairwise_ssim_mean"] == pytest.approx(1.0)
    d, i = nearest_neighbour_distances(ref[:3], ref, device="cpu", feat_size=16)
    assert np.allclose(d, 0, atol=1e-3) and i.tolist() == [0, 1, 2]
    # a mirrored copy of a training image is a copy: caught only when the reference includes flips
    flipped = ref[:1, :, :, ::-1].copy()
    d_flip, i_flip = nearest_neighbour_distances(flipped, ref, device="cpu", feat_size=16)
    assert d_flip[0] == pytest.approx(0, abs=1e-3) and i_flip[0] == len(ref)  # index N + 0 = image 0, flipped
    d_noflip, _ = nearest_neighbour_distances(flipped, ref, device="cpu", feat_size=16, include_flips=False)
    assert d_noflip[0] > 0.5
    assert memorisation_report(flipped, ref, ref[5:], device="cpu", feat_size=16)["reference"] == "train+hflip"
    save_nearest_neighbour_figure(flipped, ref, i_flip, tmp_path / "nn_flip.png", n=1, dists=d_flip)
    rep = memorisation_report(ref[:3] + 0.01, ref, ref[5:], device="cpu", feat_size=16)
    assert set(rep) >= {"fake_to_train_nn_dist", "test_to_train_nn_dist", "frac_fake_closer_than_test_p5"}
    save_nearest_neighbour_figure(ref[:3], ref, i, tmp_path / "nn.png", n=2, dists=d)
    assert (tmp_path / "nn.png").exists()


def test_export_pngs(tmp_path):
    imgs = np.random.RandomState(0).rand(3, 3, 8, 8).astype(np.float32)
    dirs = export_pngs(imgs, tmp_path, ("flair", "t1ce", "t2"))
    assert set(dirs) == {"flair", "t1ce", "t2", "rgb"}
    assert len(list(dirs["flair"].glob("*.png"))) == 3 and len(list(dirs["rgb"].glob("*.png"))) == 3


def test_vae_reconstruction_metrics(processed_dir, tmp_path):
    ds = SliceDataset(processed_dir, "test", hflip=False)
    res = evaluate_vae_reconstruction(VAEWrapper(DummyVAE()), ds, ("flair", "t1ce", "t2"), device="cpu", batch_size=4, num_workers=0, use_lpips=False)
    assert res["n"] == len(ds)
    for key in ("flair", "t1ce", "t2", "rgb"):
        assert 0 <= res["metrics"][key]["ssim"]["mean"] <= 1 and res["metrics"][key]["psnr"]["mean"] > 5
    save_reconstruction_examples(VAEWrapper(DummyVAE()), ds, tmp_path / "rec.png", [0, 1], device="cpu")
    assert (tmp_path / "rec.png").exists()
