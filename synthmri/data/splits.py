"""Patient-level train/val/test splits.

All slices of a patient go to the same split, so no test patient's anatomy is ever seen during
training (slice-level splits leak near-identical neighbouring slices between splits).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from synthmri.utils.io import load_json, save_json

SPLIT_NAMES = ("train", "val", "test")


def make_patient_splits(
    patient_ids: list[str], val_frac: float, test_frac: float, seed: int
) -> dict[str, list[str] | dict]:
    ids = sorted(set(patient_ids))
    n = len(ids)
    if n < 3:
        raise ValueError("Need at least 3 patients to build train/val/test splits")
    if not 0 <= val_frac < 1 or not 0 <= test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be in [0, 1) and sum to < 1")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    test = sorted(ids[i] for i in perm[:n_test])
    val = sorted(ids[i] for i in perm[n_test : n_test + n_val])
    train = sorted(ids[i] for i in perm[n_test + n_val :])
    splits = {
        "train": train,
        "val": val,
        "test": test,
        "meta": {"seed": seed, "val_frac": val_frac, "test_frac": test_frac, "n_patients": n},
    }
    assert_disjoint(splits)
    return splits


def assert_disjoint(splits: dict) -> None:
    sets = {k: set(splits[k]) for k in SPLIT_NAMES}
    for a in SPLIT_NAMES:
        for b in SPLIT_NAMES:
            if a < b and sets[a] & sets[b]:
                raise ValueError(f"Splits {a} and {b} overlap: {sorted(sets[a] & sets[b])[:5]}")


def save_splits(splits: dict, path: str | Path) -> None:
    save_json(splits, path)


def load_splits(path: str | Path) -> dict:
    splits = load_json(path)
    assert_disjoint(splits)
    return splits
