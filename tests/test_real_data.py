"""Checks on the actually prepared dataset (skipped when it is not present)."""

from pathlib import Path

import cv2
import pytest

from weldsight.data.dataset import load_manifest

PROC = Path(__file__).resolve().parents[1] / "data/processed/riawelc_v1"
pytestmark = pytest.mark.skipif(
    not (PROC / "manifest.csv").exists(), reason="run weldsight-prepare first"
)


def test_no_specimen_or_radiograph_in_two_splits():
    df, _ = load_manifest(PROC)
    assert df.groupby("group_id")["split"].nunique().max() == 1
    assert df.groupby("radiograph_id")["split"].nunique().max() == 1


def test_no_duplicate_pixels_across_splits():
    df, _ = load_manifest(PROC)
    assert df["md5_source"].is_unique


def test_every_class_in_every_split():
    df, card = load_manifest(PROC)
    for s in ("train", "val", "test"):
        for c in card["class_names"]:
            assert df.loc[df.split == s, f"cls_{c}"].sum() > 0, (s, c)


def test_patch_size():
    df, card = load_manifest(PROC)
    for p in df["image"].sample(20, random_state=0):
        assert cv2.imread(str(PROC / p), cv2.IMREAD_GRAYSCALE).shape == (card["patch_size"],) * 2
