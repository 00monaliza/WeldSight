from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from weldsight.config import load_config
from weldsight.data.dataset import BasicAugment, WeldPatchDataset, load_manifest
from weldsight.data.patching import fit_patch, patch_labels, tile
from weldsight.data.prepare import prepare
from weldsight.data.sources import RiawelcSource
from weldsight.data.split import check_no_group_leak, stratified_group_split

from .conftest import CLASSES

REPO = Path(__file__).resolve().parents[1]


def _cfg(raw, out, **data):
    cfg = load_config(REPO / "configs/experiment/riawelc_baseline.yaml")
    cfg["data"].update(raw_dir=str(raw), processed_dir=str(out), **data)
    cfg["data"]["split"]["n_trials"] = 300
    return cfg


# ---------------------------------------------------------------- patching
@pytest.mark.parametrize("h,w", [(256, 256), (300, 600), (100, 700), (1000, 257)])
def test_tile_shapes_and_coverage(h, w):
    img = np.random.default_rng(0).integers(0, 255, (h, w), dtype=np.uint8)
    mask = np.zeros((h, w), np.uint8)
    mask[-5:, -5:] = 1  # bottom-right corner must be covered
    patches = tile(img, mask, 256, 256)
    assert all(p.image.shape == (256, 256) and p.mask.shape == (256, 256) for p in patches)
    assert sum(int(p.mask.sum()) for p in patches) >= 25


def test_fit_patch_resize_and_pad():
    p = np.zeros((227, 227), np.uint8)
    assert fit_patch(p, 256, "resize").shape == (256, 256)
    assert fit_patch(p, 256, "pad").shape == (256, 256)


def test_patch_labels_min_pixels():
    m = np.zeros((8, 8), np.uint8)
    m[0, :3] = 2
    assert patch_labels(m, 3, 1).tolist() == [0, 1, 0]
    assert patch_labels(m, 3, 5).tolist() == [0, 0, 0]


# ---------------------------------------------------------------- split
def test_group_split_no_leak_and_deterministic():
    rng = np.random.default_rng(0)
    groups = [f"g{i % 20}" for i in range(400)]
    labels = rng.integers(0, 2, (400, 3)).astype(np.uint8)
    a = stratified_group_split(groups, labels, seed=1, n_trials=200)
    b = stratified_group_split(groups, labels, seed=1, n_trials=200)
    assert a == b
    assert set(a.values()) == {"train", "val", "test"}
    df = pd.DataFrame({"group_id": groups, "split": [a[g] for g in groups]})
    check_no_group_leak(df)


def test_rare_class_present_in_val_and_test():
    # class 2 exists in only 3 of 12 groups -> must still reach val and test
    groups, labels = [], []
    for g in range(12):
        for _ in range(10):
            groups.append(f"g{g}")
            labels.append([1, 0, int(g in (0, 5, 9))])
    s = stratified_group_split(groups, np.array(labels, np.uint8), seed=0, n_trials=500)
    rare_splits = {s[f"g{g}"] for g in (0, 5, 9)}
    assert {"val", "test"} <= rare_splits


# ---------------------------------------------------------------- RIAWELC end-to-end
def test_riawelc_source_parses_names(fake_riawelc):
    recs = RiawelcSource(fake_riawelc, CLASSES).records()
    assert len(recs) == 8 * 2 * 4 * 3 + 1
    r = recs[0]
    assert r.group_id.startswith("RRT-") and r.radiograph_id.startswith(r.group_id + "_Img")


def test_prepare_riawelc(fake_riawelc, tmp_path):
    out = prepare(_cfg(fake_riawelc, tmp_path / "proc"))
    df, card = load_manifest(out)
    # duplicate " - Copia" file removed
    assert card["duplicates_dropped"] == 1 and len(df) == 8 * 2 * 4 * 3
    # no specimen or radiograph shared between splits (the original split leaks)
    for col in ("group_id", "radiograph_id"):
        assert df.groupby(col)["split"].nunique().max() == 1
    tr, te = (set(df.loc[df.split == s, "radiograph_id"]) for s in ("train", "test"))
    assert not tr & te
    # every patch is 256x256 uint8
    for p in df["image"].sample(10, random_state=0):
        im = cv2.imread(str(out / p), cv2.IMREAD_UNCHANGED)
        assert im.shape == (256, 256) and im.dtype == np.uint8
    # label-only supervision, one-hot or all-zero
    assert card["supervision"] == "label"
    assert df[[f"cls_{c}" for c in CLASSES]].sum(axis=1).max() == 1


def test_prepare_is_reproducible(fake_riawelc, tmp_path):
    a = prepare(_cfg(fake_riawelc, tmp_path / "a"))
    b = prepare(_cfg(fake_riawelc, tmp_path / "b"))
    assert load_manifest(a)[1]["data_version"] == load_manifest(b)[1]["data_version"]


def test_dataset_shapes(fake_riawelc, tmp_path):
    out = prepare(_cfg(fake_riawelc, tmp_path / "proc"))
    ds = WeldPatchDataset(out, "train", CLASSES, BasicAugment())
    item = ds[0]
    assert item["image"].shape == (1, 256, 256) and item["image"].dtype == torch.float32
    assert item["labels"].shape == (3,)
    assert item["mask"].shape == (256, 256) and (item["mask"] == -1).all()
    assert not item["has_mask"]


# ---------------------------------------------------------------- polygon masks (SWRD format)
def test_prepare_swrd_masks(fake_swrd, tmp_path):
    cfg = load_config(REPO / "configs/experiment/swrd_baseline.yaml")
    cfg["data"].update(
        raw_dir=str(fake_swrd),
        processed_dir=str(tmp_path / "proc_swrd"),
        bg_keep_ratio={"train": 1.0, "val": 1.0, "test": 1.0},
    )
    cfg["data"]["split"]["n_trials"] = 200
    out = prepare(cfg)
    df, card = load_manifest(out)
    assert card["supervision"] == "mask"
    check_no_group_leak(df)
    classes = card["class_names"]
    for _, row in df.iterrows():
        m = cv2.imread(str(out / row["mask"]), cv2.IMREAD_GRAYSCALE)
        assert m.shape == (256, 256) and m.max() <= len(classes)
        for k, c in enumerate(classes, start=1):
            assert row[f"px_{c}"] == (m == k).sum()
            assert row[f"cls_{c}"] == int((m == k).sum() >= cfg["data"]["min_defect_pixels"])
    assert df["cls_porosity"].sum() > 0 and df["cls_crack"].sum() > 0


def test_swrd_coco_layout_in_separate_folders(tmp_path):
    """COCO json in annotations/, images in images/ -> matched by file name."""
    import json

    from weldsight.data.sources import SwrdSource

    root = tmp_path / "coco"
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    ims, anns = [], []
    for i in range(3):
        cv2.imwrite(str(root / "images" / f"W_{i}.tif"), np.full((300, 600), 128, np.uint16))
        ims.append({"id": i, "file_name": f"images\\W_{i}.tif", "width": 600, "height": 300})
        anns.append(
            {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "segmentation": [[10, 10, 60, 10, 60, 40, 10, 40]],
                "bbox": [10, 10, 50, 30],
            }
        )
    coco = {"images": ims, "annotations": anns, "categories": [{"id": 1, "name": "气孔"}]}
    (root / "annotations" / "all.json").write_text(json.dumps(coco, ensure_ascii=False))
    src = SwrdSource(root, ["porosity", "crack"], label_map={"气孔": "porosity"})
    recs = src.records()
    assert len(recs) == 3 and src.diag["unmatched"] == []
    img, mask = src.load(recs[0])
    assert img.dtype == np.uint8 and mask.max() == 1 and (mask == 1).sum() > 1000
