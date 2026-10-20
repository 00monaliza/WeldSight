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
    for s, per_class in card["groups_with_class_per_split"].items():
        part = df[df.split == s]
        for c, n in per_class.items():
            assert n == part.loc[part[f"cls_{c}"] > 0, "group_id"].nunique()
            assert n <= card["radiographs_with_class_per_split"][s][c]


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


def test_remote_subset_from_zip(tmp_path, fake_swrd):
    """Subset extraction: all annotations, N seeded images, 16-bit -> 8-bit png."""
    import zipfile

    from weldsight.data.remote_subset import extract_subset
    from weldsight.data.sources import SwrdSource

    zpath = tmp_path / "swrd.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for f in sorted(fake_swrd.iterdir()):
            if f.suffix == ".png":  # store as 16-bit tif like the real scans
                im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.uint16) * 256
                ok, buf = cv2.imencode(".tif", im)
                z.writestr(f"SWRD/img/{f.stem}.tif", buf.tobytes())
            else:
                z.write(f, f"SWRD/ann/{f.name}")
    out = tmp_path / "subset"
    with zipfile.ZipFile(zpath) as z:
        a = extract_subset(z, out, max_images=4, seed=1)
    with zipfile.ZipFile(zpath) as z:
        assert extract_subset(z, tmp_path / "subset2", max_images=4, seed=1) == a
    assert len(a) == 4 and len(list(out.rglob("*.json"))) == 6
    pngs = list(out.rglob("*.png"))
    assert len(pngs) == 4 and cv2.imread(str(pngs[0]), cv2.IMREAD_UNCHANGED).dtype == np.uint8
    recs = SwrdSource(out, ["porosity", "crack"]).records()
    assert len(recs) == 4  # json in ann/, png in img/ -> matched by name


def _drive_like_server(data: bytes, fail_every: int = 0):
    """HTTP server that answers like Google Drive: HEAD says Content-Length 0, GET
    serves byte ranges, every `fail_every`-th GET fails with 500, and once
    `stats["quota"]` is set every answer is the HTML quota page."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    stats = {"get": 0, "failed": 0, "quota": False}

    class DriveLike(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            stats["get"] += 1
            if stats["quota"]:
                page = b"<html><title>Google Drive - Quota exceeded</title></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            if fail_every and stats["get"] % fail_every == 0:
                stats["failed"] += 1
                self.send_error(500)
                return
            lo, hi = self.headers["Range"][len("bytes=") :].split("-")
            if not lo:  # suffix range: last N bytes
                lo, hi = len(data) - int(hi), len(data) - 1
            else:
                lo, hi = int(lo), min(int(hi) if hi else len(data) - 1, len(data) - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(data)}")
            self.send_header("Content-Length", str(hi - lo + 1))
            self.end_headers()
            self.wfile.write(data[lo : hi + 1])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), DriveLike)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/swrd.zip", stats


def test_remote_zip_when_head_reports_zero_length(tmp_path, fake_swrd):
    """Google Drive answers HEAD with Content-Length: 0; the size must come from Content-Range."""
    import zipfile

    from weldsight.data.remote_subset import extract_subset, open_zip

    zpath = tmp_path / "swrd.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for f in sorted(fake_swrd.iterdir()):
            z.write(f, f"SWRD/{f.name}")
    server, url, _ = _drive_like_server(zpath.read_bytes())
    try:
        with open_zip(url, None) as z:
            assert sorted(z.namelist()) == sorted(f"SWRD/{f.name}" for f in fake_swrd.iterdir())
            chosen = extract_subset(z, tmp_path / "subset", max_images=2, seed=0)
    finally:
        server.shutdown()
    assert len(chosen) == 2 and len(list((tmp_path / "subset").rglob("*.png"))) == 2


def test_remote_zip_batches_annotations_and_retries_5xx(tmp_path, monkeypatch):
    """Many small json files cost a few requests, and sporadic 500s are retried."""
    import json
    import zipfile

    import weldsight.data.remote_subset as rs

    monkeypatch.setattr(rs, "BACKOFF", 0.0)
    # noise does not compress: the archive is far larger than remotezip's 64 KB first read
    noise = np.random.default_rng(0).integers(0, 65535, (300, 400), dtype=np.uint16)
    ok, tif = cv2.imencode(".tif", noise)
    zpath = tmp_path / "swrd.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for i in range(200):
            z.writestr(f"jsons/img_{i}.json", json.dumps({"shapes": [], "i": i}))
        for i in range(5):
            z.writestr(f"images/img_{i}.tif", tif.tobytes())
    server, url, stats = _drive_like_server(zpath.read_bytes(), fail_every=4)
    try:
        with rs.open_zip(url, None) as z:
            stats["get"] = 0
            chosen = rs.extract_subset(z, tmp_path / "out", max_images=3, seed=0)
    finally:
        server.shutdown()
    out = tmp_path / "out"
    assert len(list(out.rglob("*.json"))) == 200 and len(chosen) == 3
    assert json.loads((out / "jsons/img_7.json").read_text())["i"] == 7
    assert len(list(out.rglob("*.png"))) == 3
    assert stats["failed"] > 0  # the failures happened and were survived
    assert stats["get"] < 20  # 200 json + 3 images, not one request per json


def test_swrd_real_layout_labels_groups_and_ann_filter(tmp_path):
    """SWXD_Data.zip layout: A_/B_ crops of one film share a group, full-image json files
    are not extracted next to the crops, and non-defect labels are dropped."""
    import json
    import zipfile

    from weldsight.data.remote_subset import extract_subset
    from weldsight.data.sources import SwrdSource

    cfg = load_config(REPO / "configs/experiment/swrd_baseline.yaml")["data"]
    sq = [[10, 10], [60, 10], [60, 60], [10, 60]]
    shapes = [
        {"label": "气孔", "shape_type": "polygon", "points": sq},
        {"label": "夹钨", "shape_type": "polygon", "points": sq},
        {"label": "焊缝", "shape_type": "polygon", "points": sq},
        {"label": "伪缺陷", "shape_type": "polygon", "points": sq},
    ]
    ok, png = cv2.imencode(".tif", np.full((100, 200), 30000, np.uint16))
    zpath = tmp_path / "swxd.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for film in ["DJ-RT-20220810-33", "DJ-RT-20220810-34", "DJ-RT-20230101-5"]:
            z.writestr(f"Raw_data/images/{film}.tif", png.tobytes())
            z.writestr(f"Raw_data/json/{film}.json", json.dumps({"shapes": []}))
            for half in "AB":
                z.writestr(f"crop_weld_data/crop_weld_images/L/1/{half}_{film}.tif", png.tobytes())
                z.writestr(
                    f"crop_weld_data/crop_weld_jsons/L/1/{half}_{film}.json",
                    json.dumps({"shapes": shapes, "imagePath": f"{film}.tif"}),
                )
    out = tmp_path / "raw"
    with zipfile.ZipFile(zpath) as z:
        extract_subset(z, out, include="crop_weld_images/", ann_include="crop_weld_jsons/")
    assert not (out / "Raw_data").exists()

    src = SwrdSource(out, cfg["class_names"], **cfg["source_opts"])
    recs = src.records()
    assert len(recs) == 6 and not src.diag["unmatched"]
    assert {r.group_id for r in recs} == {"DJ-RT-20220810", "DJ-RT-20230101"}
    assert {c for r in recs for c in r.present_classes()} == {"porosity", "inclusion"}
    assert src.diag["ignored"] == {"焊缝": 6, "伪缺陷": 6}


def test_remote_zip_quota_page_stops_without_retrying(tmp_path, fake_swrd, monkeypatch):
    import zipfile

    import weldsight.data.remote_subset as rs

    monkeypatch.setattr(rs, "BACKOFF", 0.0)
    zpath = tmp_path / "swrd.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for f in sorted(fake_swrd.iterdir()):
            z.write(f, f"SWRD/{f.name}")
    server, url, stats = _drive_like_server(zpath.read_bytes())
    try:
        z = rs.open_zip(url, None)
        stats.update(get=1, quota=True)
        with pytest.raises(rs.DownloadBlocked, match="Quota exceeded"):
            rs.read_range(z, 0, 10)
    finally:
        server.shutdown()
    assert stats["get"] == 2  # one request, no retries


def test_select_images_keeps_previous_subset_and_prefers_labels(tmp_path):
    """Subset growth: images of an earlier run stay, rare-label images come first."""
    import json
    import zipfile

    from weldsight.data.remote_subset import extract_subset

    ok, tif = cv2.imencode(".tif", np.full((20, 30), 30000, np.uint16))
    zpath = tmp_path / "a.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for i in range(40):
            label = "咬边" if i in (3, 17, 29) else "气孔"
            z.writestr(f"jsons/img_{i}.json", json.dumps({"shapes": [{"label": label}]}))
            z.writestr(f"images/img_{i}.tif", tif.tobytes())
    out = tmp_path / "out"
    with zipfile.ZipFile(zpath) as z:
        first = extract_subset(z, out, max_images=5, seed=0)
        plain = extract_subset(z, tmp_path / "plain", max_images=5, seed=0)
        grown = extract_subset(z, out, max_images=12, seed=1, prefer_labels={"咬边"})
    assert first == plain  # no keep/prefer: same as a plain seeded sample
    assert set(first) <= set(grown) and len(grown) == 12
    assert {f"images/img_{i}.tif" for i in (3, 17, 29)} <= set(grown)
    assert len(list(out.rglob("*.png"))) == 12
