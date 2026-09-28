"""Prepare a dataset: raw files -> deduplicated, group-split 256x256 patches + manifest.

Usage:
    uv run weldsight-prepare --config configs/data/riawelc.yaml [--download] [key=value ...]

Outputs (under `data.processed_dir`):
    patches/<split>/<patch_id>.png   uint8 grayscale, 256x256
    masks/<split>/<patch_id>.png     uint8 indexed mask (only for mask datasets)
    manifest.csv                     one row per patch (split, group, labels, ...)
    dataset_card.json                counts, provenance, data version hash
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from weldsight.config import load_config
from weldsight.data.patching import fit_patch, patch_labels, tile
from weldsight.data.sources import Record, build_source, md5_of_file
from weldsight.data.split import check_no_group_leak, stratified_group_split
from weldsight.utils import file_sha256, write_json


# --------------------------------------------------------------------------- download
def download_riawelc(raw_dir: Path, url: str) -> None:
    """Clone the RIAWELC repo and extract the multi-part RAR5 archive."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if any(raw_dir.rglob("*.png")):
        print(f"[download] RIAWELC already extracted in {raw_dir}")
        return
    repo = raw_dir / "_repo"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", url, str(repo)], check=True)
    part1 = next(repo.rglob("*part01.rar"))
    for tool in (["unrar", "x", "-o+", "-inul"], ["7zz", "x", "-y"], ["unar", "-f"]):
        if shutil.which(tool[0]):
            cmd = [*tool, str(part1)]
            if tool[0] == "unar":
                cmd += ["-o", str(raw_dir)]
            subprocess.run(cmd, check=True, cwd=raw_dir)
            break
    else:
        sys.exit(
            "Need a RAR5 extractor: `apt-get install unrar` (Linux/Colab/Kaggle), "
            "`brew install rar` or 7-Zip >= 21 (`7zz`)."
        )
    shutil.rmtree(repo, ignore_errors=True)


def manual_download_hint(cfg: dict) -> str:
    return (
        f"Dataset '{cfg['source']}' must be downloaded manually from {cfg.get('url')} "
        f"(license: {cfg.get('license')}) and unpacked into {cfg['raw_dir']}."
    )


# --------------------------------------------------------------------------- prepare
def deduplicate(records: list[Record]) -> tuple[list[Record], dict]:
    """Drop byte-identical images; keep the first by (is_copy, record_id)."""
    by_hash: dict[str, list[Record]] = {}
    for r in records:
        by_hash.setdefault(md5_of_file(r.image_path), []).append(r)
    kept, conflicts, dropped = [], 0, 0
    for h, rs in by_hash.items():
        labels = {tuple(sorted(r.present_classes())) for r in rs}
        if len(labels) > 1:  # identical pixels, different labels -> unusable
            conflicts += len(rs)
            continue
        rs.sort(key=lambda r: (r.meta.get("is_copy", False), r.record_id))
        rs[0].meta["md5"] = h
        kept.append(rs[0])
        dropped += len(rs) - 1
    return kept, {"duplicates_dropped": dropped, "conflicting_label_dropped": conflicts}


def prepare(cfg: dict) -> Path:
    dcfg = cfg["data"]
    t0 = time.time()
    rng = np.random.default_rng(cfg.get("seed", 0))
    out = Path(dcfg["processed_dir"])
    raw = Path(dcfg["raw_dir"]).resolve()
    if out.resolve() == raw or out.resolve() in raw.parents:
        raise ValueError(f"processed_dir {out} must not contain raw_dir {raw}")
    if out.exists() and dcfg.get("overwrite", True):
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    source = build_source(dcfg)
    classes = source.class_names
    K = len(classes)
    records = source.records()
    n_raw = len(records)
    if not records:
        raise FileNotFoundError(f"No records found by '{source.name}' adapter in {dcfg['raw_dir']}")
    dedup_stats = {}
    if dcfg.get("deduplicate", True):
        records, dedup_stats = deduplicate(records)
    print(f"[prepare] {n_raw} raw records -> {len(records)} after dedup {dedup_stats}")

    # ---- split by group, stratified on record-level class presence
    rec_labels = np.array(
        [[int(c in r.present_classes()) for c in classes] for r in records], dtype=np.uint8
    )
    split_cfg = dcfg["split"]
    group_split = stratified_group_split(
        [r.group_id for r in records],
        rec_labels,
        tuple(split_cfg["fractions"]),
        seed=split_cfg.get("seed", 0),
        n_trials=split_cfg.get("n_trials", 5000),
    )

    size = dcfg["patch_size"]
    stride = dcfg.get("stride", {"train": size, "val": size, "test": size})
    bg_keep = dcfg.get("bg_keep_ratio", {"train": 1.0, "val": 1.0, "test": 1.0})
    min_px = dcfg.get("min_defect_pixels", 1)
    rows = []
    for rec in tqdm(records, desc="patching"):
        split = group_split[rec.group_id]
        img, mask = source.load(rec)
        if rec.kind == "patch":
            patches = [(fit_patch(img, size, dcfg.get("fit_mode", "resize")), None, 0, 0)]
        else:
            patches = [
                (p.image, p.mask, p.row, p.col) for p in tile(img, mask, size, stride[split])
            ]
        for pim, pm, y, x in patches:
            if pm is not None:
                lab = patch_labels(pm, K, min_px)
            else:
                lab = np.array([int(c in (rec.labels or [])) for c in classes], np.uint8)
            if pm is not None and lab.sum() == 0 and rng.random() > bg_keep[split]:
                continue
            pid = rec.record_id if rec.kind == "patch" else f"{rec.record_id}_y{y}_x{x}"
            ip = out / "patches" / split / f"{pid}.png"
            ip.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ip), pim)
            mp = ""
            if pm is not None:
                mpp = out / "masks" / split / f"{pid}.png"
                mpp.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mpp), pm)
                mp = str(mpp.relative_to(out))
            row = {
                "patch_id": pid,
                "image": str(ip.relative_to(out)),
                "mask": mp,
                "has_mask": int(pm is not None),
                "split": split,
                "group_id": rec.group_id,
                "radiograph_id": rec.radiograph_id,
                "record_id": rec.record_id,
                "source": source.name,
                "boxes_only": int(rec.boxes_only),
                "y": y,
                "x": x,
                "md5_source": rec.meta.get("md5", ""),
            }
            for k, c in enumerate(classes):
                row[f"cls_{c}"] = int(lab[k])
                if pm is not None:
                    row[f"px_{c}"] = int((pm == k + 1).sum())
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["split", "patch_id"]).reset_index(drop=True)
    check_no_group_leak(df)
    manifest = out / "manifest.csv"
    df.to_csv(manifest, index=False)

    card = {
        "name": dcfg["name"],
        "source": source.name,
        "url": dcfg.get("url"),
        "license": dcfg.get("license"),
        "citation": dcfg.get("citation"),
        "class_names": classes,
        "supervision": "mask" if df["has_mask"].any() else "label",
        "patch_size": size,
        "n_records_raw": n_raw,
        "n_records": len(records),
        **dedup_stats,
        "n_patches": len(df),
        "groups_per_split": {
            s: sorted(df.loc[df.split == s, "group_id"].unique().tolist())
            for s in ("train", "val", "test")
        },
        "radiographs_per_split": df.groupby("split")["radiograph_id"].nunique().to_dict(),
        "patches_per_split": df["split"].value_counts().to_dict(),
        "positives_per_split": {
            s: {c: int(df.loc[df.split == s, f"cls_{c}"].sum()) for c in classes}
            for s in ("train", "val", "test")
        },
        "data_version": file_sha256(manifest)[:16],
        "prepare_config": dcfg,
        "seed": cfg.get("seed", 0),
        "prepared_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
    }
    write_json(card, out / "dataset_card.json")
    print(f"[prepare] wrote {len(df)} patches, data_version={card['data_version']}")
    print(pd.crosstab(df["split"], [df[f"cls_{c}"] for c in classes]).to_string())
    return out


def inspect(cfg: dict) -> None:
    """Print what the adapter sees (labels, groups) without writing anything."""
    source = build_source(cfg["data"])
    root = Path(cfg["data"]["raw_dir"])
    files = [f for f in root.rglob("*") if f.is_file()] if root.exists() else []
    print(f"raw_dir: {root.resolve()} (exists={root.exists()}), {len(files)} files")
    print(
        "files by extension:",
        dict(Counter(f.suffix.lower() or "<none>" for f in files).most_common(15)),
    )
    print("sample paths:", *[str(f.relative_to(root)) for f in sorted(files)[:10]], sep="\n  ")
    archives = [f for f in files if f.suffix.lower() in {".zip", ".rar", ".7z", ".tar", ".gz"}]
    if archives:
        print(f"WARNING: {len(archives)} archives not unpacked, e.g. {archives[0].name}")
    recs = source.records()
    diag = getattr(source, "diag", None)
    if diag:
        print(f"images found: {diag['images_found']}, annotation files: {diag['annotation_files']}")
        if diag["unmatched"]:
            print(
                f"annotations without a matching image: {len(diag['unmatched'])}, e.g.",
                diag["unmatched"][:3],
            )
        if diag["unparsed"]:
            print(
                f"annotation files in an unknown format: {len(diag['unparsed'])}, e.g.",
                diag["unparsed"][:3],
            )
            first = Path(str(diag["unparsed"][0]).split(": ")[0])
            if first.exists():
                print("first unknown file starts with:", first.read_text(errors="replace")[:400])
    labels = Counter(c for r in recs for c in r.present_classes())
    print(f"{len(recs)} records, {len({r.group_id for r in recs})} groups")
    print("label counts:", dict(labels))
    unknown = set(labels) - set(source.class_names)
    if unknown:
        print("WARNING: labels not in class_names (add them to label_map):", unknown)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--download", action="store_true", help="download raw data if possible")
    ap.add_argument("--inspect", action="store_true", help="only list records/labels")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    dcfg = cfg["data"]
    if args.download:
        if dcfg["source"] == "riawelc":
            download_riawelc(Path(dcfg["raw_dir"]), dcfg["url"])
        else:
            print(manual_download_hint(dcfg))
    if args.inspect:
        inspect(cfg)
        return
    prepare(cfg)


if __name__ == "__main__":
    main()
