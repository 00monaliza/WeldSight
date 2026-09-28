"""Pull a reproducible subset out of a huge remote zip (e.g. SWRD, ~116 GB) without
downloading it.

The zip's central directory and the selected members are read with HTTP Range
requests (`remotezip`). All annotation files are extracted as-is; images are
decoded and immediately stored as 8-bit PNG (percentile windowing, the same as
`read_gray`), which shrinks 16-bit 1200-dpi TIFF scans by ~5-10x. Pixel geometry is
unchanged, so polygon coordinates stay valid.

    uv run weldsight-remote-subset --url URL --list            # contents only
    uv run weldsight-remote-subset --url URL --max-images 600  # subset
    uv run weldsight-remote-subset --zip local.zip ...          # same from a local zip

The list of extracted images is written to `<out>/_subset.txt` (seeded, resumable).
"""

from __future__ import annotations

import argparse
import random
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from tqdm import tqdm

from weldsight.data.sources import to_uint8

ANN_EXTS = {".json", ".xml", ".txt", ".csv", ".yaml", ".yml"}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def describe(zf: zipfile.ZipFile) -> str:
    infos = [i for i in zf.infolist() if not i.is_dir()]
    by_ext = Counter(PurePosixPath(i.filename).suffix.lower() or "<none>" for i in infos)
    size_ext = Counter()
    for i in infos:
        size_ext[PurePosixPath(i.filename).suffix.lower() or "<none>"] += i.file_size
    dirs = Counter(str(PurePosixPath(i.filename).parent) for i in infos)
    lines = [f"{len(infos)} files, {sum(i.file_size for i in infos) / 1e9:.1f} GB uncompressed"]
    lines += [f"  {e:8s} {n:7d} files {size_ext[e] / 1e9:8.2f} GB" for e, n in by_ext.most_common()]
    lines.append("folders (files per folder, top 30):")
    lines += [f"  {n:6d}  {d}" for d, n in dirs.most_common(30)]
    lines.append("first 20 paths:")
    lines += [f"  {i.filename}" for i in infos[:20]]
    return "\n".join(lines)


def select_images(
    names: list[str],
    max_images: int | None,
    seed: int,
    include: str | None,
    annotated_stems: set[str],
) -> list[str]:
    imgs = [n for n in names if PurePosixPath(n).suffix.lower() in IMG_EXTS]
    if include:
        imgs = [n for n in imgs if re.search(include, n)]
    # prefer images that have an annotation file with the same stem; with COCO-style
    # single-file annotations every image counts as annotated
    if annotated_stems:
        ann = [n for n in imgs if PurePosixPath(n).stem.lower() in annotated_stems]
        imgs = ann or imgs
    imgs = sorted(imgs)
    if max_images and len(imgs) > max_images:
        imgs = sorted(random.Random(seed).sample(imgs, max_images))
    return imgs


def extract_subset(
    zf: zipfile.ZipFile,
    out: Path,
    max_images: int | None = None,
    seed: int = 0,
    include: str | None = None,
) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    names = [i.filename for i in zf.infolist() if not i.is_dir()]
    anns = [n for n in names if PurePosixPath(n).suffix.lower() in ANN_EXTS]
    for n in tqdm(anns, desc="annotations"):
        dst = out / n
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(zf.read(n))
    stems = {PurePosixPath(n).stem.lower() for n in anns if not n.lower().endswith(".txt")}
    chosen = select_images(names, max_images, seed, include, stems)
    (out / "_subset.txt").write_text("\n".join(chosen) + "\n")
    for n in tqdm(chosen, desc="images -> 8-bit png"):
        dst = (out / n).with_suffix(".png")
        if dst.exists():
            continue
        img = cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"WARNING: cannot decode {n}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), to_uint8(img), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return chosen


def open_zip(url: str | None, path: str | None) -> zipfile.ZipFile:
    if path:
        return zipfile.ZipFile(path)
    from remotezip import RemoteZip

    return RemoteZip(url, support_suffix_range=False)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="direct download URL that supports HTTP Range")
    src.add_argument("--zip", help="local zip path")
    ap.add_argument("--out", default="data/raw/swrd")
    ap.add_argument("--list", action="store_true", help="only print the archive contents")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--include", default=None, help="regex on member paths, e.g. 'T-joint'")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    with open_zip(args.url, args.zip) as zf:
        report = describe(zf)
        print(report)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "_zip_listing.txt").write_text(
            report + "\n\nALL MEMBERS:\n" + "\n".join(i.filename for i in zf.infolist())
        )
        if args.list:
            return
        chosen = extract_subset(zf, Path(args.out), args.max_images, args.seed, args.include)
        print(f"extracted {len(chosen)} images into {args.out}")


if __name__ == "__main__":
    main()
