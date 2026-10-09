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

Google Drive: anonymous downloads of a popular file hit "Too many users ...". Read it through
the Drive API as yourself instead (no Drive storage needed): put an OAuth access token in
$GDRIVE_TOKEN and use https://www.googleapis.com/drive/v3/files/<ID>?alt=media as --url.

The list of extracted images is written to `<out>/_subset.txt` (seeded, resumable).
"""

from __future__ import annotations

import argparse
import os
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


def _range_size_fetcher():
    """remotezip asks for the file size with HEAD, which some endpoints (Drive API) do not
    answer with Content-Length; a 1-byte GET returns it in Content-Range instead."""
    import requests
    from remotezip import RemoteFetcher, RemoteZipError

    class RangeSizeFetcher(RemoteFetcher):
        def get_file_size(self):
            res = requests.get(self._url, **self.prepare_request((0, 0)))
            res.raise_for_status()
            total = res.headers.get("Content-Range", "").rpartition("/")[2]
            if not total.isdigit():
                raise RemoteZipError(
                    f"no file size in response ({res.status_code}, "
                    f"{res.headers.get('Content-Type')})"
                )
            return int(total)

    return RangeSizeFetcher


def open_zip(url: str | None, path: str | None, token: str | None = None) -> zipfile.ZipFile:
    if path:
        return zipfile.ZipFile(path)
    from remotezip import RemoteZip

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        return RemoteZip(
            url, support_suffix_range=False, fetcher=_range_size_fetcher(), headers=headers
        )
    except zipfile.BadZipFile as e:
        raise SystemExit(
            f"{e}: the server did not return a zip. Google Drive usually answers with an HTML "
            "page (quota / virus-scan warning); use the Drive API URL with $GDRIVE_TOKEN."
        ) from e


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
    ap.add_argument(
        "--token-env", default="GDRIVE_TOKEN", help="env var with an OAuth bearer token (optional)"
    )
    args = ap.parse_args(argv)
    with open_zip(args.url, args.zip, os.environ.get(args.token_env)) as zf:
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
