"""Pull a reproducible subset out of a huge remote zip (e.g. SWRD, ~116 GB) without
downloading it.

The zip's central directory and the selected members are read with HTTP Range
requests (`remotezip`). All annotation files are extracted as-is; images are
decoded and immediately stored as 8-bit PNG (percentile windowing, the same as
`read_gray`), which shrinks 16-bit 1200-dpi TIFF scans by ~5-10x. Pixel geometry is
unchanged, so polygon coordinates stay valid.

    uv run weldsight-remote-subset --url URL --list            # contents only
    uv run weldsight-remote-subset --url URL --max-images 600  # subset
    uv run weldsight-remote-subset --url URL --max-images 600 \
        --include crop_weld_images/ --ann-include crop_weld_jsons/  # SWRD weld crops only
    uv run weldsight-remote-subset --zip local.zip ...          # same from a local zip

The list of extracted images is written to `<out>/_subset.txt` (seeded, resumable).
Members are fetched as byte ranges: neighbouring small files (annotations) share one
request, and every request is retried with backoff, since Google Drive answers bursts of
requests with occasional 5xx errors.
"""

from __future__ import annotations

import argparse
import bisect
import random
import re
import struct
import time
import zipfile
import zlib
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from tqdm import tqdm

from weldsight.data.sources import to_uint8

try:
    from remotezip import RemoteZipError
except ImportError:  # pragma: no cover - remotezip is a dependency
    RemoteZipError = OSError

ANN_EXTS = {".json", ".xml", ".txt", ".csv", ".yaml", ".yml"}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAX_BLOCK = 32 * 2**20  # bytes per range request when batching small members
MAX_GAP = 2**20  # unwanted bytes we accept inside one batched request
RETRIES = 8
BACKOFF = 2.0  # seconds, doubled after every failed attempt, capped at 60 s
TRANSIENT = (OSError, RemoteZipError, zipfile.BadZipFile, zlib.error)


class DownloadBlocked(RuntimeError):
    """The server answered a range request with a page instead of file bytes (Google
    Drive's quota page). Not retried: it lasts hours, not seconds."""


def retry(fn: Callable, what: str):
    for attempt in range(RETRIES):
        try:
            return fn()
        except TRANSIENT as e:
            if attempt == RETRIES - 1:
                raise RuntimeError(
                    f"{what}: still failing after {RETRIES} attempts ({e}). "
                    "Re-run the same command later: finished files are skipped."
                ) from e
            wait = min(60.0, BACKOFF * 2**attempt)
            print(f"\n{what}: {e!s:.120} -> retry {attempt + 1}/{RETRIES - 1} in {wait:.0f} s")
            time.sleep(wait)


def read_range(zf: zipfile.ZipFile, lo: int, hi: int) -> bytes:
    """Bytes lo..hi (inclusive) of the archive, as one stateless request."""
    fetch = getattr(zf.fp, "_fetch_fun", None)  # RemoteZip: a fresh HTTP Range request
    if fetch is not None:
        data = fetch((lo, hi), stream=False).read(hi - lo + 1)
    else:
        zf.fp.seek(lo)
        data = zf.fp.read(hi - lo + 1)
    if len(data) != hi - lo + 1:
        raise OSError(f"short read: {len(data)} of {hi - lo + 1} bytes")
    return data


def member_bytes(block: bytes, base: int, info: zipfile.ZipInfo) -> bytes:
    """Decompress one member from a block of raw archive bytes starting at `base`."""
    off = info.header_offset - base
    if block[off : off + 4] != b"PK\x03\x04":
        raise zipfile.BadZipFile(f"no local header for {info.filename}")
    name_len, extra_len = struct.unpack("<HH", block[off + 26 : off + 30])
    start = off + 30 + name_len + extra_len
    comp = block[start : start + info.compress_size]
    if len(comp) != info.compress_size:
        raise zipfile.BadZipFile(f"truncated data for {info.filename}")
    dec = zipfile._get_decompressor(info.compress_type)  # None for stored members
    data = dec.decompress(comp) if dec else comp
    if zlib.crc32(data) != info.CRC:
        raise zipfile.BadZipFile(f"CRC mismatch for {info.filename}")
    return data


def read_members(
    zf: zipfile.ZipFile, infos: list[zipfile.ZipInfo]
) -> Iterator[tuple[zipfile.ZipInfo, bytes]]:
    """Yield (info, data), batching members that sit next to each other in the archive."""
    offsets = sorted(i.header_offset for i in zf.infolist()) + [zf.start_dir]

    def end_of(info: zipfile.ZipInfo) -> int:  # offset of the next member
        return offsets[bisect.bisect_right(offsets, info.header_offset)]

    blocks: list[list[zipfile.ZipInfo]] = []
    for info in sorted(infos, key=lambda i: i.header_offset):
        if blocks:
            first, last = blocks[-1][0], blocks[-1][-1]
            near = info.header_offset - end_of(last) <= MAX_GAP
            if near and end_of(info) - first.header_offset <= MAX_BLOCK:
                blocks[-1].append(info)
                continue
        blocks.append([info])
    for blk in blocks:
        lo, hi = blk[0].header_offset, end_of(blk[-1]) - 1

        def fetch(lo=lo, hi=hi, blk=blk):
            raw = read_range(zf, lo, hi)
            return [(i, member_bytes(raw, lo, i)) for i in blk]

        what = blk[0].filename if len(blk) == 1 else f"{len(blk)} files from {blk[0].filename}"
        yield from retry(fetch, what)


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
    ann_include: str | None = None,
) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    by_name = {i.filename: i for i in zf.infolist() if not i.is_dir()}
    names = list(by_name)
    anns = [n for n in names if PurePosixPath(n).suffix.lower() in ANN_EXTS]
    if ann_include:
        anns = [n for n in anns if re.search(ann_include, n)]
    todo = [by_name[n] for n in anns if not (out / n).exists()]
    with tqdm(total=len(anns), initial=len(anns) - len(todo), desc="annotations") as bar:
        for info, data in read_members(zf, todo):
            dst = out / info.filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            bar.update()
    stems = {PurePosixPath(n).stem.lower() for n in anns if not n.lower().endswith(".txt")}
    chosen = select_images(names, max_images, seed, include, stems)
    (out / "_subset.txt").write_text("\n".join(chosen) + "\n")
    todo = [by_name[n] for n in chosen if not (out / n).with_suffix(".png").exists()]
    with tqdm(total=len(chosen), initial=len(chosen) - len(todo), desc="images -> png") as bar:
        for info in todo:  # one request per image: they are large
            ((_, data),) = read_members(zf, [info])
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            bar.update()
            if img is None:
                print(f"WARNING: cannot decode {info.filename}")
                continue
            dst = (out / info.filename).with_suffix(".png")
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), to_uint8(img), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return chosen


def open_zip(url: str | None, path: str | None) -> zipfile.ZipFile:
    if path:
        return zipfile.ZipFile(path)
    import requests
    from remotezip import RemoteFetcher, RemoteZip

    class RangeSizeFetcher(RemoteFetcher):
        def _request(self, kwargs):
            res = (self._session or requests).get(self._url, stream=True, **kwargs)
            res.raise_for_status()
            if "Content-Range" not in res.headers:
                body = res.raw.read(4096, decode_content=True).decode("utf-8", "replace")
                res.close()
                title = re.search(r"<title>(.*?)</title>", body, re.S | re.I)
                text = title.group(1).strip() if title else " ".join(body.split())[:200]
                raise DownloadBlocked(
                    f"server sent HTTP {res.status_code} "
                    f"{res.headers.get('Content-Type')} instead of a byte range: {text!r}. "
                    "Google Drive does this when the file's download quota is used up. "
                    "Re-run later (finished files are skipped) or read the archive through "
                    "a shortcut in your own Drive: --zip /content/drive/MyDrive/SWXD_Data.zip"
                )
            return res.raw, res.headers["Content-Range"]

        # Google Drive answers HEAD with Content-Length: 0, so remotezip's HEAD-based
        # size (used when suffix ranges are off) is 0. Read the total from Content-Range.
        def get_file_size(self) -> int:
            raw, content_range = self._request(self.prepare_request((0, 0)))
            raw.close()
            return int(content_range.rsplit("/", 1)[1])

    return RemoteZip(url, fetcher=RangeSizeFetcher, support_suffix_range=False, timeout=120)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="direct download URL that supports HTTP Range")
    src.add_argument("--zip", help="local zip path")
    ap.add_argument("--out", default="data/raw/swrd")
    ap.add_argument("--list", action="store_true", help="only print the archive contents")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--include", default=None, help="regex on image paths, e.g. 'T-joint'")
    ap.add_argument(
        "--ann-include",
        default=None,
        help="regex on annotation paths; keeps e.g. full-image labels away from crops",
    )
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
        chosen = extract_subset(
            zf, Path(args.out), args.max_images, args.seed, args.include, args.ann_include
        )
        print(f"extracted {len(chosen)} images into {args.out}")


if __name__ == "__main__":
    main()
