"""Dataset adapters.

Every adapter turns a raw public dataset into a list of `Record`s with a common
schema, so the rest of the pipeline (split -> patching -> training) is dataset
agnostic. Two kinds of supervision exist in the public data:

* `mask`   - pixel masks (GDXray W0002, rasterised SWRD polygons);
* `label`  - image/patch-level class labels only (RIAWELC).

Class index 0 in masks is always background; defect classes are 1..K in the order
of `class_names`.
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Record:
    record_id: str
    image_path: Path
    group_id: str  # unit of the train/val/test split (weld specimen / source radiograph)
    radiograph_id: str  # unit of referral to the inspector
    kind: str  # "patch" (already a patch) or "full" (full radiograph -> tile it)
    labels: list[str] | None = None  # image-level labels (label-only supervision)
    mask_path: Path | None = None  # binary or indexed mask image
    polygons: list[tuple[str, np.ndarray]] | None = None  # (class, Nx2 xy points)
    boxes_only: bool = False  # True when "polygons" are really bounding boxes
    meta: dict = field(default_factory=dict)

    def present_classes(self) -> set[str]:
        if self.labels is not None:
            return set(self.labels)
        if self.polygons is not None:
            return {c for c, _ in self.polygons}
        return set(self.meta.get("mask_classes", []))


def read_gray(path: Path, window: tuple[float, float] | None = (0.5, 99.5)) -> np.ndarray:
    """Read any 8/16-bit grayscale image and return uint8.

    16-bit radiographs are windowed with robust percentiles; 8-bit images are
    returned unchanged so that published patches are bit-identical.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, window) if window else (img.min(), img.max())
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return (img * 255).round().astype(np.uint8)


def md5_of_file(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


class Source:
    name: str = "base"

    def __init__(self, root: str | Path, class_names: list[str], **kwargs):
        self.root = Path(root)
        self.class_names = list(class_names)
        self.opts = kwargs

    def records(self) -> list[Record]:  # pragma: no cover - interface
        raise NotImplementedError

    def load(self, rec: Record) -> tuple[np.ndarray, np.ndarray | None]:
        """Return (uint8 image HxW, uint8 mask HxW with 0=bg,1..K or None)."""
        img = read_gray(rec.image_path)
        if rec.polygons is not None:
            return img, rasterize(rec.polygons, img.shape, self.class_names)
        if rec.mask_path is not None:
            m = cv2.imread(str(rec.mask_path), cv2.IMREAD_GRAYSCALE)
            if m.shape != img.shape:
                m = cv2.resize(m, img.shape[::-1], interpolation=cv2.INTER_NEAREST)
            # binary mask (0/255 or 0/1) of a single-class dataset -> class 1
            if len(self.class_names) == 1:
                m = (m > 0).astype(np.uint8)
            return img, m
        return img, None


def rasterize(polys, shape, class_names) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    # draw larger regions first so that small defects inside them stay visible
    order = sorted(polys, key=lambda p: -cv2.contourArea(p[1].astype(np.float32)))
    for cls, pts in order:
        if cls not in class_names:
            continue
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], class_names.index(cls) + 1)
    return mask


# --------------------------------------------------------------------------- RIAWELC
RIAWELC_NAME = re.compile(
    r"^(?P<specimen>.+?)_Img(?P<img>\d+)_A(?P<a>\d+)_S(?P<s>\d+)_\[(?P<r>\d+)\]\[(?P<c>\d+)\]"
    r"(?P<copy>.*)\.png$"
)
# Folder -> class. Verified against per-class counts in the RIAWELC paper
# (CR 7635, PO 6320, LP 4452, ND 6000) and by visual inspection.
RIAWELC_FOLDERS = {
    "Difetto1": "crack",
    "Difetto2": "porosity",
    "Difetto4": "lack_of_penetration",
    "NoDifetto": None,
}


class RiawelcSource(Source):
    """RIAWELC (Totino et al., 2022): 24,407 patches of 227x227 px, 4 classes.

    The patch file name encodes the source radiograph
    (`RRT-26R_Img2_A80_S4_[12][84].png` -> specimen RRT-26R, film Img2), which
    lets us re-split by specimen. The published training/validation/testing
    folders are ignored on purpose: they share 108/109 radiographs and 2,443
    byte-identical files across splits.
    """

    name = "riawelc"

    def records(self) -> list[Record]:
        group_by = self.opts.get("group_by", "specimen")  # specimen | radiograph
        recs = []
        files = sorted(self.root.rglob("*.png"))
        if not files:
            raise FileNotFoundError(f"No RIAWELC png files under {self.root}")
        for f in files:
            folder = f.parent.name
            if folder not in RIAWELC_FOLDERS:
                continue
            m = RIAWELC_NAME.match(f.name)
            if not m:
                raise ValueError(f"Unexpected RIAWELC file name: {f.name}")
            cls = RIAWELC_FOLDERS[folder]
            radiograph = f"{m['specimen']}_Img{m['img']}"
            recs.append(
                Record(
                    record_id=f"{radiograph}_S{m['s']}_r{m['r']}_c{m['c']}{'_copy' if m['copy'] else ''}",
                    image_path=f,
                    group_id=m["specimen"] if group_by == "specimen" else radiograph,
                    radiograph_id=radiograph,
                    kind="patch",
                    labels=[cls] if cls else [],
                    meta={
                        "orig_split": f.parent.parent.name,
                        "specimen": m["specimen"],
                        "is_copy": bool(m["copy"]),
                        "grid_s": int(m["s"]),
                        "grid_r": int(m["r"]),
                        "grid_c": int(m["c"]),
                    },
                )
            )
        return recs


# --------------------------------------------------------------------------- GDXray
class GdxraySource(Source):
    """GDXray Welds: W0001 (10 radiographs) + W0002 (binary defect masks).

    Only one class ("defect") because W0002 masks carry no defect type.
    W0003 (67 radiographs, mostly unlabelled) is not used for supervision.
    """

    name = "gdxray"

    def records(self) -> list[Record]:
        img_dir = next(self.root.rglob("W0001"), None)
        mask_dir = next(self.root.rglob("W0002"), None)
        if img_dir is None or mask_dir is None:
            raise FileNotFoundError(f"W0001/W0002 not found under {self.root}")
        recs = []
        for f in sorted(img_dir.glob("W0001_*.png")):
            idx = f.stem.split("_")[1]
            mp = mask_dir / f"W0002_{idx}.png"
            if not mp.exists():
                continue
            rid = f"W0001_{idx}"
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            recs.append(
                Record(
                    rid,
                    f,
                    rid,
                    rid,
                    "full",
                    mask_path=mp,
                    meta={
                        "mask_classes": self.class_names if m is not None and m.max() > 0 else []
                    },
                )
            )
        return recs


# --------------------------------------------------------------------------- SWRD
class SwrdSource(Source):
    """SWRD (Zhao et al., J. Nondestruct. Eval. 2025): ~3.6k seam-weld radiographs,
    polygon annotations for 6 defect types.

    Expected layout (configurable globs): images next to LabelMe-style JSON files
    (`shapes[*].label/points/shape_type`). Pascal-VOC XML with only `bndbox`
    is also accepted, but then masks are boxes and `boxes_only=True` is recorded.
    NOTE: written against the published description of the dataset; verify the
    label names with `weldsight-prepare --inspect` after download and adjust
    `label_map` in configs/data/swrd.yaml.
    """

    name = "swrd"
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def records(self) -> list[Record]:
        label_map: dict = self.opts.get("label_map", {})
        group_regex = self.opts.get("group_regex")  # e.g. "^(.*)_\\d+$" to merge crops
        # index every image in the tree by lower-case stem: annotation and image files
        # often live in different folders (images/, labels/, json/ ...)
        images: dict[str, Path] = {}
        for f in sorted(self.root.rglob("*")):
            if f.suffix.lower() in self.IMG_EXTS and f.is_file():
                images.setdefault(f.stem.lower(), f)
        self.diag = {
            "images_found": len(images),
            "annotation_files": 0,
            "unmatched": [],
            "unparsed": [],
        }

        items: list[tuple[Path, list, bool]] = []  # (image, polygons, boxes_only)
        anns = sorted(list(self.root.rglob("*.json")) + list(self.root.rglob("*.xml")))
        self.diag["annotation_files"] = len(anns)
        for ann in anns:
            try:
                if ann.suffix == ".xml":
                    polys, boxes_only = self._parse_voc(ann)
                    items.append((self._match(ann.stem, None, images, ann), polys, boxes_only))
                    continue
                d = json.loads(ann.read_text(encoding="utf-8"))
                if isinstance(d, dict) and {"images", "annotations"} <= set(d):
                    for stem, polys in self._parse_coco(d):  # one file, many images
                        items.append((self._match(stem, None, images, ann), polys, False))
                elif isinstance(d, dict) and "shapes" in d:
                    polys, _ = self._parse_labelme(d)
                    items.append(
                        (self._match(ann.stem, d.get("imagePath"), images, ann), polys, False)
                    )
                else:
                    self.diag["unparsed"].append(str(ann))
            except Exception as e:  # noqa: BLE001 - report and continue
                self.diag["unparsed"].append(f"{ann}: {e}")

        recs = []
        for img, polys, boxes_only in items:
            if img is None:
                continue
            polys = [(label_map.get(c, c), p) for c, p in polys]
            gid = img.stem
            if group_regex:
                mm = re.match(group_regex, img.stem)
                gid = mm.group(1) if mm else img.stem
            recs.append(
                Record(img.stem, img, gid, img.stem, "full", polygons=polys, boxes_only=boxes_only)
            )
        return recs

    def _match(self, stem: str, image_path: str | None, images: dict, ann: Path) -> Path | None:
        for key in (stem, Path(image_path.replace("\\", "/")).stem if image_path else None):
            if key and key.lower() in images:
                return images[key.lower()]
        self.diag["unmatched"].append(str(ann) if stem == ann.stem else f"{ann}:{stem}")
        return None

    @staticmethod
    def _parse_coco(d: dict):
        cats = {c["id"]: str(c["name"]).strip() for c in d.get("categories", [])}
        by_img: dict[int, list] = {im["id"]: [] for im in d["images"]}
        for a in d["annotations"]:
            name = cats.get(a.get("category_id"), str(a.get("category_id")))
            seg = a.get("segmentation")
            if isinstance(seg, list) and seg:
                for poly in seg:
                    pts = np.asarray(poly, np.float32).reshape(-1, 2)
                    if len(pts) >= 3:
                        by_img[a["image_id"]].append((name, pts))
            elif a.get("bbox"):
                x, y, w, h = a["bbox"]
                pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
                by_img[a["image_id"]].append((name, pts))
        for im in d["images"]:
            yield Path(im["file_name"].replace("\\", "/")).stem, by_img[im["id"]]

    @staticmethod
    def _parse_json(p: Path):
        return SwrdSource._parse_labelme(json.loads(p.read_text(encoding="utf-8")))

    @staticmethod
    def _parse_labelme(d: dict):
        polys = []
        for s in d.get("shapes", []):
            pts = np.asarray(s["points"], np.float32)
            if s.get("shape_type") == "rectangle" and len(pts) == 2:
                (x0, y0), (x1, y1) = pts
                pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
            if len(pts) >= 3:
                polys.append((str(s["label"]).strip(), pts))
        return polys, False

    @staticmethod
    def _parse_voc(p: Path):
        root = ET.parse(p).getroot()
        polys = []
        for obj in root.iter("object"):
            b = obj.find("bndbox")
            x0, y0, x1, y1 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
            polys.append((obj.find("name").text.strip(), pts))
        return polys, True


SOURCES = {c.name: c for c in (RiawelcSource, GdxraySource, SwrdSource)}


def build_source(cfg: dict) -> Source:
    d = dict(cfg)
    kind = d.pop("source")
    root = d.pop("raw_dir")
    classes = d.pop("class_names")
    opts = d.pop("source_opts", {}) or {}
    return SOURCES[kind](root, classes, **opts)
