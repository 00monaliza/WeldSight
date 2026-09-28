"""Tiling full radiographs into fixed-size patches and resizing ready-made patches."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Patch:
    image: np.ndarray  # uint8 HxW
    mask: np.ndarray | None  # uint8 HxW, 0 = background
    row: int  # top-left y in the source image (after padding)
    col: int  # top-left x


def _starts(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)  # cover the border without padding
    return starts


def tile(
    image: np.ndarray, mask: np.ndarray | None, size: int = 256, stride: int = 256
) -> list[Patch]:
    """Cut an image (and mask) into `size`x`size` patches.

    Images smaller than `size` along an axis are reflect-padded (mask with 0).
    The last row/column of patches is shifted inwards so no padding is needed for
    large images; with stride == size this gives at most one overlapping strip.
    """
    h, w = image.shape[:2]
    ph, pw = max(0, size - h), max(0, size - w)
    if ph or pw:
        image = np.pad(image, ((0, ph), (0, pw)), mode="reflect")
        if mask is not None:
            mask = np.pad(mask, ((0, ph), (0, pw)), mode="constant")
    h, w = image.shape[:2]
    out = []
    for y in _starts(h, size, stride):
        for x in _starts(w, size, stride):
            out.append(
                Patch(
                    image[y : y + size, x : x + size].copy(),
                    None if mask is None else mask[y : y + size, x : x + size].copy(),
                    y,
                    x,
                )
            )
    return out


def fit_patch(image: np.ndarray, size: int = 256, mode: str = "resize") -> np.ndarray:
    """Bring a ready-made patch (e.g. RIAWELC 227x227) to `size`x`size`."""
    h, w = image.shape[:2]
    if (h, w) == (size, size):
        return image
    if mode == "resize":
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    if mode == "pad":
        if h > size or w > size:
            raise ValueError(f"Cannot pad {h}x{w} patch down to {size}")
        return np.pad(image, ((0, size - h), (0, size - w)), mode="reflect")
    raise ValueError(f"Unknown fit mode {mode!r}")


def patch_labels(mask: np.ndarray, num_classes: int, min_pixels: int = 1) -> np.ndarray:
    """Multi-hot presence vector (K defect classes) from an indexed mask."""
    counts = np.bincount(mask.ravel(), minlength=num_classes + 1)[1 : num_classes + 1]
    return (counts >= min_pixels).astype(np.uint8)
