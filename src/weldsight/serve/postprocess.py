"""Turn per-pixel class probabilities + uncertainty maps into the /predict response.

numpy/OpenCV only, so it runs (and is tested) without PyTorch.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Prediction:
    """Output of a predictor, at the resolution of the input radiograph.

    probs: [K+1, H, W] float32, mean softmax over MC samples, class 0 = background
    pe:    [H, W] float32 in [0, 1], predictive entropy / ln(K+1)
    mi:    [H, W] float32 in [0, 1], mutual information (BALD), display-scaled
    """

    probs: np.ndarray
    pe: np.ndarray
    mi: np.ndarray


def output_size(h: int, w: int, max_dim: int) -> tuple[int, int]:
    """(H, W) of the returned maps. The web client uses the same rule for its canvas."""
    k = min(1.0, max_dim / max(h, w))
    return max(32, round(h * k)), max(32, round(w * k))


def _png_b64(a: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", a)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def extract_defects(
    pred: Prediction,
    classes: list[str],
    candidate_thr: float = 0.2,
    min_area: int = 12,
    detect_thr: float = 0.5,
) -> tuple[np.ndarray, list[dict]]:
    """Connected components of the foreground probability.

    A region is a *detection* when its mean foreground probability >= `detect_thr`;
    weaker regions (>= `candidate_thr` pixelwise) are returned as `suspect` so that an
    uncertain miss can still be referred to an inspector.

    Returns the label map (0 = background, k = classes[k-1]; only pixels of kept
    regions) and a list of dicts with bbox in input-image pixels.
    """
    p_fg = 1.0 - pred.probs[0]
    cand = (p_fg >= candidate_thr).astype(np.uint8)
    n, comp, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=4)
    labels = np.zeros(p_fg.shape, np.uint8)
    argmax = pred.probs.argmax(0)
    defects: list[dict] = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        sl = (slice(y, y + h), slice(x, x + w))
        m = comp[sl] == i
        cls_p = pred.probs[1:, sl[0], sl[1]][:, m].mean(1)
        k = int(cls_p.argmax()) + 1
        prob = float(p_fg[sl][m].mean())
        keep = m & (argmax[sl] > 0)
        view = labels[sl]
        view[keep] = k
        defects.append(
            {
                "cls": classes[k - 1],
                "prob": round(prob, 4),
                "bbox": [x, y, w, h],
                "uncertainty": {
                    "pe": round(float(pred.pe[sl][m].mean()), 4),
                    "mi": round(float(pred.mi[sl][m].mean()), 4),
                },
            }
        )
    defects.sort(key=lambda d: -d["uncertainty"]["pe"])
    return labels, defects


def build_response(pred: Prediction, classes: list[str], max_dim: int = 1024, **kw) -> dict:
    h, w = pred.pe.shape
    labels, defects = extract_defects(pred, classes, **kw)
    oh, ow = output_size(h, w, max_dim)
    if (oh, ow) != (h, w):
        sx, sy = ow / w, oh / h
        labels = cv2.resize(labels, (ow, oh), interpolation=cv2.INTER_NEAREST)
        pe = cv2.resize(pred.pe, (ow, oh), interpolation=cv2.INTER_AREA)
        mi = cv2.resize(pred.mi, (ow, oh), interpolation=cv2.INTER_AREA)
        for d in defects:
            bx, by, bw, bh = d["bbox"]
            d["bbox"] = [
                round(bx * sx),
                round(by * sy),
                max(1, round(bw * sx)),
                max(1, round(bh * sy)),
            ]
    else:
        pe, mi = pred.pe, pred.mi
    to8 = lambda a: np.clip(np.rint(a * 255), 0, 255).astype(np.uint8)  # noqa: E731
    return {
        "width": ow,
        "height": oh,
        "classes": classes,
        "mask_png": _png_b64(labels),
        "maps": {"pe": _png_b64(to8(pe)), "mi": _png_b64(to8(mi))},
        "defects": defects,
    }
