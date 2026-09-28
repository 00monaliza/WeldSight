"""Segmentation and detection metrics (per class)."""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


class SegConfusion:
    """Pixel confusion matrix accumulated over a dataset (0 = background)."""

    def __init__(self, num_classes_incl_bg: int):
        self.n = num_classes_incl_bg
        self.cm = np.zeros((self.n, self.n), np.int64)
        # object-level recall: GT connected components hit by a same-class prediction
        self.comp_total = np.zeros(self.n, np.int64)
        self.comp_hit = np.zeros(self.n, np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """pred, gt: int arrays [B,H,W] or [H,W]."""
        valid = gt >= 0
        idx = self.n * gt[valid].astype(np.int64) + pred[valid].astype(np.int64)
        self.cm += np.bincount(idx, minlength=self.n**2).reshape(self.n, self.n)
        for p, g in zip(
            np.atleast_3d(pred).reshape(-1, *pred.shape[-2:]),
            np.atleast_3d(gt).reshape(-1, *gt.shape[-2:]),
            strict=True,
        ):
            for k in range(1, self.n):
                gk = (g == k).astype(np.uint8)
                if not gk.any():
                    continue
                ncomp, lab = cv2.connectedComponents(gk, connectivity=8)
                pk = p == k
                for c in range(1, ncomp):
                    self.comp_total[k] += 1
                    self.comp_hit[k] += bool(pk[lab == c].any())

    def iou(self) -> np.ndarray:
        tp = np.diag(self.cm).astype(float)
        denom = self.cm.sum(0) + self.cm.sum(1) - tp
        return np.divide(tp, denom, out=np.full(self.n, np.nan), where=denom > 0)

    def pixel_recall(self) -> np.ndarray:
        tp = np.diag(self.cm).astype(float)
        gt = self.cm.sum(1)
        return np.divide(tp, gt, out=np.full(self.n, np.nan), where=gt > 0)

    def pixel_precision(self) -> np.ndarray:
        tp = np.diag(self.cm).astype(float)
        pr = self.cm.sum(0)
        return np.divide(tp, pr, out=np.full(self.n, np.nan), where=pr > 0)

    def component_recall(self) -> np.ndarray:
        return np.divide(
            self.comp_hit, self.comp_total, out=np.full(self.n, np.nan), where=self.comp_total > 0
        )

    def summary(self, class_names: list[str]) -> dict:
        iou, rec, prec, crec = (
            self.iou(),
            self.pixel_recall(),
            self.pixel_precision(),
            self.component_recall(),
        )
        out = {}
        for k, c in enumerate(class_names, start=1):
            out[c] = {
                "iou": iou[k],
                "pixel_recall": rec[k],
                "pixel_precision": prec[k],
                "component_recall": crec[k],
                "n_components": int(self.comp_total[k]),
            }
        out["_mean"] = {
            "iou": float(np.nanmean(iou[1:])),
            "component_recall": float(np.nanmean(crec[1:])),
        }
        return out


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true, y_pred = y_true.astype(bool), y_pred.astype(bool)
    return {
        "tp": int((y_true & y_pred).sum()),
        "fp": int((~y_true & y_pred).sum()),
        "fn": int((y_true & ~y_pred).sum()),
        "tn": int((~y_true & ~y_pred).sum()),
    }


def _safe_div(a, b):
    return float(a / b) if b else float("nan")


def detection_metrics(
    y_true: np.ndarray, prob: np.ndarray, threshold: float, class_names: list[str]
) -> dict:
    """Per-class patch-level recall/precision/F1/AUROC/AP for multi-label outputs."""
    y_pred = prob >= threshold
    out = {}
    for k, c in enumerate(class_names):
        cnt = binary_counts(y_true[:, k], y_pred[:, k])
        r = _safe_div(cnt["tp"], cnt["tp"] + cnt["fn"])
        p = _safe_div(cnt["tp"], cnt["tp"] + cnt["fp"])
        both = 0 < y_true[:, k].sum() < len(y_true)
        out[c] = {
            **cnt,
            "support": int(y_true[:, k].sum()),
            "recall": r,
            "precision": p,
            "f1": _safe_div(2 * p * r, p + r) if not (np.isnan(p) or np.isnan(r)) else float("nan"),
            "specificity": _safe_div(cnt["tn"], cnt["tn"] + cnt["fp"]),
            "auroc": float(roc_auc_score(y_true[:, k], prob[:, k])) if both else float("nan"),
            "ap": float(average_precision_score(y_true[:, k], prob[:, k]))
            if both
            else float("nan"),
        }
    any_true, any_pred = y_true.any(1), y_pred.any(1)
    cnt = binary_counts(any_true, any_pred)
    out["_any_defect"] = {
        **cnt,
        "recall": _safe_div(cnt["tp"], cnt["tp"] + cnt["fn"]),
        "precision": _safe_div(cnt["tp"], cnt["tp"] + cnt["fp"]),
    }
    keys = ["recall", "precision", "f1", "auroc", "ap"]
    out["_macro"] = {m: float(np.nanmean([out[c][m] for c in class_names])) for m in keys}
    return out


def cluster_bootstrap_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """95% CI of per-class recall, resampling whole radiographs (patches are not iid)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    idx_by = {c: np.flatnonzero(clusters == c) for c in uniq}
    K = y_true.shape[1]
    stats = np.full((n_boot, K), np.nan)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        t, p = y_true[idx].astype(bool), y_pred[idx].astype(bool)
        pos = t.sum(0)
        stats[b] = np.where(pos > 0, (t & p).sum(0) / np.maximum(pos, 1), np.nan)
    return np.nanpercentile(stats, [2.5, 97.5], axis=0).T  # [K, 2]
