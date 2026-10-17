"""Predictors: a trained WeldUNet (MC Dropout, tiled) and a dependency-free stub."""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from weldsight.serve.postprocess import Prediction

EPS = 1e-7
MI_GAIN = 4.0  # mutual information is tiny next to entropy; scaled for display/thresholding


class Predictor(Protocol):
    name: str
    classes: list[str]
    stub: bool
    info: dict

    def predict(self, gray: np.ndarray, mc_samples: int) -> Prediction: ...


class TorchPredictor:
    """Loads `runs/<run>/{config.yaml,best.pt}` and runs tiled MC-Dropout inference.

    The image is cut into `data.patch_size` tiles exactly like val/test data (no
    overlap, last row/column shifted inwards), each tile is normalised like training,
    and overlapping predictions are averaged. Needs a model trained **with masks**
    (e.g. SWRD); a label-only run (RIAWELC) has an untrained decoder.
    """

    stub = False

    def __init__(self, run_dir: str | Path, device: str = "auto", batch_size: int = 8):
        import torch

        from weldsight.config import load_config
        from weldsight.models.unet import build_model, enable_mc_dropout
        from weldsight.utils import select_device

        self._torch = torch
        run_dir = Path(run_dir)
        cfg = load_config(run_dir / "config.yaml")
        data = cfg["data"]
        self.classes = list(data["class_names"])
        if data.get("source") == "riawelc":
            raise ValueError(
                f"{run_dir.name} was trained on RIAWELC (patch labels, no masks): its "
                "segmentation decoder is untrained. Serve a model trained on SWRD/GDXray."
            )
        self.patch = int(data.get("patch_size", 256))
        self.norm = cfg["train"].get("normalize", "per_patch")
        self.batch_size = batch_size
        self.device = select_device(device)  # "auto" = best device of *this* host
        model = build_model(
            {**cfg, "model": {**cfg["model"], "encoder_weights": None}}, len(self.classes)
        )
        # tensors + primitives only (see train.py), so no pickle code execution is needed
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        if ckpt.get("classes") is not None and list(ckpt["classes"]) != self.classes:
            raise ValueError(
                f"class order in {run_dir.name}/config.yaml {self.classes} differs from the "
                f"checkpoint's {list(ckpt['classes'])}; refusing to mislabel predictions"
            )
        model.load_state_dict(ckpt["model"])
        self.model = enable_mc_dropout(model.to(self.device))
        self.name = f"{cfg.get('experiment', run_dir.name)} (epoch {ckpt.get('epoch', '?')})"
        self.info = {
            "device": str(self.device),
            "patch_size": self.patch,
            "normalize": self.norm,
            "run": run_dir.name,
        }

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32) / 255.0
        if self.norm == "per_patch":
            return (x - x.mean()) / (x.std() + 1e-6)
        if self.norm == "fixed":
            return (x - 0.5) / 0.25
        return x

    def predict(self, gray: np.ndarray, mc_samples: int) -> Prediction:
        from weldsight.data.patching import tile

        torch = self._torch
        h, w = gray.shape
        patches = tile(gray, None, self.patch, self.patch)
        C = len(self.classes) + 1
        ph, pw = max(h, self.patch), max(w, self.patch)  # tile() reflect-pads small images
        acc = np.zeros((C, ph, pw), np.float32)
        ee_acc, cnt = (np.zeros((ph, pw), np.float32) for _ in range(2))
        norm = math.log(C)
        # forward_mc keeps all T logit tensors: bound them by shrinking the batch as T grows
        bs = max(1, min(self.batch_size, 64 // mc_samples))
        with torch.no_grad():
            for i in range(0, len(patches), bs):
                chunk = patches[i : i + bs]
                x = torch.from_numpy(np.stack([self._normalize(p.image) for p in chunk]))[
                    :, None
                ].to(self.device)
                seg, _ = self.model.forward_mc(x, mc_samples, with_seg=True)
                S = torch.softmax(seg.float(), 2)  # [T,B,C,H,W]
                mean = S.mean(0).cpu().numpy()
                ee = (-(S * torch.log(S + EPS)).sum(2).mean(0)).cpu().numpy()
                for j, p in enumerate(chunk):
                    sl = (slice(p.row, p.row + self.patch), slice(p.col, p.col + self.patch))
                    acc[(slice(None), *sl)] += mean[j]
                    ee_acc[sl] += ee[j]
                    cnt[sl] += 1
        # Average the per-tile mean probabilities and expected entropies, then derive pe/mi from
        # the stitched mean, so where shifted tiles overlap the maps describe the returned probs.
        acc /= cnt
        ee_acc /= cnt
        pe_acc = -(acc * np.log(acc + EPS)).sum(0)
        mi_acc = np.maximum(pe_acc - ee_acc, 0)
        return Prediction(
            probs=acc[:, :h, :w],
            pe=np.clip(pe_acc[:h, :w] / norm, 0, 1),
            mi=np.clip(mi_acc[:h, :w] * MI_GAIN / norm, 0, 1),
        )


class StubPredictor:
    """Blur-difference heuristic with simulated MC noise. NOT a trained model.

    Lets the server and UI be exercised without PyTorch or a checkpoint. Every
    foreground pixel is assigned to the first class; classification is not simulated.
    """

    stub = True

    def __init__(self, classes: list[str] | None = None):
        self.classes = classes or [
            "porosity",
            "inclusion",
            "crack",
            "undercut",
            "lack_of_fusion",
            "lack_of_penetration",
        ]
        self.name = "stub (heuristic, not a trained model)"
        self.info = {"device": "cpu"}
        self._rng = np.random.default_rng(0)

    def predict(self, gray: np.ndarray, mc_samples: int) -> Prediction:
        g = gray.astype(np.float32)
        z = cv2.blur(g, (29, 29)) - cv2.blur(g, (3, 3))
        med = np.median(z[::7, ::7])
        sd = max(0.5, 1.4826 * float(np.median(np.abs(z[::7, ::7] - med))))
        z = (z - med) / sd
        idx = np.flatnonzero(z >= 2.5)  # everything else has p = 0, so only these pixels matter
        zz = z.ravel()[idx]
        sum_p, sum_h = np.zeros(idx.size, np.float32), np.zeros(idx.size, np.float32)
        for _ in range(mc_samples):  # one pass at a time: memory stays O(pixels)
            noise = self._rng.standard_normal(idx.size).astype(np.float32) * 0.9
            q = np.clip(1 / (1 + np.exp(-(zz + noise - 5.2) / 0.8)), EPS, 1 - EPS)
            sum_p += q
            sum_h += -(q * np.log(q) + (1 - q) * np.log(1 - q))
        mean = np.clip(sum_p / mc_samples, EPS, 1 - EPS)
        h_mean = -(mean * np.log(mean) + (1 - mean) * np.log(1 - mean))
        fg, pe, mi = (np.zeros(z.size, np.float32) for _ in range(3))
        fg[idx] = sum_p / mc_samples
        pe[idx] = h_mean / math.log(2)
        mi[idx] = np.clip((h_mean - sum_h / mc_samples) / 0.3, 0, 1)
        probs = np.zeros((len(self.classes) + 1, *z.shape), np.float32)
        probs[0] = 1 - fg.reshape(z.shape)
        probs[1] = fg.reshape(z.shape)
        return Prediction(probs, pe.reshape(z.shape), mi.reshape(z.shape))


class LockedPredictor:
    """Serialises predict() calls: dropout state and the GPU are not re-entrant."""

    def __init__(self, inner: Predictor):
        self._inner, self._lock = inner, threading.Lock()
        self.name, self.classes, self.stub, self.info = (
            inner.name,
            inner.classes,
            inner.stub,
            inner.info,
        )

    def predict(self, gray: np.ndarray, mc_samples: int) -> Prediction:
        with self._lock:
            return self._inner.predict(gray, mc_samples)
