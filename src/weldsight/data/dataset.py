"""PyTorch dataset over a prepared manifest."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_manifest(processed_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    processed_dir = Path(processed_dir)
    df = pd.read_csv(processed_dir / "manifest.csv", keep_default_na=False)
    card = json.loads((processed_dir / "dataset_card.json").read_text())
    return df, card


class BasicAugment:
    """Label-preserving photometric/geometric augmentation.

    Applied identically in every experiment arm (baseline / copy-paste / diffusion),
    so that the arms differ only in how *defects* are augmented.
    """

    def __init__(self, hflip=0.5, vflip=0.5, brightness=0.1, contrast=0.1, gamma=0.1, noise=0.0):
        self.hflip, self.vflip = hflip, vflip
        self.brightness, self.contrast, self.gamma, self.noise = brightness, contrast, gamma, noise

    def __call__(self, img: np.ndarray, mask: np.ndarray | None, rng: np.random.Generator):
        if rng.random() < self.hflip:
            img = img[:, ::-1]
            mask = None if mask is None else mask[:, ::-1]
        if rng.random() < self.vflip:
            img = img[::-1]
            mask = None if mask is None else mask[::-1]
        x = img.astype(np.float32) / 255.0
        if self.gamma:
            x = x ** np.exp(rng.uniform(-self.gamma, self.gamma))
        if self.contrast:
            m = x.mean()
            x = (x - m) * (1 + rng.uniform(-self.contrast, self.contrast)) + m
        if self.brightness:
            x = x + rng.uniform(-self.brightness, self.brightness)
        if self.noise:
            x = x + rng.normal(0, self.noise, x.shape).astype(np.float32)
        x = np.clip(x, 0, 1)
        return np.ascontiguousarray(x), None if mask is None else np.ascontiguousarray(mask)


class WeldPatchDataset(Dataset):
    """Returns dict(image[1,H,W] float, mask[H,W] long (or -1 filled), labels[K] float,
    has_mask bool, idx int)."""

    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        class_names: list[str],
        augment: BasicAugment | None = None,
        normalize: str = "per_patch",
        subset: float | int | None = None,
        seed: int = 0,
        df: pd.DataFrame | None = None,
    ):
        self.root = Path(processed_dir)
        if df is None:
            df, _ = load_manifest(self.root)
        df = df[df["split"] == split].reset_index(drop=True)
        if subset:
            n = int(subset * len(df)) if isinstance(subset, float) and subset <= 1 else int(subset)
            df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
        self.df = df
        self.class_names = class_names
        self.label_cols = [f"cls_{c}" for c in class_names]
        self.augment = augment
        self.normalize = normalize
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def labels(self) -> np.ndarray:
        return self.df[self.label_cols].to_numpy(np.float32)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        img = cv2.imread(str(self.root / row["image"]), cv2.IMREAD_GRAYSCALE)
        mask = None
        if row["has_mask"]:
            mask = cv2.imread(str(self.root / row["mask"]), cv2.IMREAD_GRAYSCALE)
        if self.augment is not None:
            # worker-safe randomness: numpy global state is re-seeded per worker
            x, mask = self.augment(img, mask, np.random.default_rng(np.random.randint(2**31)))
        else:
            x = img.astype(np.float32) / 255.0
        if self.normalize == "per_patch":
            x = (x - x.mean()) / (x.std() + 1e-6)
        elif self.normalize == "fixed":
            x = (x - 0.5) / 0.25
        out = {
            "image": torch.from_numpy(np.ascontiguousarray(x, np.float32))[None],
            "labels": torch.from_numpy(row[self.label_cols].to_numpy(np.float32)),
            "has_mask": torch.tensor(bool(row["has_mask"])),
            "idx": torch.tensor(i),
        }
        if mask is None:
            out["mask"] = torch.full(x.shape, -1, dtype=torch.long)
        else:
            out["mask"] = torch.from_numpy(mask.astype(np.int64))
        return out
