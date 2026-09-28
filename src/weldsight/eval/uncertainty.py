"""MC Dropout inference and uncertainty scores."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from weldsight.eval.metrics import SegConfusion
from weldsight.models.unet import enable_mc_dropout

EPS = 1e-7


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def cls_uncertainty(samples: np.ndarray) -> dict[str, np.ndarray]:
    """samples: [T, N, K] sigmoid probabilities from T stochastic passes.

    predictive_entropy = sum_k H(mean_t p)          (total uncertainty)
    expected_entropy   = mean_t sum_k H(p_t)        (aleatoric part)
    mutual_information = predictive - expected      (epistemic part, BALD)
    """
    mean = samples.mean(0)
    pe = binary_entropy(mean).sum(-1)
    ee = binary_entropy(samples).sum(-1).mean(0)
    return {
        "predictive_entropy": pe,
        "expected_entropy": ee,
        "mutual_information": np.maximum(pe - ee, 0),
        "max_prob": 0.5 - np.abs(mean - 0.5).min(-1),  # closeness of the least-sure class to 0.5
    }


@torch.no_grad()
def mc_predict(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    T: int = 20,
    with_seg: bool = True,
    num_classes: int | None = None,
    top_frac: float = 0.01,
) -> dict:
    """Run T stochastic forward passes per batch.

    Returns per-patch arrays (cls probs & uncertainties, seg-derived presence probs
    and pixel uncertainty) plus a dataset-level `SegConfusion` if masks exist.
    Pixel-level maps are not stored to keep memory O(N).
    """
    enable_mc_dropout(model)
    cls_samples, idxs, labels = [], [], []
    seg_presence, seg_unc_pe, seg_unc_mi = [], [], []
    conf = SegConfusion(num_classes + 1) if (with_seg and num_classes) else None
    any_mask = False
    for batch in tqdm(loader, desc=f"MC dropout (T={T})", leave=False):
        x = batch["image"].to(device)
        seg, cls = model.forward_mc(x, T, with_seg=with_seg)
        cls_samples.append(torch.sigmoid(cls.float()).cpu().numpy())
        idxs.append(batch["idx"].numpy())
        labels.append(batch["labels"].numpy())
        if with_seg:
            S = torch.softmax(seg.float(), 2).cpu()  # [T,B,C,H,W]
            mean = S.mean(0)
            pe = -(mean * torch.log(mean + EPS)).sum(1)  # [B,H,W]
            ee = -(S * torch.log(S + EPS)).sum(2).mean(0)
            mi = (pe - ee).clamp_min(0)
            k = max(1, int(top_frac * pe[0].numel()))
            seg_unc_pe.append(pe.flatten(1).topk(k, 1).values.mean(1).numpy())
            seg_unc_mi.append(mi.flatten(1).topk(k, 1).values.mean(1).numpy())
            seg_presence.append(mean[:, 1:].flatten(2).max(-1).values.numpy())  # [B,K]
            hm = batch["has_mask"].numpy()
            if conf is not None and hm.any():
                any_mask = True
                pred = mean.argmax(1).numpy()
                conf.update(pred[hm], batch["mask"].numpy()[hm])
    out = {
        "idx": np.concatenate(idxs),
        "labels": np.concatenate(labels),
        "cls_samples": np.concatenate(cls_samples, axis=1),  # [T,N,K]
    }
    out["cls_prob"] = out["cls_samples"].mean(0)
    out["cls_unc"] = cls_uncertainty(out["cls_samples"])
    if with_seg:
        out["seg_prob"] = np.concatenate(seg_presence)
        out["seg_unc"] = {
            "predictive_entropy": np.concatenate(seg_unc_pe),
            "mutual_information": np.concatenate(seg_unc_mi),
        }
    out["seg_confusion"] = conf if any_mask else None
    return out
