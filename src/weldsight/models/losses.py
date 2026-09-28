"""Dice + Focal segmentation loss and focal BCE for the patch-level head."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from torch import nn


class SegClsLoss(nn.Module):
    """total = w_dice*Dice + w_focal*Focal (only samples with masks) + w_cls*FocalBCE.

    Samples without masks (label-only datasets such as RIAWELC) contribute only
    to the classification term, so mixed mask/label batches are handled.
    """

    def __init__(self, dice_w=1.0, focal_w=1.0, gamma=2.0, cls_w=1.0, cls_gamma=2.0):
        super().__init__()
        self.dice = DiceLoss(mode="multiclass", from_logits=True)
        self.focal = FocalLoss(mode="multiclass", gamma=gamma)
        self.dice_w, self.focal_w, self.cls_w, self.cls_gamma = dice_w, focal_w, cls_w, cls_gamma

    def forward(self, seg_logits, cls_logits, masks, labels, has_mask) -> dict[str, torch.Tensor]:
        out = {}
        zero = cls_logits.sum() * 0.0
        if seg_logits is not None and has_mask.any():
            sl, m = seg_logits[has_mask], masks[has_mask]
            out["dice"] = self.dice(sl, m)
            out["focal"] = self.focal(sl, m)
        else:
            out["dice"] = out["focal"] = zero
        out["cls"] = focal_bce(cls_logits, labels, self.cls_gamma)
        out["total"] = (
            self.dice_w * out["dice"] + self.focal_w * out["focal"] + self.cls_w * out["cls"]
        )
        return out


def focal_bce(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if gamma:
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        bce = bce * (1 - pt) ** gamma
    return bce.mean()


def build_loss(cfg: dict) -> SegClsLoss:
    lc = cfg["loss"]
    return SegClsLoss(
        lc.get("seg_dice_weight", 1.0),
        lc.get("seg_focal_weight", 1.0),
        lc.get("focal_gamma", 2.0),
        lc.get("cls_weight", 1.0),
        lc.get("cls_focal_gamma", 2.0),
    )
