"""U-Net (segmentation_models_pytorch) with MC-Dropout-ready dropout layers and a
patch-level multi-label classification head."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn


class WeldUNet(nn.Module):
    """U-Net with a pretrained encoder and two outputs.

    * seg logits  [B, K+1, H, W]   (class 0 = background, softmax over classes)
    * cls logits  [B, K]           (multi-label "class k present in the patch")

    Dropout sits (a) on the bottleneck features, (b) before the segmentation head
    and (c) inside the classification head, so that keeping dropout active at test
    time (MC Dropout) perturbs both outputs.
    """

    def __init__(
        self,
        num_classes: int,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 1,
        decoder_dropout: float = 0.2,
        bottleneck_dropout: float = 0.2,
        cls_dropout: float = 0.3,
        arch: str = "unet",
    ):
        super().__init__()
        if arch != "unet":
            raise ValueError("Only 'unet' is implemented in stage 1")
        base = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes + 1,
            aux_params={"classes": num_classes, "dropout": cls_dropout},
        )
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.segmentation_head = base.segmentation_head
        self.classification_head = base.classification_head
        self.bottleneck_dropout = nn.Dropout2d(bottleneck_dropout)
        self.decoder_dropout = nn.Dropout2d(decoder_dropout)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, with_seg: bool = True):
        feats = list(self.encoder(x))
        feats[-1] = self.bottleneck_dropout(feats[-1])
        cls_logits = self.classification_head(feats[-1])
        seg_logits = None
        if with_seg:
            seg_logits = self.segmentation_head(self.decoder_dropout(self.decoder(feats)))
        return seg_logits, cls_logits

    def forward_mc(self, x: torch.Tensor, T: int, with_seg: bool = True):
        """T stochastic passes sharing one deterministic encoder pass.

        Valid because every dropout layer sits after the encoder (smp CNN encoders
        have no dropout), so the encoder output is identical across MC samples.
        Returns seg logits [T,B,K+1,H,W] (or None) and cls logits [T,B,K].
        """
        base = list(self.encoder(x))
        segs, clss = [], []
        for _ in range(T):
            feats = base[:-1] + [self.bottleneck_dropout(base[-1])]
            clss.append(self.classification_head(feats[-1].clone()))
            if with_seg:
                segs.append(self.segmentation_head(self.decoder_dropout(self.decoder(feats))))
        return (torch.stack(segs) if with_seg else None), torch.stack(clss)


def build_model(cfg: dict, num_classes: int) -> WeldUNet:
    m = cfg["model"]
    return WeldUNet(
        num_classes=num_classes,
        encoder_name=m["encoder_name"],
        encoder_weights=m.get("encoder_weights"),
        in_channels=m.get("in_channels", 1),
        decoder_dropout=m.get("decoder_dropout", 0.2),
        bottleneck_dropout=m.get("bottleneck_dropout", 0.2),
        cls_dropout=m.get("cls_dropout", 0.3),
        arch=m.get("arch", "unet"),
    )


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """Eval mode everywhere (BatchNorm uses running stats) except dropout layers."""
    model.eval()
    for mod in model.modules():
        if isinstance(mod, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            mod.train()
    return model
