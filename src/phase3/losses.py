"""Phase 3 training losses, one per method ("method-faithful", Tier A).

  l1               U-Net (Phase 2) and Retinexformer (official BasicSR config: L1Loss, weight 1, mean).
  cidnet_official  HVI-CIDNet train.py / data/options.py defaults:
                     loss = loss_rgb + HVI_weight * loss_hvi,
                     loss_x = L1 + 0.5 * (1 - SSIM) + 50 * Edge + 0.01 * Perceptual(VGG-19 conv1_2/2_2/3_4/4_4, MSE)
                   computed on RGB and on HVI (the network's own RGB->HVI transform), HVI_weight = 1.0.
  darkir_paper     DarkIR paper (no public training config): L1 + 0.01 * VGG-19 feature loss (5 slices, L1,
                   weights 1/32..1) + 50 * Edge (Laplacian-pyramid MSE) + low-light guide loss on the side output
                   (EnhanceLoss: side upsampled x8 nearest vs reference, L1 weight 1 + its internal 0.01 * VGG).
                   Weights from the paper text; the enhance-loss weight is not stated in the paper (1.0 used here).

All loss modules are the vendored originals (third_party/*/loss*), called exactly as the official code does.
Each loss returns (total, parts) where parts is a dict of float components for logging.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

LOSS_NAMES = ("l1", "cidnet_official", "darkir_paper")


class L1Objective(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="mean")

    def forward(self, pred, target, side=None, model=None):
        loss = self.l1(pred, target)
        return loss, {"l1": float(loss.detach())}


class CIDNetObjective(nn.Module):
    def __init__(self, l1_weight=1.0, d_weight=0.5, e_weight=50.0, p_weight=1e-2, hvi_weight=1.0):
        super().__init__()
        from third_party.hvi_cidnet.loss.losses import SSIM, EdgeLoss, L1Loss, PerceptualLoss
        self.L1 = L1Loss(loss_weight=l1_weight, reduction="mean")
        self.D = SSIM(weight=d_weight)
        self.E = EdgeLoss(loss_weight=e_weight)
        # official init_loss(): PerceptualLoss(..., perceptual_weight=1.0, criterion='mse'); train loop multiplies by P_weight
        self.P = PerceptualLoss({"conv1_2": 1, "conv2_2": 1, "conv3_4": 1, "conv4_4": 1}, perceptual_weight=1.0,
                                criterion="mse")
        self.p_weight, self.hvi_weight = p_weight, hvi_weight

    def _space(self, a, b):
        l1, d, e = self.L1(a, b), self.D(a, b), self.E(a, b)
        p = self.P(a, b)[0]
        return l1 + d + e + self.p_weight * p, (l1, d, e, p)

    def forward(self, pred, target, side=None, model=None):
        if model is None or not hasattr(model, "hvi"):
            raise ValueError("cidnet_official needs the CIDNet wrapper (for its HVI transform)")
        loss_rgb, prgb = self._space(pred, target)
        loss_hvi, phvi = self._space(model.hvi(pred), model.hvi(target))
        loss = loss_rgb + self.hvi_weight * loss_hvi
        parts = {f"{s}_{k}": float(v.detach()) for s, vals in (("rgb", prgb), ("hvi", phvi))
                 for k, v in zip(("l1", "ssim", "edge", "perc"), vals)}
        return loss, parts


class DarkIRObjective(nn.Module):
    def __init__(self, perceptual_weight=1e-2, edge_weight=50.0, enhance_weight=1.0, scale_factor=8):
        super().__init__()
        from third_party.darkir.losses.loss import EdgeLoss, EnhanceLoss, L1Loss, VGGLoss
        self.pixel = L1Loss()
        self.perc = VGGLoss(loss_weight=perceptual_weight, criterion="l1", reduction="mean")
        self.edge = EdgeLoss(rank="cpu", loss_weight=edge_weight, criterion="l2", reduction="mean")
        self.enh = EnhanceLoss(loss_weight=enhance_weight, criterion="l1", reduction="mean")
        self.scale_factor = scale_factor

    def forward(self, pred, target, side=None, model=None):
        if side is None:
            raise ValueError("darkir_paper needs the side output (forward_train)")
        self.edge.kernel = self.edge.kernel.to(pred.device)
        lp, lv, le = self.pixel(pred, target), self.perc(pred, target), self.edge(pred, target)
        # official calculate_loss(): all_losses['enhance_loss'](outside_batch, high_batch, scale_factor=scale_factor)
        ln = self.enh(side, target, scale_factor=self.scale_factor)
        loss = lp + lv + le + ln
        return loss, {"l1": float(lp.detach()), "perc": float(lv.detach()), "edge": float(le.detach()),
                      "enhance": float(ln.detach())}


def build_phase3_loss(name: str) -> nn.Module:
    if name == "l1":
        return L1Objective()
    if name == "cidnet_official":
        return CIDNetObjective()
    if name == "darkir_paper":
        return DarkIRObjective()
    raise ValueError(f"unknown Phase-3 loss {name!r}; available {LOSS_NAMES}")
