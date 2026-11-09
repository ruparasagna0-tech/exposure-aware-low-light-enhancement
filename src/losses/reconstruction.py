"""Supervised reconstruction losses.

Currently only L1 (mean absolute error) between the enhanced prediction and the
high-light reference, both in [0, 1]. Other terms (perceptual, SSIM, colour,
...) are intentionally not included at the baseline stage.
"""

import torch.nn as nn

LOSSES = {"l1": nn.L1Loss}


def build_loss(loss_cfg: dict) -> nn.Module:
    """Build a loss from a config dict like ``{"name": "l1"}``."""
    name = loss_cfg["name"].lower()
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSSES)}")
    return LOSSES[name](reduction="mean")
