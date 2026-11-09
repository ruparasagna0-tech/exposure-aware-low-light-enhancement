"""Shared evaluation loop (used for validation during training and for testing)."""

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .image_metrics import psnr, ssim


def quantize_uint8(x: torch.Tensor) -> torch.Tensor:
    """Round a [0, 1] tensor to the 256 levels of an 8-bit image (stays in [0, 1])."""
    return torch.round(x.clamp(0, 1) * 255.0) / 255.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    quantize: bool = False,
    on_output: Optional[Callable[[str, torch.Tensor], None]] = None,
) -> Dict[str, object]:
    """Run ``model`` over ``loader`` and compute per-image PSNR/SSIM (range [0, 1]).

    Args:
        loss_fn: if given, also report the mean loss (on un-quantized output).
        quantize: round predictions to 8-bit before computing metrics, so the
            numbers match what is measured on saved PNG files.
        on_output: optional callback ``(name, pred[3,H,W] on CPU)`` per image,
            e.g. to save enhanced images.

    Returns:
        ``{"loss", "psnr", "ssim"}`` means plus ``"per_image"`` list of dicts.
    """
    was_training = model.training
    model.eval()
    per_image, loss_sum, n = [], 0.0, 0
    for batch in loader:
        low = batch["low"].to(device)
        high = batch["high"].to(device)
        pred = model(low)
        if loss_fn is not None:
            loss_sum += loss_fn(pred, high).item() * low.shape[0]
        if quantize:
            pred = quantize_uint8(pred)
        p, s = psnr(pred, high), ssim(pred, high)
        for i, name in enumerate(batch["name"]):
            per_image.append({"name": name, "psnr": p[i].item(), "ssim": s[i].item()})
            if on_output is not None:
                on_output(name, pred[i].cpu())
        n += low.shape[0]
    if was_training:
        model.train()

    return {
        "loss": loss_sum / n if loss_fn is not None else None,
        "psnr": sum(r["psnr"] for r in per_image) / n,
        "ssim": sum(r["ssim"] for r in per_image) / n,
        "per_image": per_image,
    }
