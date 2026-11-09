"""Per-domain evaluation for Phase 2 (PSNR / SSIM / optional LPIPS / optional loss).

Extends the conventions of src/metrics/evaluation.py (kept unchanged for the
LOL-v1 baseline): metrics on [0, 1] RGB, optional 8-bit quantization of
predictions (used for final testing so numbers match saved PNGs), per-image
records.  Adds: a custom forward function (e.g. tiled inference), LPIPS, extra
per-item metadata (level, scene id, selected exposure, ...), and group means.
"""

from collections import defaultdict
from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .evaluation import quantize_uint8
from .image_metrics import psnr, ssim
from .large_image import check_metric_values

META_SKIP = ("low", "high")


def _item(v, i):
    x = v[i]
    return x.item() if torch.is_tensor(x) else x


@torch.no_grad()
def evaluate_domain(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    forward_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    loss_fn: Optional[nn.Module] = None,
    lpips_fn: Optional[Callable] = None,
    quantize: bool = False,
    metric_device: Optional[torch.device] = None,
    on_output: Optional[Callable[[Dict, torch.Tensor], None]] = None,
    ssim_fn: Callable = ssim,
) -> Dict[str, object]:
    """``ssim_fn``: e.g. large_image.ssim_separable for very large images (same definition).
    Every metric value is sanity-checked (impossible values raise instead of being averaged).

    Returns {"n", "loss", "psnr", "ssim", "lpips", "per_image": [...]}."""
    was_training = model.training
    model.eval()
    mdev = metric_device or device
    rows = []
    for batch in loader:
        low = batch["low"]
        pred = forward_fn(model, low) if forward_fn else model(low.to(device))
        pred, high = pred.to(mdev), batch["high"].to(mdev)
        losses = [loss_fn(pred[i:i + 1], high[i:i + 1]).item() for i in range(len(pred))] if loss_fn else None
        if quantize:
            pred = quantize_uint8(pred)
        p, s = psnr(pred, high), ssim_fn(pred, high)
        lp = lpips_fn(pred, high) if lpips_fn else None
        for i in range(len(pred)):
            r = {k: _item(v, i) for k, v in batch.items() if k not in META_SKIP}
            r.update(psnr=p[i].item(), ssim=s[i].item())
            if lp is not None:
                r["lpips"] = lp[i].item()
            check_metric_values(str(r.get("name")), r["psnr"], r["ssim"], r.get("lpips"))
            if losses is not None:
                r["loss"] = losses[i]
            rows.append(r)
            if on_output is not None:
                on_output(r, pred[i].detach().cpu())
    if was_training:
        model.train()
    return summarize(rows)


def summarize(rows: Sequence[Dict]) -> Dict[str, object]:
    def mean(k):
        vals = [r[k] for r in rows if k in r]
        return sum(vals) / len(vals) if vals else None

    return {"n": len(rows), "loss": mean("loss"), "psnr": mean("psnr"), "ssim": mean("ssim"),
            "lpips": mean("lpips"), "per_image": list(rows)}


def group_summaries(rows: Sequence[Dict], key: str) -> Dict[str, Dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return {g: {k: v for k, v in summarize(rs).items() if k != "per_image"} for g, rs in sorted(groups.items())}
