"""Memory-bounded SSIM for very large images (e.g. 20-24 MP SICE) + metric sanity checks.

``ssim_separable`` computes EXACTLY the same quantity as ``image_metrics.ssim`` (same 11x11
Gaussian window sigma 1.5, K1/K2, data range 1, 'valid' filtering, mean over all valid
positions and RGB channels): the 2-D window is the outer product of the same 1-D Gaussian,
so filtering is done separably (row then column), and the image is processed in horizontal
strips (with window-1 rows of overlap, so every valid output position is computed once).
Only floating-point summation order differs (accumulated in float64); the smoke test checks
agreement with ``image_metrics.ssim``.
"""

import math

import torch
import torch.nn.functional as F

from .image_metrics import DATA_RANGE, _prepare


def _gaussian_1d(size, sigma, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


@torch.no_grad()
def ssim_separable(pred, target, window_size=11, sigma=1.5, k1=0.01, k2=0.03, strip=1024):
    """Per-image SSIM, identical definition to image_metrics.ssim; inputs in [0, 1]."""
    pred, target = _prepare(pred, target)
    b, c, h, w = pred.shape
    if min(h, w) < window_size:
        raise ValueError(f"Image smaller than SSIM window ({window_size})")
    g = _gaussian_1d(window_size, sigma, pred.device, pred.dtype)
    gh = g.view(1, 1, 1, -1).expand(c, 1, 1, window_size).contiguous()
    gv = g.view(1, 1, -1, 1).expand(c, 1, window_size, 1).contiguous()
    c1, c2 = (k1 * DATA_RANGE) ** 2, (k2 * DATA_RANGE) ** 2

    def filt(x):
        return F.conv2d(F.conv2d(x, gh, groups=c), gv, groups=c)

    out_rows = h - window_size + 1
    total = torch.zeros(b, dtype=torch.float64, device=pred.device)
    count = 0
    for r0 in range(0, out_rows, strip):
        r1 = min(r0 + strip, out_rows)
        x = pred[:, :, r0:r1 + window_size - 1]
        y = target[:, :, r0:r1 + window_size - 1]
        mu_x, mu_y = filt(x), filt(y)
        sxx = filt(x * x) - mu_x**2
        syy = filt(y * y) - mu_y**2
        sxy = filt(x * y) - mu_x * mu_y
        m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
        total += m.double().sum(dim=(1, 2, 3))
        count += m[0].numel()
    return (total / count).float()


def check_metric_values(name, psnr_v, ssim_v, lpips_v=None):
    """Raise if a metric value is impossible (guards against silent numerical/device failures)."""
    if not math.isfinite(psnr_v):
        raise RuntimeError(f"{name}: non-finite PSNR {psnr_v}")
    if not (-1.0 - 1e-4 <= ssim_v <= 1.0 + 1e-4):
        raise RuntimeError(f"{name}: SSIM {ssim_v} outside [-1, 1]")
    if lpips_v is not None and not (math.isfinite(lpips_v) and -1e-6 <= lpips_v <= 2.0):
        raise RuntimeError(f"{name}: LPIPS {lpips_v} outside sanity range [0, 2]")
