"""Full-reference image metrics: PSNR and SSIM.

EXPECTED RANGE: all inputs are float tensors in [0, 1] (the range produced by
the dataset loader and the model's sigmoid output), shape [B, C, H, W] or
[C, H, W]. ``data_range`` is fixed to 1.0 accordingly. Inputs outside [0, 1]
raise an error instead of being silently rescaled, so e.g. 0-255 images are
caught.

Both metrics are computed on RGB (all channels), per image, and return one
value per image; average them over the dataset.

Note for comparing with published numbers: LLIE papers differ in SSIM details
(RGB vs. grayscale/Y, Gaussian vs. uniform window, uint8 rounding). This
implementation follows Wang et al. (2004): 11x11 Gaussian window, sigma=1.5,
K1=0.01, K2=0.03, no padding, averaged over RGB channels.
"""

import torch
import torch.nn.functional as F

DATA_RANGE = 1.0
_RANGE_TOL = 1e-4


def _prepare(pred: torch.Tensor, target: torch.Tensor):
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    if pred.dim() == 3:
        pred, target = pred.unsqueeze(0), target.unsqueeze(0)
    if pred.dim() != 4:
        raise ValueError(f"Expected [B,C,H,W] or [C,H,W], got {tuple(pred.shape)}")
    for name, t in (("pred", pred), ("target", target)):
        lo, hi = t.min().item(), t.max().item()
        if lo < -_RANGE_TOL or hi > DATA_RANGE + _RANGE_TOL:
            raise ValueError(
                f"{name} values in [{lo:.4f}, {hi:.4f}]; metrics expect [0, 1]"
            )
    return pred.float().clamp(0, 1), target.float().clamp(0, 1)


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB, per image. Inputs in [0, 1].

    PSNR = 10 * log10(DATA_RANGE^2 / MSE). MSE is clamped at ``eps`` so identical
    images give a finite 100 dB instead of inf.
    """
    pred, target = _prepare(pred, target)
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1).clamp_min(eps)
    return 10.0 * torch.log10(DATA_RANGE**2 / mse)


def _gaussian_window(size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = g[:, None] @ g[None, :]  # separable 2-D Gaussian
    return window.expand(channels, 1, size, size).contiguous()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Structural similarity, per image (mean over pixels and channels). Inputs in [0, 1]."""
    pred, target = _prepare(pred, target)
    c = pred.shape[1]
    if min(pred.shape[-2:]) < window_size:
        raise ValueError(f"Image smaller than SSIM window ({window_size})")
    win = _gaussian_window(window_size, sigma, c, pred.device, pred.dtype)
    c1, c2 = (k1 * DATA_RANGE) ** 2, (k2 * DATA_RANGE) ** 2

    def filt(x):  # per-channel Gaussian filtering, 'valid' region only
        return F.conv2d(x, win, groups=c)

    mu_x, mu_y = filt(pred), filt(target)
    sigma_x = filt(pred * pred) - mu_x**2
    sigma_y = filt(target * target) - mu_y**2
    sigma_xy = filt(pred * target) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.flatten(1).mean(dim=1)
