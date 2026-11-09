"""LPIPS (Zhang et al. 2018) via the official ``lpips`` package.

Approved setting: AlexNet backbone (``net="alex"``, lpips v0.1 linear weights).
Inputs are the project's usual float tensors in [0, 1] (same range check as
PSNR/SSIM, no silent rescaling); ``normalize=True`` makes the package map them to
[-1, 1] internally, which is the range LPIPS expects.  Lower is better.
Computed on RGB at the same resolution as PSNR/SSIM (whole image, spatially averaged).
First use downloads torchvision's ImageNet AlexNet weights (internet required once).
"""

import torch

from .image_metrics import _prepare


class LPIPSMetric:
    def __init__(self, net: str = "alex", device: torch.device = torch.device("cpu")):
        import lpips  # imported lazily so the rest of the project works without it

        self.net = net
        self.device = device
        self.model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-image LPIPS distance, shape [B]."""
        pred, target = _prepare(pred, target)  # validates [0, 1], adds batch dim if needed
        d = self.model(pred.to(self.device), target.to(self.device), normalize=True)
        return d.flatten().float().cpu()
