"""Phase 3 model wrappers with one common interface.

    model(x)              -> enhanced image; in eval() mode clamped to [0, 1] (what evaluation/export use)
    model.forward_train(x)-> (pred, side): raw training output (+ side output for DarkIR's guide loss, else None)

Each wrapper reproduces the official test-time handling of its network:
  - retinexformer : reflect-pad to a multiple of 4, crop, clamp [0, 1]
                    (Retinexformer Enhancement/test_from_dataset.py, factor = 4)
  - hvi_cidnet    : reflect-pad to a multiple of 8 (3 bilinear 0.5x downsamplings), crop; output already in [0, 1]
                    because PHVIT clamps
  - darkir_m      : the network zero-pads internally (check_image_size), clamp [0, 1]
  - unet          : the Phase-2 BaselineUNet (reflect-pads internally, sigmoid output)

FP32 islands under mixed precision (no effect when autocast is off):
  - HVI-CIDNet: RGB->HVI (HVIT) and HVI->RGB (PHVIT). They use eps = 1e-8, which underflows in FP16.
  - DarkIR: FreMLP (rfft2 / angle / irfft2). FP16 FFT on CUDA is limited to power-of-two sizes and complex-half
    support is partial.
These are applied by wrapping bound methods of the vendored modules; the vendored files are unchanged for this.
"""

import functools
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.baseline_unet import BaselineUNet
from third_party.darkir.archs.arch_model import FreMLP
from third_party.darkir.archs.DarkIR import DarkIR
from third_party.hvi_cidnet.net.CIDNet import CIDNet
from third_party.retinexformer.RetinexFormer_arch import RetinexFormer

# Official configurations (verified 2026-10-14; see experiments/phase3_model_selection/STAGE1_PLAN.md)
RETINEXFORMER_CFG = dict(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])  # LOL-v2 YAMLs
DARKIR_M_CFG = dict(img_channel=3, width=32, middle_blk_num_enc=2, middle_blk_num_dec=2, enc_blk_nums=[1, 2, 3],
                    dec_blk_nums=[3, 1, 1], dilations=[1, 4, 9], extra_depth_wise=True)  # options/test/*.yml
CIDNET_CFG = dict(channels=[36, 36, 72, 144], heads=[1, 2, 4, 8], norm=False)  # CIDNet defaults (as trained)

# Expected parameter counts: exact counts of this code, with the paper figure and the paper's unit convention.
EXPECTED_PARAMS = {
    "unet": {"count": 1_925_667, "paper": "1.93 M (Phase-1/2 baseline)", "paper_value": 1_925_667, "unit": 1},
    "unet_expaug": {"count": 1_925_667, "paper": "1.93 M", "paper_value": 1_925_667, "unit": 1},
    "retinexformer": {"count": 1_605_701, "paper": "1.61 M", "paper_value": 1.61, "unit": 1e6},
    "hvi_cidnet": {"count": 1_975_569, "paper": "1.88 M (net_test.py divides by 2**20)", "paper_value": 1.88, "unit": 2 ** 20},
    "darkir_m": {"count": 3_321_638, "paper": "3.31 M", "paper_value": 3.31, "unit": 1e6},
}


def _fp32(fn: Callable) -> Callable:
    """Run ``fn`` with autocast disabled and floating tensor args cast to float32."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        dev = next((a.device.type for a in args if torch.is_tensor(a)), "cpu")
        with torch.autocast(device_type=dev, enabled=False):
            args = [a.float() if torch.is_tensor(a) and a.is_floating_point() else a for a in args]
            return fn(*args, **kwargs)
    return wrapped


def _pad_to(x: torch.Tensor, m: int, mode: str = "reflect") -> Tuple[torch.Tensor, int, int]:
    h, w = x.shape[-2:]
    ph, pw = (m - h % m) % m, (m - w % m) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode=mode)
    return x, h, w


class Phase3Model(nn.Module):
    name = "base"
    pad_multiple = 1
    clamp_eval = True

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def _raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.pad_multiple > 1:
            xp, h, w = _pad_to(x, self.pad_multiple)
            return self.net(xp)[..., :h, :w], None
        return self.net(x), None

    def forward_train(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self._raw(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self._raw(x)
        return y.clamp(0, 1) if (self.clamp_eval and not self.training) else y


class UNetModel(Phase3Model):
    name = "unet"

    def __init__(self):
        super().__init__(BaselineUNet(in_channels=3, out_channels=3, base_channels=32, depth=3))


class RetinexformerModel(Phase3Model):
    name = "retinexformer"
    pad_multiple = 4

    def __init__(self):
        super().__init__(RetinexFormer(**RETINEXFORMER_CFG))


class CIDNetModel(Phase3Model):
    name = "hvi_cidnet"
    pad_multiple = 8

    def __init__(self):
        super().__init__(CIDNet(**CIDNET_CFG))
        trans = self.net.trans
        trans.HVIT = _fp32(trans.HVIT)    # instance attributes shadow the class methods
        trans.PHVIT = _fp32(trans.PHVIT)

    def hvi(self, x: torch.Tensor) -> torch.Tensor:
        """RGB -> HVI with the network's (learnable-k) transform; used by the official loss."""
        return self.net.HVIT(x)


class DarkIRModel(Phase3Model):
    name = "darkir_m"

    def __init__(self):
        super().__init__(DarkIR(**DARKIR_M_CFG))
        for m in self.net.modules():
            if isinstance(m, FreMLP):
                m.forward = _fp32(m.forward)

    def _raw(self, x):
        return self.net(x), None

    def forward_train(self, x):
        side, out = self.net(x, side_loss=True)
        return out, side


MODELS: Dict[str, type] = {"unet": UNetModel, "unet_expaug": UNetModel, "retinexformer": RetinexformerModel,
                           "hvi_cidnet": CIDNetModel, "darkir_m": DarkIRModel}


def build_phase3_model(name: str) -> Phase3Model:
    if name not in MODELS:
        raise ValueError(f"unknown Phase-3 model {name!r}; available {sorted(MODELS)}")
    return MODELS[name]()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
