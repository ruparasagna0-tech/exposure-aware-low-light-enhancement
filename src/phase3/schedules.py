"""Per-step optimizers and LR schedules, each official recipe rescaled to our step budget.

Official recipes (verified 2026-10-14) and the rescaling used for a budget of T optimizer steps:
  unet          Phase 2: Adam 2e-4 (0.9, 0.999), wd 0, constant LR, no clipping.
  retinexformer BasicSR LOL-v2 YAML: Adam 2e-4 (0.9, 0.999); CosineAnnealingRestartCyclicLR periods [46000, 104000]
                (= 150,000 iters), restart_weights [1, 1], eta_mins [3e-4, 1e-6]; grad-norm clip 0.01.
                Rescaled: periods = [46000, 104000] * T / 150000. (Cycle 1 anneals TOWARD 3e-4, i.e. the LR rises
                from 2e-4; this upstream setting is reproduced as is.)
  hvi_cidnet    train.py: Adam 1e-4; GradualWarmupScheduler (multiplier 1, 3 of 1,000 epochs) then
                CosineAnnealingRestartLR to eta_min 1e-7; grad-norm clip 0.01.
                Rescaled: warmup = 3/1000 * T steps (linear from 0), cosine over the remaining steps.
  darkir_m      paper: AdamW 5e-4, betas (0.9, 0.9), weight decay 1e-3, cosine to 1e-6; no clipping stated.
                Rescaled: cosine over T steps.
Not reproduced (data protocol must be identical across models): Retinexformer mixup, CIDNet random gamma,
official crop/batch sizes (we use 256 crops, batch 4 = 2 real + 2 synthetic for every model).
"""

import math
from typing import Dict

import torch

RECIPES: Dict[str, Dict] = {
    "unet": {"optimizer": "adam", "lr": 2e-4, "betas": (0.9, 0.999), "weight_decay": 0.0, "schedule": "constant",
             "grad_clip": None},
    "retinexformer": {"optimizer": "adam", "lr": 2e-4, "betas": (0.9, 0.999), "weight_decay": 0.0,
                      "schedule": "cyclic_cosine", "periods_frac": (46000 / 150000, 104000 / 150000),
                      "restart_weights": (1.0, 1.0), "eta_mins": (3e-4, 1e-6), "grad_clip": 0.01},
    "hvi_cidnet": {"optimizer": "adam", "lr": 1e-4, "betas": (0.9, 0.999), "weight_decay": 0.0,
                   "schedule": "warmup_cosine", "warmup_frac": 3 / 1000, "eta_min": 1e-7, "grad_clip": 0.01},
    "darkir_m": {"optimizer": "adamw", "lr": 5e-4, "betas": (0.9, 0.9), "weight_decay": 1e-3, "schedule": "cosine",
                 "eta_min": 1e-6, "grad_clip": None},
}
RECIPES["unet_expaug"] = RECIPES["unet"]


def build_optimizer(name: str, params):
    r = RECIPES[name]
    if r["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=r["lr"], betas=r["betas"], weight_decay=r["weight_decay"])
    if r["optimizer"] == "adamw":
        return torch.optim.AdamW(params, lr=r["lr"], betas=r["betas"], weight_decay=r["weight_decay"])
    raise ValueError(r["optimizer"])


def lr_at(name: str, step: int, total_steps: int) -> float:
    """LR for 0-based optimizer step ``step`` (the LR used FOR that step)."""
    r, base = RECIPES[name], RECIPES[name]["lr"]
    s = r["schedule"]
    if s == "constant":
        return base
    if s == "cosine":
        return r["eta_min"] + 0.5 * (base - r["eta_min"]) * (1 + math.cos(math.pi * min(step, total_steps) / total_steps))
    if s == "cyclic_cosine":  # BasicSR CosineAnnealingRestartCyclicLR, per-iteration
        periods = [max(1, round(f * total_steps)) for f in r["periods_frac"]]
        cum = [sum(periods[:i + 1]) for i in range(len(periods))]
        idx = next((i for i, c in enumerate(cum) if step <= c), len(cum) - 1)
        start = 0 if idx == 0 else cum[idx - 1]
        eta = r["eta_mins"][idx]
        return eta + r["restart_weights"][idx] * 0.5 * (base - eta) * (1 + math.cos(math.pi * (step - start) / periods[idx]))
    if s == "warmup_cosine":  # GradualWarmupScheduler(multiplier=1) then cosine to eta_min
        w = max(1, round(r["warmup_frac"] * total_steps))
        if step < w:
            return base * step / w
        t = (step - w) / max(1, total_steps - w)
        return r["eta_min"] + 0.5 * (base - r["eta_min"]) * (1 + math.cos(math.pi * min(t, 1.0)))
    raise ValueError(s)


def set_lr(optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr
