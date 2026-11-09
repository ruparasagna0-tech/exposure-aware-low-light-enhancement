"""Shared Phase-3 training engine (used by scripts/train_phase3.py and scripts/phase3_pilot.py).

Keeps the Phase-2 protocol pieces unchanged: build_train_val (splits, 256 crops, hflip, optional exposure
augmentation), BalancedDomainBatchSampler (every batch = 2 LOL-v2-real + 2 LOL-v2-synthetic), per-epoch reseeding
(resume-safe batch stream), evaluate_domain for full-resolution validation.

Mixed precision: FP16 autocast + GradScaler only when ``amp=True`` and the device is CUDA (T4 has no bf16).
Gradient clipping (if the recipe has one) is applied to unscaled gradients (scaler.unscale_ first).
Never deep-copy an engine/model (wrapped bound methods): build a new one and load a state_dict.
"""

import hashlib
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from src.datasets.balanced_sampler import BalancedDomainBatchSampler
from src.datasets.phase2 import SUBSETS, build_train_val
from src.metrics.domain_evaluation import evaluate_domain
from src.utils import make_generator, seed_worker

from .losses import build_phase3_loss
from .models import build_phase3_model, count_params
from .schedules import RECIPES, build_optimizer, lr_at, set_lr

LOSS_OF = {"unet": "l1", "unet_expaug": "l1", "retinexformer": "l1", "hvi_cidnet": "cidnet_official",
           "darkir_m": "darkir_paper"}


def state_hash(model: nn.Module) -> str:
    h = hashlib.md5()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Engine:
    def __init__(self, cfg: dict, model_name: str, device: torch.device, amp: bool, total_steps: int,
                 data_root: Optional[str] = None, build_data: bool = True):
        self.cfg, self.name, self.device, self.total_steps = cfg, model_name, device, total_steps
        self.amp = bool(amp) and device.type == "cuda"
        self.model = build_phase3_model(model_name).to(device)
        self.objective = build_phase3_loss(LOSS_OF[model_name]).to(device)
        self.val_l1 = nn.L1Loss()
        self.opt = build_optimizer(model_name, self.model.parameters())
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.grad_clip = RECIPES[model_name]["grad_clip"]
        self.k = cfg["sampler"]["per_domain"]
        self.expected_domains = ["lol_v2_real"] * self.k + ["lol_v2_synthetic"] * self.k
        if build_data:
            (self.train_real, self.train_syn), (self.val_real, self.val_syn) = build_train_val(cfg, data_root)
            self.concat = ConcatDataset([self.train_real, self.train_syn])

    # ------------------------------------------------------------------ data
    def epoch_loader(self, epoch: int, steps: int, num_workers: int) -> DataLoader:
        """Batches of 1-based ``epoch`` exactly as the Phase-2 trainer generates them."""
        seed = self.cfg["seed"]
        torch.manual_seed(seed * 1000 + epoch)
        sampler = BalancedDomainBatchSampler([len(self.train_real), len(self.train_syn)], self.k, steps, seed)
        sampler.set_epoch(epoch - 1)
        return DataLoader(self.concat, batch_sampler=sampler, num_workers=num_workers, worker_init_fn=seed_worker,
                          generator=make_generator(seed * 1000 + epoch), pin_memory=self.device.type == "cuda")

    def val_loaders(self, num_workers: int) -> Dict[str, DataLoader]:
        return {s: DataLoader(d, batch_size=1, shuffle=False, num_workers=num_workers)
                for s, d in zip(SUBSETS, (self.val_real, self.val_syn))}

    # ------------------------------------------------------------------ training
    def train_step(self, batch: dict, global_step: int) -> dict:
        doms = list(batch["domain"])
        if doms != self.expected_domains:
            raise RuntimeError(f"unexpected batch composition {doms}")
        low = batch["low"].to(self.device, non_blocking=True)
        high = batch["high"].to(self.device, non_blocking=True)
        lr = lr_at(self.name, global_step, self.total_steps)
        set_lr(self.opt, lr)
        self.model.train()
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            pred, side = self.model.forward_train(low)
            loss, parts = self.objective(pred, high, side=side, model=self.model)
        if not torch.isfinite(loss):
            if not self.amp:
                raise FloatingPointError(f"non-finite loss {float(loss)} at step {global_step}")
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        grad_norm = None
        if self.grad_clip:
            self.scaler.unscale_(self.opt)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip))
        scale_before = self.scaler.get_scale() if self.amp else 1.0
        self.scaler.step(self.opt)
        self.scaler.update()
        skipped = bool(self.amp and self.scaler.get_scale() < scale_before)
        with torch.no_grad():
            per = (pred.float() - high).abs().flatten(1).mean(1).tolist()
        return {"loss": float(loss.detach()), "parts": parts, "lr": lr, "skipped": skipped, "grad_norm": grad_norm,
                "per_sample_l1": per, "domains": doms}

    @torch.no_grad()
    def validate(self, loaders: Dict[str, DataLoader]) -> dict:
        val = {s: evaluate_domain(self.model, loaders[s], self.device, loss_fn=self.val_l1) for s in SUBSETS}
        out = {}
        for s in SUBSETS:
            out.update({f"val_{s}_loss": val[s]["loss"], f"val_{s}_psnr": val[s]["psnr"], f"val_{s}_ssim": val[s]["ssim"]})
        out["val_psnr_mean"] = (val["real"]["psnr"] + val["synthetic"]["psnr"]) / 2
        return out

    # ------------------------------------------------------------------ checkpoint state
    def state(self) -> dict:
        return {"optimizer_state": self.opt.state_dict(), "scaler_state": self.scaler.state_dict(),
                "model_name": self.name, "amp": self.amp}

    def load_state(self, ck: dict) -> None:
        if ck.get("model_name") != self.name:
            raise ValueError(f"checkpoint model {ck.get('model_name')} != {self.name}")
        if bool(ck.get("amp")) != self.amp:
            raise ValueError(f"checkpoint AMP={ck.get('amp')} but this run AMP={self.amp}")
        self.model.load_state_dict(ck["model_state"])
        self.opt.load_state_dict(ck["optimizer_state"])
        self.scaler.load_state_dict(ck["scaler_state"])

    def n_params(self) -> int:
        return count_params(self.model)
