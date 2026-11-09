"""Smoke test for the baseline (NO training run, nothing written to disk).

Checks:
  1. metric sanity (identical images, range guard)
  2. forward pass on one LOL-v1 training batch: shapes, dtype, output range
  3. ONE optimisation step: loss, backward, gradients, optimizer update
  4. forward + PSNR/SSIM on one full-resolution validation image

Usage (from project root):
    .venv/bin/python scripts/smoke_test_baseline.py
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.datasets import build_lol_v1_dataset  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.metrics import psnr, ssim  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import count_parameters, load_config, make_generator, select_device, set_seed  # noqa: E402

failures = []


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        failures.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_lol_v1.yaml")
    ap.add_argument("--device", help="override train.device")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], cfg["train"]["deterministic"])
    device = select_device(args.device or cfg["train"]["device"])
    bs, crop = cfg["train"]["batch_size"], cfg["data"]["crop_size"]
    print(f"Device: {device} | torch {torch.__version__}")

    print("\n[1] Metric sanity")
    x = torch.rand(2, 3, 64, 64)
    check(torch.allclose(psnr(x, x), torch.full((2,), 100.0)), "PSNR(x, x) = 100 dB (capped)")
    check(torch.allclose(ssim(x, x), torch.ones(2), atol=1e-5), "SSIM(x, x) = 1")
    noisy = (x + 0.1 * torch.randn_like(x)).clamp(0, 1)
    p, s = psnr(noisy, x).mean().item(), ssim(noisy, x).mean().item()
    check(15 < p < 25 and s < 1, f"noisy (sigma=0.1): PSNR {p:.2f} dB, SSIM {s:.3f}")
    try:
        psnr(x * 255, x)
        check(False, "0-255 input rejected")
    except ValueError:
        check(True, "0-255 input rejected (no silent rescaling)")

    print("\n[2] Model + forward pass on one training batch")
    model = build_model(cfg["model"]).to(device)
    print(f"  model: {cfg['model']} | trainable params: {count_parameters(model):,}")
    train_ds = build_lol_v1_dataset(cfg, "train")
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0,
                        generator=make_generator(cfg["seed"]))
    batch = next(iter(loader))
    low, high = batch["low"].to(device), batch["high"].to(device)
    print(f"  batch names: {batch['name']}")
    model.train()
    pred = model(low)
    check(tuple(low.shape) == (bs, 3, crop, crop), f"input shape {tuple(low.shape)}")
    check(pred.shape == low.shape, f"output shape {tuple(pred.shape)}")
    check(pred.dtype == torch.float32, f"output dtype {pred.dtype}")
    lo, hi = pred.min().item(), pred.max().item()
    check(0.0 <= lo and hi <= 1.0, f"output range [{lo:.4f}, {hi:.4f}] within [0, 1]")

    print("\n[3] One training step (L1 + Adam)")
    loss_fn = build_loss(cfg["loss"])
    t = cfg["train"]
    opt = torch.optim.Adam(model.parameters(), lr=t["learning_rate"], betas=tuple(t["betas"]),
                           weight_decay=t["weight_decay"])
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    loss = loss_fn(pred, high)
    check(loss.ndim == 0 and torch.isfinite(loss).item(), f"loss computes: L1 = {loss.item():.4f}")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    missing = [n for n, g in grads.items() if g is None]
    nonfinite = [n for n, g in grads.items() if g is not None and not torch.isfinite(g).all()]
    zero = [n for n, g in grads.items() if g is not None and g.abs().sum().item() == 0]
    gnorm = torch.norm(torch.stack([g.norm() for g in grads.values() if g is not None])).item()
    check(not missing, f"every parameter received a gradient ({len(grads)} tensors)"
          + (f" missing: {missing[:5]}" if missing else ""))
    check(not nonfinite, "all gradients finite")
    check(not zero, "no all-zero gradient tensors" + (f" (zero: {zero[:5]})" if zero else ""))
    print(f"  total gradient L2 norm: {gnorm:.4e}")

    opt.step()
    changed = sum(not torch.equal(before[n], p.detach()) for n, p in model.named_parameters())
    check(changed == len(before), f"optimizer updated {changed}/{len(before)} parameter tensors")
    with torch.no_grad():
        loss_after = loss_fn(model(low), high).item()
    print(f"  L1 on same batch: before {loss.item():.4f} -> after one step {loss_after:.4f} (informational)")

    print("\n[4] Full-resolution validation forward + metrics")
    val_ds = build_lol_v1_dataset(cfg, "val")
    item = val_ds[0]
    model.eval()
    with torch.no_grad():
        vp = model(item["low"].unsqueeze(0).to(device))
    vh = item["high"].unsqueeze(0).to(device)
    check(vp.shape == vh.shape == (1, 3, 400, 600), f"val output shape {tuple(vp.shape)} ({item['name']})")
    print(f"  untrained-model val metrics: PSNR {psnr(vp, vh).item():.3f} dB, "
          f"SSIM {ssim(vp, vh).item():.4f} (meaningless before training)")

    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST: {len(failures)} check(s) FAILED")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("SMOKE TEST: all checks passed (no training performed, nothing saved)")


if __name__ == "__main__":
    main()
