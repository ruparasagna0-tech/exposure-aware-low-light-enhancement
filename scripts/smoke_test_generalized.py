"""Phase-2 smoke test (NO training run). Writes only <output.dir>/logs/smoke_test_report_<device>.json
(temporary checkpoints go to a temp dir that is deleted).

Checks: imports/config, integrity checks (MD5, counts, splits, leakage, isolation, readability),
loaders (shapes/dtype/range/RGB), splits, balanced sampler (2+2, 112 steps, determinism, epoch
coverage), forward, L1, one optimizer step, PSNR/SSIM/LPIPS, tiling (exactness + one SICE
sequence at native resolution), checkpoint save/load, LOL-v1 baseline untouched + its smoke test.

Usage: .venv/bin/python scripts/smoke_test_generalized.py [--data-root DIR] [--skip-md5]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader  # noqa: E402

REPORT = []


def check(name, ok, detail=""):
    REPORT.append({"check": name, "passed": bool(ok), "detail": str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail != "" else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/generalized_lol_v2.yaml")
    ap.add_argument("--data-root")
    ap.add_argument("--skip-md5", action="store_true")
    ap.add_argument("--device", help="override train.device (use cpu on Macs; MPS is unreliable here)")
    ap.add_argument("--skip-baseline-check", action="store_true",
                    help="e.g. on Colab, where LOL-v1 data / baseline files are not present")
    args = ap.parse_args()
    t_start = time.time()

    print(f"[1] imports + config  (t={time.time() - t_start:.0f}s)")
    from src.datasets.balanced_sampler import BalancedDomainBatchSampler
    from src.datasets.phase2 import SUBSETS, build_test, build_train_val, integrity_checks, load_json, split_file
    from src.losses import build_loss
    from src.metrics import psnr, ssim
    from src.metrics.domain_evaluation import evaluate_domain
    from src.metrics.lpips_metric import LPIPSMetric
    from src.metrics.large_image import check_metric_values, ssim_separable
    from src.models import build_model
    from src.utils import (count_parameters, load_config, load_model_from_checkpoint, resolve_path,
                           save_checkpoint, select_device, set_seed)
    from src.utils.tiling import tiled_forward
    cfg = load_config(args.config)
    check("config loads", cfg["experiment_name"] == "generalized_lol_v2")
    t, sm = cfg["train"], cfg["sampler"]
    check("budget: batch 4 = 2 real + 2 synthetic; 112 steps x 50 epochs",
          t["batch_size"] == 4 and sm["per_domain"] == 2 and sm["steps_per_epoch"] == 112 and t["epochs"] == 50)
    check("model/loss/optimizer identical to LOL-v1 baseline",
          cfg["model"] == load_config("configs/baseline_lol_v1.yaml")["model"] and cfg["loss"]["name"] == "l1"
          and t["learning_rate"] == 2e-4 and t["weight_decay"] == 0 and t["scheduler"] == "none"
          and cfg["data"]["crop_size"] == 256 and cfg["data"]["hflip"] and not cfg["data"]["vflip"] and cfg["seed"] == 42)
    set_seed(cfg["seed"])
    device = select_device(args.device or t["device"])
    print(f"  device: {device}")

    print(f"[2] integrity checks  (t={time.time() - t_start:.0f}s)")
    for c, ok, d in integrity_checks(cfg, args.data_root, verify_md5=not args.skip_md5):
        check(c, ok, d if not ok or "pairs" in c or "count" in c or "scene" in c else "")

    print(f"[3] loaders  (t={time.time() - t_start:.0f}s)")
    (tr_r, tr_s), (va_r, va_s) = build_train_val(cfg, args.data_root)
    tests = build_test(cfg, args.data_root)
    for name, ds, crop in [("train real", tr_r, 256), ("train synthetic", tr_s, 256), ("val real", va_r, None),
                           ("val synthetic", va_s, None)] + [(k, v, None) for k, v in tests.items()]:
        it = ds[0]
        lo, hi = it["low"], it["high"]
        ok = (lo.dtype == torch.float32 and lo.shape == hi.shape and lo.shape[0] == 3
              and 0 <= float(lo.min()) and float(lo.max()) <= 1 and 0 <= float(hi.min()) and float(hi.max()) <= 1
              and (crop is None or tuple(lo.shape[1:]) == (crop, crop)))
        check(f"loader {name}: n={len(ds)}", ok, f"item {it['name']} {tuple(lo.shape)} range "
              f"[{float(lo.min()):.3f},{float(lo.max()):.3f}]")

    print(f"[4] splits  (t={time.time() - t_start:.0f}s)")
    for s in SUBSETS:
        sp = load_json(split_file(cfg, s))
        check(f"{s} split: {sp['counts']['train_pairs']} train / {sp['counts']['val_pairs']} val pairs, "
              f"{sp['counts']['val_scenes']} val scenes", sp["counts"]["val_scenes"] == cfg["splits"][s]["n_val_scenes"])

    print(f"[5] balanced sampler  (t={time.time() - t_start:.0f}s)")
    sizes = [len(tr_r), len(tr_s)]
    smp = BalancedDomainBatchSampler(sizes, sm["per_domain"], sm["steps_per_epoch"], cfg["seed"])
    ok_comp, per_epoch = True, []
    for e in range(3):
        bs = smp.epoch_batches(e)
        comp = all(all(i < sizes[0] for i in b[:2]) and all(i >= sizes[0] for i in b[2:]) and len(b) == 4 for b in bs)
        ok_comp &= comp and len(bs) == 112
        per_epoch.append((sum(i < sizes[0] for b in bs for i in b), sum(i >= sizes[0] for b in bs for i in b)))
    check("every batch = 2 real + 2 synthetic; 112 batches/epoch", ok_comp, f"samples/epoch (real, syn) {per_epoch}")
    smp2 = BalancedDomainBatchSampler(sizes, sm["per_domain"], sm["steps_per_epoch"], cfg["seed"])
    check("sampler deterministic (same seed -> identical batches; epoch regenerable)",
          smp.epoch_batches(7) == smp2.epoch_batches(7))
    stream = [i for e in range(3) for b in smp.epoch_batches(e) for i in b[:2]]
    first_cycle = stream[:sizes[0]]
    check("real domain: no repeat before every real train pair was used once",
          len(set(first_cycle)) == sizes[0], f"{sizes[0]} real pairs, epochs to cover ~{sizes[0] / 224:.2f}")
    loader = DataLoader(ConcatDataset([tr_r, tr_s]), batch_sampler=smp, num_workers=0)
    doms = [list(b["domain"]) for _, b in zip(range(3), loader)]
    check("DataLoader batches carry domains [real, real, syn, syn]",
          all(d == ["lol_v2_real"] * 2 + ["lol_v2_synthetic"] * 2 for d in doms), doms[0])

    print(f"[6-10] model, loss, optimizer step, metrics  (t={time.time() - t_start:.0f}s)")
    batch = next(iter(DataLoader(ConcatDataset([tr_r, tr_s]), batch_sampler=smp, num_workers=0)))
    model = build_model(cfg["model"]).to(device)
    check("U-Net parameters ~1.93M (same as baseline)", count_parameters(model) == 1925667, count_parameters(model))
    low, high = batch["low"].to(device), batch["high"].to(device)
    pred = model(low)
    check("forward [4,3,256,256] -> same, float32, [0,1]",
          tuple(pred.shape) == (4, 3, 256, 256) and pred.dtype == torch.float32 and 0 <= float(pred.detach().min()) <= float(pred.detach().max()) <= 1)
    loss_fn = build_loss(cfg["loss"])
    opt = torch.optim.Adam(model.parameters(), lr=t["learning_rate"], betas=tuple(t["betas"]), weight_decay=t["weight_decay"])
    loss = loss_fn(pred, high)
    before = [p.detach().clone() for p in model.parameters()]
    opt.zero_grad()
    loss.backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step()
    changed = sum(not torch.equal(a, p.detach()) for a, p in zip(before, model.parameters()))
    check("L1 loss + backward + one Adam step", grads_ok and changed == len(before), f"L1 {loss.item():.4f}, {changed} tensors updated")
    with torch.no_grad():
        pr = model(low)
    lp = LPIPSMetric(cfg["eval"]["lpips_net"], device)
    x = torch.rand(2, 3, 64, 64)
    check("PSNR/SSIM identity (100 dB cap / 1.0)", torch.allclose(psnr(x, x), torch.full((2,), 100.0))
          and torch.allclose(ssim(x, x), torch.ones(2), atol=1e-5))
    check("LPIPS identity = 0 and batch works", float(lp(x, x).abs().max()) < 1e-6 and lp(pr, high).shape == (4,),
          f"LPIPS(pred, ref) batch {[round(v, 3) for v in lp(pr, high).tolist()]}")
    small = torch.utils.data.Subset(va_r, [0, 1])
    r = evaluate_domain(model, DataLoader(small, batch_size=1), device, loss_fn=loss_fn, lpips_fn=lp, quantize=True)
    check("evaluate_domain on 2 val images (PSNR/SSIM/LPIPS/loss)", r["n"] == 2 and r["lpips"] is not None,
          {k: round(r[k], 4) for k in ("psnr", "ssim", "lpips", "loss")})

    print(f"[9] tiled inference  (t={time.time() - t_start:.0f}s)")
    img = torch.rand(1, 3, 1100, 1700)
    rec = tiled_forward(lambda z: z, img, tile=512, overlap=64)
    check("tiling with identity fn reconstructs exactly (linear blend weights normalise)",
          float((rec - img).abs().max()) < 1e-5, f"max err {float((rec - img).abs().max()):.2e}")
    model.eval()
    lolimg = va_r[0]["low"][None]
    with torch.no_grad():
        full = model(lolimg.to(device)).cpu()
    til = tiled_forward(model, lolimg, tile=256, overlap=64, device=device)
    check("model: tiled (256/64) vs full-image on 600x400 (context effect only)", til.shape == full.shape,
          f"mean abs diff {float((til - full).abs().mean()):.2e}, max {float((til - full).abs().max()):.2e}")
    a_, b_ = va_r[0]["low"][None], va_r[0]["high"][None]
    d_ss = abs(ssim(a_, b_).item() - ssim_separable(a_, b_).item())
    check("strip-wise separable SSIM == baseline SSIM (same definition)", d_ss < 1e-4, f"|diff| {d_ss:.2e}")
    sice = tests["sice"]
    t0 = time.time()
    it = sice[0]
    with torch.no_grad():
        out = tiled_forward(model, it["low"][None], tile=cfg["eval"]["sice"]["tile"],
                            overlap=cfg["eval"]["sice"]["overlap"], device=device).clamp(0, 1)
    t_inf = time.time() - t0
    hi = it["high"][None]
    vals = {}
    for dev in dict.fromkeys([device, torch.device("cpu")]):  # compute device, then CPU reference
        t1 = time.time()
        o, h_ = out.to(dev), hi.to(dev)
        lpm = lp if dev == device else LPIPSMetric(cfg["eval"]["lpips_net"], dev)
        vals[str(dev)] = (psnr(o, h_).item(), ssim_separable(o, h_).item(), lpm(o, h_).item(), time.time() - t1)
    pv, sv, lv, tm = vals[str(device)]
    try:
        check_metric_values("sice", pv, sv, lv)
        sane = True
    except RuntimeError as e:
        sane, pv = False, str(e)
    cpu = vals["cpu"]
    agree = all(abs(x - y) <= tol for x, y, tol in zip(vals[str(device)][:3], cpu[:3], (1e-3, 1e-4, 1e-3)))
    check(f"SICE {it['name']} native {tuple(it['low'].shape[1:])}: tiled 512/64 inference + PSNR/SSIM/LPIPS sane",
          out.shape == hi.shape and sane, f"input {it['exposure_file']}; PSNR {vals[str(device)][0]:.3f} SSIM "
          f"{sv:.4f} LPIPS {lv:.4f}; inference {t_inf:.1f}s, metrics {tm:.1f}s on {device}")
    check(f"SICE metrics on {device} agree with CPU reference", agree, f"cpu: PSNR {cpu[0]:.3f} SSIM {cpu[1]:.4f} "
          f"LPIPS {cpu[2]:.4f} ({cpu[3]:.1f}s)")

    print(f"[11] checkpoint save/load  (t={time.time() - t_start:.0f}s)")
    with tempfile.TemporaryDirectory() as td:
        pth = os.path.join(td, "ckpt.pth")
        save_checkpoint(pth, model, cfg, optimizer_state=opt.state_dict(), epoch=1, history=[], best_val_psnr_mean=0.0)
        m2, ck = load_model_from_checkpoint(pth, device)
        with torch.no_grad():
            same = torch.equal(m2(low), model.eval()(low))
        check("checkpoint save -> load reproduces outputs; stores config + optimizer", same and "optimizer_state" in ck)

    print(f"[13] LOL-v1 baseline untouched  (t={time.time() - t_start:.0f}s)")
    if args.skip_baseline_check:
        print("  (skipped by --skip-baseline-check)")
    else:
        snap = json.load(open(resolve_path("experiments/generalized_lol_v2/logs/pre_phase2_file_md5.json")))["files"]
        changed = [f for f, m in snap.items() if f != "requirements.txt"
                   and (not os.path.exists(f) or hashlib.md5(open(f, "rb").read()).hexdigest() != m)]
        check(f"{len(snap) - 1} pre-existing files byte-identical (requirements.txt excluded: lpips added)", not changed, changed[:5])
        for path in ("experiments/baseline_lol_v1", "results/baseline_lol_v1"):
            check(f"{path} not created/overwritten locally", not os.path.exists(path))
        rc = subprocess.run([sys.executable, "scripts/smoke_test_baseline.py"], capture_output=True, text=True)
        check("LOL-v1 baseline smoke test still passes", rc.returncode == 0, rc.stdout.strip().splitlines()[-1])

    failed = [r for r in REPORT if not r["passed"]]
    out = resolve_path(cfg["output"]["dir"]) / "logs"
    out.mkdir(parents=True, exist_ok=True)
    report = out / f"smoke_test_report_{device.type}.json"
    with open(report, "w") as f:
        json.dump({"device": str(device), "runtime_s": round(time.time() - t_start, 1), "passed": not failed,
                   "checks": REPORT}, f, indent=1)
    print("=" * 70)
    print(f"SMOKE TEST: {'ALL ' + str(len(REPORT)) + ' CHECKS PASSED' if not failed else str(len(failed)) + ' FAILED'} "
          f"({time.time() - t_start:.0f}s); report: {report}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
