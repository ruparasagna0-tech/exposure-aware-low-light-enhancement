"""Phase 2: train ONE baseline U-Net on LOL-v2-real + LOL-v2-synthetic with balanced 2+2 batches.

Protocol (configs/generalized_lol_v2.yaml; approved decisions):
- model/loss/optimizer identical to the LOL-v1 baseline (U-Net ~1.93M, L1, Adam 2e-4, wd 0, no scheduler)
- every batch = 2 LOL-v2-real + 2 LOL-v2-synthetic training crops (256, paired hflip)
- exactly 112 optimizer steps per epoch x 50 epochs = 5,600 steps
- validation after each epoch on the scene-disjoint val subsets (full resolution, deterministic)
- best checkpoint = highest (real_val_PSNR + synthetic_val_PSNR) / 2; official test sets,
  SICE and LoLI-Street are never touched here
- integrity checks run first; any failure stops before training

Usage (from project root):
    python scripts/train_generalized.py
    python scripts/train_generalized.py --data-root /content/phase2
    python scripts/train_generalized.py --resume experiments/generalized_lol_v2/checkpoints/last.pth

Pipeline-test options (NOT for the real run; recorded in run_info): --output-dir, --max-steps-per-epoch.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader  # noqa: E402

from src.datasets.balanced_sampler import BalancedDomainBatchSampler  # noqa: E402
from src.datasets.phase2 import SUBSETS, build_train_val, integrity_checks, paths, sice_selection_file, split_file  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.metrics.domain_evaluation import evaluate_domain  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import (  # noqa: E402
    count_parameters, ensure_outside_data, load_checkpoint, load_config, make_generator, resolve_path,
    save_checkpoint, save_config, seed_worker, select_device, set_seed,
)


def file_md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def write_history(history, out_dir):
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, "logs", "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/generalized_lol_v2.yaml")
    ap.add_argument("--data-root", help="override data.root (e.g. /content/phase2 on Colab)")
    ap.add_argument("--device", help="override train.device")
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--resume", help="continue from a last.pth")
    ap.add_argument("--output-dir", help="PIPELINE TEST ONLY: write outputs elsewhere")
    ap.add_argument("--max-steps-per-epoch", type=int, help="PIPELINE TEST ONLY: truncate epochs")
    ap.add_argument("--epochs", type=int, help="PIPELINE TEST ONLY: override epoch count")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["train"]["device"] = args.device
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    t, sm = cfg["train"], cfg["sampler"]
    if t["batch_size"] != sm["per_domain"] * len(SUBSETS):
        sys.exit(f"batch_size {t['batch_size']} != {len(SUBSETS)} domains x per_domain {sm['per_domain']}")
    if t.get("scheduler", "none") != "none" or t["optimizer"].lower() != "adam":
        sys.exit("Phase-2 protocol: Adam, no scheduler")
    steps = args.max_steps_per_epoch or sm["steps_per_epoch"]
    test_mode = bool(args.output_dir or args.max_steps_per_epoch or args.epochs is not None)

    set_seed(cfg["seed"], deterministic=t["deterministic"])
    device = select_device(t["device"])
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None

    out = ensure_outside_data(resolve_path(args.output_dir or cfg["output"]["dir"]))
    for sub in ("checkpoints", "logs", "evaluation", "splits"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if (out / "checkpoints" / "last.pth").exists() and not args.resume:
        sys.exit(f"{out}/checkpoints/last.pth exists: refusing to overwrite a previous run (use --resume).")

    # ---------------- integrity checks (stop on any failure)
    print("Running integrity checks ...", flush=True)
    checks = integrity_checks(cfg, args.data_root)
    with open(out / "logs" / "integrity_checks.json", "w") as f:
        json.dump([{"check": c, "passed": ok, "detail": d} for c, ok, d in checks], f, indent=1)
    failed = [c for c, ok, _ in checks if not ok]
    for c, ok, d in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {c}" + (f" -- {d}" if not ok else ""))
    if failed:
        sys.exit(f"STOP: {len(failed)} integrity check(s) failed; not training.")

    # ---------------- data
    (train_real, train_syn), (val_real, val_syn) = build_train_val(cfg, args.data_root)
    concat = ConcatDataset([train_real, train_syn])
    sampler = BalancedDomainBatchSampler([len(train_real), len(train_syn)], sm["per_domain"], steps, cfg["seed"])
    nw = t["num_workers"]
    val_loaders = {s: DataLoader(d, batch_size=1, shuffle=False, num_workers=nw)
                   for s, d in zip(SUBSETS, (val_real, val_syn))}

    # ---------------- model / loss / optimizer (identical to LOL-v1 baseline)
    model = build_model(cfg["model"]).to(device)
    loss_fn = build_loss(cfg["loss"])
    opt = torch.optim.Adam(model.parameters(), lr=t["learning_rate"], betas=tuple(t["betas"]),
                           weight_decay=t["weight_decay"])
    start_epoch, best, history = 1, float("-inf"), []
    if args.resume:
        ck = load_checkpoint(args.resume)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer_state"])
        start_epoch, best, history = ck["epoch"] + 1, ck["best_val_psnr_mean"], ck["history"]
        print(f"Resumed from {args.resume} at epoch {ck['epoch']} (best mean val PSNR {best:.3f})")

    p = paths(cfg, args.data_root)
    save_config(cfg, out / "config.yaml")
    run_info = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "pipeline_test_mode": test_mode,
        "seed": cfg["seed"], "device": str(device), "gpu": gpu,
        "model": cfg["model"], "trainable_parameters": count_parameters(model),
        "loss": cfg["loss"], "optimizer": {"name": "adam", "lr": t["learning_rate"], "betas": t["betas"],
                                           "weight_decay": t["weight_decay"], "scheduler": "none"},
        "budget": {"epochs": t["epochs"], "steps_per_epoch": steps, "total_steps": t["epochs"] * steps,
                   "batch_size": t["batch_size"]},
        "sampler": {"type": "BalancedDomainBatchSampler", "domains": list(SUBSETS), "per_domain_per_batch": sm["per_domain"],
                    "samples_per_domain_per_epoch": steps * sm["per_domain"]},
        "data": {"root": p["root"], "manifest_md5": file_md5(p["manifest"]),
                 "train_pairs": {"real": len(train_real), "synthetic": len(train_syn)},
                 "val_pairs": {"real": len(val_real), "synthetic": len(val_syn)},
                 "split_files_md5": {s: file_md5(split_file(cfg, s)) for s in SUBSETS},
                 "sice_selection_md5": file_md5(sice_selection_file(cfg))},
        "augmentation": {"crop": cfg["data"]["crop_size"], "hflip": cfg["data"]["hflip"], "vflip": cfg["data"]["vflip"]},
        "selection": "max (real_val_PSNR + synthetic_val_PSNR) / 2",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform()},
        "resumed_from": args.resume,
    }
    name = f"run_info_resume_epoch{start_epoch}.json" if args.resume else "run_info.json"
    with open(out / "logs" / name, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"Device {device}{' (' + gpu + ')' if gpu else ''} | params {run_info['trainable_parameters']:,} | "
          f"train real {len(train_real)} / synthetic {len(train_syn)} | val real {len(val_real)} / synthetic {len(val_syn)} | "
          f"{steps} steps/epoch x {t['epochs']} epochs")

    t_run = time.time()
    for epoch in range(start_epoch, t["epochs"] + 1):
        # Per-epoch seeding -> identical batches/augmentation whether or not the run was resumed.
        torch.manual_seed(cfg["seed"] * 1000 + epoch)
        sampler.set_epoch(epoch - 1)
        loader = DataLoader(concat, batch_sampler=sampler, num_workers=nw, worker_init_fn=seed_worker,
                            generator=make_generator(cfg["seed"] * 1000 + epoch), pin_memory=device.type == "cuda")
        model.train()
        t0 = time.time()
        n_batches, dom_count = 0, {"lol_v2_real": 0, "lol_v2_synthetic": 0}
        loss_sum, dom_loss = 0.0, {"lol_v2_real": 0.0, "lol_v2_synthetic": 0.0}
        for batch in loader:
            doms = list(batch["domain"])
            if doms != ["lol_v2_real"] * sm["per_domain"] + ["lol_v2_synthetic"] * sm["per_domain"]:
                sys.exit(f"STOP: unexpected batch composition {doms}")
            low, high = batch["low"].to(device, non_blocking=True), batch["high"].to(device, non_blocking=True)
            pred = model(low)
            loss = loss_fn(pred, high)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n_batches += 1
            loss_sum += loss.item()
            with torch.no_grad():
                per = (pred - high).abs().flatten(1).mean(1)
                for d, v in zip(doms, per.tolist()):
                    dom_count[d] += 1
                    dom_loss[d] += v
            if t["log_every"] and n_batches % t["log_every"] == 0:
                print(f"  epoch {epoch} step {n_batches}/{steps} L1 {loss_sum / n_batches:.4f}", flush=True)
        if n_batches != steps:
            sys.exit(f"STOP: epoch produced {n_batches} optimizer steps, expected {steps}")

        val = {s: evaluate_domain(model, val_loaders[s], device, loss_fn=loss_fn) for s in SUBSETS}
        mean_psnr = (val["real"]["psnr"] + val["synthetic"]["psnr"]) / 2
        is_best = mean_psnr > best
        best = max(best, mean_psnr)
        row = {
            "epoch": epoch, "optimizer_steps_total": epoch * steps, "batches": n_batches,
            "samples_lol_v2_real": dom_count["lol_v2_real"], "samples_lol_v2_synthetic": dom_count["lol_v2_synthetic"],
            "train_loss": loss_sum / n_batches,
            "train_loss_real": dom_loss["lol_v2_real"] / dom_count["lol_v2_real"],
            "train_loss_synthetic": dom_loss["lol_v2_synthetic"] / dom_count["lol_v2_synthetic"],
            "val_real_loss": val["real"]["loss"], "val_real_psnr": val["real"]["psnr"], "val_real_ssim": val["real"]["ssim"],
            "val_synthetic_loss": val["synthetic"]["loss"], "val_synthetic_psnr": val["synthetic"]["psnr"],
            "val_synthetic_ssim": val["synthetic"]["ssim"],
            "val_psnr_mean": mean_psnr, "is_best": is_best,
            "lr": opt.param_groups[0]["lr"], "time_s": round(time.time() - t0, 1),
        }
        history.append(row)
        write_history(history, out)
        state = dict(optimizer_state=opt.state_dict(), epoch=epoch, history=history, best_val_psnr_mean=best,
                     metrics={k: row[k] for k in row if k.startswith("val_")})
        save_checkpoint(out / "checkpoints" / "last.pth", model, cfg, **state)
        if is_best:
            save_checkpoint(out / "checkpoints" / "best.pth", model, cfg, **state)
        print(f"Epoch {epoch:03d}/{t['epochs']} | steps {n_batches} (real {dom_count['lol_v2_real']}, syn "
              f"{dom_count['lol_v2_synthetic']}) | train L1 {row['train_loss']:.4f} | val PSNR real "
              f"{row['val_real_psnr']:.3f} syn {row['val_synthetic_psnr']:.3f} mean {mean_psnr:.3f} | SSIM real "
              f"{row['val_real_ssim']:.4f} syn {row['val_synthetic_ssim']:.4f} | {row['time_s']}s"
              + ("  *best*" if is_best else ""), flush=True)

    torch.save(load_checkpoint(out / "checkpoints" / "last.pth"), out / "checkpoints" / "final.pth")
    b = max(history, key=lambda h: h["val_psnr_mean"])
    summary = {
        "gpu": gpu, "device": str(device), "pipeline_test_mode": test_mode,
        "epochs_completed": history[-1]["epoch"], "optimizer_steps_total": history[-1]["optimizer_steps_total"],
        "train_time_s_this_session": round(time.time() - t_run, 1),
        "train_time_s_sum_of_epochs": round(sum(h["time_s"] for h in history), 1),
        "best_epoch": b["epoch"], "best_val_psnr_mean": b["val_psnr_mean"],
        "best_val_real_psnr": b["val_real_psnr"], "best_val_synthetic_psnr": b["val_synthetic_psnr"],
        "best_val_real_ssim": b["val_real_ssim"], "best_val_synthetic_ssim": b["val_synthetic_ssim"],
        "final_train_loss": history[-1]["train_loss"],
        "samples_per_domain_total": {"lol_v2_real": sum(h["samples_lol_v2_real"] for h in history),
                                     "lol_v2_synthetic": sum(h["samples_lol_v2_synthetic"] for h in history)},
        "best_checkpoint": str(out / "checkpoints" / "best.pth"),
    }
    with open(out / "logs" / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Best mean val PSNR {b['val_psnr_mean']:.3f} dB at epoch {b['epoch']} "
          f"(real {b['val_real_psnr']:.3f}, synthetic {b['val_synthetic_psnr']:.3f}).")


if __name__ == "__main__":
    main()
