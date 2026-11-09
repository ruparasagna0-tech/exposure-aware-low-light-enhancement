"""Train the baseline U-Net (L1) on LOL-v1 train, validate on the 35-pair val split.

Intended for Google Colab GPU. Selection of the best checkpoint uses validation
PSNR only; the eval15 test set is never touched here.

Usage (from project root):
    python scripts/train_baseline.py --config configs/baseline_lol_v1.yaml
    python scripts/train_baseline.py --data-root /content/drive/MyDrive/.../LOLdataset
    python scripts/train_baseline.py --resume experiments/baseline_lol_v1/checkpoints/last.pth

Outputs:
    <checkpoint_dir>/best.pth   best validation PSNR
    <checkpoint_dir>/last.pth   latest epoch (used for --resume)
    <log_dir>/history.json|csv  per-epoch train loss, val loss, PSNR, SSIM
    <log_dir>/run_info.json     seed, config, split, optimizer, loss, environment
    <log_dir>/config.yaml       resolved config actually used
"""

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.datasets import build_lol_v1_dataset  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import (  # noqa: E402
    count_parameters,
    ensure_outside_data,
    load_checkpoint,
    load_config,
    make_generator,
    resolve_path,
    save_checkpoint,
    save_config,
    seed_worker,
    select_device,
    set_seed,
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline_lol_v1.yaml")
    ap.add_argument("--data-root", help="override data.root")
    ap.add_argument("--device", help="override train.device (auto|cuda|mps|cpu)")
    ap.add_argument("--epochs", type=int, help="override train.epochs")
    ap.add_argument("--num-workers", type=int, help="override train.num_workers")
    ap.add_argument("--resume", help="path to a last.pth checkpoint to continue from")
    return ap.parse_args()


def apply_overrides(cfg, args):
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.device:
        cfg["train"]["device"] = args.device
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    return cfg


def build_optimizer(model, t):
    if t["optimizer"].lower() != "adam":
        raise ValueError(f"Only 'adam' is supported for the baseline, got {t['optimizer']}")
    return torch.optim.Adam(
        model.parameters(), lr=t["learning_rate"], betas=tuple(t["betas"]), weight_decay=t["weight_decay"]
    )


def write_history(history, log_dir):
    with open(os.path.join(log_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(log_dir, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    t = cfg["train"]

    # --- Reproducibility: seed everything before building data/model. ---
    set_seed(cfg["seed"], deterministic=t["deterministic"])
    device = select_device(t["device"])
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None

    ckpt_dir = ensure_outside_data(resolve_path(cfg["output"]["checkpoint_dir"]))
    log_dir = ensure_outside_data(resolve_path(cfg["output"]["log_dir"]))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- Data: existing LOL-v1 loader + paired transforms. ---
    train_ds = build_lol_v1_dataset(cfg, cfg["data"]["train_split"])
    val_ds = build_lol_v1_dataset(cfg, cfg["data"]["val_split"])
    nw = t["num_workers"]
    train_loader = DataLoader(
        train_ds,
        batch_size=t["batch_size"],
        shuffle=True,
        drop_last=True,  # keep every step at the full batch size
        num_workers=nw,
        worker_init_fn=seed_worker,
        generator=make_generator(cfg["seed"]),  # reproducible shuffling/worker seeds
        pin_memory=device.type == "cuda",
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["eval"]["batch_size"], shuffle=False, num_workers=nw)

    # --- Model / loss / optimizer. ---
    model = build_model(cfg["model"]).to(device)
    loss_fn = build_loss(cfg["loss"])
    optimizer = build_optimizer(model, t)
    n_params = count_parameters(model)

    start_epoch, best_psnr, history = 1, float("-inf"), []
    if args.resume:
        ck = load_checkpoint(args.resume)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        start_epoch = ck["epoch"] + 1
        best_psnr = ck.get("best_val_psnr", best_psnr)
        history = ck.get("history", [])
        print(f"Resumed from {args.resume} (epoch {ck['epoch']}, best val PSNR {best_psnr:.3f})")

    # --- Record everything needed to reproduce / describe this run. ---
    save_config(cfg, log_dir / "config.yaml")
    run_info = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "seed": cfg["seed"],
        "device": str(device),
        "model": cfg["model"],
        "trainable_parameters": n_params,
        "loss": cfg["loss"],
        "optimizer": {"name": t["optimizer"], "lr": t["learning_rate"], "betas": t["betas"],
                      "weight_decay": t["weight_decay"]},
        "epochs": t["epochs"],
        "batch_size": t["batch_size"],
        "crop_size": cfg["data"]["crop_size"],
        "augmentation": {"hflip": cfg["data"]["hflip"], "vflip": cfg["data"]["vflip"]},
        "dataset": {
            "root": str(resolve_path(cfg["data"]["root"])),
            "split_seed": cfg["data"]["split_seed"],
            "train_count": len(train_ds),
            "val_count": len(val_ds),
            "val_names": val_ds.names,
            "train_names": train_ds.names,
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform(), "gpu": gpu_name},
        "resumed_from": args.resume,
    }
    # Keep the original run_info on resume; record the resume separately.
    info_name = f"run_info_resume_epoch{start_epoch}.json" if args.resume else "run_info.json"
    with open(log_dir / info_name, "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"Device: {device}" + (f" ({gpu_name})" if gpu_name else "")
          + f" | params: {n_params:,} | train {len(train_ds)} / val {len(val_ds)} "
          f"| {len(train_loader)} iters/epoch")

    # --- Training loop. ---
    run_t0 = time.time()
    for epoch in range(start_epoch, t["epochs"] + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for it, batch in enumerate(train_loader, 1):
            low = batch["low"].to(device, non_blocking=True)
            high = batch["high"].to(device, non_blocking=True)

            pred = model(low)
            loss = loss_fn(pred, high)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * low.shape[0]
            n += low.shape[0]
            if t["log_every"] and it % t["log_every"] == 0:
                print(f"  epoch {epoch} iter {it}/{len(train_loader)} train L1 {loss_sum / n:.4f}")

        train_loss = loss_sum / n
        val = evaluate(model, val_loader, device, loss_fn=loss_fn)  # full-res, deterministic
        is_best = val["psnr"] > best_psnr
        best_psnr = max(best_psnr, val["psnr"])

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_psnr": val["psnr"],
            "val_ssim": val["ssim"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": round(time.time() - t0, 1),
        })
        write_history(history, log_dir)

        state = dict(optimizer_state=optimizer.state_dict(), epoch=epoch, history=history,
                     best_val_psnr=best_psnr,
                     metrics={"val_loss": val["loss"], "val_psnr": val["psnr"], "val_ssim": val["ssim"]})
        save_checkpoint(ckpt_dir / "last.pth", model, cfg, **state)
        if is_best:
            save_checkpoint(ckpt_dir / "best.pth", model, cfg, **state)

        print(f"Epoch {epoch:03d}/{t['epochs']} | train L1 {train_loss:.4f} | val L1 {val['loss']:.4f} | "
              f"PSNR {val['psnr']:.3f} dB | SSIM {val['ssim']:.4f} | {time.time() - t0:.1f}s"
              + ("  *best*" if is_best else ""))

    # last.pth after the final epoch is the final checkpoint; also keep an explicit copy.
    final = load_checkpoint(ckpt_dir / "last.pth")
    torch.save(final, ckpt_dir / "final.pth")

    # Summary of the whole run (selection by validation PSNR only).
    best = max(history, key=lambda h: h["val_psnr"])
    summary = {
        "gpu": gpu_name,
        "device": str(device),
        "epochs_completed": history[-1]["epoch"],
        "train_time_s_this_session": round(time.time() - run_t0, 1),
        "train_time_s_sum_of_epochs": round(sum(h["time_s"] for h in history), 1),
        "best_epoch": best["epoch"],
        "best_val_psnr": best["val_psnr"],
        "best_val_ssim": best["val_ssim"],
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "best_checkpoint": str(ckpt_dir / "best.pth"),
        "history": str(log_dir / "history.json"),
    }
    with open(log_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done in {summary['train_time_s_sum_of_epochs'] / 60:.1f} min. Best val PSNR {best_psnr:.3f} dB "
          f"at epoch {best['epoch']}. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
