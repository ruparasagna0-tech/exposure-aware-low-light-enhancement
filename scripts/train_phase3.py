"""Phase 3 / Stage 1: train one candidate model under the Phase-2 protocol (Tier A).

Same protocol as scripts/train_generalized.py (which is unchanged): integrity checks first; every batch =
2 LOL-v2-real + 2 LOL-v2-synthetic crops; 112 steps x 50 epochs; full-resolution validation after every epoch on
the scene-disjoint val subsets; best.pth = max (real_val_PSNR + synthetic_val_PSNR) / 2; test sets, SICE and
LoLI-Street are never read. Per model: the method's own loss (src/phase3/losses.py) and optimizer/schedule
(src/phase3/schedules.py, rescaled to the step budget).

Checkpoints: last.pth after every epoch, best.pth on validation improvement (atomic writes). --resume continues
from last.pth; the batch stream is regenerated per epoch, so a resumed run sees the same batches.

Precision must be chosen explicitly (--amp on|off); the config value 'pilot_gated' is refused.

Usage (Colab):
    python scripts/train_phase3.py --config configs/phase3/tierA_retinexformer.yaml --data-root /content/phase2 \
        --run-dir /content/drive/MyDrive/LowLight-Generalization/experiments/phase3_model_selection/stage1/runs/tierA_retinexformer_s42 \
        --device cuda --amp off
Pipeline-test only (recorded as such): --epochs, --steps-per-epoch, --skip-integrity-md5.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.datasets.phase2 import integrity_checks, paths, split_file  # noqa: E402
from src.phase3.engine import Engine, state_hash  # noqa: E402
from src.phase3.models import EXPECTED_PARAMS  # noqa: E402
from src.phase3.schedules import RECIPES  # noqa: E402
from src.utils import ensure_outside_data, load_checkpoint, load_config, save_checkpoint, select_device, set_seed  # noqa: E402

VENDORED = sorted(str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "third_party").rglob("*.py"))
CODE = ["src/phase3/models.py", "src/phase3/losses.py", "src/phase3/schedules.py", "src/phase3/engine.py",
        "scripts/train_phase3.py"] + VENDORED


def md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def atomic_json(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-root")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", choices=["on", "off"], required=True)
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--resume", help="path to last.pth of this run")
    ap.add_argument("--epochs", type=int, help="PIPELINE TEST ONLY")
    ap.add_argument("--steps-per-epoch", type=int, help="PIPELINE TEST ONLY")
    ap.add_argument("--skip-integrity-md5", action="store_true", help="PIPELINE TEST ONLY")
    ap.add_argument("--no-val", action="store_true", help="PIPELINE TEST ONLY: skip validation (no best.pth)")
    ap.add_argument("--stop-after-epoch", type=int, help="PIPELINE TEST ONLY: exit after this epoch (simulated disconnect)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name, t, sm = cfg["model"]["name"], cfg["train"], cfg["sampler"]
    epochs = args.epochs or t["epochs"]
    steps = args.steps_per_epoch or sm["steps_per_epoch"]
    test_mode = bool(args.epochs or args.steps_per_epoch or args.skip_integrity_md5 or args.no_val
                     or args.stop_after_epoch)
    nw = t["num_workers"] if args.num_workers is None else args.num_workers
    device = select_device(args.device)
    if device.type == "mps":
        sys.exit("MPS refused (invalid values on large images in Phase-2 testing); use cuda or cpu.")
    set_seed(cfg["seed"], deterministic=t["deterministic"])
    out = ensure_outside_data(Path(args.run_dir).expanduser().resolve())
    for protected in ("experiments/generalized_lol_v2", "experiments/phase3_model_selection/stage0"):
        p = (PROJECT_ROOT / protected).resolve()
        if out == p or p in out.parents:
            sys.exit(f"refusing to write inside {p}")
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    if (out / "checkpoints" / "last.pth").exists() and not args.resume:
        sys.exit(f"{out}/checkpoints/last.pth exists: refusing to overwrite (use --resume).")

    checks = integrity_checks(cfg, args.data_root, verify_md5=not args.skip_integrity_md5)
    atomic_json([{"check": c, "passed": ok, "detail": d} for c, ok, d in checks], out / "logs" / "integrity_checks.json")
    if any(not ok for _, ok, _ in checks):
        sys.exit("STOP: integrity check(s) failed: " + ", ".join(c for c, ok, _ in checks if not ok))

    eng = Engine(cfg, name, device, amp=args.amp == "on", total_steps=epochs * steps, data_root=args.data_root)
    n = eng.n_params()
    if n != EXPECTED_PARAMS[name]["count"]:
        sys.exit(f"STOP: {name} has {n} parameters, expected {EXPECTED_PARAMS[name]['count']}")
    start_epoch, best, history = 1, float("-inf"), []
    cfg_md5 = md5(args.config)
    if args.resume:
        ck = load_checkpoint(args.resume)
        if ck.get("config_md5") != cfg_md5:
            sys.exit("STOP: config changed since the checkpoint was written")
        eng.load_state(ck)
        start_epoch, best, history = ck["epoch"] + 1, ck["best_val_psnr_mean"], ck["history"]
        print(f"Resumed {args.resume} at epoch {ck['epoch']} (best mean val PSNR {best:.3f})", flush=True)

    p = paths(cfg, args.data_root)
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120).stdout
    except Exception as e:  # noqa: BLE001
        freeze = f"unavailable: {e!r}"
    run_info = {
        "started": datetime.now().isoformat(timespec="seconds"), "pipeline_test_mode": test_mode,
        "model": name, "params": n, "expected_params": EXPECTED_PARAMS[name], "loss": cfg["loss"],
        "recipe": RECIPES[name], "amp": eng.amp, "seed": cfg["seed"], "deterministic": t["deterministic"],
        "budget": {"epochs": epochs, "steps_per_epoch": steps, "total_steps": epochs * steps,
                   "batch_size": t["batch_size"]},
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                        "platform": platform.platform()},
        "config": args.config, "config_md5": cfg_md5, "code_md5": {f: md5(PROJECT_ROOT / f) for f in CODE},
        "data": {"root": p["root"], "manifest_md5": md5(p["manifest"]),
                 "split_files_md5": {s: md5(split_file(cfg, s)) for s in ("real", "synthetic")}},
        "resumed_from": args.resume,
    }
    tag = f"_resume_epoch{start_epoch}" if args.resume else ""
    atomic_json(run_info, out / "logs" / f"run_info{tag}.json")
    (out / "logs" / f"pip_freeze{tag}.txt").write_text(freeze)
    print(f"{name}: {n:,} params | AMP {eng.amp} | {steps} steps x {epochs} epochs | device {run_info['gpu'] or device}",
          flush=True)

    val_loaders = eng.val_loaders(nw)
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        sums = {"loss": 0.0, "lol_v2_real": 0.0, "lol_v2_synthetic": 0.0}
        counts = {"lol_v2_real": 0, "lol_v2_synthetic": 0}
        skipped, nb = 0, 0
        names_md5 = hashlib.md5()
        for batch in eng.epoch_loader(epoch, steps, nw):
            names_md5.update("|".join(batch["name"]).encode())
            r = eng.train_step(batch, (epoch - 1) * steps + nb)
            nb += 1
            sums["loss"] += r["loss"]
            skipped += r["skipped"]
            for d, v in zip(r["domains"], r["per_sample_l1"]):
                sums[d] += v
                counts[d] += 1
            if t["log_every"] and nb % t["log_every"] == 0:
                print(f"  epoch {epoch} step {nb}/{steps} loss {sums['loss'] / nb:.4f} lr {r['lr']:.2e}", flush=True)
        if nb != steps:
            sys.exit(f"STOP: epoch produced {nb} steps, expected {steps}")
        train_time = time.time() - t0
        row = {"epoch": epoch, "optimizer_steps_total": epoch * steps, "train_loss": sums["loss"] / nb,
               "train_l1_real": sums["lol_v2_real"] / counts["lol_v2_real"],
               "train_l1_synthetic": sums["lol_v2_synthetic"] / counts["lol_v2_synthetic"],
               "amp_skipped_steps": skipped, "lr_end": r["lr"], "train_time_s": round(train_time, 1),
               "batch_names_md5": names_md5.hexdigest()}
        is_best = False
        if not args.no_val:
            tv = time.time()
            row.update(eng.validate(val_loaders))
            row["val_time_s"] = round(time.time() - tv, 1)
            is_best = row["val_psnr_mean"] > best
            best = max(best, row["val_psnr_mean"])
        row["is_best"] = is_best
        history.append(row)
        atomic_json(history, out / "history.json")
        with open(out / "logs" / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for h in history for k in h}))
            w.writeheader()
            w.writerows(history)
        state = dict(eng.state(), epoch=epoch, history=history, best_val_psnr_mean=best, config_md5=cfg_md5,
                     metrics={k: v for k, v in row.items() if k.startswith("val_")})
        tc = time.time()
        save_checkpoint(out / "checkpoints" / "last.pth", eng.model, cfg, **state)
        if is_best:
            save_checkpoint(out / "checkpoints" / "best.pth", eng.model, cfg, **state)
        print(f"Epoch {epoch:03d}/{epochs} | loss {row['train_loss']:.4f} | "
              + (f"val PSNR real {row['val_real_psnr']:.3f} syn {row['val_synthetic_psnr']:.3f} mean "
                 f"{row['val_psnr_mean']:.3f} | " if not args.no_val else "")
              + f"train {row['train_time_s']}s ckpt {time.time() - tc:.1f}s skipped {skipped}"
              + ("  *best*" if is_best else ""), flush=True)
        if args.stop_after_epoch and epoch >= args.stop_after_epoch and epoch < epochs:
            print(f"PIPELINE TEST: stopping after epoch {epoch} (simulated disconnect)", flush=True)
            return

    summary = {"model": name, "pipeline_test_mode": test_mode, "epochs_completed": history[-1]["epoch"],
               "optimizer_steps_total": history[-1]["optimizer_steps_total"], "amp": eng.amp,
               "final_state_md5": state_hash(eng.model), "gpu": run_info["gpu"]}
    if not args.no_val:
        b = max(history, key=lambda h: h["val_psnr_mean"])
        summary.update(best_epoch=b["epoch"], best_val_psnr_mean=b["val_psnr_mean"],
                       best_val_real_psnr=b["val_real_psnr"], best_val_synthetic_psnr=b["val_synthetic_psnr"])
    atomic_json(summary, out / "logs" / "train_summary.json")
    print("Done.", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
