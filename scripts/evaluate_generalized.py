"""Phase 2: evaluate the BEST validation checkpoint on every test domain (once).

Domains (reported separately, never only as one combined average):
  seen   : lol_v2_real_test (100), lol_v2_synthetic_test (100)            full-image inference
  unseen : sice (58 official test sequences, native resolution, tiled 512/64 linear blend)
           loli_street_val (2,997 pairs) + per level: dense / moderate / light
Metrics: PSNR, SSIM (project conventions, [0,1], 8-bit-quantized predictions), LPIPS (AlexNet).

Outputs (<output.dir>/evaluation/):
  metrics_summary.json / .md          per-domain n, mean PSNR / SSIM / LPIPS (+ checkpoint info)
  per_image/<domain>.csv              per-image / per-sequence metrics
  sice_evaluation_manifest.json       selected exposure, brightness, -1 EV info, Label rotation, metrics
  qualitative/<domain>.png            Input | Output | Reference (display-only downscaling for SICE)
Refuses to run if metrics_summary.json exists (test sets are evaluated once).

Usage (from project root):
    python scripts/evaluate_generalized.py [--data-root /content/phase2]
Pipeline-test options: --checkpoint, --output-dir, --limit.
"""

import argparse
import csv
import json
import os
import random
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from src.datasets.loli_street import LEVELS  # noqa: E402
from src.datasets.phase2 import build_test, load_json, sice_selection_file  # noqa: E402
from src.metrics.domain_evaluation import evaluate_domain, group_summaries  # noqa: E402
from src.metrics.large_image import ssim_separable  # noqa: E402
from src.metrics.lpips_metric import LPIPSMetric  # noqa: E402
from src.metrics import ssim  # noqa: E402
from src.utils import ensure_outside_data, load_config, load_model_from_checkpoint, resolve_path, select_device  # noqa: E402
from src.utils.tiling import tiled_forward  # noqa: E402

DISPLAY_LONG_SIDE = 900


def small(t):
    """Display-only downscale of a [3,H,W] tensor (figures only; metrics use full resolution)."""
    h, w = t.shape[-2:]
    s = DISPLAY_LONG_SIDE / max(h, w)
    if s >= 1:
        return t
    return F.interpolate(t[None], size=(round(h * s), round(w * s)), mode="area")[0]


def save_rows(rows, path):
    keys = [k for k in rows[0].keys()]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/generalized_lol_v2.yaml")
    ap.add_argument("--checkpoint", help="default: <output.dir>/checkpoints/best.pth")
    ap.add_argument("--data-root")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--metric-device", help="device for PSNR/SSIM/LPIPS (default: --device)")
    ap.add_argument("--allow-mps", action="store_true", help="MPS produced invalid metric values in testing; refused by default")
    ap.add_argument("--output-dir", help="PIPELINE TEST ONLY")
    ap.add_argument("--limit", type=int, help="PIPELINE TEST ONLY: first N items per domain")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = ensure_outside_data(resolve_path(args.output_dir or cfg["output"]["dir"]))
    ev_dir = out / "evaluation"
    if (ev_dir / "metrics_summary.json").exists():
        sys.exit(f"{ev_dir}/metrics_summary.json exists: test domains were already evaluated for this run.")
    for sub in ("per_image", "qualitative"):
        (ev_dir / sub).mkdir(parents=True, exist_ok=True)
    ckpt_path = resolve_path(args.checkpoint or os.path.join(cfg["output"]["dir"], "checkpoints", "best.pth"))
    device = select_device(args.device)
    if device.type == "mps" and not args.allow_mps:
        sys.exit("Refusing MPS: it produced invalid values on large images during testing. Use --device cpu/cuda.")
    mdev = select_device(args.metric_device) if args.metric_device else device
    model, ckpt = load_model_from_checkpoint(ckpt_path, device)
    lp = LPIPSMetric(cfg["eval"]["lpips_net"], mdev)
    sc = cfg["eval"]["sice"]
    q = cfg["eval"]["qualitative_per_domain"]
    print(f"Checkpoint {ckpt_path} (epoch {ckpt.get('epoch')}, val {ckpt.get('metrics')}) | device {device}, metrics on {mdev}")

    datasets = build_test(cfg, args.data_root)
    tiled = lambda m, x: tiled_forward(m, x, tile=sc["tile"], overlap=sc["overlap"], device=device)  # noqa: E731
    results, summary = {}, {}
    rng = random.Random(0)
    for name, ds in datasets.items():
        idx = list(range(len(ds)))[: args.limit] if args.limit else list(range(len(ds)))
        pick = set(rng.sample(idx, min(q, len(idx))))  # qualitative examples (fixed seed)
        pick_names = {ds.names[i] for i in pick}
        shown = {}

        def keep(row, pred, _pick=pick_names, _shown=shown):
            if row["name"] in _pick:
                _shown[row["name"]] = small(pred)

        loader = DataLoader(Subset(ds, idx), batch_size=1, shuffle=False, num_workers=0)
        t0 = time.time()
        r = evaluate_domain(model, loader, device, forward_fn=tiled if name == "sice" else None,
                            lpips_fn=lp, quantize=cfg["eval"]["quantize"], on_output=keep, metric_device=mdev,
                            ssim_fn=ssim_separable if name == "sice" else ssim)
        results[name] = r
        summary[name] = {k: r[k] for k in ("n", "psnr", "ssim", "lpips")}
        if name == "loli_street_val":
            for lvl, s in group_summaries(r["per_image"], "level").items():
                summary[f"loli_street_val_{lvl}"] = {k: s[k] for k in ("n", "psnr", "ssim", "lpips")}
        save_rows(r["per_image"], ev_dir / "per_image" / f"{name}.csv")
        print(f"{name:24s} n={r['n']:5d}  PSNR {r['psnr']:.3f}  SSIM {r['ssim']:.4f}  LPIPS {r['lpips']:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

        # qualitative figure: Input | Output | Reference
        rows = sorted(pick, key=lambda i: ds.names[i])
        fig, axes = plt.subplots(len(rows), 3, figsize=(13, 3.6 * len(rows)), squeeze=False)
        for k, i in enumerate(rows):
            item = ds[i]
            m = next(x for x in r["per_image"] if x["name"] == item["name"])
            for ax, img, title in zip(axes[k], (small(item["low"]), shown[item["name"]], small(item["high"])),
                                      ("Input (low-light)", f"Output  PSNR {m['psnr']:.2f} / SSIM {m['ssim']:.3f} / "
                                       f"LPIPS {m['lpips']:.3f}", "Reference" + (" (dataset Label)" if name == "sice" else ""))):
                ax.imshow(img.clamp(0, 1).permute(1, 2, 0).numpy())
                ax.set_title(title, fontsize=9)
                ax.axis("off")
            axes[k, 0].set_title(f"{item['name']}" + (f" [{item.get('level')}]" if "level" in item else "")
                                 + "\nInput (low-light)", fontsize=9)
        fig.suptitle(f"{name} — best checkpoint (epoch {ckpt.get('epoch')})", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(ev_dir / "qualitative" / f"{name}.png", dpi=90)
        plt.close(fig)

    # SICE manifest: selection info + metrics per sequence
    sel = {s["seq_id"]: s for s in load_json(sice_selection_file(cfg))["sequences"]}
    sice_rows = []
    for rrow in results["sice"]["per_image"]:
        s = sel[rrow["seq_id"]]
        sice_rows.append({**{k: s[k] for k in ("seq_id", "n_exposures", "selected_file", "selected_brightness",
                                                "excluded_too_dark", "minus_1ev_index_per_testing_index_txt",
                                                "minus_1ev_brightness", "label_file", "label_rotation_deg", "size")},
                          "psnr": rrow["psnr"], "ssim": rrow["ssim"], "lpips": rrow["lpips"]})
    with open(ev_dir / "sice_evaluation_manifest.json", "w") as f:
        json.dump({"reference_note": "SICE Label = dataset-provided reference, not independently verified ground truth",
                   "input_rule": "darkest exposure with mean brightness >= 5 (0-255)",
                   "inference": f"native resolution, tiled {sc['tile']}/{sc['overlap']} linear blending",
                   "sequences": sice_rows}, f, indent=1)

    meta = {"checkpoint": str(ckpt_path), "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_val_metrics": ckpt.get("metrics"), "selection": "max (real_val_PSNR + synthetic_val_PSNR)/2",
            "metrics": "PSNR/SSIM on [0,1] RGB of 8-bit-quantized outputs (SICE: exact strip-wise separable SSIM); LPIPS AlexNet (normalize=True)",
            "device": str(device), "metric_device": str(mdev),
            "limited_to_first_n": args.limit}
    with open(ev_dir / "metrics_summary.json", "w") as f:
        json.dump({"meta": meta, "domains": summary}, f, indent=1)
    order = ["lol_v2_real_test", "lol_v2_synthetic_test", "sice", "loli_street_val"] + [f"loli_street_val_{l}" for l in LEVELS]
    md = ["| Domain | Role | n | PSNR (dB) | SSIM | LPIPS |", "|---|---|---:|---:|---:|---:|"]
    for k in [k for k in order if k in summary]:
        s = summary[k]
        role = "seen" if k.startswith("lol_v2") else "unseen"
        md.append(f"| {k} | {role} | {s['n']} | {s['psnr']:.3f} | {s['ssim']:.4f} | {s['lpips']:.4f} |")
    with open(ev_dir / "metrics_summary.md", "w") as f:
        f.write(f"Checkpoint: {ckpt_path} (epoch {ckpt.get('epoch')})\n\n" + "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
