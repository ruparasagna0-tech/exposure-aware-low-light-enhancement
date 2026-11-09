"""Phase 3 progress demo (CPU): validation metrics, training curves and side-by-side enhanced images of the Tier-A
models, for a supervisor progress review. NOT a final evaluation.

Data used:
  - LOL-v2 validation splits (64 real + 92 synthetic; the data used for checkpoint selection, so slightly optimistic):
    metrics for every image + fixed-rule example panels.
  - LoLI-Street Val (development data, already inspected in Stage 0): image panels only (no metrics), 2 scenes x
    3 darkness levels, chosen by a fixed rule (Stage-0 worst LoLI scene 30231 + one scene drawn with seed 0).
  - Never: LOL-v2 Test (reserved for the final evaluation), SICE Part 2 (sealed).
Models: Phase-2 U-Net (E0, `stage0/source_phase2/checkpoints/best.pth`) and the Tier-A `best.pth` of each model found
under --runs (FP32 inference on CPU, outputs quantized to 8 bit as in the Phase-2 evaluation).
Each Tier-A checkpoint is verified: model name, epoch and stored validation PSNR must match its train_summary.json, and
the validation PSNR recomputed here (unquantized, the trainer's definition) must match the stored value within 0.05 dB.

Usage: .venv/bin/python scripts/phase3_progress_demo.py --runs ~/Downloads/tierA_runs [--out <new dir>]
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.datasets.loli_street import LoLIStreetValDataset  # noqa: E402
from src.datasets.phase2 import build_train_val, paths  # noqa: E402
from src.metrics import psnr, ssim  # noqa: E402
from src.metrics.colour_metrics import pointwise_metrics  # noqa: E402
from src.metrics.evaluation import quantize_uint8  # noqa: E402
from src.metrics.lpips_metric import LPIPSMetric  # noqa: E402
from src.phase3.models import build_phase3_model  # noqa: E402
from src.utils import load_checkpoint, load_config, load_model_from_checkpoint  # noqa: E402

E0_CKPT = PROJECT_ROOT / "experiments/phase3_model_selection/stage0/source_phase2/checkpoints/best.pth"
E0_VAL = {"real": 17.4546537399292, "synthetic": 19.32794589581697, "mean": 18.391299817873083, "epoch": 45}
PHASE2_HISTORY = PROJECT_ROOT / "experiments/phase3_model_selection/stage0/source_phase2/history.json"
ORDER = ["unet_expaug", "retinexformer", "hvi_cidnet", "darkir_m"]
NAMES = {"e0": "U-Net (Phase 2)", "unet_expaug": "U-Net + exp. aug", "retinexformer": "Retinexformer",
         "hvi_cidnet": "HVI-CIDNet", "darkir_m": "DarkIR-m (re-impl.)"}


def to_u8(t):
    return (t.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()


def load_models(runs: Path):
    models = {}
    e0, ck = load_model_from_checkpoint(E0_CKPT)
    models["e0"] = {"model": e0.eval(), "epoch": ck["epoch"], "val": E0_VAL}
    for m in ORDER:
        d = runs / f"tierA_{m}_s42"
        if not (d / "checkpoints/best.pth").exists():
            continue
        ck = load_checkpoint(d / "checkpoints/best.pth")
        summ = json.load(open(d / "logs/train_summary.json"))
        if ck.get("model_name") != m or ck["epoch"] != summ["best_epoch"]:
            sys.exit(f"{m}: checkpoint model/epoch {ck.get('model_name')}/{ck['epoch']} != train_summary best_epoch {summ['best_epoch']}")
        if abs(ck["metrics"]["val_psnr_mean"] - summ["best_val_psnr_mean"]) > 1e-9:
            sys.exit(f"{m}: checkpoint val PSNR differs from train_summary")
        net = build_phase3_model(m)
        net.load_state_dict(ck["model_state"])
        models[m] = {"model": net.eval(), "epoch": ck["epoch"],
                     "val": {"real": ck["metrics"]["val_real_psnr"], "synthetic": ck["metrics"]["val_synthetic_psnr"],
                             "mean": ck["metrics"]["val_psnr_mean"]},
                     "history": json.load(open(d / "history.json"))}
    return models


@torch.no_grad()
def evaluate(models, val_sets, lp):
    rows, keep = [], {}
    for subset, ds in val_sets.items():
        for i in range(len(ds)):
            item = ds[i]
            low, high = item["low"][None], item["high"][None]
            ref8 = to_u8(high[0])
            for k, mm in models.items():
                pred = mm["model"](low).clamp(0, 1)
                raw_psnr = float(psnr(pred, high))           # trainer's validation definition (unquantized)
                q = quantize_uint8(pred)
                c = pointwise_metrics(to_u8(q[0]), ref8)
                rows.append({"subset": subset, "name": item["name"], "model": k, "psnr_raw": raw_psnr,
                             "psnr": float(psnr(q, high)), "ssim": float(ssim(q, high)), "lpips": float(lp(q, high)[0]),
                             "de00": c["de00_mean"], "de00_colour_only": c["de00_leq_mean"], "dL": c["dL_mean"],
                             "chroma_ratio": c["chroma_ratio"]})
                keep[(subset, item["name"], k)] = to_u8(q[0])
            keep[(subset, item["name"], "input")] = to_u8(low[0])
            keep[(subset, item["name"], "reference")] = ref8
        print(f"  {subset}: {len(ds)} images", flush=True)
    return rows, keep


def summarise(rows, models):
    out = []
    for k in models:
        for subset in ("real", "synthetic", "all"):
            r = [x for x in rows if x["model"] == k and (subset == "all" or x["subset"] == subset)]
            mean = lambda f: float(np.mean([x[f] for x in r]))  # noqa: E731
            out.append({"model": NAMES[k], "key": k, "subset": subset, "n": len(r), "psnr": mean("psnr"),
                        "ssim": mean("ssim"), "lpips": mean("lpips"), "de00": mean("de00"),
                        "de00_colour_only": mean("de00_colour_only"), "dL": mean("dL"), "chroma_ratio": mean("chroma_ratio"),
                        "psnr_raw": mean("psnr_raw")})
    return out


def check_reload(rows, models):
    """Recomputed mean (unquantized) validation PSNR per subset must equal the checkpoint's stored value."""
    res = {}
    for k, mm in models.items():
        for subset in ("real", "synthetic"):
            got = float(np.mean([x["psnr_raw"] for x in rows if x["model"] == k and x["subset"] == subset]))
            res[f"{k}/{subset}"] = {"recomputed": got, "stored": mm["val"][subset], "diff": got - mm["val"][subset]}
    bad = {k: v for k, v in res.items() if abs(v["diff"]) > 0.05}
    return res, bad


def panel(keep, keys, models, rows, path, title):
    cols = ["input"] + list(models) + ["reference"]
    fig, axes = plt.subplots(len(keys), len(cols), figsize=(2.6 * len(cols), 2.2 * len(keys) + 0.6), squeeze=False)
    for r, (subset, name) in enumerate(keys):
        for c, k in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(keep[(subset, name, k)])
            ax.set_xticks([]), ax.set_yticks([])
            if k in models:
                x = next(x for x in rows if x["model"] == k and x["subset"] == subset and x["name"] == name)
                ax.set_xlabel(f"{x['psnr']:.2f} dB | ΔE {x['de00']:.1f}", fontsize=8)
            if r == 0:
                ax.set_title({"input": "Input", "reference": "Reference"}.get(k, NAMES.get(k, k)), fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{subset}\n{name}", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def loli_panels(models, cfg, out):
    ds = LoLIStreetValDataset(paths(cfg)["loli"])
    by_id = {}
    for i, p in enumerate(ds.pairs):
        by_id.setdefault(p["scene_id"], {})[p["level"]] = i
    ids = sorted(by_id)
    scenes = [30231, random.Random(0).choice([s for s in ids if s != 30231])]
    cols = ["input"] + list(models) + ["reference"]
    for sid in scenes:
        fig, axes = plt.subplots(3, len(cols), figsize=(2.6 * len(cols), 7.2), squeeze=False)
        for r, lvl in enumerate(("dense", "moderate", "light")):
            item = ds[by_id[sid][lvl]]
            low, high = item["low"][None], item["high"][None]
            ref8 = to_u8(high[0])
            for c, k in enumerate(cols):
                ax = axes[r, c]
                if k == "input":
                    img, lab = to_u8(low[0]), f"{lvl}"
                elif k == "reference":
                    img, lab = ref8, ""
                else:
                    q = quantize_uint8(models[k]["model"](low).clamp(0, 1))
                    img = to_u8(q[0])
                    m = pointwise_metrics(img, ref8)
                    lab = f"ΔL* {m['dL_mean']:+.1f} | ΔE {m['de00_mean']:.1f}"
                ax.imshow(img)
                ax.set_xticks([]), ax.set_yticks([])
                ax.set_xlabel(lab, fontsize=8)
                if r == 0:
                    ax.set_title({"input": "Input", "reference": "Reference"}.get(k, NAMES.get(k, k)), fontsize=9)
        fig.suptitle(f"LoLI-Street scene {sid}: the same scene at three darkness levels (development data, not a test set)\n"
                     "under each output: ΔL* (> 0 = brighter than the reference) | ΔE00 colour error", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"loli_street_scene_{sid}_darkness_levels.png", dpi=130)
        plt.close(fig)
    return scenes


def curves(models, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    p2 = json.load(open(PHASE2_HISTORY))
    ax.plot([e["epoch"] for e in p2], [e["val_psnr_mean"] for e in p2], label=NAMES["e0"], color="grey")
    for k, mm in models.items():
        if "history" in mm:
            h = mm["history"]
            ax.plot([e["epoch"] for e in h], [e["val_psnr_mean"] for e in h], label=NAMES[k])
    ax.set_xlabel("epoch (112 steps each)"), ax.set_ylabel("mean validation PSNR (dB)")
    ax.set_title("LOL-v2 validation PSNR during training (same data, steps and selection rule)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "training_curves_val_psnr.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="folder containing tierA_<model>_s42/ downloaded from Drive")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, help="CODE CHECK ONLY: first N validation images per subset")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    out = Path(args.out or PROJECT_ROOT / f"experiments/phase3_model_selection/stage1/progress_demo_{time.strftime('%Y%m%d_%H%M')}")
    if out.exists():
        sys.exit(f"{out} exists")
    out.mkdir(parents=True)
    cfg = load_config("configs/generalized_lol_v2.yaml")
    models = load_models(Path(args.runs).expanduser())
    print("models:", {k: f"epoch {v['epoch']}" for k, v in models.items()}, flush=True)
    (_, _), (val_real, val_syn) = build_train_val(cfg)
    vs = {"real": val_real, "synthetic": val_syn}
    if args.limit:
        from torch.utils.data import Subset
        vs = {k: Subset(v, range(args.limit)) for k, v in vs.items()}
        for k, v in vs.items():
            v.names = [v.dataset.names[i] for i in range(args.limit)]
    lp = LPIPSMetric("alex", torch.device("cpu"))
    t0 = time.time()
    rows, keep = evaluate(models, vs, lp)
    summary = summarise(rows, models)
    reload_check, bad = check_reload(rows, models) if not args.limit else ({}, {})
    with open(out / "validation_metrics_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "validation_metrics_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    rng = random.Random(0)
    picks = [("real", n) for n in rng.sample(list(vs["real"].names), min(3, len(vs["real"].names)))] + \
            [("synthetic", n) for n in rng.sample(list(vs["synthetic"].names), min(3, len(vs["synthetic"].names)))]
    panel(keep, picks, models, rows, out / "validation_examples.png",
          "LOL-v2 validation examples (chosen with random seed 0, not hand-picked). Under each: PSNR | colour error ΔE00")
    scenes = loli_panels(models, cfg, out) if not args.limit else []
    curves(models, out)
    md = ["# Phase 3 progress demo — validation metrics (NOT a final evaluation)", "",
          "LOL-v2 validation (64 real + 92 synthetic images; also used for checkpoint selection, so slightly optimistic). "
          "One training run (seed 42) per model, 5,600 steps, FP32. Outputs quantized to 8 bit. "
          "ΔE00 = CIEDE2000 colour difference (lower is better); colour-only = with brightness equalised; "
          "ΔL* > 0 = brighter than the reference.", "",
          "| Model | Subset | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | ΔE00 ↓ | ΔE00 colour-only ↓ | ΔL* | chroma ratio |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        sub = "all 156 (image-weighted)" if s["subset"] == "all" else s["subset"]
        md.append(f"| {s['model']} | {sub} | {s['psnr']:.2f} | {s['ssim']:.4f} | {s['lpips']:.4f} | {s['de00']:.2f} | "
                  f"{s['de00_colour_only']:.2f} | {s['dL']:+.2f} | {s['chroma_ratio']:.2f} |")
    md += ["", "**Selection metric** (mean of the real and synthetic validation PSNR, as used for checkpoint selection):",
           ""] + [f"- {NAMES[k]}: {np.mean([x['psnr_raw'] for x in rows if x['model'] == k and x['subset'] == 'real']) / 2 + np.mean([x['psnr_raw'] for x in rows if x['model'] == k and x['subset'] == 'synthetic']) / 2:.2f} dB (best epoch {mm['epoch']})"
                  for k, mm in models.items()]
    md += ["", f"Checkpoint reload check (recomputed vs stored validation PSNR, tolerance 0.05 dB): "
               f"{'PASS' if not bad else 'FAIL ' + str(bad)}",
           f"Figures: validation_examples.png, training_curves_val_psnr.png"
           + (f", LoLI-Street darkness-level panels for scenes {scenes} (development data)" if scenes else ""),
           f"CPU time: {time.time() - t0:.0f} s"]
    (out / "SUMMARY.md").write_text("\n".join(md) + "\n")
    json.dump({"reload_check": reload_check, "failed": bad}, open(out / "reload_check.json", "w"), indent=1)
    print("\n".join(md))
    print("written:", out)


if __name__ == "__main__":
    main()
