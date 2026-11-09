"""Plot Phase-2 training curves from <output.dir>/history.json -> <output.dir>/evaluation/training_curves.png.

Panels: training L1 (overall + per domain), validation L1 per domain,
validation PSNR per domain + mean (selection metric; best epoch dashed), validation SSIM per domain.

Usage: python scripts/plot_generalized_curves.py [--output-dir DIR]
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.utils.config import load_config, resolve_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/generalized_lol_v2.yaml")
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve_path(args.output_dir or cfg["output"]["dir"])
    h = json.load(open(out / "history.json"))
    ep = [r["epoch"] for r in h]
    best = max(h, key=lambda r: r["val_psnr_mean"])
    panels = [
        ("Training L1", [("train_loss", "all"), ("train_loss_real", "real"), ("train_loss_synthetic", "synthetic")]),
        ("Validation L1", [("val_real_loss", "real"), ("val_synthetic_loss", "synthetic")]),
        ("Validation PSNR (dB)", [("val_real_psnr", "real"), ("val_synthetic_psnr", "synthetic"),
                                  ("val_psnr_mean", "mean (selection)")]),
        ("Validation SSIM", [("val_real_ssim", "real"), ("val_synthetic_ssim", "synthetic")]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (title, series) in zip(axes.ravel(), panels):
        for key, label in series:
            ax.plot(ep, [r[key] for r in h], marker="o", ms=2.5, lw=1.4, label=label)
        ax.axvline(best["epoch"], color="gray", ls="--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{cfg['experiment_name']} — best mean val PSNR {best['val_psnr_mean']:.2f} dB at epoch {best['epoch']}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    (out / "evaluation").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "evaluation" / "training_curves.png", dpi=120)
    print(f"Saved {out / 'evaluation' / 'training_curves.png'}")


if __name__ == "__main__":
    main()
