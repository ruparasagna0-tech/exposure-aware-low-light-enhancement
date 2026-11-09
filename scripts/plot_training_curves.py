"""Plot training curves from <log_dir>/history.json.

Panels: train L1 loss, val L1 loss, val PSNR, val SSIM (best val-PSNR epoch marked).

Usage (from project root):
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --history path/to/history.json --out path/to/dir

Output: <results_dir>/training_curves.png
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

from src.utils.config import ensure_outside_data, load_config, resolve_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline_lol_v1.yaml")
    ap.add_argument("--history", help="default: <log_dir>/history.json")
    ap.add_argument("--out", help="default: output.results_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    hist_path = resolve_path(args.history or os.path.join(cfg["output"]["log_dir"], "history.json"))
    out_dir = ensure_outside_data(resolve_path(args.out or cfg["output"]["results_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(hist_path) as f:
        h = json.load(f)
    ep = [r["epoch"] for r in h]
    best = max(h, key=lambda r: r["val_psnr"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("train_loss", "Training loss (L1)", "tab:blue"),
        ("val_loss", "Validation loss (L1)", "tab:orange"),
        ("val_psnr", "Validation PSNR (dB)", "tab:green"),
        ("val_ssim", "Validation SSIM", "tab:red"),
    ]
    for ax, (key, title, color) in zip(axes.ravel(), panels):
        ax.plot(ep, [r[key] for r in h], color=color, marker="o", ms=3, lw=1.5)
        ax.axvline(best["epoch"], color="gray", ls="--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    # The loss panels share a y-range so train and val are directly comparable.
    lo = min(min(r["train_loss"], r["val_loss"]) for r in h)
    hi = max(max(r["train_loss"], r["val_loss"]) for r in h)
    pad = 0.05 * (hi - lo or 1)
    for ax in axes[0]:
        ax.set_ylim(lo - pad, hi + pad)

    fig.suptitle(
        f"{cfg.get('experiment_name', 'run')} — best val PSNR {best['val_psnr']:.2f} dB "
        f"(SSIM {best['val_ssim']:.4f}) at epoch {best['epoch']} (dashed line)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out_dir / "training_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
