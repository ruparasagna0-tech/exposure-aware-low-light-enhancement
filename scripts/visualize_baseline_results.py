"""Side-by-side test examples: low-light input | enhanced output | reference.

Runs the checkpoint directly on the LOL-v1 eval15 test images (no need to run
evaluate_baseline.py first).

Usage (from project root):
    python scripts/visualize_baseline_results.py
    python scripts/visualize_baseline_results.py --names 1.png 22.png --checkpoint path/to/best.pth

Output: <results_dir>/<out-name> (default test_comparison.png)
"""

import argparse
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.datasets import build_lol_v1_dataset  # noqa: E402
from src.metrics import psnr, quantize_uint8, ssim  # noqa: E402
from src.utils import (  # noqa: E402
    ensure_outside_data,
    load_config,
    load_model_from_checkpoint,
    resolve_path,
    select_device,
)


def to_img(t):
    return t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline_lol_v1.yaml")
    ap.add_argument("--checkpoint", help="default: <checkpoint_dir>/best.pth")
    ap.add_argument("--data-root", help="override data.root")
    ap.add_argument("--out", help="default: output.results_dir")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=4, help="number of random test examples")
    ap.add_argument("--names", nargs="*", help="specific test filenames instead of random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-name", default="test_comparison.png", help="figure filename")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt_path = resolve_path(args.checkpoint or os.path.join(cfg["output"]["checkpoint_dir"], "best.pth"))
    out_dir = ensure_outside_data(resolve_path(args.out or cfg["output"]["results_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    model, ckpt = load_model_from_checkpoint(ckpt_path, device)
    ds = build_lol_v1_dataset(cfg, cfg["data"]["test_split"], root=args.data_root)

    if args.names:
        idxs = [ds.names.index(n) for n in args.names]
    else:
        idxs = sorted(random.Random(args.seed).sample(range(len(ds)), min(args.n, len(ds))))

    fig, axes = plt.subplots(len(idxs), 3, figsize=(16, 4.2 * len(idxs)), squeeze=False)
    for r, i in enumerate(idxs):
        item = ds[i]
        low, high = item["low"].to(device), item["high"].to(device)
        with torch.no_grad():
            pred = quantize_uint8(model(low.unsqueeze(0))[0])
        p, s = psnr(pred, high).item(), ssim(pred, high).item()

        for ax, img, title in zip(
            axes[r],
            (low, pred, high),
            ("Low-light input", f"Enhanced output  (PSNR {p:.2f} dB, SSIM {s:.3f})", "High-light reference"),
        ):
            ax.imshow(to_img(img))
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(f"TEST\n{item['name']}", fontsize=12, rotation=0, ha="right", va="center")

    fig.suptitle(f"Baseline U-Net on LOL-v1 eval15 — checkpoint epoch {ckpt.get('epoch')}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out_dir / args.out_name
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
