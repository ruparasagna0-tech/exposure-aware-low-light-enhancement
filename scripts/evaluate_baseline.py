"""Evaluate a trained baseline checkpoint on the LOL-v1 eval15 TEST set.

Predictions are rounded to 8-bit before PSNR/SSIM so the reported numbers match
the saved PNG files. Metrics use range [0, 1] (see src/metrics/image_metrics.py).

Usage (from project root):
    python scripts/evaluate_baseline.py
    python scripts/evaluate_baseline.py --checkpoint path/to/best.pth --per-image

Outputs (default <results_dir> = results/baseline_lol_v1/):
    test_outputs/<name>.png   enhanced test images
    test_metrics.json|csv     mean + per-image PSNR/SSIM
"""

import argparse
import csv
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.datasets import build_lol_v1_dataset  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.utils import (  # noqa: E402
    ensure_outside_data,
    load_config,
    load_model_from_checkpoint,
    resolve_path,
    select_device,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline_lol_v1.yaml")
    ap.add_argument("--checkpoint", help="default: <checkpoint_dir>/best.pth")
    ap.add_argument("--data-root", help="override data.root")
    ap.add_argument("--out", help="default: output.results_dir")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--per-image", action="store_true", help="print per-image metrics")
    ap.add_argument("--no-save-images", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt_path = resolve_path(args.checkpoint or os.path.join(cfg["output"]["checkpoint_dir"], "best.pth"))
    out_dir = ensure_outside_data(resolve_path(args.out or cfg["output"]["results_dir"]))
    img_dir = out_dir / "test_outputs"
    img_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    model, ckpt = load_model_from_checkpoint(ckpt_path, device)
    test_ds = build_lol_v1_dataset(cfg, cfg["data"]["test_split"], root=args.data_root)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Checkpoint: {ckpt_path} (epoch {ckpt.get('epoch')}) | device {device} | test images {len(test_ds)}")

    def save_png(name, pred):
        arr = (pred.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        Image.fromarray(arr).save(img_dir / name)

    res = evaluate(model, loader, device, quantize=True,
                   on_output=None if args.no_save_images else save_png)

    if args.per_image:
        for r in res["per_image"]:
            print(f"  {r['name']:>10}  PSNR {r['psnr']:7.3f} dB  SSIM {r['ssim']:.4f}")
    print(f"TEST (LOL-v1 eval15, n={len(res['per_image'])}): "
          f"mean PSNR {res['psnr']:.3f} dB | mean SSIM {res['ssim']:.4f}")

    summary = {
        "checkpoint": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "split": "test (eval15)",
        "metric_range": "[0, 1], predictions quantized to 8-bit",
        "ssim": "Gaussian 11x11, sigma 1.5, RGB mean",
        "mean_psnr": res["psnr"],
        "mean_ssim": res["ssim"],
        "per_image": res["per_image"],
    }
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "test_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "psnr", "ssim"])
        w.writeheader()
        w.writerows(res["per_image"])
        w.writerow({"name": "MEAN", "psnr": res["psnr"], "ssim": res["ssim"]})
    print(f"Saved metrics to {out_dir}" + ("" if args.no_save_images else f" and images to {img_dir}"))


if __name__ == "__main__":
    main()
