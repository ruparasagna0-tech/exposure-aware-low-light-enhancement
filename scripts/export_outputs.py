"""Phase 3 / Stage 0: export enhanced images of a checkpoint and reproduce its recorded metrics.

NEW script. It does not change scripts/evaluate_generalized.py or any existing output. It reuses the
same code path as the original evaluation (``build_test`` datasets, ``evaluate_domain``, 8-bit
quantization, tiled 512/64 SICE inference, separable SSIM for SICE, LPIPS-Alex), and additionally saves
every quantized prediction as a lossless 8-bit RGB PNG.

Outputs (default --out-dir experiments/phase3_model_selection/stage0/outputs/phase2_unet_best):
    <domain>/<stem>.png           enhanced image, exactly the tensor the metrics were computed on
    per_image/<domain>.csv        PSNR / SSIM / LPIPS recomputed here (+ metadata)
    export_manifest.json          checkpoint MD5, config, code MD5s, versions, device, per-file MD5s, timings
Each domain is written to <domain>.partial/ and renamed when complete; an existing domain is never
overwritten (rerunning skips completed domains).

Then compare against the recorded results with scripts/stage0_reproduction_report.py.

Usage:
    python scripts/export_outputs.py --checkpoint experiments/phase3_model_selection/stage0/source_phase2/checkpoints/best.pth
Pipeline test only: --limit N --out-dir <scratch dir>
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import PIL  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from src.datasets.phase2 import build_test  # noqa: E402
from src.metrics import ssim  # noqa: E402
from src.metrics.domain_evaluation import evaluate_domain  # noqa: E402
from src.metrics.large_image import ssim_separable  # noqa: E402
from src.metrics.lpips_metric import LPIPSMetric  # noqa: E402
from src.utils import ensure_outside_data, load_config, load_model_from_checkpoint, resolve_path, select_device  # noqa: E402
from src.utils.tiling import tiled_forward  # noqa: E402

DOMAINS = ("lol_v2_real_test", "lol_v2_synthetic_test", "sice", "loli_street_val")
DEFAULT_OUT = "experiments/phase3_model_selection/stage0/outputs/phase2_unet_best"
CODE_FILES = ["scripts/export_outputs.py", "src/datasets/phase2.py", "src/datasets/lol_v2.py", "src/datasets/sice.py",
              "src/datasets/loli_street.py", "src/datasets/paired_list.py", "src/datasets/paired_transforms.py",
              "src/metrics/domain_evaluation.py", "src/metrics/evaluation.py", "src/metrics/image_metrics.py",
              "src/metrics/large_image.py", "src/metrics/lpips_metric.py", "src/models/baseline_unet.py",
              "src/utils/tiling.py", "src/utils/checkpoint.py", "configs/generalized_lol_v2.yaml"]
ALEXNET_CACHE = Path(torch.hub.get_dir()) / "checkpoints" / "alexnet-owt-7be5be79.pth"


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def output_stem(name: str) -> str:
    """Dataset item name -> PNG file name (lossless; extension always .png)."""
    return os.path.splitext(name)[0] + ".png"


def tensor_to_uint8(pred: torch.Tensor) -> np.ndarray:
    """[3, H, W] quantized prediction in [0, 1] -> uint8 [H, W, 3] (R, G, B order). Exact for quantized input."""
    if pred.dim() != 3 or pred.shape[0] != 3:
        raise ValueError(f"expected [3, H, W], got {tuple(pred.shape)}")
    if float(pred.min()) < 0 or float(pred.max()) > 1:
        raise ValueError("prediction outside [0, 1]")
    arr = torch.round(pred.detach().cpu().double() * 255.0)
    if not torch.allclose(arr, pred.detach().cpu().double() * 255.0, atol=1e-3):
        raise ValueError("prediction is not 8-bit quantized")
    return arr.to(torch.uint8).permute(1, 2, 0).contiguous().numpy()


def save_png(arr: np.ndarray, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    Image.fromarray(arr, mode="RGB").save(path, format="PNG", compress_level=6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/generalized_lol_v2.yaml")
    ap.add_argument("--data-root")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--device", default="cpu", help="cpu or cuda (MPS refused: invalid values on large images)")
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=DOMAINS)
    ap.add_argument("--threads", type=int, help="torch CPU threads")
    ap.add_argument("--limit", type=int, help="PIPELINE TEST ONLY: first N items per domain")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = select_device(args.device)
    if device.type == "mps":
        sys.exit("Refusing MPS (produced invalid metric values on large images in Phase-2 testing).")
    if not ALEXNET_CACHE.is_file():
        sys.exit(f"LPIPS backbone weights not cached at {ALEXNET_CACHE}; refusing to download anything.")
    out = ensure_outside_data(resolve_path(args.out_dir))
    for protected in ("experiments/generalized_lol_v2", "experiments/generalized_lol_v2_expaug"):
        p = (PROJECT_ROOT / protected).resolve()
        if out == p or p in out.parents:
            sys.exit(f"Refusing to write inside the existing experiment folder {p}")
    (out / "per_image").mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    ckpt_path = resolve_path(args.checkpoint)
    model, ckpt = load_model_from_checkpoint(ckpt_path, device)
    if ckpt["config"]["model"] != cfg["model"]:
        sys.exit(f"checkpoint model config {ckpt['config']['model']} != {args.config} {cfg['model']}")
    lp = LPIPSMetric(cfg["eval"]["lpips_net"], device)
    sc = cfg["eval"]["sice"]
    tiled = lambda m, x: tiled_forward(m, x, tile=sc["tile"], overlap=sc["overlap"], device=device)  # noqa: E731
    datasets = build_test(cfg, args.data_root)

    manifest_path = out / "export_manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else {"domains": {}}
    ck_md5 = md5(ckpt_path)
    if manifest.get("checkpoint_md5") not in (None, ck_md5):
        sys.exit(f"{out} was exported from a different checkpoint ({manifest['checkpoint_md5']})")
    manifest.update({
        "purpose": "Stage 0: baseline recovery + output export (no training, no new model)",
        "evidence_status": "LOL-v2 Test: seen-domain test. SICE and LoLI-Street Val: previously evaluated and "
                           "inspected development/evaluation evidence, not untouched final test sets.",
        "checkpoint": str(ckpt_path), "checkpoint_md5": ck_md5, "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_metrics": ckpt.get("metrics"), "config": args.config,
        "data_root": str(resolve_path(args.data_root or cfg["data"]["root"])),
        "inference": {"quantize_uint8": cfg["eval"]["quantize"], "sice": f"tiled {sc['tile']}/{sc['overlap']} linear blend",
                      "others": "full image, single forward pass"},
        "png": "8-bit RGB, lossless, values = round(255 * quantized prediction)",
        "code_md5": {f: md5(PROJECT_ROOT / f) for f in CODE_FILES},
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                     "pillow": PIL.__version__, "platform": platform.platform()},
        "device": str(device), "torch_threads": torch.get_num_threads(), "limit": args.limit,
    })

    for name in args.domains:
        final_dir, part_dir = out / name, out / f"{name}.partial"
        if final_dir.exists():
            print(f"{name}: already exported, skipping (never overwritten)")
            continue
        if part_dir.exists():
            shutil.rmtree(part_dir)  # incomplete earlier attempt of THIS script (never a final folder)
        part_dir.mkdir(parents=True)
        ds = datasets[name]
        idx = list(range(len(ds)))[: args.limit] if args.limit else list(range(len(ds)))
        saved = {}

        def keep(row, pred, _dir=part_dir, _saved=saved):
            fn = output_stem(row["name"])
            if fn in _saved:
                raise RuntimeError(f"duplicate output name {fn}")
            save_png(tensor_to_uint8(pred), _dir / fn)
            _saved[fn] = row["name"]

        t0 = time.time()
        loader = DataLoader(Subset(ds, idx), batch_size=1, shuffle=False, num_workers=0)
        r = evaluate_domain(model, loader, device, forward_fn=tiled if name == "sice" else None, lpips_fn=lp,
                            quantize=cfg["eval"]["quantize"], on_output=keep,
                            ssim_fn=ssim_separable if name == "sice" else ssim)
        rows = [dict(x, output_file=f"{name}/{output_stem(x['name'])}") for x in r["per_image"]]
        with open(out / "per_image" / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        files = {fn: {"item": item, "md5": md5(part_dir / fn)} for fn, item in sorted(saved.items())}
        os.replace(part_dir, final_dir)
        manifest["domains"][name] = {"n": r["n"], "psnr": r["psnr"], "ssim": r["ssim"], "lpips": r["lpips"],
                                     "seconds": round(time.time() - t0, 1), "files": files}
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=1, default=str)
        print(f"{name:24s} n={r['n']:5d}  PSNR {r['psnr']:.3f}  SSIM {r['ssim']:.4f}  LPIPS {r['lpips']:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
