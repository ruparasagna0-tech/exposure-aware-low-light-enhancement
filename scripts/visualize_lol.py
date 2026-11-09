"""Visual sanity check of the LOL-v1 pipeline (read-only; no training).

Saves one figure per split to results/visualization/:
    train: full-res reference with crop box | low (as loaded) | low (brightened) | high
    val/test: low (as loaded) | low (brightened) | high

"Brightened" is a display-only gain so dark low-light content can be compared
with the reference by eye; it is never used as data.

For training examples the crop box and flip are recovered independently from
the transformed low and high tensors (by locating them in the original
images), which checks that both got the same crop/flip.

Usage (from project root):
    .venv/bin/python scripts/visualize_lol.py
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
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402

from src.datasets import LOLv1Dataset, build_paired_transforms  # noqa: E402

DEFAULT_ROOT = os.environ.get(
    "LOL_V1_ROOT", os.path.join(PROJECT_ROOT, "data", "LOL", "LOLdataset")
)
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "results", "visualization")


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) float in [0,1] -> (H,W,3) uint8 (exact inverse of ToTensor)."""
    return (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def load_raw(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def brighten(low: np.ndarray, high: np.ndarray) -> tuple:
    """Display-only gain matching the low image's mean to the reference's."""
    gain = float(high.mean()) / max(float(low.mean()), 1e-6)
    return np.clip(low.astype(np.float32) * gain, 0, 255).astype(np.uint8), gain


def locate_crop(raw: np.ndarray, crop: np.ndarray):
    """Find (top, left, hflipped) such that crop == raw[top:, left:] (maybe flipped).

    Returns a list of all exact matches (normally exactly one).
    """
    h, w, _ = crop.shape
    matches = []
    for flipped in (False, True):
        target = crop[:, ::-1] if flipped else crop
        # Candidate positions from the first row, then confirm the full block.
        win = sliding_window_view(raw, (w, 3), axis=(1, 2))[:, :, 0]  # H, W-w+1, w, 3
        cand = np.argwhere((win == target[0]).all(axis=(-1, -2)))
        for top, left in cand:
            if top + h <= raw.shape[0] and np.array_equal(
                raw[top:top + h, left:left + w], target
            ):
                matches.append((int(top), int(left), flipped))
    return matches


def pick(n_total: int, n: int, seed: int):
    return sorted(random.Random(seed).sample(range(n_total), n))


def plot_train(ds, indices, crop, out_path):
    cols = ["Reference (full res) + crop box", "Low (loaded)", "Low (brightened, display only)", "High / reference"]
    fig, axes = plt.subplots(len(indices), 4, figsize=(18, 4.2 * len(indices)))
    report = []
    for r, idx in enumerate(indices):
        item = ds[idx]
        name = item["name"]
        low, high = to_uint8(item["low"]), to_uint8(item["high"])
        low_path, high_path = ds.get_paths(idx)
        raw_low, raw_high = load_raw(low_path), load_raw(high_path)

        m_low, m_high = locate_crop(raw_low, low), locate_crop(raw_high, high)
        box = m_high[0] if len(m_high) == 1 else None
        aligned = box is not None and box in m_low
        report.append((name, box, aligned, len(m_low), len(m_high)))

        bright, gain = brighten(low, high)
        axes[r, 0].imshow(raw_high)
        if box:
            top, left, flipped = box
            axes[r, 0].add_patch(Rectangle((left, top), crop, crop, fill=False, ec="red", lw=2))
        axes[r, 1].imshow(low)
        axes[r, 2].imshow(bright)
        axes[r, 3].imshow(high)
        if box:
            info = f"crop top={box[0]}, left={box[1]} | hflip={'YES' if box[2] else 'no'}"
        else:
            info = "crop NOT located"
        axes[r, 0].set_ylabel(f"TRAIN\n{name}", fontsize=12, rotation=0, ha="right", va="center")
        axes[r, 0].set_title(f"{cols[0]}\n{info}", fontsize=9)
        axes[r, 1].set_title(f"{cols[1]}  {low.shape[1]}x{low.shape[0]}", fontsize=9)
        axes[r, 2].set_title(f"{cols[2]}  x{gain:.1f}", fontsize=9)
        axes[r, 3].set_title(
            f"{cols[3]}  | low/high same crop+flip: {'YES' if aligned else 'NO'}", fontsize=9
        )
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"LOL-v1 TRAIN — training transform: RandomCrop({crop}) + RandomHFlip(p=0.5), no VFlip", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return report


def plot_eval(ds, indices, split, out_path):
    fig, axes = plt.subplots(len(indices), 3, figsize=(16, 4.2 * len(indices)), squeeze=False)
    report = []
    for r, idx in enumerate(indices):
        item = ds[idx]
        name = item["name"]
        low, high = to_uint8(item["low"]), to_uint8(item["high"])
        low_path, high_path = ds.get_paths(idx)
        # Deterministic conversion only => must equal the raw files exactly.
        exact = np.array_equal(low, load_raw(low_path)) and np.array_equal(high, load_raw(high_path))
        report.append((name, exact))

        bright, gain = brighten(low, high)
        axes[r, 0].imshow(low)
        axes[r, 1].imshow(bright)
        axes[r, 2].imshow(high)
        axes[r, 0].set_ylabel(f"{split.upper()}\n{name}", fontsize=12, rotation=0, ha="right", va="center")
        axes[r, 0].set_title(f"Low (loaded)  {low.shape[1]}x{low.shape[0]}", fontsize=9)
        axes[r, 1].set_title(f"Low (brightened, display only)  x{gain:.1f}", fontsize=9)
        axes[r, 2].set_title(f"High / reference  | identical to file: {'YES' if exact else 'NO'}", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"LOL-v1 {split.upper()} — deterministic: ToTensor only (no crop, no flip)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="Path to LOLdataset/")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output folder for figures")
    ap.add_argument("--split-seed", type=int, default=42, help="train/val split seed (as in loader)")
    ap.add_argument("--seed", type=int, default=0, help="seed for example choice and augmentation")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=4)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=2)
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.root))
    out = os.path.abspath(os.path.expanduser(args.out))
    data_dir = os.path.join(PROJECT_ROOT, "data")
    if os.path.commonpath([out, data_dir]) == data_dir or os.path.commonpath([out, root]) == root:
        sys.exit(f"Refusing to write inside the dataset folder: {out}")
    os.makedirs(out, exist_ok=True)

    torch.manual_seed(args.seed)
    ds = {
        "train": LOLv1Dataset(root, "train", build_paired_transforms("train", crop_size=args.crop), seed=args.split_seed),
        "val": LOLv1Dataset(root, "val", build_paired_transforms("val"), seed=args.split_seed),
        "test": LOLv1Dataset(root, "test", build_paired_transforms("test"), seed=args.split_seed),
    }
    print(f"Loaded: train={len(ds['train'])}, val={len(ds['val'])}, test={len(ds['test'])}")
    print(f"Train transform: {ds['train'].transform}")
    print(f"Val/test transform: {ds['val'].transform}")

    ok = True
    path = os.path.join(out, "lol_v1_train_examples.png")
    rep = plot_train(ds["train"], pick(len(ds["train"]), args.n_train, args.seed), args.crop, path)
    print(f"\nSaved {path}")
    for name, box, aligned, nl, nh in rep:
        print(f"  {name:>8}  crop(top,left,hflip)={box}  low/high same crop+flip={aligned}  (matches low={nl}, high={nh})")
        ok &= aligned

    for split, n in (("val", args.n_val), ("test", args.n_test)):
        path = os.path.join(out, f"lol_v1_{split}_examples.png")
        rep = plot_eval(ds[split], pick(len(ds[split]), n, args.seed), split, path)
        print(f"\nSaved {path}")
        for name, exact in rep:
            print(f"  {name:>8}  tensors identical to original files (no crop/flip)={exact}")
            ok &= exact

    print("\nRESULT:", "all visual sanity checks passed" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
