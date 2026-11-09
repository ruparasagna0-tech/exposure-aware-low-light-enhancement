"""Verify the LOL-v1 data pipeline (read-only; no training).

Usage (from project root):
    .venv/bin/python scripts/check_lol_loader.py
    .venv/bin/python scripts/check_lol_loader.py --root /path/to/LOLdataset

The root can also be set with the LOL_V1_ROOT environment variable.
Exits with code 1 if any check fails.
"""

import argparse
import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.datasets import (  # noqa: E402
    LOLv1Dataset,
    PairedToTensor,
    build_paired_transforms,
    find_pairs,
)

DEFAULT_ROOT = os.environ.get(
    "LOL_V1_ROOT", os.path.join(PROJECT_ROOT, "data", "LOL", "LOLdataset")
)
EXPECTED_COUNTS = {"train": 450, "val": 35, "test": 15}


class Checker:
    def __init__(self):
        self.failures = []

    def check(self, cond: bool, msg: str):
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            self.failures.append(msg)


def snapshot(root: str):
    """(relative path, size, mtime) for every file under root."""
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            st = os.stat(p)
            out[os.path.relpath(p, root)] = (st.st_size, st.st_mtime_ns)
    return out


def check_raw_files(c: Checker, root: str):
    print("\n[1] Raw files: pairing, readability, dimensions")
    for folder in ("our485", "eval15"):
        split_dir = os.path.join(root, folder)
        try:
            names = find_pairs(split_dir)
        except Exception as e:  # noqa: BLE001
            c.check(False, f"{folder}: pairing failed: {e}")
            continue
        c.check(True, f"{folder}: {len(names)} low/high pairs with matching filenames")

        sizes, modes, unreadable, mismatched = Counter(), Counter(), [], []
        for n in names:
            dims = []
            for sub in ("low", "high"):
                p = os.path.join(split_dir, sub, n)
                try:
                    with Image.open(p) as im:
                        im.load()  # full decode, not just the header
                        dims.append(im.size)
                        sizes[im.size] += 1
                        modes[im.mode] += 1
                except Exception as e:  # noqa: BLE001
                    unreadable.append(f"{sub}/{n}: {e}")
            if len(dims) == 2 and dims[0] != dims[1]:
                mismatched.append(f"{n}: low {dims[0]} vs high {dims[1]}")
        c.check(not unreadable, f"{folder}: all {2 * len(names)} images readable"
                + (f" (bad: {unreadable[:5]})" if unreadable else ""))
        c.check(not mismatched, f"{folder}: low/high dimensions match per pair"
                + (f" (bad: {mismatched[:5]})" if mismatched else ""))
        print(f"         sizes (W x H): {dict(sizes)}  modes: {dict(modes)}")


def check_splits(c: Checker, root: str, seed: int):
    print(f"\n[2] Splits (seed={seed})")
    ds = {s: LOLv1Dataset(root, s, seed=seed) for s in EXPECTED_COUNTS}
    for s, n in EXPECTED_COUNTS.items():
        c.check(len(ds[s]) == n, f"{s} count = {len(ds[s])} (expected {n})")

    train, val = set(ds["train"].names), set(ds["val"].names)
    all485 = set(find_pairs(os.path.join(root, "our485")))
    c.check(not (train & val), "train and val are disjoint")
    c.check(train | val == all485, "train + val cover all of our485")

    again = LOLv1Dataset(root, "val", seed=seed)
    c.check(again.names == ds["val"].names, "val split is reproducible with same seed")
    print(f"         val filenames: {ds['val'].names}")
    return ds


def check_tensors(c: Checker, ds: dict):
    print("\n[3] Tensors (full pass, default transform = PairedToTensor)")
    for s, d in ds.items():
        shapes, bad = Counter(), []
        vmin, vmax = float("inf"), float("-inf")
        for i in range(len(d)):
            item = d[i]
            low, high, name = item["low"], item["high"], item["name"]
            shapes[tuple(low.shape)] += 1
            if not (isinstance(name, str) and name == d.names[i]):
                bad.append(f"{i}: bad name {name!r}")
            if low.shape != high.shape or low.ndim != 3 or low.shape[0] != 3:
                bad.append(f"{name}: shape low {tuple(low.shape)} high {tuple(high.shape)}")
            if low.dtype != torch.float32 or high.dtype != torch.float32:
                bad.append(f"{name}: dtype {low.dtype}/{high.dtype}")
            lo = min(low.min().item(), high.min().item())
            hi = max(low.max().item(), high.max().item())
            vmin, vmax = min(vmin, lo), max(vmax, hi)
        c.check(not bad, f"{s}: {len(d)} items, shape (3,H,W) matching, float32, names ok"
                + (f" (bad: {bad[:5]})" if bad else ""))
        c.check(0.0 <= vmin and vmax <= 1.0, f"{s}: value range [{vmin:.4f}, {vmax:.4f}] within [0, 1]")
        print(f"         shapes: {dict(shapes)}")


def check_transforms(c: Checker, root: str, seed: int, crop: int):
    print(f"\n[4] Paired transforms (train crop={crop})")
    torch.manual_seed(0)
    tf = build_paired_transforms("train", crop_size=crop)
    d = LOLv1Dataset(root, "train", transform=tf, seed=seed)
    item = d[0]
    c.check(tuple(item["low"].shape) == (3, crop, crop)
            and item["low"].shape == item["high"].shape,
            f"train crop shape = {tuple(item['low'].shape)}")

    # Feed the SAME image as low and high: identical crop/flip => identical outputs.
    with Image.open(d.get_paths(0)[1]) as im:
        img = im.convert("RGB")
    ok = True
    for _ in range(50):
        a, b = tf(img, img)
        ok &= torch.equal(a, b)
    c.check(ok, "random crop/flip applied identically to low and high (50 trials)")

    ttf = build_paired_transforms("test")
    ref = PairedToTensor()(img, img)[0]
    c.check(all(torch.equal(ttf(img, img)[0], ref) for _ in range(5)),
            "test transform is deterministic (ToTensor only)")


def check_dataloader(c: Checker, root: str, seed: int, crop: int, batch_size: int):
    print(f"\n[5] DataLoader (batch_size={batch_size})")
    for s in EXPECTED_COUNTS:
        tf = build_paired_transforms(s, crop_size=crop if s == "train" else None)
        d = LOLv1Dataset(root, s, transform=tf, seed=seed)
        bs = batch_size if s != "test" else 1
        batch = next(iter(DataLoader(d, batch_size=bs, shuffle=(s == "train"), num_workers=0)))
        c.check(batch["low"].shape == batch["high"].shape and len(batch["name"]) == bs,
                f"{s}: batch low {tuple(batch['low'].shape)}, {len(batch['name'])} names")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="Path to LOLdataset/")
    ap.add_argument("--seed", type=int, default=42, help="train/val split seed")
    ap.add_argument("--crop", type=int, default=256, help="train crop size for transform check")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.root))
    print(f"LOL-v1 root: {root}")
    if not os.path.isdir(root):
        print("FAIL: root does not exist")
        sys.exit(1)

    c = Checker()
    before = snapshot(root)
    check_raw_files(c, root)
    ds = check_splits(c, root, args.seed)
    check_tensors(c, ds)
    check_transforms(c, root, args.seed, args.crop)
    check_dataloader(c, root, args.seed, args.crop, args.batch_size)

    print("\n[6] Dataset integrity")
    c.check(snapshot(root) == before, "no dataset files added, removed or modified")

    print("\n" + "=" * 60)
    if c.failures:
        print(f"RESULT: {len(c.failures)} check(s) FAILED")
        for f in c.failures:
            print(f"  - {f}")
        sys.exit(1)
    print("RESULT: all checks passed")


if __name__ == "__main__":
    main()
