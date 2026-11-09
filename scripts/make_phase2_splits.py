"""Generate (or re-verify) the fixed Phase-2 split artefacts in <splits.dir>:

  lol_v2_real_split.json        scene-disjoint train/val of LOL-v2-real Train      (thr 0.85, 25 val scenes)
  lol_v2_synthetic_split.json   scene-disjoint train/val of LOL-v2-synthetic Train (thr 0.95, 89 val scenes)
  sice_test_selection.json      per official SICE test sequence: selected input exposure
                                (darkest with mean brightness >= 5), -1 EV info, Label orientation

If a file already exists, it is regenerated in memory and must match exactly
(reproducibility check); nothing is overwritten.

Usage (from project root):
    .venv/bin/python scripts/make_phase2_splits.py [--data-root data_prepared/phase2]
"""

import argparse
import json
import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402

from src.datasets.lol_v2 import lol_v2_pairs  # noqa: E402
from src.datasets.phase2 import paths, sice_selection_file, split_file  # noqa: E402
from src.datasets.scene_split import group_scenes, split_scenes  # noqa: E402
from src.datasets.sice import read_test_ids, select_sequence  # noqa: E402
from src.utils import load_config  # noqa: E402

COMPARE_KEYS = ("train_names", "val_names", "scenes", "val_scene_indices")


def make_lol_split(cfg, p, subset):
    c = cfg["splits"][subset]
    train_pairs = lol_v2_pairs(p["lol_v2"], subset, "Train")
    test_pairs = lol_v2_pairs(p["lol_v2"], subset, "Test")
    g = group_scenes(train_pairs, c["corr_threshold"], tuple(c["thumb_size"]))
    train, val, val_idx = split_scenes(g["scenes"], c["n_val_scenes"], c["seed"])

    # leakage checks against the official Test set (same signature method)
    gt = group_scenes(test_pairs, c["corr_threshold"], tuple(c["thumb_size"]))
    th = g["thumbs"]
    tt = gt["thumbs"]
    V = [n for n in val]
    T = [q["name"] for q in test_pairs]
    VL, VH = np.stack([th[n][0] for n in V]), np.stack([th[n][1] for n in V])
    TL, TH = np.stack([tt[n][0] for n in T]), np.stack([tt[n][1] for n in T])
    vt = np.maximum(VL @ TL.T, VH @ TH.T)
    AL, AH = np.stack([th[q["name"]][0] for q in train_pairs]), np.stack([th[q["name"]][1] for q in train_pairs])
    at = np.maximum(AL @ TL.T, AH @ TH.T)
    test_px = {m for v in gt["pixel_md5"].values() for m in v}
    val_px = {m for n in V for m in g["pixel_md5"][n]}
    sizes = Counter(len(s) for s in g["scenes"])
    return {
        "dataset": f"LOL-v2-{subset}",
        "official_split": "Train",
        "seed": c["seed"],
        "method": "connected components of: pixel-identical low OR reference, or thumbnail Pearson corr "
                  ">= corr_threshold (low-low or reference-reference); scenes ordered by first member, "
                  "val scenes = random.Random(seed).sample(range(n_scenes), n_val_scenes)",
        "corr_threshold": c["corr_threshold"],
        "thumb_size": list(c["thumb_size"]),
        "n_scenes": len(g["scenes"]),
        "n_val_scenes": c["n_val_scenes"],
        "counts": {"train_pairs": len(train), "val_pairs": len(val), "total_pairs": len(train_pairs),
                   "train_scenes": len(g["scenes"]) - len(val_idx), "val_scenes": len(val_idx),
                   "val_fraction_pairs": round(len(val) / len(train_pairs), 4)},
        "scene_size_histogram": dict(sorted(sizes.items())),
        "leakage_checks": {
            "val_test_max_corr": round(float(vt.max()), 4),
            "val_test_pixel_identical": len(val_px & test_px),
            "info_all_train_test_max_corr": round(float(at.max()), 4),
        },
        "val_scene_indices": val_idx,
        "train_names": train,
        "val_names": val,
        "scenes": g["scenes"],
    }


def make_sice_selection(cfg, p):
    mb = cfg["eval"]["sice"]["min_brightness"]
    ids = read_test_ids(p["sice"])
    seqs = []
    for i in ids:
        seqs.append(select_sequence(p["sice"], i, mb))
        print(f"  SICE {i}: selected {seqs[-1]['selected_file']} (brightness {seqs[-1]['selected_brightness']:.1f})",
              flush=True)
    return {
        "dataset": "SICE Dataset_Part1 official test sequences (testing index.txt)",
        "rule": "darkest exposure with mean brightness >= min_brightness; brightness = mean of full-resolution "
                "8-bit grayscale (PIL 'L'); exposure order NOT taken from filenames",
        "min_brightness": mb,
        "reference": "dataset-provided Label image (not independently verified ground truth)",
        "minus_1ev_note": "testing index.txt: -1 EV input = #3 of 7 images, #4 of 9 images (recorded for reference only)",
        "n_sequences": len(seqs),
        "sequences": seqs,
    }


def save_or_verify(path, data, keys):
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        diff = [k for k in keys if old.get(k) != data.get(k)]
        if diff:
            sys.exit(f"EXISTING {path} differs from regenerated result in {diff} -> not overwriting; investigate.")
        print(f"verified (reproduced identically): {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    print(f"written: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/generalized_lol_v2.yaml")
    ap.add_argument("--data-root")
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = paths(cfg, args.data_root)
    for s in ("real", "synthetic"):
        sp = make_lol_split(cfg, p, s)
        print(f"{s}: {sp['n_scenes']} scenes; {sp['counts']}; leakage {sp['leakage_checks']}")
        save_or_verify(split_file(cfg, s), sp, COMPARE_KEYS)
    sel = make_sice_selection(cfg, p)
    save_or_verify(sice_selection_file(cfg), sel, ("sequences", "min_brightness"))


if __name__ == "__main__":
    main()
