"""Phase-2 dataset wiring (from configs/generalized_lol_v2.yaml) and pre-training integrity checks.

Roles (fixed by the approved protocol):
  train : LOL-v2-real Train (minus val scenes) + LOL-v2-synthetic Train (minus val scenes)
  val   : scene-disjoint subsets of those Train sets (checkpoint selection only)
  test  : LOL-v2-real Test, LOL-v2-synthetic Test (seen domains);
          SICE 58 official test sequences, LoLI-Street Val (unseen domains)
LOL-v1 (data/LOL) is never read here.
"""

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

from PIL import Image

from ..utils.config import PROJECT_ROOT, resolve_path
from .lol_v2 import LOLv2Dataset, lol_v2_pairs
from .loli_street import INVALID_VAL_IDS, LEVELS, LoLIStreetValDataset
from .exposure_augment import build_input_exposure_aug
from .paired_transforms import PairedCompose, build_paired_transforms
from .sice import SICETestDataset, read_test_ids

EXPECTED = {"real_train": 689, "real_test": 100, "synthetic_train": 900, "synthetic_test": 100,
            "sice_sequences": 58, "loli_ids": 999, "loli_pairs": 2997}
SUBSETS = ("real", "synthetic")


def paths(cfg: dict, root: Optional[str] = None) -> Dict[str, str]:
    r = str(resolve_path(root or cfg["data"]["root"]))
    d = cfg["data"]
    return {"root": r, "lol_v2": os.path.join(r, d["lol_v2_dir"]), "sice": os.path.join(r, d["sice_dir"]),
            "loli": os.path.join(r, d["loli_dir"]), "splits": str(resolve_path(cfg["splits"]["dir"])),
            "manifest": os.path.join(r, "manifest.json")}


def split_file(cfg: dict, subset: str) -> str:
    return os.path.join(str(resolve_path(cfg["splits"]["dir"])), f"lol_v2_{subset}_split.json")


def sice_selection_file(cfg: dict) -> str:
    return os.path.join(str(resolve_path(cfg["splits"]["dir"])), "sice_test_selection.json")


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_train_val(cfg: dict, root: Optional[str] = None):
    """-> (train_datasets[real, synthetic], val_datasets[real, synthetic]) in SUBSETS order."""
    p, d = paths(cfg, root), cfg["data"]
    train_tf = build_paired_transforms("train", crop_size=d["crop_size"], hflip=d["hflip"], vflip=d["vflip"])
    # Optional input-only exposure augmentation (off unless data.input_exposure_aug.enabled is true);
    # appended after crop -> hflip -> ToTensor, training inputs only.
    aug = build_input_exposure_aug(d.get("input_exposure_aug"))
    if aug is not None:
        train_tf = PairedCompose(train_tf.transforms + [aug])
    tr, va = [], []
    for s in SUBSETS:
        sp = load_json(split_file(cfg, s))
        tr.append(LOLv2Dataset(p["lol_v2"], s, "Train", names=sp["train_names"], transform=train_tf))
        va.append(LOLv2Dataset(p["lol_v2"], s, "Train", names=sp["val_names"], transform=build_paired_transforms("val")))
    return tr, va


def build_test(cfg: dict, root: Optional[str] = None) -> Dict[str, object]:
    p = paths(cfg, root)
    tf = build_paired_transforms("test")
    return {
        "lol_v2_real_test": LOLv2Dataset(p["lol_v2"], "real", "Test", transform=tf),
        "lol_v2_synthetic_test": LOLv2Dataset(p["lol_v2"], "synthetic", "Test", transform=tf),
        "sice": SICETestDataset(p["sice"], load_json(sice_selection_file(cfg))["sequences"], transform=tf),
        "loli_street_val": LoLIStreetValDataset(p["loli"], transform=tf),
    }


# ----------------------------------------------------------------------------- integrity checks
def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_image(path: str) -> Tuple[bool, str, Tuple[int, int]]:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return im.mode == "RGB", im.mode, im.size
    except Exception as e:  # noqa: BLE001
        return False, repr(e), (0, 0)


def integrity_checks(cfg: dict, root: Optional[str] = None, verify_md5: bool = True,
                     verify_images: bool = True) -> List[Tuple[str, bool, str]]:
    """Returns [(check, passed, detail)]. Any failure must stop training."""
    res: List[Tuple[str, bool, str]] = []
    add = lambda name, ok, detail="": res.append((name, bool(ok), str(detail)))  # noqa: E731
    p = paths(cfg, root)

    # 1. manifest / checksums
    if not os.path.isfile(p["manifest"]):
        add("manifest present", False, p["manifest"])
        return res
    man = load_json(p["manifest"])
    md5_of = {r["dest"]: r["md5"] for r in man["files"]}
    missing = [k for k in md5_of if not os.path.isfile(os.path.join(p["root"], k))]
    add("all manifest files present", not missing, f"{len(md5_of)} files, missing {missing[:5]}")
    if verify_md5 and not missing:
        bad = [k for k, v in md5_of.items() if _md5(os.path.join(p["root"], k)) != v]
        add("prepared files match manifest MD5 (= source bytes)", not bad, f"mismatches {bad[:5]}")

    # 2. counts / pairing (loaders raise on missing counterparts or orphans)
    try:
        pairs = {f"{s}_{sp.lower()}": lol_v2_pairs(p["lol_v2"], s, sp) for s in SUBSETS for sp in ("Train", "Test")}
        for k, v in pairs.items():
            add(f"LOL-v2 {k} pair count", len(v) == EXPECTED[k], f"{len(v)} (expected {EXPECTED[k]})")
    except Exception as e:  # noqa: BLE001
        add("LOL-v2 pairing", False, repr(e))
        return res
    try:
        loli = LoLIStreetValDataset(p["loli"]).pairs
        ids = {q["scene_id"] for q in loli}
        lv = {lvl: sum(q["level"] == lvl for q in loli) for lvl in LEVELS}
        add("LoLI-Street Val pairs", len(loli) == EXPECTED["loli_pairs"], f"{len(loli)} pairs, levels {lv}")
        add("LoLI-Street Val ids (30727 excluded)", len(ids) == EXPECTED["loli_ids"] and not ids & INVALID_VAL_IDS,
            f"{len(ids)} ids")
        add("LoLI-Street every id has all 3 levels", all(v == len(ids) for v in lv.values()), lv)
    except Exception as e:  # noqa: BLE001
        add("LoLI-Street Val pairing", False, repr(e))
    try:
        test_ids = read_test_ids(p["sice"])
        sel = load_json(sice_selection_file(cfg))
        seqs = sel["sequences"]
        add("SICE official test list = 58 sequences", len(test_ids) == EXPECTED["sice_sequences"], len(test_ids))
        add("SICE selection covers exactly the test list", [s["seq_id"] for s in seqs] == test_ids,
            f"{len(seqs)} selected")
        mb = cfg["eval"]["sice"]["min_brightness"]
        add("SICE selection rule/threshold matches config",
            sel["min_brightness"] == mb and all(s["selected_brightness"] >= mb for s in seqs), f">= {mb}")
        miss = [s["seq_id"] for s in seqs if not os.path.isfile(os.path.join(p["sice"], s["label_file"]))
                or not os.path.isfile(os.path.join(p["sice"], "Dataset_Part1", str(s["seq_id"]), s["selected_file"]))]
        add("SICE selected inputs and Labels exist", not miss, miss[:5])
        add("SICE Label orientation resolved", all(s["label_rotation_deg"] in (0, 90, 270) for s in seqs),
            {r: sum(s["label_rotation_deg"] == r for s in seqs) for r in (0, 90, 270)})
    except Exception as e:  # noqa: BLE001
        add("SICE test list / selection", False, repr(e))

    # 3. splits: complete, disjoint, scene-disjoint, consistent with config
    train_md5, val_md5 = set(), set()
    for s in SUBSETS:
        try:
            sp = load_json(split_file(cfg, s))
        except Exception as e:  # noqa: BLE001
            add(f"{s} split file", False, repr(e))
            continue
        c = cfg["splits"][s]
        add(f"{s} split settings match config",
            sp["seed"] == c["seed"] and sp["corr_threshold"] == c["corr_threshold"]
            and sp["n_val_scenes"] == c["n_val_scenes"] and list(sp["thumb_size"]) == list(c["thumb_size"]),
            f"seed {sp['seed']}, thr {sp['corr_threshold']}, val scenes {sp['n_val_scenes']}")
        tr, va = set(sp["train_names"]), set(sp["val_names"])
        allnames = {q["name"] for q in pairs[f"{s}_train"]}
        add(f"{s} train/val disjoint and complete", not tr & va and tr | va == allnames,
            f"train {len(tr)}, val {len(va)}, official Train {len(allnames)}")
        scene_of = {nm: k for k, members in enumerate(sp["scenes"]) for nm in members}
        vs = {scene_of[n] for n in va}
        ts = {scene_of[n] for n in tr}
        add(f"{s} train/val scene-disjoint", not vs & ts, f"{len(ts)} train scenes, {len(vs)} val scenes")
        rel = lambda path: os.path.relpath(path, p["root"])  # noqa: E731
        for q in pairs[f"{s}_train"]:
            (val_md5 if q["name"] in va else train_md5).update({md5_of.get(rel(q["low"])), md5_of.get(rel(q["high"]))})
        test_md5 = {md5_of.get(rel(q[k])) for q in pairs[f"{s}_test"] for k in ("low", "high")}
        vt = {md5_of.get(rel(q[k])) for q in pairs[f"{s}_train"] if q["name"] in va for k in ("low", "high")} & test_md5
        add(f"{s} val/test: no identical files", not vt, f"{len(vt)} shared")
        lk = sp.get("leakage_checks", {})
        add(f"{s} val/test: no near-duplicate scenes (split-time check)",
            lk.get("val_test_max_corr", 1.0) < c["corr_threshold"] and lk.get("val_test_pixel_identical", 1) == 0,
            lk)

    # 4. training pool isolation (no SICE / LoLI-Street / LOL-v1)
    ext_md5 = {r["md5"] for r in man["files"] if r["dest"].startswith(("SICE/", "LoLI-Street/"))}
    add("training/val files not identical to any SICE/LoLI-Street file", not (train_md5 | val_md5) & ext_md5)
    lol_root = os.path.realpath(p["lol_v2"])
    outside = [q["low"] for s in SUBSETS for q in pairs[f"{s}_train"]
               if not os.path.realpath(q["low"]).startswith(lol_root)]
    add("training pool paths only under LOL-v2", not outside, outside[:3])
    v1 = os.path.realpath(os.path.join(PROJECT_ROOT, "data", "LOL"))
    add("LOL-v1 folder not used", not os.path.realpath(p["root"]).startswith(v1))

    # 5. image readability / RGB / pair sizes
    if verify_images:
        bad, size_mis = [], []
        all_pairs = [q for v in pairs.values() for q in v]
        try:
            all_pairs += LoLIStreetValDataset(p["loli"]).pairs
        except Exception:  # noqa: BLE001
            pass
        for q in all_pairs:
            a, b = _check_image(q["low"]), _check_image(q["high"])
            if not (a[0] and b[0]):
                bad.append(q["low"])
            elif a[2] != b[2]:
                size_mis.append(q["name"])
        try:
            for s in load_json(sice_selection_file(cfg))["sequences"]:
                d = os.path.join(p["sice"], "Dataset_Part1", str(s["seq_id"]))
                for f in s["files"]:
                    if not _check_image(os.path.join(d, f))[0]:
                        bad.append(os.path.join(d, f))
                if not _check_image(os.path.join(p["sice"], s["label_file"]))[0]:
                    bad.append(s["label_file"])
        except Exception as e:  # noqa: BLE001
            bad.append(repr(e))
        add("all images readable and RGB", not bad, bad[:5])
        add("paired low/reference sizes match", not size_mis, size_mis[:5])
    return res
