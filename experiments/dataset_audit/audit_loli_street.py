"""READ-ONLY structure + overlap audit of LoLI-Street (data/LoLI-Street/*.zip).

Nothing is extracted: the ZIP is read in memory; SICE is streamed from its RAR via
`bsdtar -cf - @archive` (as in audit_sice_structure.py); LOL-v2 read from its ZIP.

Checks
  integrity   : ZipFile.testzip() (CRC of every member)
  structure   : folders, counts, naming, orphan files, per-image format/mode/size
  exact dups  : byte MD5 and decoded-pixel MD5
  pairing     : low/high filename correspondence, equal sizes, structural correlation
                (brightness-normalised 32x32 thumbnails), full-res best shift on a sample
  levels      : whether dense/moderate/light share one reference per id
  scenes      : near-duplicate search (normalised thumbnails) within/between splits;
                threshold calibrated on known low/high pairs vs random non-pairs
  YOLO        : class list, label format, low vs high label agreement, orphans
  cross-data  : vs LOL-v1, LOL-v2 (real + synthetic), SICE (labels + exposures)

Outputs (next to this script): loli_street_structure_audit.json / .txt
Usage: .venv/bin/python experiments/dataset_audit/audit_loli_street.py
"""

import glob
import hashlib
import io
import json
import os
import random
import re
import subprocess
import tarfile
import time
import zipfile
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOLI_DIR = os.path.join(PROJECT_ROOT, "data", "LoLI-Street")
V1_ROOT = os.path.join(PROJECT_ROOT, "data", "LOL", "LOLdataset")
V2_ZIP = os.path.join(PROJECT_ROOT, "data", "LOL-v2", "LOL-v2.zip")
SICE_DIR = os.path.join(PROJECT_ROOT, "data", "SICE")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TH = 32          # thumbnail side (centre-square crop -> TH x TH grayscale)
SHIFT = 3
rng = random.Random(0)


def sq_thumb(img: Image.Image) -> np.ndarray:
    g = img.convert("L")
    w, h = g.size
    s = min(w, h)
    g = g.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    t = np.asarray(g.resize((TH, TH), Image.Resampling.BOX), dtype=np.float32).ravel()
    t -= t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


def best_shift(a, b, r=SHIFT):
    best, bm = None, np.inf
    h, w = a.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            aa = a[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)]
            bb = b[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)]
            m = ((aa - bb) ** 2).mean()
            if m < bm:
                best, bm = (dy, dx), m
    return best


def topk_matches(A, B, thr, same=False, chunk=2048):
    """For each row of A: best match index/corr in B, and all (i, j, corr) >= thr."""
    best_j = np.empty(len(A), dtype=np.int64)
    best_c = np.empty(len(A), dtype=np.float32)
    hits = []
    for s in range(0, len(A), chunk):
        C = A[s:s + chunk] @ B.T
        if same:
            for k in range(C.shape[0]):
                C[k, s + k] = -1
        best_j[s:s + chunk] = C.argmax(axis=1)
        best_c[s:s + chunk] = C.max(axis=1)
        ii, jj = np.where(C >= thr)
        hits += [(int(s + i), int(j), float(C[i, j])) for i, j in zip(ii, jj) if not same or s + i < j]
    return best_j, best_c, hits


def main():
    t0 = time.time()
    zips = sorted(f for f in os.listdir(LOLI_DIR) if f.lower().endswith(".zip"))
    assert len(zips) == 1, zips
    zpath = os.path.join(LOLI_DIR, zips[0])
    z = zipfile.ZipFile(zpath)
    res = {"file": zpath, "file_size": os.path.getsize(zpath),
           "other_files_in_dir": sorted(set(os.listdir(LOLI_DIR)) - {zips[0]})}

    print("CRC test ...", flush=True)
    bad = z.testzip()
    infos = z.infolist()
    res["integrity"] = {"testzip_first_bad_member": bad, "members": len(infos),
                        "dir_entries": sum(i.is_dir() for i in infos),
                        "uncompressed_bytes": sum(i.file_size for i in infos),
                        "compress_types": dict(Counter(i.compress_type for i in infos))}
    names = [i.filename for i in infos if not i.is_dir()]
    root = names[0].split("/")[0]
    rel = [n[len(root) + 1:] for n in names]
    res["root_folder"] = root
    res["folders"] = dict(sorted(Counter(os.path.dirname(r) for r in rel).items()))
    res["extensions"] = dict(Counter(os.path.splitext(r)[1] for r in rel))

    # ------------------------------------------------------------ images
    img_rel = [r for r in rel if r.lower().endswith((".jpg", ".jpeg", ".png"))]
    rec = {}
    for k, r in enumerate(img_rel):
        data = z.read(f"{root}/{r}")
        im = Image.open(io.BytesIO(data))
        fmt, mode, size = im.format, im.mode, im.size
        arr = np.asarray(im.convert("RGB"))
        rec[r] = {"fmt": fmt, "mode": mode, "size": size, "bmd5": hashlib.md5(data).hexdigest(),
                  "pmd5": hashlib.md5(arr.tobytes()).hexdigest(), "lum": float(arr.mean()),
                  "th": sq_thumb(Image.fromarray(arr))}
        if (k + 1) % 5000 == 0:
            print(f"  decoded {k + 1}/{len(img_rel)} ({time.time() - t0:.0f}s)", flush=True)

    def split_of(r):
        return r.split("/")[0]

    def stats(rs):
        return {"images": len(rs), "formats": dict(Counter(rec[r]["fmt"] for r in rs)),
                "modes": dict(Counter(rec[r]["mode"] for r in rs)),
                "sizes": dict(Counter("x".join(map(str, rec[r]["size"])) for r in rs)),
                "mean_lum": round(float(np.mean([rec[r]["lum"] for r in rs])), 2) if rs else None}

    groups = defaultdict(list)
    for r in img_rel:
        groups[os.path.dirname(r)].append(r)
    res["image_folders"] = {g: stats(v) for g, v in sorted(groups.items())}
    lum_by = defaultdict(list)
    for r in img_rel:
        m = re.match(r"(dense|moderate|light)_", os.path.basename(r))
        lum_by[f"{os.path.dirname(r)}:{m.group(1) if m else 'test'}"].append(rec[r]["lum"])
    res["mean_luminance_by_folder_and_level"] = {k: round(float(np.mean(v)), 2) for k, v in sorted(lum_by.items())}

    # ------------------------------------------------------------ pairing
    pairing = {}
    pair_list = []
    for sp in ("Train", "Val"):
        lows = {os.path.basename(r) for r in groups.get(f"{sp}/low", [])}
        highs = {os.path.basename(r) for r in groups.get(f"{sp}/high", [])}
        common = sorted(lows & highs)
        size_mis = [n for n in common if rec[f"{sp}/low/{n}"]["size"] != rec[f"{sp}/high/{n}"]["size"]]
        corr = np.array([float(rec[f"{sp}/low/{n}"]["th"] @ rec[f"{sp}/high/{n}"]["th"]) for n in common])
        pair_list += [(sp, n) for n in common]
        low_corr = sorted(zip(corr.tolist(), common))[:15]
        pairing[sp] = {"low_files": len(lows), "high_files": len(highs), "pairs": len(common),
                       "only_low": sorted(lows - highs)[:20], "only_high": sorted(highs - lows)[:20],
                       "size_mismatches": size_mis[:20], "n_size_mismatches": len(size_mis),
                       "low_high_identical_pixels": sum(rec[f"{sp}/low/{n}"]["pmd5"] == rec[f"{sp}/high/{n}"]["pmd5"] for n in common),
                       "pair_corr_percentiles_1_5_50": [round(float(np.percentile(corr, p)), 3) for p in (1, 5, 50)],
                       "pair_corr_lowest15": low_corr}
    # random non-pair baseline for calibration
    tr = [n for s, n in pair_list if s == "Train"]
    nonpair = np.array([float(rec[f"Train/low/{a}"]["th"] @ rec[f"Train/high/{b}"]["th"])
                        for a, b in ((rng.choice(tr), rng.choice(tr)) for _ in range(20000))
                        if a.split("_")[1] != b.split("_")[1]])
    pairing["calibration"] = {
        "true_pair_corr_p1": pairing["Train"]["pair_corr_percentiles_1_5_50"][0],
        "random_nonpair_corr_p99": round(float(np.percentile(nonpair, 99)), 3),
        "random_nonpair_corr_p999": round(float(np.percentile(nonpair, 99.9)), 3),
    }
    # full-res alignment sample
    sample = rng.sample(pair_list, 300)
    shifts = Counter()
    for sp, n in sample:
        a = np.asarray(Image.open(io.BytesIO(z.read(f"{root}/{sp}/low/{n}"))).convert("L"), np.float32)
        b = np.asarray(Image.open(io.BytesIO(z.read(f"{root}/{sp}/high/{n}"))).convert("L"), np.float32)
        a = (a - a.mean()) / (a.std() + 1e-6)
        b = (b - b.mean()) / (b.std() + 1e-6)
        shifts[str(best_shift(a, b))] += 1
    pairing["alignment_sample_300_best_shift"] = dict(shifts)
    res["pairing"] = pairing

    # ------------------------------------------------------------ darkness levels share a reference?
    lv = {}
    for sp, (lo, hi) in (("Train", (1, 10000)), ("Val", (30001, 31000))):
        same_all, same_some, n_ids = 0, 0, 0
        low_same = 0
        for i in range(lo, hi + 1):
            hs = [rec.get(f"{sp}/high/{L}_{i}.jpg", {}).get("pmd5") for L in ("dense", "moderate", "light")]
            ls = [rec.get(f"{sp}/low/{L}_{i}.jpg", {}).get("pmd5") for L in ("dense", "moderate", "light")]
            if None in hs:
                continue
            n_ids += 1
            same_all += len(set(hs)) == 1
            same_some += len(set(hs)) < 3
            low_same += len(set(ls)) < 3
        lv[sp] = {"ids": n_ids, "all_three_high_identical": same_all, "at_least_two_high_identical": same_some,
                  "ids_with_identical_low_across_levels": low_same}
    res["levels"] = lv

    # ------------------------------------------------------------ exact duplicates
    def dup_groups(key, rs):
        g = defaultdict(list)
        for r in rs:
            g[rec[r][key]].append(r)
        return [v for v in g.values() if len(v) > 1]

    # unique references (collapse identical highs), plus lows and test
    uniq_high = {}
    for r in img_rel:
        if "/high/" in r:
            uniq_high.setdefault(rec[r]["pmd5"], r)
    highs_u = sorted(uniq_high.values())
    tests = sorted(r for r in img_rel if r.startswith("Test/"))
    lows = sorted(r for r in img_rel if "/low/" in r)
    pm_split = defaultdict(set)
    for r in img_rel:
        pm_split[rec[r]["pmd5"]].add(split_of(r) + ("/" + r.split("/")[1] if "/" in r[5:] and not r.startswith("Test") else ""))
    cross_exact = Counter(" & ".join(sorted(v)) for v in pm_split.values() if len({x.split("/")[0] for x in v}) > 1)
    res["exact_duplicates"] = {
        "unique_high_images": len(highs_u),
        "high_duplicate_groups_beyond_levels": sum(
            1 for g in dup_groups("pmd5", [r for r in img_rel if "/high/" in r])
            if len({re.search(r"_(\d+)", os.path.basename(x)).group(1) for x in g}) > 1),
        "low_duplicate_groups": len(dup_groups("pmd5", lows)),
        "test_duplicate_groups": dup_groups("pmd5", tests)[:20],
        "n_test_duplicate_groups": len(dup_groups("pmd5", tests)),
        "cross_split_exact_pixel_groups": dict(cross_exact),
    }

    # ------------------------------------------------------------ scene near-duplicates
    thr = max(0.95, pairing["calibration"]["random_nonpair_corr_p999"] + 0.02)
    Hu = np.stack([rec[r]["th"] for r in highs_u])
    bj, bc, hits = topk_matches(Hu, Hu, thr, same=True)
    sp_h = [split_of(r) for r in highs_u]
    ids_h = [re.search(r"_(\d+)", os.path.basename(r)).group(1) for r in highs_u]
    scene_pairs = Counter(" <-> ".join(sorted((sp_h[i], sp_h[j]))) for i, j, _ in hits)
    tv = [(round(c, 3), highs_u[i], highs_u[j]) for i, j, c in hits if sp_h[i] != sp_h[j]]
    tv.sort(reverse=True)
    idgap = Counter(min(abs(int(ids_h[i]) - int(ids_h[j])), 999) // 1 if sp_h[i] == sp_h[j] else -1 for i, j, _ in hits)
    # test (low-light) vs all references and lows: use low->high calibration (true pair p1)
    Tt = np.stack([rec[r]["th"] for r in tests])
    tj, tc, thits = topk_matches(Tt, Hu, thr)
    Lall = np.stack([rec[r]["th"] for r in lows])
    tlj, tlc, _ = topk_matches(Tt, Lall, 2.0)
    res["scene_near_duplicates"] = {
        "threshold": thr,
        "high_high_pairs_ge_thr_by_split": dict(scene_pairs),
        "train_val_examples": tv[:30],
        "n_train_val_pairs": len(tv),
        "within_split_id_gap_hist_top": dict(Counter({k: v for k, v in idgap.items() if k >= 0}).most_common(10)),
        "test_best_corr_vs_references_percentiles_50_90_99_max": [round(float(np.percentile(tc, p)), 3) for p in (50, 90, 99)] + [round(float(tc.max()), 3)],
        "test_images_with_reference_match_ge_thr": int((tc >= thr).sum()),
        "test_top_matches_vs_references": sorted(((round(float(tc[i]), 3), tests[i], highs_u[tj[i]]) for i in range(len(tests))), reverse=True)[:20],
        "test_top_matches_vs_train_val_lows": sorted(((round(float(tlc[i]), 3), tests[i], lows[tlj[i]]) for i in range(len(tests))), reverse=True)[:20],
    }

    # ------------------------------------------------------------ YOLO annotations
    ann = [r for r in rel if r.startswith("YOLO Annotations/")]
    cls_files = [r for r in ann if r.endswith("class_names.txt")]
    cls_text = {r: z.read(f"{root}/{r}").decode(errors="replace") for r in cls_files}
    cls_names = [ln.split(":", 1)[1].strip() for ln in list(cls_text.values())[0].splitlines() if ":" in ln]
    lab = [r for r in ann if "/Labels/" in r]
    lab_map = {}
    for r in lab:
        sp = r.split("/")[1]
        side = "high" if "high" in r.split("/")[2] else "low"
        lab_map[(sp, side, os.path.basename(r)[:-4])] = r
    bad_lines, n_boxes, empty, cls_hist, same_lr, n_cmp = 0, 0, 0, Counter(), 0, 0
    content = {}
    for key, r in lab_map.items():
        txt = z.read(f"{root}/{r}").decode(errors="replace")
        content[key] = txt
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        empty += not lines
        for ln in lines:
            p = ln.split()
            ok = len(p) == 5 and p[0].isdigit() and all(0 <= float(x) <= 1 for x in p[1:])
            bad_lines += not ok
            if ok:
                n_boxes += 1
                cls_hist[int(p[0])] += 1
    for (sp, side, stem) in lab_map:
        if side == "high" and (sp, "low", stem) in lab_map:
            n_cmp += 1
            same_lr += content[(sp, "high", stem)].strip() == content[(sp, "low", stem)].strip()
    img_stems = {(r.split("/")[0], r.split("/")[1], os.path.basename(r)[:-4]) for r in img_rel if "/" in r[5:]}
    lab_stems = set(lab_map)
    res["yolo"] = {
        "class_files": {k: len(v.splitlines()) for k, v in cls_text.items()},
        "class_files_identical": len(set(cls_text.values())) == 1,
        "n_classes": len(cls_names),
        "class_names_first10": cls_names[:10],
        "label_files": len(lab_map),
        "empty_label_files": empty,
        "malformed_lines": bad_lines,
        "boxes": n_boxes,
        "class_histogram_top15": [(cls_names[c] if c < len(cls_names) else c, n) for c, n in cls_hist.most_common(15)],
        "low_high_label_files_identical": f"{same_lr}/{n_cmp}",
        "images_without_label": len(img_stems - lab_stems),
        "labels_without_image": len(lab_stems - img_stems),
        "test_labels": sum(1 for r in ann if "/Test" in r),
    }

    # ------------------------------------------------------------ cross-dataset
    print("cross-dataset thumbnails ...", flush=True)
    ext = []  # (dataset, name, bmd5, pmd5, thumb)
    for sp in ("our485", "eval15"):
        for q in ("low", "high"):
            for f in glob.glob(os.path.join(V1_ROOT, sp, q, "*.png")):
                data = open(f, "rb").read()
                im = Image.open(io.BytesIO(data)).convert("RGB")
                ext.append(("LOL-v1", f"{sp}/{q}/{os.path.basename(f)}", hashlib.md5(data).hexdigest(),
                            hashlib.md5(np.asarray(im).tobytes()).hexdigest(), sq_thumb(im)))
    z2 = zipfile.ZipFile(V2_ZIP)
    for n in z2.namelist():
        if n.endswith(".png"):
            data = z2.read(n)
            im = Image.open(io.BytesIO(data)).convert("RGB")
            ext.append(("LOL-v2-real" if "Real_captured" in n else "LOL-v2-synthetic", n, hashlib.md5(data).hexdigest(),
                        hashlib.md5(np.asarray(im).tobytes()).hexdigest(), sq_thumb(im)))
    for arc in sorted(glob.glob(os.path.join(SICE_DIR, "*.rar"))):
        proc = subprocess.Popen(["bsdtar", "-cf", "-", "--format=pax", "@" + arc], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            for m in tf:
                if m.isfile() and m.name.lower().endswith((".jpg", ".png")):
                    data = tf.extractfile(m).read()
                    im = Image.open(io.BytesIO(data))
                    if im.format == "JPEG":
                        im.draft("RGB", (im.width // 8, im.height // 8))
                    ext.append(("SICE", f"{os.path.basename(arc)}:{m.name}", hashlib.md5(data).hexdigest(), None,
                                sq_thumb(im.convert("RGB"))))
        proc.wait()
    all_loli = sorted(img_rel)
    loli_b = {rec[r]["bmd5"] for r in all_loli}
    loli_p = {rec[r]["pmd5"] for r in all_loli}
    L2 = np.stack([rec[r]["th"] for r in all_loli])
    cross = {}
    for ds in ("LOL-v1", "LOL-v2-real", "LOL-v2-synthetic", "SICE"):
        items = [e for e in ext if e[0] == ds]
        E = np.stack([e[4] for e in items])
        j, c, h = topk_matches(E, L2, thr)
        cross[ds] = {
            "images_compared": len(items),
            "byte_identical": sum(e[2] in loli_b for e in items),
            "pixel_identical": sum(e[3] in loli_p for e in items if e[3]),
            "near_dup_pairs_ge_thr": len(h),
            "images_with_near_dup": int((c >= thr).sum()),
            "max_corr": round(float(c.max()), 3),
            "top5": sorted(((round(float(c[i]), 3), items[i][1], all_loli[j[i]]) for i in range(len(items))), reverse=True)[:5],
        }
    res["cross_dataset"] = cross
    res["runtime_s"] = round(time.time() - t0, 1)

    with open(os.path.join(OUT_DIR, "loli_street_structure_audit.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)

    T = ["LoLI-Street structure + overlap audit (READ-ONLY; read from ZIP in memory)", "=" * 72, "",
         f"File: {zpath} ({res['file_size'] / 1e9:.2f} GB); other files in dir: {res['other_files_in_dir']}",
         f"Integrity: {res['integrity']}", f"Root folder: {root}", "", "Folders (file counts):"]
    T += [f"  {k}: {v}" for k, v in res["folders"].items()]
    T += ["", "Image folders:"] + [f"  {k}: {v}" for k, v in res["image_folders"].items()]
    T += ["", "Mean luminance by folder/level:"] + [f"  {k}: {v}" for k, v in res["mean_luminance_by_folder_and_level"].items()]
    T += ["", "Pairing:"] + [f"  {k}: {v}" for k, v in pairing.items()]
    T += ["", f"Darkness levels share reference: {lv}", "", f"Exact duplicates: {res['exact_duplicates']}", "",
          "Scene near-duplicates:"] + [f"  {k}: {v}" for k, v in res["scene_near_duplicates"].items()]
    T += ["", "YOLO annotations:"] + [f"  {k}: {v}" for k, v in res["yolo"].items()]
    T += ["", "Cross-dataset:"] + [f"  {k}: {v}" for k, v in cross.items()] + ["", f"runtime {res['runtime_s']} s"]
    with open(os.path.join(OUT_DIR, "loli_street_structure_audit.txt"), "w") as f:
        f.write("\n".join(T) + "\n")
    print("\n".join(T))


if __name__ == "__main__":
    main()
