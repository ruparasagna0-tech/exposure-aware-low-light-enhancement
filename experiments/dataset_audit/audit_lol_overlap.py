"""READ-ONLY overlap audit: LOL-v1 vs LOL-v2 (real + synthetic).

- LOL-v2 is read directly from the ZIP in memory (nothing extracted).
- LOL-v1 is read from data/LOL/LOLdataset (nothing modified).
- Only writes overlap_audit.json / overlap_audit.txt next to this script.

Evidence levels:
  1. EXACT  : MD5 of decoded RGB pixels (independent of filename / PNG encoding).
              Counted as UNIQUE images (LOL re-uses one reference image for several
              low-light exposures of the same scene, so raw match events over-count).
  2. SCENE  : pairs are linked if they share an exact low or reference image, or if
              their brightness-normalised grayscale thumbnails correlate >= SCENE_THR
              (reference-vs-reference or low-vs-low). Connected components = scenes.
              Leakage is judged at scene level, not only file level.
  3. ALIGN  : for every LOL-v1 pair whose low image exists in LOL-v2-real, the LOL-v1
              high is compared with the paired LOL-v2 normal (exact? PSNR, best
              integer shift within +-4 px).

Usage (from project root):
    .venv/bin/python experiments/dataset_audit/audit_lol_overlap.py
"""

import hashlib
import io
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from src.datasets.lol_v1 import find_pairs, train_val_indices  # noqa: E402  (read-only use)

V1_ROOT = os.path.join(PROJECT_ROOT, "data", "LOL", "LOLdataset")
V2_ZIP = os.path.join(PROJECT_ROOT, "data", "LOL-v2", "LOL-v2.zip")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

THUMB_RECT = (96, 64)   # (W, H) thumbnails for 600x400 real images
THUMB_SQ = (64, 64)     # thumbnails for comparison against 384x384 synthetic
SCENE_THR = 0.95
BORDER_LO = 0.80        # report cross-scene links in [BORDER_LO, SCENE_THR) for manual review
SHIFT = 4


def pixel_md5(img):
    return hashlib.md5(img.convert("RGB").tobytes()).hexdigest()


def thumb(img, size):
    """Zero-mean / unit-norm grayscale thumbnail; dot product = Pearson correlation."""
    t = np.asarray(img.convert("L").resize(size, Image.Resampling.BOX), dtype=np.float64).ravel()
    t -= t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


def center_square(img):
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def best_shift(a, b, r=SHIFT):
    best, best_m = None, np.inf
    h, w = a.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            aa = a[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)]
            bb = b[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)]
            m = ((aa - bb) ** 2).mean()
            if m < best_m:
                best, best_m = (dy, dx), m
    return best


def psnr_u8(a, b):
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def natural(s):
    return (len(s), s)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main():
    z = zipfile.ZipFile(V2_ZIP)
    entries = z.namelist()
    files = [n for n in entries if not n.endswith("/")]
    structure = dict(sorted(Counter(os.path.dirname(n) for n in files).items()))
    extra = [n for n in entries if "__MACOSX" in n or os.path.basename(n).startswith(".")]

    # ------------------------------------------------------------ pairs
    # Each pair: dataset, subset, id, low/high image records.
    pairs = []

    def rec(img, need_sq=False):
        r = {"md5": pixel_md5(img), "size": img.size}
        if img.size == (600, 400):
            r["rect"] = thumb(img, THUMB_RECT)
        r["sq_full"] = thumb(img, THUMB_SQ)
        if need_sq:
            r["sq_crop"] = thumb(center_square(img), THUMB_SQ)
        return r

    # LOL-v1 with our logical train/val split (read-only use of the loader's split function)
    our = find_pairs(os.path.join(V1_ROOT, "our485"))
    tr_idx, va_idx = train_val_indices(len(our), 35, 42)
    v1_subset = {our[i]: "train450" for i in tr_idx} | {our[i]: "val35" for i in va_idx}
    for sp in ("our485", "eval15"):
        for f in sorted(find_pairs(os.path.join(V1_ROOT, sp)), key=natural):
            p = {"ds": "LOL-v1", "folder": sp, "subset": v1_subset.get(f, "eval15") if sp == "our485" else "eval15",
                 "id": f, "low_path": f"{sp}/low/{f}", "high_path": f"{sp}/high/{f}"}
            with Image.open(os.path.join(V1_ROOT, sp, "low", f)) as im:
                p["low"] = rec(im, need_sq=True)
            with Image.open(os.path.join(V1_ROOT, sp, "high", f)) as im:
                p["high"] = rec(im, need_sq=True)
            pairs.append(p)
    n_v1 = len(pairs)

    for sub in ("Real_captured", "Synthetic"):
        for split in ("Train", "Test"):
            lows = sorted(os.path.basename(n) for n in files if n.startswith(f"LOL-v2/{sub}/{split}/Low/"))
            for f in lows:
                if sub == "Real_captured":
                    k = int(re.search(r"(\d+)", f).group(1))
                    nf = f"normal{k:05d}.png"
                    pid = k
                else:
                    nf, pid = f, f
                lp, hp = f"LOL-v2/{sub}/{split}/Low/{f}", f"LOL-v2/{sub}/{split}/Normal/{nf}"
                p = {"ds": "LOL-v2-real" if sub == "Real_captured" else "LOL-v2-synthetic",
                     "folder": split, "subset": split, "id": pid, "low_path": lp, "high_path": hp}
                p["low"] = rec(Image.open(io.BytesIO(z.read(lp))))
                p["high"] = rec(Image.open(io.BytesIO(z.read(hp))))
                pairs.append(p)
    print(f"Decoded {len(pairs)} pairs ({2 * len(pairs)} images)")

    v1 = [p for p in pairs if p["ds"] == "LOL-v1"]
    real = [p for p in pairs if p["ds"] == "LOL-v2-real"]
    syn = [p for p in pairs if p["ds"] == "LOL-v2-synthetic"]

    # ------------------------------------------------------------ 1. exact (unique images)
    v2_md5 = defaultdict(set)  # md5 -> set of v2 image paths
    for p in real + syn:
        v2_md5[p["low"]["md5"]].add(p["low_path"])
        v2_md5[p["high"]["md5"]].add(p["high_path"])

    def v2_sub(path):  # "Real_captured/Train/Low"
        return "/".join(path.split("/")[1:4])

    rows = []
    mappings = {}
    for folder in ("our485", "eval15"):
        for kind, tkind in (("low", "Low"), ("high", "Normal")):
            src = [p for p in v1 if p["folder"] == folder]
            for tsplit in ("Train", "Test"):
                tgt = f"Real_captured/{tsplit}/{tkind}"
                src_hit, tgt_hit, mp = set(), set(), []
                for p in src:
                    for h in sorted(v2_md5.get(p[kind]["md5"], ())):
                        if v2_sub(h) == tgt:
                            src_hit.add(p["id"])
                            tgt_hit.add(h)
                            mp.append([p[f"{kind}_path"], h])
                tgt_total = sum(1 for n in files if n.startswith(f"LOL-v2/{tgt}/"))
                rows.append({"source": "LOL-v1", "target": "LOL-v2-real", "source_subset": f"{folder}/{kind}",
                             "target_subset": f"{tsplit}/{tkind}", "exact_source_images": len(src_hit),
                             "exact_target_images": len(tgt_hit), "source_total": len(src), "target_total": tgt_total})
                mappings[f"{folder}/{kind} -> {tgt}"] = mp
            s_hit = sum(1 for p in src if any(v2_sub(h).startswith("Synthetic") for h in v2_md5.get(p[kind]["md5"], ())))
            rows.append({"source": "LOL-v1", "target": "LOL-v2-synthetic", "source_subset": f"{folder}/{kind}",
                         "target_subset": "Train+Test/Low+Normal", "exact_source_images": s_hit,
                         "exact_target_images": s_hit, "source_total": len(src), "target_total": 2 * len(syn)})

    # ------------------------------------------------------------ 2. scene grouping (real 600x400 pairs)
    rp = v1 + real
    uf = UnionFind(len(rp))
    by_md5 = defaultdict(list)
    for i, p in enumerate(rp):
        by_md5[("L", p["low"]["md5"])].append(i)
        by_md5[("H", p["high"]["md5"])].append(i)
    for idx in by_md5.values():
        for j in idx[1:]:
            uf.union(idx[0], j)
    H = np.stack([p["high"]["rect"] for p in rp])
    L = np.stack([p["low"]["rect"] for p in rp])
    CH, CL = H @ H.T, L @ L.T
    C = np.maximum(CH, CL)
    np.fill_diagonal(C, -1)
    ii, jj = np.where(np.triu(C >= SCENE_THR, 1))
    for a, b in zip(ii, jj):
        uf.union(int(a), int(b))
    scene_of = [uf.find(i) for i in range(len(rp))]
    # renumber scenes
    sid = {s: k for k, s in enumerate(sorted(set(scene_of)))}
    for i, p in enumerate(rp):
        p["scene"] = sid[scene_of[i]]

    def subset_key(p):
        return f"{p['ds']}:{p['subset']}"

    keys = ["LOL-v1:train450", "LOL-v1:val35", "LOL-v1:eval15", "LOL-v2-real:Train", "LOL-v2-real:Test"]
    scenes_in = {k: {p["scene"] for p in rp if subset_key(p) == k} for k in keys}
    pairs_in = {k: [p for p in rp if subset_key(p) == k] for k in keys}
    scene_matrix = {a: {b: len(scenes_in[a] & scenes_in[b]) for b in keys} for a in keys}
    # pairs of subset A whose scene also occurs in subset B
    pair_leak = {a: {b: sum(1 for p in pairs_in[a] if p["scene"] in scenes_in[b]) for b in keys if b != a}
                 for a in keys}

    v1_scenes = scenes_in["LOL-v1:train450"] | scenes_in["LOL-v1:val35"] | scenes_in["LOL-v1:eval15"]
    real_clean = [p for p in real if p["scene"] not in v1_scenes]
    real_clean_by_split = Counter(p["subset"] for p in real_clean)
    real_clean_train_scenes = {p["scene"] for p in real_clean if p["subset"] == "Train"}
    real_clean_test = [p for p in real_clean if p["subset"] == "Test"]
    real_clean_test_indep = [p for p in real_clean_test if p["scene"] not in real_clean_train_scenes]

    # borderline links (not merged) for manual review
    border = []
    for a, b in zip(*np.where(np.triu((C >= BORDER_LO) & (C < SCENE_THR), 1))):
        if rp[a]["scene"] != rp[b]["scene"]:
            border.append((float(C[a, b]), f"{rp[a]['ds']}:{rp[a]['subset']}:{rp[a]['id']}",
                           f"{rp[b]['ds']}:{rp[b]['subset']}:{rp[b]['id']}"))
    border.sort(reverse=True)
    different_scene_p99 = float(np.percentile(C[np.triu_indices(len(rp), 1)][
        np.array([rp[a]["scene"] != rp[b]["scene"] for a, b in zip(*np.triu_indices(len(rp), 1))])], 99))

    # ------------------------------------------------------------ 3. alignment of overlapping pairs
    real_by_id = {p["id"]: p for p in real}
    align = []
    for p in v1:
        hits = [h for h in v2_md5.get(p["low"]["md5"], ()) if h.startswith("LOL-v2/Real_captured/")]
        if not hits:
            align.append({"v1": p["low_path"], "v2_low": None})
            continue
        h = sorted(hits)[0]
        k = int(re.search(r"low(\d+)", h).group(1))
        q = real_by_id[k]
        a = np.asarray(Image.open(os.path.join(V1_ROOT, p["high_path"])).convert("RGB"))
        b = np.asarray(Image.open(io.BytesIO(z.read(q["high_path"]))).convert("RGB"))
        align.append({
            "v1_pair": f"{p['folder']}/{p['id']}", "v1_subset": p["subset"],
            "v2_low": h, "v2_normal": q["high_path"], "v2_split": q["subset"],
            "v2_id_equals_v1_number": k == int(p["id"][:-4]),
            "low_matches": len(hits),
            "high_exact": p["high"]["md5"] == q["high"]["md5"],
            "high_psnr_db": round(psnr_u8(a, b), 2),
            "high_best_shift": best_shift(a.mean(axis=2), b.mean(axis=2)),
            "same_scene": p["scene"] == q["scene"],
        })
    al = [x for x in align if x["v2_low"]]

    # ------------------------------------------------------------ 4. synthetic
    S = np.stack([p[k]["sq_full"] for p in syn for k in ("low", "high")])
    syn_names = [p[f"{k}_path"] for p in syn for k in ("low", "high")]
    V = [(p[f"{k}_path"], p[k]) for p in v1 for k in ("low", "high")]
    Vf = np.stack([r["sq_full"] for _, r in V])
    Vc = np.stack([r["sq_crop"] for _, r in V])
    CS = np.maximum(Vf @ S.T, Vc @ S.T)
    best_j = CS.argmax(axis=1)
    syn_top = sorted(((float(CS[i, best_j[i]]), V[i][0], syn_names[best_j[i]]) for i in range(len(V))), reverse=True)[:10]
    syn_exact = sum(1 for p in v1 for k in ("low", "high")
                    if any(h.startswith("LOL-v2/Synthetic/") for h in v2_md5.get(p[k]["md5"], ())))
    # synthetic internal: train/test scene overlap (exact + corr on references)
    SH = np.stack([p["high"]["sq_full"] for p in syn])
    CSS = SH @ SH.T
    np.fill_diagonal(CSS, -1)
    syn_tr = [i for i, p in enumerate(syn) if p["subset"] == "Train"]
    syn_te = [i for i, p in enumerate(syn) if p["subset"] == "Test"]
    syn_tt_exact = len({p["high"]["md5"] for p in syn if p["subset"] == "Train"}
                       & {p["high"]["md5"] for p in syn if p["subset"] == "Test"})
    syn_tt_corr = int((CSS[np.ix_(syn_te, syn_tr)].max(axis=1) >= SCENE_THR).sum())
    syn_internal_dup = sum(c - 1 for c in Counter(p["high"]["md5"] for p in syn).values() if c > 1)

    # ------------------------------------------------------------ output
    totals = {
        "v1_images_total": 2 * len(v1),
        "v1_images_with_exact_copy_in_v2_real": sum(
            1 for p in v1 for k in ("low", "high")
            if any(h.startswith("LOL-v2/Real_captured/") for h in v2_md5.get(p[k]["md5"], ()))),
        "v1_low_exact": sum(1 for x in al),
        "v1_high_exact": sum(1 for x in al if x["high_exact"]),
        "v1_pairs_total": len(v1),
        "v1_pairs_whose_scene_is_in_v2_real": sum(1 for p in v1 if p["scene"] in {q["scene"] for q in real}),
        "v2_real_images_total": 2 * len(real),
        "v2_real_images_exactly_equal_to_some_v1_image": len(
            {h for p in v1 for k in ("low", "high") for h in v2_md5.get(p[k]["md5"], ())
             if h.startswith("LOL-v2/Real_captured/")}),
        "v2_real_pairs_total": len(real),
        "v2_real_pairs_sharing_scene_with_v1": len(real) - len(real_clean),
        "v2_real_pairs_clean": dict(real_clean_by_split),
        "v2_real_clean_test_pairs_scene_independent_of_clean_train": len(real_clean_test_indep),
        "scenes_total_real": len(sid),
    }
    result = {
        "zip": V2_ZIP, "zip_structure": structure, "zip_extra_entries": extra,
        "method": {"exact": "MD5 of decoded RGB pixels; counts are unique images",
                   "scene": f"union of exact-shared low/reference + thumbnail corr >= {SCENE_THR} {THUMB_RECT}",
                   "alignment": f"v1 high vs paired v2 normal: exact / PSNR / best shift within +-{SHIFT}px",
                   "synthetic": f"{THUMB_SQ} thumbnails, v1 full-frame and centre-square, max corr"},
        "table": rows, "totals": totals,
        "scene_subsets": {k: {"pairs": len(pairs_in[k]), "scenes": len(scenes_in[k])} for k in keys},
        "scene_overlap_matrix_shared_scenes": scene_matrix,
        "pairs_of_A_whose_scene_occurs_in_B": pair_leak,
        "different_scene_corr_p99": different_scene_p99,
        "borderline_links_not_merged": border,
        "borderline_links_by_subset_pair": dict(Counter(
            " <-> ".join(sorted((x.rsplit(":", 1)[0], y.rsplit(":", 1)[0]))) for _, x, y in border)),
        "n_borderline_links": len(border),
        "v2_real_clean_ids": {s: sorted(p["id"] for p in real_clean if p["subset"] == s) for s in ("Train", "Test")},
        "v2_real_clean_test_ids_independent": sorted(p["id"] for p in real_clean_test_indep),
        "alignment_summary": {
            "overlapping_pairs": len(al),
            "v2_id_equals_v1_number": sum(x["v2_id_equals_v1_number"] for x in al),
            "high_exact": sum(x["high_exact"] for x in al),
            "best_shift_zero": sum(tuple(x["high_best_shift"]) == (0, 0) for x in al),
            "same_scene": sum(x["same_scene"] for x in al),
            "non_exact_high": [(x["v1_pair"], x["v2_normal"].split("/")[-1], x["high_psnr_db"]) for x in al
                               if not x["high_exact"]],
        },
        "alignment": align,
        "v1_to_v2_real_mapping": [(x["v1_pair"], x["v1_subset"], x["v2_split"], x["v2_low"].split("/")[-1],
                                   x["v2_normal"].split("/")[-1], "high_exact" if x["high_exact"] else
                                   f"high_differs_{x['high_psnr_db']}dB") for x in al],
        "exact_mappings": mappings,
        "synthetic": {"sizes": sorted({tuple(p["low"]["size"]) for p in syn} | {tuple(p["high"]["size"]) for p in syn}),
                      "exact_matches_with_v1_images": syn_exact,
                      "top10_corr_v1_vs_synthetic": syn_top,
                      "train_test_shared_reference_exact": syn_tt_exact,
                      "test_pairs_with_train_scene_corr_ge_thr": syn_tt_corr,
                      "internal_duplicate_references": syn_internal_dup},
    }
    with open(os.path.join(OUT_DIR, "overlap_audit.json"), "w") as f:
        json.dump(result, f, indent=1, default=lambda o: list(o) if isinstance(o, (tuple, set)) else str(o))

    L_ = ["LOL-v1 vs LOL-v2 overlap audit (READ-ONLY)", "=" * 70, "", "ZIP structure:"]
    L_ += [f"  {d}: {c}" for d, c in structure.items()] + [f"  junk entries: {len(extra)}", ""]
    L_ += ["Exact overlap (unique images, decoded-pixel MD5):",
           f"  {'source':14s} {'target':34s} {'src hit':>7s} {'src tot':>7s} {'%src':>6s} {'tgt hit':>7s} {'tgt tot':>7s} {'%tgt':>6s}"]
    for r in rows:
        L_.append(f"  {r['source_subset']:14s} {r['target'] + ' ' + r['target_subset']:34s} "
                  f"{r['exact_source_images']:7d} {r['source_total']:7d} {100 * r['exact_source_images'] / r['source_total']:5.1f}% "
                  f"{r['exact_target_images']:7d} {r['target_total']:7d} {100 * r['exact_target_images'] / r['target_total']:5.1f}%")
    L_ += ["", "Totals:"] + [f"  {k}: {v}" for k, v in totals.items()]
    L_ += ["", f"Scene-level (real 600x400; threshold {SCENE_THR}; different-scene corr p99 = {different_scene_p99:.3f}):"]
    for k in keys:
        L_.append(f"  {k:20s} pairs {len(pairs_in[k]):4d}  scenes {len(scenes_in[k]):4d}")
    L_ += ["  pairs of A whose scene also occurs in B:"]
    L_.append("  " + " " * 20 + "".join(f"{b.split(':')[1]:>10s}" for b in keys))
    for a in keys:
        L_.append(f"  {a:20s}" + "".join(f"{pair_leak[a].get(b, len(pairs_in[a])) if b != a else '-':>10}" for b in keys))
    L_ += [f"  borderline cross-scene links in [{BORDER_LO}, {SCENE_THR}): {len(border)}  top: {border[:5]}"]
    L_ += ["", "Alignment of overlapping pairs:"] + [f"  {k}: {v}" for k, v in result["alignment_summary"].items()]
    L_ += ["", "Synthetic:"] + [f"  {k}: {v}" for k, v in result["synthetic"].items()]
    with open(os.path.join(OUT_DIR, "overlap_audit.txt"), "w") as f:
        f.write("\n".join(L_) + "\n")
    print("\n".join(L_))


if __name__ == "__main__":
    main()
