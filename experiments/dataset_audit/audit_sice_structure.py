"""READ-ONLY structure audit of the SICE archives (Dataset_Part1.rar, Dataset_Part2.rar).

Nothing is extracted to disk: each RAR is converted on the fly by macOS bsdtar
(libarchive) into a tar stream (`bsdtar -cf - @archive.rar`) and read member by
member in memory. Decompressing every member also verifies archive integrity.

Per image: byte MD5, format, mode, resolution, EXIF exposure fields, mean
luminance, and small normalised grayscale thumbnails (for duplicate-scene and
LOL comparison).

Writes (next to this script):
    sice_structure_audit.json   full per-sequence / per-image details
    sice_structure_audit.txt    concise report
Optional: --thumbs PATH saves small RGB previews (npz) for manual inspection.

Usage (from project root):
    .venv/bin/python experiments/dataset_audit/audit_sice_structure.py [--thumbs /tmp/x.npz]
"""

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import time
import zipfile
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SICE_DIR = os.path.join(PROJECT_ROOT, "data", "SICE")
V1_ROOT = os.path.join(PROJECT_ROOT, "data", "LOL", "LOLdataset")
V2_ZIP = os.path.join(PROJECT_ROOT, "data", "LOL-v2", "LOL-v2.zip")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

EXIF_IFD = 0x8769
TAGS = {0x829A: "exposure_time", 0x829D: "f_number", 0x8827: "iso", 0x9204: "exposure_bias",
        0x9003: "datetime_original"}
BASE_TAGS = {0x0110: "camera_model", 0x0112: "orientation"}
SCENE_THR = 0.95


def norm_thumb(gray: Image.Image, size) -> np.ndarray:
    t = np.asarray(gray.resize(size, Image.Resampling.BOX), dtype=np.float32).ravel()
    t -= t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


def center_square(img):
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def fnum(v):
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return str(v)


def inspect_image(data: bytes):
    im = Image.open(io.BytesIO(data))
    info = {"format": im.format, "mode": im.mode, "size": list(im.size)}
    try:
        ex = im.getexif()
        for k, n in BASE_TAGS.items():
            if k in ex:
                info[n] = fnum(ex[k]) if n == "orientation" else str(ex[k]).strip("\x00 ")
        sub = ex.get_ifd(EXIF_IFD)
        for k, n in TAGS.items():
            if k in sub:
                info[n] = str(sub[k]) if n == "datetime_original" else fnum(sub[k])
    except Exception:  # noqa: BLE001
        pass
    if im.format == "JPEG":
        im.draft("RGB", (max(1, im.width // 8), max(1, im.height // 8)))  # fast reduced decode
    rgb = im.convert("RGB")
    gray = rgb.convert("L")
    info["mean_lum"] = float(np.asarray(gray, dtype=np.float32).mean())
    thumbs = {
        "rect": norm_thumb(gray, (96, 64)),
        "sq": norm_thumb(center_square(gray), (64, 64)),
        "preview": np.asarray(rgb.resize((144, 96), Image.Resampling.BOX), dtype=np.uint8),
    }
    return info, thumbs


def stream_rar(path):
    """Yield (name, size, bytes) for every file in a RAR without writing to disk."""
    proc = subprocess.Popen(["bsdtar", "-cf", "-", "--format=pax", "@" + path],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
        for m in tf:
            if m.isfile():
                yield m.name, m.size, tf.extractfile(m).read()
    proc.stdout.close()
    err = proc.stderr.read().decode(errors="replace")
    rc = proc.wait()
    yield None, rc, err


def listing(path):
    out = subprocess.run(["bsdtar", "-tvf", path], capture_output=True, text=True, check=True).stdout
    files = {}
    for line in out.splitlines():
        if not line.startswith("d"):
            parts = line.split(None, 8)
            files[parts[8]] = int(parts[4])
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thumbs", help="optional .npz path for RGB previews (manual inspection)")
    args = ap.parse_args()
    t0 = time.time()

    archives = sorted(f for f in os.listdir(SICE_DIR) if not f.startswith("."))
    report = {"sice_dir": SICE_DIR, "files_present": {f: os.path.getsize(os.path.join(SICE_DIR, f)) for f in archives},
              "archives": {}}
    images = []          # per-image records
    thumbs = {}          # key -> thumbs
    text_files = {}

    for arc in archives:
        path = os.path.join(SICE_DIR, arc)
        listed = listing(path)
        seen, sizes_ok = 0, 0
        integrity = {}
        for name, size, data in stream_rar(path):
            if name is None:
                integrity = {"bsdtar_exit_code": size, "stderr": data.strip()}
                break
            seen += 1
            sizes_ok += listed.get(name) == len(data)
            if name.lower().endswith(".txt"):
                text_files[f"{arc}:{name}"] = data.decode("utf-8", errors="replace")
                continue
            parts = name.split("/")
            part = parts[0]
            if len(parts) == 3 and parts[1].isdigit():
                seq, idx, kind = int(parts[1]), int(parts[2].split(".")[0]), "exposure"
            elif len(parts) == 3 and parts[1] == "Label":
                seq, idx, kind = int(re.sub(r"\D", "", parts[2].split(".")[0])), None, "label"
            else:
                seq, idx, kind = None, None, "other"
            rec = {"archive": arc, "path": name, "part": part, "seq": seq, "idx": idx, "kind": kind,
                   "bytes": len(data), "md5": hashlib.md5(data).hexdigest()}
            try:
                info, th = inspect_image(data)
                rec.update(info)
                thumbs[(part, seq, idx if kind == "exposure" else "L")] = th
            except Exception as e:  # noqa: BLE001
                rec["decode_error"] = repr(e)
            images.append(rec)
            if seen % 500 == 0:
                print(f"  {arc}: {seen}/{len(listed)} ({time.time() - t0:.0f}s)", flush=True)
        report["archives"][arc] = {
            "listed_files": len(listed), "streamed_files": seen, "sizes_match_listing": sizes_ok,
            "uncompressed_bytes": sum(listed.values()), **integrity,
        }
        print(f"{arc}: streamed {seen}/{len(listed)} files, exit {integrity.get('bsdtar_exit_code')}", flush=True)

    # ------------------------------------------------------------ per-sequence structure
    seqs = defaultdict(lambda: {"exposures": [], "label": None})
    for r in images:
        if r["kind"] == "exposure":
            seqs[(r["part"], r["seq"])]["exposures"].append(r)
        elif r["kind"] == "label":
            seqs[(r["part"], r["seq"])]["label"] = r
    seq_summary = {}
    ordering = Counter()
    bias_patterns = Counter()
    label_vs_seq_size = Counter()
    label_lum_rank = Counter()
    exif_presence = Counter()
    for key in sorted(seqs):
        s = seqs[key]
        ex = sorted(s["exposures"], key=lambda r: r["idx"])
        lum = [round(r.get("mean_lum", float("nan")), 1) for r in ex]
        et = [r.get("exposure_time") for r in ex]
        bias = [r.get("exposure_bias") for r in ex]
        sizes = sorted({tuple(r.get("size", [])) for r in ex})
        mono = all(a <= b for a, b in zip(lum, lum[1:])) and len(lum) > 1
        ordering["lum_increasing_with_index" if mono else "lum_not_monotonic"] += 1
        et_num = [x for x in et if isinstance(x, float)]
        if len(et_num) == len(et) and len(et) > 1:
            ordering["exposure_time_increasing" if all(a <= b for a, b in zip(et_num, et_num[1:]))
                     else "exposure_time_not_monotonic"] += 1
        bias_patterns[str(bias)] += 1
        for r in ex:
            exif_presence["exposure_time" if "exposure_time" in r else "no_exposure_time"] += 1
            exif_presence["exposure_bias" if "exposure_bias" in r else "no_exposure_bias"] += 1
        lab = s["label"]
        if lab:
            label_vs_seq_size["label_size_equals_sequence" if [tuple(lab.get("size", []))] == sizes
                              else "label_size_differs"] += 1
            rank = sum(1 for x in lum if x < lab.get("mean_lum", 0))
            label_lum_rank[f"{rank}/{len(lum)}"] += 1
            # which exposure is structurally closest to the label (informational only)
            lt = thumbs.get((key[0], key[1], "L"))
            corr = [float(lt["rect"] @ thumbs[(key[0], key[1], r["idx"])]["rect"]) for r in ex] if lt else []
        else:
            corr = []
        seq_summary[f"{key[0]}/{key[1]}"] = {
            "n_exposures": len(ex), "sizes": sizes, "modes": sorted({r.get("mode") for r in ex}),
            "mean_lum_by_index": lum, "exposure_time_by_index": et, "exposure_bias_by_index": bias,
            "iso_by_index": [r.get("iso") for r in ex], "f_number_by_index": [r.get("f_number") for r in ex],
            "camera": sorted({r.get("camera_model", "?") for r in ex}),
            "label": lab and {k: lab.get(k) for k in ("path", "format", "mode", "size", "mean_lum", "bytes")},
            "label_corr_with_each_exposure": [round(c, 3) for c in corr],
        }

    # ------------------------------------------------------------ duplicates
    md5_groups = defaultdict(list)
    for r in images:
        md5_groups[r["md5"]].append(r["path"] if r["archive"].endswith("1.rar") else "P2:" + r["path"])
    dup_files = [g for g in md5_groups.values() if len(g) > 1]

    keys = sorted(seqs)
    lab_keys = [k for k in keys if (k[0], k[1], "L") in thumbs]
    LT = np.stack([thumbs[(k[0], k[1], "L")]["rect"] for k in lab_keys])
    # also compare the middle exposure of each sequence (structure without label processing)
    mid = {k: sorted(seqs[k]["exposures"], key=lambda r: r["idx"])[len(seqs[k]["exposures"]) // 2]["idx"] for k in keys}
    MT = np.stack([thumbs[(k[0], k[1], mid[k])]["rect"] for k in lab_keys])
    C = np.maximum(LT @ LT.T, MT @ MT.T)
    np.fill_diagonal(C, -1)
    dup_scene = [(float(C[i, j]), f"{lab_keys[i][0]}/{lab_keys[i][1]}", f"{lab_keys[j][0]}/{lab_keys[j][1]}")
                 for i, j in zip(*np.where(np.triu(C >= SCENE_THR, 1)))]
    dup_scene.sort(reverse=True)
    border = [(float(C[i, j]), f"{lab_keys[i][0]}/{lab_keys[i][1]}", f"{lab_keys[j][0]}/{lab_keys[j][1]}")
              for i, j in zip(*np.where(np.triu((C >= 0.85) & (C < SCENE_THR), 1)))]
    border.sort(reverse=True)
    diff_p99 = float(np.percentile(C[np.triu_indices(len(lab_keys), 1)], 99))

    # ------------------------------------------------------------ vs LOL (thumbnail level)
    def lol_thumbs():
        out = []
        for sp in ("our485", "eval15"):
            d = os.path.join(V1_ROOT, sp, "high")
            for f in sorted(os.listdir(d)):
                if f.endswith(".png"):
                    with Image.open(os.path.join(d, f)) as im:
                        g = im.convert("L")
                        out.append((f"LOL-v1/{sp}/high/{f}", norm_thumb(g, (96, 64)), norm_thumb(center_square(g), (64, 64))))
        z = zipfile.ZipFile(V2_ZIP)
        for n in z.namelist():
            if "/Normal/" in n and n.endswith(".png"):
                g = Image.open(io.BytesIO(z.read(n))).convert("L")
                out.append((n, norm_thumb(g, (96, 64)), norm_thumb(center_square(g), (64, 64))))
        return out

    lol = lol_thumbs()
    lol_real = [(n, r) for n, r, _ in lol if "Synthetic" not in n]
    lol_syn = [(n, s) for n, _, s in lol if "Synthetic" in n]
    R_ = np.stack([r for _, r in lol_real])
    Sy = np.stack([s for _, s in lol_syn])
    LS = np.stack([thumbs[(k[0], k[1], "L")]["sq"] for k in lab_keys])
    MS = np.stack([thumbs[(k[0], k[1], mid[k])]["sq"] for k in lab_keys])
    c_real = np.maximum(LT @ R_.T, MT @ R_.T)
    c_syn = np.maximum(LS @ Sy.T, MS @ Sy.T)
    lol_top = []
    for name, cm, ref in (("LOL-real", c_real, lol_real), ("LOL-v2-synthetic", c_syn, lol_syn)):
        j = cm.argmax(axis=1)
        best = sorted(((float(cm[i, j[i]]), f"{lab_keys[i][0]}/{lab_keys[i][1]}", ref[j[i]][0])
                       for i in range(len(lab_keys))), reverse=True)
        lol_top.append({"target": name, "n_ge_thr": int(sum(b[0] >= SCENE_THR for b in best)), "top10": best[:10]})
    lol_md5 = set()
    for sp in ("our485", "eval15"):
        for q in ("low", "high"):
            d = os.path.join(V1_ROOT, sp, q)
            for f in os.listdir(d):
                if f.endswith(".png"):
                    lol_md5.add(hashlib.md5(open(os.path.join(d, f), "rb").read()).hexdigest())
    z = zipfile.ZipFile(V2_ZIP)
    for n in z.namelist():
        if n.endswith(".png"):
            lol_md5.add(hashlib.md5(z.read(n)).hexdigest())
    lol_byte_dups = [r["path"] for r in images if r["md5"] in lol_md5]

    # ------------------------------------------------------------ assemble
    def stats(recs):
        return {
            "images": len(recs),
            "formats": dict(Counter(r.get("format") for r in recs)),
            "extensions": dict(Counter(r["path"].rsplit(".", 1)[-1] for r in recs)),
            "modes": dict(Counter(r.get("mode") for r in recs)),
            "resolutions": dict(Counter("x".join(map(str, r.get("size", []))) for r in recs).most_common()),
            "cameras": dict(Counter(r.get("camera_model", "(none)") for r in recs)),
            "orientation_tag": dict(Counter(str(r.get("orientation", "(none)")) for r in recs)),
            "decode_errors": [r["path"] for r in recs if "decode_error" in r],
        }

    parts = {}
    for part in sorted({r["part"] for r in images}):
        ex = [r for r in images if r["part"] == part and r["kind"] == "exposure"]
        lb = [r for r in images if r["part"] == part and r["kind"] == "label"]
        ks = [k for k in keys if k[0] == part]
        parts[part] = {
            "sequences": len(ks),
            "exposures_per_sequence": dict(sorted(Counter(len(seqs[k]["exposures"]) for k in ks).items())),
            "exposure_images": stats(ex),
            "label_images": stats(lb),
            "sequences_with_label": sum(1 for k in ks if seqs[k]["label"]),
            "mixed_resolution_sequences": [f"{k[0]}/{k[1]}" for k in ks if len(seq_summary[f"{k[0]}/{k[1]}"]["sizes"]) > 1],
        }

    result = {
        **report,
        "text_files": text_files,
        "parts": parts,
        "ordering_checks": dict(ordering),
        "exif_presence": dict(exif_presence),
        "exposure_bias_patterns_top": dict(bias_patterns.most_common(15)),
        "label_size_vs_sequence": dict(label_vs_seq_size),
        "label_mean_lum_rank_within_sequence": dict(sorted(label_lum_rank.items())),
        "duplicates": {
            "identical_files": dup_files,
            "scene_threshold": SCENE_THR,
            "different_sequence_corr_p99": diff_p99,
            "duplicate_scene_pairs": dup_scene,
            "borderline_pairs_0.85_0.95": border[:100],
            "n_borderline": len(border),
        },
        "vs_lol": {"byte_identical_files": lol_byte_dups, "thumbnail_nearest": lol_top},
        "sequences": seq_summary,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT_DIR, "sice_structure_audit.json"), "w") as f:
        json.dump(result, f, indent=1, default=str)
    if args.thumbs:
        np.savez_compressed(args.thumbs, keys=np.array([f"{k[0]}|{k[1]}|{k[2]}" for k in thumbs]),
                            previews=np.stack([v["preview"] for v in thumbs.values()]))

    # ------------------------------------------------------------ text report
    T = ["SICE structure audit (READ-ONLY, streamed from RAR, nothing extracted)", "=" * 70, ""]
    T += ["Files in data/SICE:"] + [f"  {k}: {v / 1e9:.2f} GB" for k, v in report["files_present"].items()]
    T += ["", "Integrity:"] + [f"  {k}: {v}" for k, v in report["archives"].items()]
    for part, p in parts.items():
        T += ["", f"{part}: {p['sequences']} sequences, {p['exposure_images']['images']} exposure images, "
                  f"{p['label_images']['images']} label images",
              f"  exposures/sequence: {p['exposures_per_sequence']}",
              f"  exposure images: formats {p['exposure_images']['formats']} modes {p['exposure_images']['modes']}",
              f"    resolutions: {p['exposure_images']['resolutions']}",
              f"    cameras: {p['exposure_images']['cameras']}",
              f"  labels: formats {p['label_images']['formats']} ext {p['label_images']['extensions']} modes {p['label_images']['modes']}",
              f"    resolutions: {p['label_images']['resolutions']}",
              f"  sequences with mixed resolutions: {len(p['mixed_resolution_sequences'])}",
              f"  decode errors: {p['exposure_images']['decode_errors'] + p['label_images']['decode_errors']}"]
    T += ["", f"Ordering checks: {dict(ordering)}", f"EXIF presence: {dict(exif_presence)}",
          f"Top exposure-bias patterns: {dict(bias_patterns.most_common(6))}",
          f"Label size vs sequence: {dict(label_vs_seq_size)}",
          f"Label mean-luminance rank within sequence (#exposures darker than label / n): {dict(sorted(label_lum_rank.items()))}",
          "", f"Identical files (byte MD5): {len(dup_files)} groups {dup_files[:5]}",
          f"Duplicate-scene sequence pairs (corr >= {SCENE_THR}; different-sequence p99 = {diff_p99:.3f}): {len(dup_scene)}",
          f"  {dup_scene[:15]}", f"Borderline [0.85, 0.95): {len(border)}  top {border[:8]}",
          "", f"vs LOL byte-identical files: {len(lol_byte_dups)}"]
    for x in lol_top:
        T.append(f"vs {x['target']}: sequences with corr >= {SCENE_THR}: {x['n_ge_thr']}; top {x['top10'][:5]}")
    T += ["", "Text files:"] + [f"--- {k}\n{v}" for k, v in text_files.items()] + ["", f"runtime {result['runtime_s']} s"]
    with open(os.path.join(OUT_DIR, "sice_structure_audit.txt"), "w") as f:
        f.write("\n".join(T) + "\n")
    print("\n".join(T))


if __name__ == "__main__":
    main()
