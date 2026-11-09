"""Phase 5 diagnostic data pack (Mac, CPU, data preparation only; no model is run).

    .venv/bin/python -I scripts/prepare_phase5_diag_data.py --tmp <scratch dir>

Builds data_prepared/phase5_diag/ (+ phase5_diag_data.zip) with:
  SICE_Part1_900/<seq>/<k>.jpg, SICE_Part1_900/<seq>/label.jpg
      every sequence and every exposure of SICE **Part 1 only**, area-downscaled so the long side is 900 px
      (input and label resized identically), re-saved as JPEG q95 4:4:4. SICE Part 2 is never opened:
      the script refuses any archive whose name is not Dataset_Part1.rar and any member outside Dataset_Part1/.
  LoLI-Street/Val/{low,high}/<level>_<id>.jpg   official Val pairs, bytes unchanged, id 30727 excluded (all black)
  LoLI-Street/Test/Test_<n>.jpg                 official real low-light test images (no reference), bytes unchanged
  manifest.json                                 MD5 of every file, counts, settings, source archive MD5s

Writes only into new paths; refuses to overwrite an existing pack.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LONG_SIDE = 900
JPEG_Q = 95
LOLI_EXCLUDED = {"30727"}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def target_size(w, h, long_side=LONG_SIDE):
    s = long_side / max(w, h)
    if s >= 1:
        return w, h
    return max(1, round(w * s)), max(1, round(h * s))


def resize_save(src, dst, size):
    with Image.open(src) as im:
        im = im.convert("RGB")
        if im.size != size:
            im = im.resize(size, Image.Resampling.BOX)
        im.save(dst, quality=JPEG_Q, subsampling=0)


def sice_part1(rar, tmp, out):
    if Path(rar).name != "Dataset_Part1.rar":
        sys.exit(f"refusing {rar}: only Dataset_Part1.rar may be opened (SICE Part 2 is sealed)")
    listing = subprocess.run(["bsdtar", "-tf", rar], capture_output=True, text=True, check=True).stdout.split("\n")
    bad = [m for m in listing if m and m != "Dataset_Part1" and not m.startswith("Dataset_Part1/")]
    if bad:
        sys.exit(f"unexpected members outside Dataset_Part1/: {bad[:3]}")
    x = Path(tmp) / "sice_part1_x"
    if not x.exists():
        x.mkdir(parents=True)
        subprocess.run(["bsdtar", "-xf", rar, "-C", str(x)], check=True)
    base = x / "Dataset_Part1"
    seqs = sorted((d for d in base.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name))
    rec = {}
    for i, d in enumerate(seqs):
        exps = sorted((p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")), key=lambda p: int(p.stem))
        lab = base / "Label" / f"{d.name}.JPG"
        if not lab.exists():
            cands = [p for p in (base / "Label").glob(f"{d.name}.*")]
            lab = cands[0] if cands else None
        if lab is None or not exps:
            rec[d.name] = {"skipped": "no label or no exposures"}
            continue
        with Image.open(exps[0]) as im:
            w, h = im.size
        with Image.open(lab) as im:
            lw, lh = im.size
        size = target_size(w, h)
        od = out / d.name
        od.mkdir(parents=True)
        for p in exps:
            resize_save(p, od / f"{int(p.stem)}.jpg", size)
        resize_save(lab, od / "label.jpg", size)
        rec[d.name] = {"n_exposures": len(exps), "orig_size": [w, h], "label_orig_size": [lw, lh],
                       "label_size_differs": [w, h] != [lw, lh], "size": list(size)}
        if i % 25 == 0:
            print(f"  SICE seq {d.name} ({i + 1}/{len(seqs)})", flush=True)
    return rec


def loli(zip_path, out):
    z = zipfile.ZipFile(zip_path)
    n_val, n_test, skipped = 0, 0, []
    for name in z.namelist():
        parts = name.split("/")
        if name.endswith("/") or len(parts) < 3:
            continue
        if parts[1] == "Val" and len(parts) == 4 and parts[2] in ("low", "high"):
            stem = Path(parts[3]).stem
            if stem.split("_")[-1] in LOLI_EXCLUDED:
                skipped.append(name)
                continue
            dst = out / "Val" / parts[2] / parts[3]
            n_val += 1
        elif parts[1] == "Test" and len(parts) == 3:
            dst = out / "Test" / parts[2]
            n_test += 1
        else:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with z.open(name) as s, open(dst, "wb") as d:
            shutil.copyfileobj(s, d)
    return {"val_files": n_val, "test_files": n_test, "excluded": skipped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sice-rar", default=str(ROOT / "data/SICE/Dataset_Part1.rar"))
    ap.add_argument("--loli-zip", default=str(ROOT / "data/LoLI-Street/archive (3).zip"))
    ap.add_argument("--out", default=str(ROOT / "data_prepared/phase5_diag"))
    ap.add_argument("--tmp", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"exists, not overwritten: {out}")
    t0 = time.time()
    stage = out.with_name(out.name + ".partial")
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "SICE_Part1_900").mkdir(parents=True)
    sice = sice_part1(a.sice_rar, a.tmp, stage / "SICE_Part1_900")
    lol = loli(a.loli_zip, stage / "LoLI-Street")
    files = {str(p.relative_to(stage)): md5(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    man = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "settings": {"sice_long_side": LONG_SIDE, "sice_resample": "PIL BOX (area)", "sice_jpeg_quality": JPEG_Q,
                        "sice_jpeg_subsampling": "4:4:4", "loli": "bytes unchanged", "loli_excluded_ids": sorted(LOLI_EXCLUDED)},
           "sources": {"sice_part1_rar": {"path": a.sice_rar, "md5": md5(a.sice_rar)},
                       "loli_zip": {"path": a.loli_zip, "md5": md5(a.loli_zip)},
                       "sice_part2": "NOT OPENED (sealed for the final evaluation)"},
           "sice": sice, "loli": lol,
           "counts": {"sice_sequences": sum(1 for v in sice.values() if "skipped" not in v),
                      "sice_exposures": sum(v.get("n_exposures", 0) for v in sice.values()),
                      "loli_val_files": lol["val_files"], "loli_test_files": lol["test_files"]},
           "files": files}
    json.dump(man, open(stage / "manifest.json", "w"), indent=1)
    stage.rename(out)
    zp = out.with_name("phase5_diag_data.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out).as_posix())
    print(json.dumps(man["counts"]), f"zip {zp} md5 {md5(zp)} {zp.stat().st_size / 2**30:.2f} GB",
          f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
