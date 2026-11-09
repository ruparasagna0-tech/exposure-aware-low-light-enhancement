"""Prepare Phase-2 data: copy ONLY the required files, byte-for-byte, into data_prepared/phase2/.

Sources are never modified (read-only access to the original archives):
    data/LOL-v2/LOL-v2.zip             -> LOL-v2/Real_captured/..., LOL-v2/Synthetic/...   (all)
    data/SICE/Dataset_Part1.rar        -> SICE/Dataset_Part1/<id>/*, SICE/Dataset_Part1/Label/<id>.*
                                          for the 58 official test sequences (+ "testing index.txt"
                                          from Dataset_Part2.rar, which defines that list)
    data/LoLI-Street/<archive>.zip     -> LoLI-Street/Val/{low,high}/{dense,moderate,light}_<id>.jpg
                                          for the 999 valid Val ids (30727 excluded: all-black)

Every written file is re-read and its MD5 compared with the MD5 of the source bytes.
Writes data_prepared/phase2/manifest.json (source, member, dest, bytes, md5 per file).
Optionally (--zip) writes data_prepared/phase2.zip (stored, no recompression) for Colab upload.

Usage (from project root):
    .venv/bin/python scripts/prepare_phase2_data.py [--zip]
"""

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT_ROOT, "data")
OUT = os.path.join(PROJECT_ROOT, "data_prepared", "phase2")

sys.path.insert(0, PROJECT_ROOT)
from src.datasets.loli_street import INVALID_VAL_IDS as LOLI_INVALID_IDS, LEVELS as LOLI_LEVELS  # noqa: E402
from src.datasets.sice import parse_test_index  # noqa: E402


class Writer:
    def __init__(self):
        self.records = []

    def write(self, dest_rel, data: bytes, source: str, member: str):
        dest = os.path.join(OUT, dest_rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        src_md5 = hashlib.md5(data).hexdigest()
        with open(dest, "wb") as f:
            f.write(data)
        with open(dest, "rb") as f:
            dst_md5 = hashlib.md5(f.read()).hexdigest()
        if dst_md5 != src_md5:
            sys.exit(f"MD5 mismatch after copy: {dest_rel}")
        self.records.append({"dest": dest_rel, "source": os.path.relpath(source, PROJECT_ROOT),
                             "member": member, "bytes": len(data), "md5": src_md5})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="store_true", help="also write data_prepared/phase2.zip for upload")
    args = ap.parse_args()
    if os.path.exists(os.path.join(OUT, "manifest.json")):
        sys.exit(f"{OUT} already prepared (manifest.json exists). Remove it manually to re-run.")
    t0 = time.time()
    w = Writer()

    # ---------------- LOL-v2 (all of Real_captured + Synthetic)
    src = os.path.join(DATA, "LOL-v2", "LOL-v2.zip")
    with zipfile.ZipFile(src) as z:
        for n in z.namelist():
            if n.endswith(".png"):
                w.write(n, z.read(n), src, n)  # zip read verifies CRC
    print(f"LOL-v2: {len(w.records)} files ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- SICE: test list from Dataset_Part2.rar, images from Dataset_Part1.rar
    p2 = os.path.join(DATA, "SICE", "Dataset_Part2.rar")
    idx_bytes = subprocess.run(["bsdtar", "-xOf", p2, "testing index.txt"], capture_output=True, check=True).stdout
    w.write("SICE/testing index.txt", idx_bytes, p2, "testing index.txt")
    test_ids = parse_test_index(idx_bytes.decode("utf-8", errors="replace"))
    if len(test_ids) != 58:
        sys.exit(f"Expected 58 SICE test ids, parsed {len(test_ids)}")
    wanted = set(test_ids)
    p1 = os.path.join(DATA, "SICE", "Dataset_Part1.rar")
    n_before = len(w.records)
    proc = subprocess.Popen(["bsdtar", "-cf", "-", "--format=pax", "@" + p1],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            parts = m.name.split("/")
            if len(parts) != 3 or parts[0] != "Dataset_Part1":
                continue
            if parts[1].isdigit() and int(parts[1]) in wanted:
                pass
            elif parts[1] == "Label" and re.sub(r"\D", "", parts[2].split(".")[0]).isdigit() \
                    and int(re.sub(r"\D", "", parts[2].split(".")[0])) in wanted:
                pass
            else:
                continue
            w.write(f"SICE/{m.name}", tf.extractfile(m).read(), p1, m.name)
    proc.stdout.close()
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        sys.exit(f"bsdtar failed on {p1}: {err}")
    print(f"SICE: {len(w.records) - n_before} files for {len(test_ids)} sequences ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- LoLI-Street Val (valid ids only)
    zips = sorted(glob.glob(os.path.join(DATA, "LoLI-Street", "*.zip")))
    if len(zips) != 1:
        sys.exit(f"Expected exactly one LoLI-Street zip, found {zips}")
    src = zips[0]
    n_before = len(w.records)
    with zipfile.ZipFile(src) as z:
        root = "LoLI-Street Dataset"
        for side in ("low", "high"):
            for lvl in LOLI_LEVELS:
                for i in range(30001, 31001):
                    if i in LOLI_INVALID_IDS:
                        continue
                    member = f"{root}/Val/{side}/{lvl}_{i}.jpg"
                    w.write(f"LoLI-Street/Val/{side}/{lvl}_{i}.jpg", z.read(member), src, member)
    print(f"LoLI-Street Val: {len(w.records) - n_before} files ({time.time() - t0:.0f}s)", flush=True)

    counts = {}
    for r in w.records:
        key = "/".join(r["dest"].split("/")[:3]) if not r["dest"].startswith("SICE") else "SICE"
        counts[key] = counts.get(key, 0) + 1
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Byte-for-byte copies of the required Phase-2 files; md5 verified after writing.",
        "sice_test_ids": test_ids,
        "loli_excluded_ids": sorted(LOLI_INVALID_IDS),
        "counts": dict(sorted(counts.items())),
        "files": w.records,
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest["counts"], indent=1))

    if args.zip:
        zpath = os.path.join(os.path.dirname(OUT), "phase2.zip")
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for r in w.records:
                zf.write(os.path.join(OUT, r["dest"]), f"phase2/{r['dest']}")
            zf.write(os.path.join(OUT, "manifest.json"), "phase2/manifest.json")
        print(f"Upload zip: {zpath} ({os.path.getsize(zpath) / 1e9:.2f} GB)")
    print(f"Done: {len(w.records)} files in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
