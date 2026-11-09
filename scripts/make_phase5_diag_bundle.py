"""Build the Phase 5 diagnostic code bundle, the weights pack and the Colab notebook (inference only).

    .venv/bin/python -I scripts/make_phase5_diag_bundle.py      (refuses to overwrite)

Upload to Drive  MyDrive/LowLight-Generalization/phase5_inputs/ :
  phase5_diag_bundle_<stamp>.zip      code (src/, third_party/, scripts/phase5_diagnostic.py) + BUNDLE_MANIFEST.json
  phase5_diag_weights_<stamp>.zip     our checkpoints + official weights (models/phase5_diag_weights), MD5-checked by zoo
  phase5_diag_data.zip                from scripts/prepare_phase5_diag_data.py (SICE Part 1 + LoLI-Street; no Part 2)
"""

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _code(src):
    """Notebook code cell (same definition as in scripts/make_phase3_bundle.py)."""
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n")}


def _md(src):
    """Notebook markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


OUT_DIR = ROOT / "experiments/phase5_llie_direction/bundle"
INCLUDE = ["src", "third_party"]
FILES = ["scripts/phase5_diagnostic.py"]
WEIGHTS_DIR = ROOT / "models/phase5_diag_weights"
DATA_ZIP = ROOT / "data_prepared/phase5_diag_data.zip"
LPIPS = "0.1.4"
PYIQA = "0.1.16"
PINNED = ["torch", "torchvision", "torchaudio", "numpy", "pillow", "scipy", "opencv-python", "opencv-python-headless", "lpips"]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def notebook(bundle, bundle_md5, weights, weights_md5, data_md5):
    cells = [
        _md(f"""# Phase 5 diagnostic (inference only): over-enhancement across exposure levels

Upload to Drive `MyDrive/LowLight-Generalization/phase5_inputs/`: `{bundle}`, `{weights}`, `phase5_diag_data.zip`.
Runtime: **T4 GPU**. Run cells 1-6 in order in one session. Outputs go to
`experiments/phase5_llie_direction/diagnostic/` on Drive. Reruns skip finished (dataset, condition) parts.

No training, no parameter updates, **no SICE Part 2** (it is not in the data pack and the script refuses it)."""),
        _code("""
# 1. GPU
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'
print(gpu, torch.__version__)
assert 'T4' in gpu, 'expected a Tesla T4 runtime: STOP'
"""),
        _code(f"""
# 2. Drive, integrity of the three inputs, extraction
from google.colab import drive; drive.mount('/content/drive')
import glob, hashlib, json, os, subprocess, sys, time, zipfile
DRIVE = '/content/drive/MyDrive/LowLight-Generalization'
IN = f'{{DRIVE}}/phase5_inputs'
OUT = f'{{DRIVE}}/experiments/phase5_llie_direction/diagnostic'
os.makedirs(f'{{OUT}}/session_records', exist_ok=True); os.makedirs(f'{{OUT}}/logs', exist_ok=True)
def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
for name, h in (('{bundle}', '{bundle_md5}'), ('{weights}', '{weights_md5}'), ('phase5_diag_data.zip', '{data_md5}')):
    assert md5(f'{{IN}}/{{name}}') == h, f'{{name}} MD5 mismatch: STOP'
    print('OK', name)
PROJ, W, DATA = '/content/p5proj', '/content/p5weights', '/content/p5data'
zipfile.ZipFile(f'{{IN}}/{bundle}').extractall(PROJ)
man = json.load(open(f'{{PROJ}}/BUNDLE_MANIFEST.json'))['files']
assert not [f for f, h in man.items() if md5(f'{{PROJ}}/{{f}}') != h], 'bundle file MD5 mismatch: STOP'
if not os.path.isdir(W): zipfile.ZipFile(f'{{IN}}/{weights}').extractall(W)
if not os.path.isdir(DATA): zipfile.ZipFile(f'{{IN}}/phase5_diag_data.zip').extractall(DATA)
dm = json.load(open(f'{{DATA}}/manifest.json'))
bad = [f for f, h in dm['files'].items() if md5(f'{{DATA}}/{{f}}') != h]
assert not bad, f'data file MD5 mismatch: {{bad[:3]}}: STOP'
assert not glob.glob(f'{{DATA}}/**/*Part2*', recursive=True), 'SICE Part 2 found in the data pack: STOP'
print('bundle', len(man), 'files; data', dm['counts'])
STAMP = time.strftime('%Y%m%d_%H%M%S')
"""),
        _code(f"""
# 3. Packages: lpips (as Phase 2) and pyiqa {PYIQA} for NIQE / BRISQUE, with core packages pinned to this session's versions
import subprocess, sys
def freeze():
    return dict(l.split('==', 1) for l in subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True,
                text=True).stdout.splitlines() if '==' in l)
before = freeze()
pins = {{k: v for k, v in before.items() if k.lower() in {[p.lower() for p in PINNED]}}}
open('/content/constraints.txt', 'w').write('\\n'.join(f'{{k}}=={{v}}' for k, v in pins.items()))
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'lpips=={LPIPS}', 'pyiqa=={PYIQA}', '-c', '/content/constraints.txt'],
                   capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-3000:])
assert r.returncode == 0, 'pip could not install lpips/pyiqa without changing pinned packages: STOP (paste this output)'
after = freeze()
changed = {{k: [before.get(k), after.get(k)] for k in set(before) | set(after) if before.get(k) != after.get(k)}}
assert all(before.get(k) == after.get(k) for k in pins), 'a pinned core package changed: STOP'
rec = {{'pinned_unchanged': pins, 'installed_or_changed': changed, 'python': sys.version}}
json.dump(rec, open(f'{{OUT}}/session_records/packages_{{STAMP}}.json', 'w'), indent=1)
open(f'{{OUT}}/session_records/pip_freeze_{{STAMP}}.txt', 'w').write('\\n'.join(f'{{k}}=={{v}}' for k, v in sorted(after.items())))
print('changed packages:', changed)
"""),
        _code("""
# 4. Stage runner (logs to Drive; stops on any failure)
def run_stage(stage, *extra):
    log = f'{OUT}/logs/diag_{stage}_{time.strftime("%Y%m%d_%H%M%S")}.log'
    cmd = [sys.executable, 'scripts/phase5_diagnostic.py', stage, '--data', DATA, '--weights', W, '--out', OUT, *extra]
    with open(log, 'w') as f:
        p = subprocess.Popen(cmd, cwd=PROJ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end=''); f.write(line); f.flush()
        rc = p.wait()
    assert rc == 0, f'stage {stage} FAILED (exit {rc}); log: {log}. STOP'
    print(f'stage {stage} OK; log: {log}')
"""),
        _code("""
# 5. Inference on all datasets x conditions (resumable; about 1.5-3 h on a T4, estimate)
run_stage('run')
"""),
        _code("""
# 6. Report: tables, plots, panels (CPU; a few minutes)
run_stage('report')
print(open(f'{OUT}/DIAG_REPORT.md').read()[:20000])
"""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    if not DATA_ZIP.exists():
        sys.exit(f"missing {DATA_ZIP}: run scripts/prepare_phase5_diag_data.py first")
    stamp = time.strftime("%Y%m%d_%H%M")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zb, zw = OUT_DIR / f"phase5_diag_bundle_{stamp}.zip", OUT_DIR / f"phase5_diag_weights_{stamp}.zip"
    nb = ROOT / f"notebooks/colab_phase5_diag_{stamp}.ipynb"
    if zb.exists() or zw.exists() or nb.exists():
        sys.exit("exists, not overwritten")
    files = [ROOT / f for f in FILES]
    for d in INCLUDE:
        files += [p for p in (ROOT / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                  and p.suffix != ".pyc" and p.name != ".DS_Store"]
    files = sorted(set(files))
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": {str(p.relative_to(ROOT)): md5(p) for p in files}}
    with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, str(p.relative_to(ROOT)))
        z.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=1))
    with zipfile.ZipFile(zw, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(WEIGHTS_DIR.rglob("*")):
            if p.is_file() and p.name != ".DS_Store":
                z.write(p, str(p.relative_to(WEIGHTS_DIR)))
    hb, hw, hd = md5(zb), md5(zw), md5(DATA_ZIP)
    nb.write_text(json.dumps(notebook(zb.name, hb, zw.name, hw, hd), indent=1))
    print(json.dumps({"bundle": [str(zb), hb, len(files)], "weights": [str(zw), hw, round(zw.stat().st_size / 2**20, 1)],
                      "data": [str(DATA_ZIP), hd, round(DATA_ZIP.stat().st_size / 2**30, 2)], "notebook": [str(nb), md5(nb)]},
                     indent=1))


if __name__ == "__main__":
    main()
