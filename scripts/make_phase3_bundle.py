"""Build the code bundle for the Colab T4 pilot (code only: no data, no weights, no results).

Contents: src/, scripts/{train_phase3,phase3_pilot}.py, configs/generalized_lol_v2.yaml, configs/phase3/,
third_party/ (vendored MIT code + LICENSE + PATCHES.diff), experiments/generalized_lol_v2/splits/,
requirements-phase3.txt, and BUNDLE_MANIFEST.json (MD5 of every file). The notebook verifies the zip MD5 and every
file MD5 after unzipping into /content/proj.

Usage: .venv/bin/python scripts/make_phase3_bundle.py   (refuses to overwrite an existing bundle)
"""

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments/phase3_model_selection/stage1/bundle"
INCLUDE = ["src", "third_party", "configs/phase3", "experiments/generalized_lol_v2/splits"]
FILES = ["scripts/train_phase3.py", "scripts/phase3_pilot.py", "configs/generalized_lol_v2.yaml",
         "requirements-phase3.txt",
         "experiments/phase3_model_selection/stage0/source_phase2/history.json"]  # Phase-2 epoch times (P1)
BUNDLE_PREFIX = "phase3_pilot_bundle_v2_1_"


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    files = [ROOT / f for f in FILES]
    for d in INCLUDE:
        files += [p for p in (ROOT / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                  and p.suffix not in (".pyc",) and p.name != ".DS_Store"]
    files = sorted(set(files))
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "files": {str(p.relative_to(ROOT)): md5(p) for p in files}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zp = OUT_DIR / f"{BUNDLE_PREFIX}{time.strftime('%Y%m%d_%H%M')}.zip"
    if zp.exists():
        sys.exit(f"{zp} exists")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, str(p.relative_to(ROOT)))
        z.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=1))
    zmd5 = md5(zp)
    nb = ROOT / f"notebooks/colab_phase3_t4_pilot_v2_1_{zp.stem.split(BUNDLE_PREFIX)[1]}.ipynb"
    if nb.exists():
        sys.exit(f"{nb} exists")
    nb.write_text(json.dumps(notebook(zp.name, zmd5), indent=1))
    print(f"{zp}\n  files {len(files)}  size {zp.stat().st_size / 2**20:.2f} MB  md5 {zmd5}\n{nb}")


DATA_ZIP_MD5 = "3afbe90bb3026faa39591bf820ee3916"  # data_prepared/phase2.zip (verified in Stage 0 / Phase 2)


def _code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n")}


def _md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def notebook(bundle_name: str, bundle_md5: str) -> dict:
    cells = [_md(f"""
# Phase 3 / Stage 1: T4 GPU feasibility pilot (throwaway weights, no evaluation)
Runs `scripts/phase3_pilot.py` from bundle **{bundle_name}** (MD5 `{bundle_md5}`).
- **Runtime > Change runtime type > T4 GPU.** Run cells top to bottom. Each model has its own cell and summary.
- Nothing in this notebook trains a Tier-A model or reads test sets / SICE / LoLI-Street.
- Results go to a NEW folder on Drive: `.../experiments/phase3_model_selection/stage1/pilots/pilot_<timestamp>/`.
- **If Colab disconnects:** rerun cells 1-3, set `RESUME_PILOT_DIR` in cell 4 to the existing pilot folder (cell 4 then
  skips P0/P1), and rerun only the model cells that have no summary yet. Each attempt writes a new timestamped subfolder,
  so nothing is overwritten.
- Upload first: the bundle zip to `MyDrive/LowLight-Generalization/phase3_bundles/`.
- **Integrity scope:** cell 4 runs the FULL check (structural checks + per-file MD5 of every prepared file). Model cells
  repeat only the structural checks and cite cell 4's report (`integrity_md5_reference`); the pilot refuses that shortcut
  if the reference is not a passed full check from this same VM session (after a disconnect, model cells run the full
  check again automatically).
- **Stopping a cell** terminates the pilot and all its sub-processes (process group) and saves a partial report with
  `status: interrupted` (also when a DataLoader worker dies first from the same stop). Every invocation writes
  `pilot_run_status.json` (stages P0 / P1 / model: running, complete, interrupted, failed, error); cell 9a lists all
  attempts, and summaries always describe the LATEST attempt of each model.
"""),
        _code("""
# 1. GPU
import torch, subprocess
print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| available', torch.cuda.is_available())
assert torch.cuda.is_available(), 'No GPU: Runtime -> Change runtime type -> T4 GPU'
print(torch.cuda.get_device_name(0)); print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
left = subprocess.run(['pgrep', '-af', 'phase3_pilot.py|train_phase3.py'], capture_output=True, text=True).stdout.strip()
assert not left, f'leftover pilot/training processes found (restart the runtime first):\\n{left}'
print('no leftover pilot processes')
"""),
        _code(f"""
# 2. Drive, bundle and data (all MD5-verified; nothing on Drive is modified)
from google.colab import drive
drive.mount('/content/drive')
import hashlib, json, os, zipfile, subprocess
DRIVE = '/content/drive/MyDrive/LowLight-Generalization'
BUNDLE = f'{{DRIVE}}/phase3_bundles/{bundle_name}'
def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    return h.hexdigest()
assert md5(BUNDLE) == '{bundle_md5}', 'bundle MD5 mismatch: re-upload the bundle'
if not os.path.isdir('/content/proj'):
    zipfile.ZipFile(BUNDLE).extractall('/content/proj')
man = json.load(open('/content/proj/BUNDLE_MANIFEST.json'))['files']
bad = [f for f, h in man.items() if md5(f'/content/proj/{{f}}') != h]
assert not bad, f'bundle file MD5 mismatch: {{bad[:5]}}'
print('bundle OK:', len(man), 'files')
DATA_ZIP = f'{{DRIVE}}/data_prepared/phase2.zip'
assert md5(DATA_ZIP) == '{DATA_ZIP_MD5}', 'phase2.zip MD5 mismatch'
if not os.path.exists('/content/phase2/manifest.json'):
    subprocess.run(['unzip', '-q', DATA_ZIP, '-d', '/content'], check=True)
print('data OK:', os.listdir('/content/phase2'))
"""),
        _code("""
# 3. Dependency (approved): einops 0.8.0. VGG-19 ImageNet weights are fetched by torchvision (hash-checked)
#    during the hvi_cidnet / darkir_m cells only (training-loss backbone).
!pip -q install einops==0.8.0
import einops, torchvision, numpy, PIL
print('einops', einops.__version__, '| torchvision', torchvision.__version__, '| numpy', numpy.__version__, '| PIL', PIL.__version__)
"""),
        _code("""
# 4. Pilot folder + P0 (environment, full MD5 data integrity, RGB pipeline) + P1 (U-Net calibration)
import time, glob
RESUME_PILOT_DIR = None   # after a disconnect: set to the existing pilot folder path to skip P0/P1
ROOT_PILOTS = f'{DRIVE}/experiments/phase3_model_selection/stage1/pilots'
import signal
def sh(args, cwd='/content/proj', grace_s=30):
    # Run a command, streaming its output. The child gets its own process group (start_new_session), so on ANY
    # interruption of this cell (stop button -> KeyboardInterrupt) the whole group - the pilot, its resume-test
    # training subprocesses and DataLoader workers - is sent SIGTERM (the pilot then saves a partial report),
    # then SIGKILL after grace_s. Nothing keeps running in the background or holds the GPU.
    p = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    try:
        for line in p.stdout:
            print(line, end='')
        rc = p.wait()
    except BaseException:
        for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 10)):
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                break
            try:
                p.wait(timeout=wait_s)
                break
            except subprocess.TimeoutExpired:
                continue
        print(f'\\nINTERRUPTED: process group {p.pid} terminated (exit {p.returncode})')
        raise
    if rc != 0:
        raise RuntimeError(f'command failed ({rc}): {args}')
PILOT = ['python', 'scripts/phase3_pilot.py', '--data-root', '/content/phase2']
if RESUME_PILOT_DIR:
    PILOT_DIR = RESUME_PILOT_DIR
    assert os.path.isdir(PILOT_DIR)
else:
    PILOT_DIR = f'{ROOT_PILOTS}/pilot_{time.strftime("%Y%m%d_%H%M")}'
    assert not os.path.exists(PILOT_DIR), 'folder exists'
    os.makedirs(PILOT_DIR)
    sh(PILOT + ['--out-dir', f'{PILOT_DIR}/setup', '--models'])
print('PILOT_DIR =', PILOT_DIR)

def attempt_state(folder):
    # run-level status written by the pilot (P0/P1/model stages); None if the attempt never started
    p = f'{folder}/pilot_run_status.json'
    return json.load(open(p)) if os.path.exists(p) else None

def summary(model):
    # the LATEST attempt folder decides (an older attempt's report is never shown as current)
    attempts = sorted(d for d in glob.glob(f'{PILOT_DIR}/{model}_*') if os.path.isdir(d))
    if not attempts:
        print(model, ': not run yet'); return None
    last = attempts[-1]
    st = attempt_state(last)
    rp = f'{last}/{model}/pilot_report.json'
    r = json.load(open(rp)) if os.path.exists(rp) else None
    print(f'=== {model}  (attempt {os.path.basename(last)}; {len(attempts)} attempt(s))  run: '
          f'{st and st.get("outcome")}  report: {r and r.get("status")}')
    if r is None or r.get('status') != 'complete':
        if st:
            print('   stopped in:', st.get('stopped_in_stage'), '| stages:', st.get('stages'), '|', st.get('error'))
        if r:
            print('   section in progress:', r.get('section_in_progress'), '|', r.get('error'), '|', r.get('interrupt'))
            print(r.get('traceback', '')[-1500:])
        return r
    p2, p34, p5, p6, p7 = r['P2_correctness'], r['P3_P4'], r['P5_inference_fp32'], r.get('P6_resume', {}), r['P7_projection']
    rows = [
        ('P2 params', f"{p2['params']:,} (expected {p2['expected']:,}; paper {p2['paper']})", p2['params_ok']),
        ('P2 loss finite / eval range', f"{p2['first_loss']:.4f} / {p2['eval_output_in_range']}", p2['loss_finite'] and p2['eval_output_in_range']),
        ('P3 FP32 s/step (wall)', f"{p34['fp32']['s_per_step_wall']:.3f}  data-wait {p34['fp32']['data_wait_share']:.0%}  peak {p34['fp32']['peak_mem_gb']:.2f} GB", None),
        ('P3 AMP  s/step (wall)', f"{p34['amp']['s_per_step_wall']:.3f}  peak {p34['amp']['peak_mem_gb']:.2f} GB  skipped {p34['amp']['skipped']}", None),
        ('P4 AMP gates', f"speed-up {p34['amp_speedup']:.2f}x  loss diff {p34['amp_loss_rel_diff']:.2%}", p34['amp_gates_pass']),
        ('Chosen precision / memory', f"{r['chosen_precision']}  max-batch peak GB {r['max_batch_peak_gb']}", r['mem_ok']),
        ('P5 val pass (median; warm-up) / 20 MP tiled', f"{p5['val_pass_s']} s ({p5.get('val_pass_first_s')} s) / {p5['sice_20mp_tiled_s']} s (peak {p5['sice_20mp_peak_mem_gb']} GB)", p5['sice_20mp_output_in_range']),
        ('P6 resume', f"batch order identical {p6.get('batch_order_identical')}  loss rel diff {p6.get('epoch2_loss_rel_diff')}  bitwise {p6.get('final_weights_bitwise_identical')}", p6.get('pass')),
        ('P7 Tier-A projection', f"{p7['tierA_hours']} h (gate <= 3 h)", p7['tierA_ok']),
        ('P7 dev-export inference', f"{p7['dev_export_inference_hours_approx']} h (gate <= 1.5 h)", p7['dev_export_ok']),
    ]
    for k, v, ok in rows:
        print(f"  {'PASS' if ok else ('FAIL' if ok is False else '    ')}  {k:34s} {v}")
    for k in ('expaug_ok', 'fremlp_runs_fp32_under_autocast'):
        if k in p2: print(f"  {'PASS' if p2[k] else 'FAIL'}  P2 {k}  {({kk: p2[kk] for kk in p2 if kk.startswith('expaug')} if k == 'expaug_ok' else '')}")
    return r

def run_model(model):
    out = f'{PILOT_DIR}/{model}_{time.strftime("%Y%m%d_%H%M%S")}'
    st = attempt_state(f'{PILOT_DIR}/setup')
    if not st or st.get('outcome') != 'complete':
        print('WARNING: setup run is not complete:', st and {k: st.get(k) for k in ('outcome', 'stages', 'stopped_in_stage')})
    setup_p0 = f'{PILOT_DIR}/setup/P0_environment.json'
    same_vm = (os.path.exists(setup_p0) and
               json.load(open(setup_p0)).get('boot_id') == open('/proc/sys/kernel/random/boot_id').read().strip())
    integ = ['--p0-md5-reference', setup_p0] if same_vm else []   # else: full MD5 check in this run
    print('integrity:', 'structural (MD5 verified in setup, same VM)' if same_vm else 'FULL (new VM session)')
    try:
        sh(PILOT + ['--out-dir', out, '--models', model, '--skip-calibration'] + integ)
    except RuntimeError as e:
        print('PILOT PROCESS FAILED:', e)
    return summary(model)

if not RESUME_PILOT_DIR:
    c = json.load(open(f'{PILOT_DIR}/setup/P1_calibration.json'))
    e0 = json.load(open(f'{PILOT_DIR}/setup/P0_environment.json'))
    print('P0:', e0['integrity_scope'], '| passed', e0['integrity_all_passed'], '| MD5 files', e0.get('integrity_md5_files'))
    print(f"P1 ({c['method']}): ratio {c['ratio']} (gate +/-{c['tolerance']:.0%}) {'PASS' if c['pass'] else 'FAIL'} | "
          f"steady epoch {c['steady_median_epoch_s']:.2f} s = train {c['steady_median_train_s']:.2f} + val "
          f"{c['steady_median_val_s']:.2f} s | Phase-2 median epoch {c['phase2_median_epoch_s']} s")
    for e in c['epochs_measured']: print('   ', e)
    print('   diagnostic (synced per-step rate):', c.get('diagnostic_instrumented'))
"""),
        _code("# 5. U-Net + exposure augmentation\nrun_model('unet_expaug');"),
        _code("# 6. Retinexformer\nrun_model('retinexformer');"),
        _code("# 7. HVI-CIDNet (downloads torchvision VGG-19 ImageNet weights on first use)\nrun_model('hvi_cidnet');"),
        _code("# 8. DarkIR-m (optional for Tier A; same VGG-19 weights)\nrun_model('darkir_m');"),
        _code("""
# 9a. Attempt log: every invocation in this pilot folder with its outcome (nothing ambiguous after an interruption)
for d in sorted(x for x in glob.glob(f'{PILOT_DIR}/*') if os.path.isdir(x)):
    st = attempt_state(d)
    print(f"{os.path.basename(d):40s}", 'no status file (never started)' if st is None else
          f"{st['outcome']:12s} stages={st['stages']} stopped_in={st.get('stopped_in_stage')}")
"""),
        _code("""
# 9. All summaries + record the VGG-19 weight file actually used
for m in ['unet_expaug', 'retinexformer', 'hvi_cidnet', 'darkir_m']:
    summary(m); print()
w = glob.glob(os.path.expanduser('~/.cache/torch/hub/checkpoints/vgg19-*.pth'))
info = {os.path.basename(p): {'bytes': os.path.getsize(p), 'md5': md5(p)} for p in w}
json.dump(info, open(f'{PILOT_DIR}/vgg19_weights_used.json', 'w'), indent=1); print(info)
print('Download the folder', PILOT_DIR, 'and share it for analysis. Do NOT start Tier A before approval.')
"""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 0}


if __name__ == "__main__":
    main()
