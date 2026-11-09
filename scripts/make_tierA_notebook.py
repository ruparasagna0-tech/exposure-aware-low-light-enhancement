"""Generate the Phase-3 Tier-A training notebook (review first; nothing runs automatically).

Approved decisions (2026-10-19): FP32 for all four models; order Retinexformer -> HVI-CIDNet -> U-Net + exposure
augmentation -> DarkIR-m (labelled: training re-implemented from the paper); one seed (42) per model in Tier A.
Uses the unchanged v2.1 bundle (trainer, configs, model code verified in the T4 pilot). Writes a NEW notebook file and
refuses to overwrite.

Usage: .venv/bin/python scripts/make_tierA_notebook.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = "phase3_pilot_bundle_v2_1_20261014_0545.zip"
BUNDLE_MD5 = "a4d8add3d686db92168e223be5423231"
DATA_ZIP_MD5 = "3afbe90bb3026faa39591bf820ee3916"
VGG_MD5 = "92881fe292bd7d2408ecff58a101fd03"        # vgg19-dcbb9e9d.pth recorded in the pilot


def _code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n")}


def _md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def notebook():
    cells = [_md(f"""
# Phase 3 / Stage 1 — **Tier A training** (approved decisions, run only after separate approval)
- **Models, in this order:** Retinexformer → HVI-CIDNet → U-Net + exposure augmentation → DarkIR-m.
- **Protocol (unchanged from Phase 2 / the pilot):** LOL-v2 real + synthetic training splits, every batch = 2 real + 2
  synthetic 256² crops, 112 steps × 50 epochs = **5,600 optimizer steps**, seed 42, deterministic cuDNN, validation
  after every epoch on the scene-disjoint val splits, `best.pth` = max mean validation PSNR. **FP32 for all models**
  (`--amp off`). Each model uses its own published loss and its recipe rescaled to 5,600 steps.
- **Never read here:** LOL-v2 Test, SICE, LoLI-Street (only checked for integrity), and **SICE Part 2, which is not
  in the data archive at all (sealed)**.
- **Labels:** DarkIR-m = *training re-implemented from the paper (no official training code)*. HVI-CIDNet = *GPU
  training not bit-reproducible on this stack; resume accepted by a pre-registered control*.
- Code: bundle **{BUNDLE}** (MD5 `{BUNDLE_MD5}`), unchanged since the pilot. Upload it to
  `MyDrive/LowLight-Generalization/phase3_bundles/` if it is not there yet.
- **One model per session is recommended.** Run cells 1–4, then the next model's cell. Each model cell refuses to
  start unless the previous models in the approved order are complete, resumes automatically from `last.pth` after a
  disconnect, and refuses to retrain a completed model.
- **If a cell is stopped or Colab disconnects:** re-run cells 1–4 (cell 1 also checks that nothing is left running),
  then the same model cell; it resumes from the last completed epoch (at most one epoch is lost).
"""),
        _code("""
# 1. GPU (must be a Tesla T4, as in the pilot) + no leftover training/pilot processes
import torch, subprocess, os
print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| available', torch.cuda.is_available())
assert torch.cuda.is_available(), 'No GPU: Runtime -> Change runtime type -> T4 GPU'
gpu = torch.cuda.get_device_name(0); print(gpu)
assert 'T4' in gpu, f'GPU is {gpu}, not a Tesla T4: STOP (Tier A timings and comparability assume a T4)'
print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
left = subprocess.run(['pgrep', '-af', 'phase3_pilot.py|train_phase3.py|train_generalized.py'],
                      capture_output=True, text=True).stdout.strip()
assert not left, f'leftover pilot/training processes found (restart the runtime first):\\n{left}'
print('no leftover processes')
"""),
        _code(f"""
# 2. Drive, bundle and data (all MD5-verified; nothing on Drive is modified)
from google.colab import drive
drive.mount('/content/drive')
import hashlib, json, zipfile
DRIVE = '/content/drive/MyDrive/LowLight-Generalization'
BUNDLE = f'{{DRIVE}}/phase3_bundles/{BUNDLE}'
def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    return h.hexdigest()
assert md5(BUNDLE) == '{BUNDLE_MD5}', 'bundle MD5 mismatch: STOP'
if not os.path.isdir('/content/proj'):
    zipfile.ZipFile(BUNDLE).extractall('/content/proj')
man = json.load(open('/content/proj/BUNDLE_MANIFEST.json'))['files']
bad = [f for f, h in man.items() if md5(f'/content/proj/{{f}}') != h]
assert not bad, f'bundle file MD5 mismatch: {{bad[:5]}}: STOP'
print('bundle OK:', len(man), 'files')
DATA_ZIP = f'{{DRIVE}}/data_prepared/phase2.zip'
assert md5(DATA_ZIP) == '{DATA_ZIP_MD5}', 'phase2.zip MD5 mismatch: STOP'
if not os.path.exists('/content/phase2/manifest.json'):
    subprocess.run(['unzip', '-q', DATA_ZIP, '-d', '/content'], check=True)
assert not any('part2' in p.lower() for p in os.listdir('/content/phase2/SICE')), 'SICE Part 2 present: STOP'
print('data OK:', os.listdir('/content/phase2'))
"""),
        _code(f"""
# 3. Dependencies: einops 0.8.0 (approved) and the approved VGG-19 loss weights (HVI-CIDNet / DarkIR-m losses only;
#    torchvision checks the file hash on download; the MD5 must equal the file recorded in the pilot)
!pip -q install einops==0.8.0
import einops, torchvision
w = torchvision.models.VGG19_Weights.IMAGENET1K_V1
_ = torchvision.models.vgg19(weights=w)
vgg = os.path.expanduser('~/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth')
assert md5(vgg) == '{VGG_MD5}', 'VGG-19 weights differ from the pilot: STOP'
print('einops', einops.__version__, '| torchvision', torchvision.__version__, '| VGG-19 OK')
"""),
        _code("""
# 4. Tier-A helpers (define only; nothing starts here)
import signal, time
ORDER = ['retinexformer', 'hvi_cidnet', 'unet_expaug', 'darkir_m']
LABELS = {
    'retinexformer': 'Retinexformer (ICCV 2023), official recipe rescaled to 5,600 steps, FP32',
    'hvi_cidnet': 'HVI-CIDNet (CVPR 2025), official loss/recipe rescaled, FP32; GPU training NOT bit-reproducible on this '
                  'stack (resume accepted by pre-registered control, HVI_CIDNET_RESUME_CONTROL.md)',
    'unet_expaug': 'Phase-2 U-Net + input exposure augmentation (soft_shoulder), L1, Adam 2e-4, FP32',
    'darkir_m': 'DarkIR-m (CVPR 2025): training RE-IMPLEMENTED from the paper (no official training code), FP32'}
RUNS = f'{DRIVE}/experiments/phase3_model_selection/stage1/runs'
EPOCHS, STEPS = 50, 5600

def run_dir(m): return f'{RUNS}/tierA_{m}_s42'

def state(m):
    # 'complete' (50 epochs, not a pipeline test), 'partial' (last.pth exists), or 'not started'
    d = run_dir(m)
    s = f'{d}/logs/train_summary.json'
    if os.path.exists(s):
        j = json.load(open(s))
        if j.get('epochs_completed') == EPOCHS and j.get('optimizer_steps_total') == STEPS and not j.get('pipeline_test_mode'):
            return 'complete'
    if os.path.exists(f'{d}/checkpoints/last.pth'):
        return 'partial'
    return 'not started'

def sh(args, cwd='/content/proj', grace_s=60):
    # streams output; the child runs in its own process group; on ANY interruption the whole group gets SIGTERM, then
    # SIGKILL. last.pth (written atomically after every epoch) is untouched, so the run can resume.
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
                p.wait(timeout=wait_s); break
            except subprocess.TimeoutExpired:
                continue
        print(f'\\nINTERRUPTED: process group {p.pid} terminated; resume by re-running cells 1-4 and this cell')
        raise
    if rc != 0:
        raise RuntimeError(f'training process failed ({rc}); nothing is retried automatically')

def run_tierA(m):
    assert m in ORDER
    for prev in ORDER[:ORDER.index(m)]:
        assert state(prev) == 'complete', f'{prev} is {state(prev)}: the approved order requires it first. STOP'
    st = state(m)
    if st == 'complete':
        print(m, 'is already complete; not retraining.'); return report(m)
    left = subprocess.run(['pgrep', '-af', 'train_phase3.py|phase3_pilot.py|train_generalized.py'],
                          capture_output=True, text=True).stdout.strip()
    assert not left, f'another training process is running:\\n{left}'
    d = run_dir(m)
    os.makedirs(f'{d}/logs', exist_ok=True)
    label = f'{d}/RUN_LABEL.json'
    if not os.path.exists(label):
        json.dump({'model': m, 'label': LABELS[m], 'tier': 'A', 'seed': 42, 'precision': 'fp32',
                   'approved': '2026-10-19', 'bundle': BUNDLE}, open(label, 'w'), indent=1)
    sess = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%S'), 'state_at_start': st,
            'boot_id': open('/proc/sys/kernel/random/boot_id').read().strip(), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'nvidia_smi': subprocess.run(['nvidia-smi', '--query-gpu=driver_version,clocks.max.sm,power.limit',
                                          '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()}
    json.dump(sess, open(f"{d}/logs/session_{time.strftime('%Y%m%d_%H%M%S')}.json", 'w'), indent=1)
    args = ['python', 'scripts/train_phase3.py', '--config', f'configs/phase3/tierA_{m}.yaml', '--run-dir', d,
            '--data-root', '/content/phase2', '--device', 'cuda', '--amp', 'off']
    if st == 'partial':
        done = len(json.load(open(f'{d}/history.json'))) if os.path.exists(f'{d}/history.json') else '?'
        print(f'RESUMING {m} after epoch {done} from {d}/checkpoints/last.pth')
        args += ['--resume', f'{d}/checkpoints/last.pth']
    print(LABELS[m])
    sh(args)
    return report(m)

def report(m):
    d = run_dir(m)
    s = json.load(open(f'{d}/logs/train_summary.json'))
    ok = (s['epochs_completed'] == EPOCHS and s['optimizer_steps_total'] == STEPS and not s['pipeline_test_mode']
          and s['amp'] is False and os.path.exists(f'{d}/checkpoints/best.pth'))
    print(f"=== {m}: {'COMPLETE' if ok else 'CHECK FAILED'} | epochs {s['epochs_completed']} | steps "
          f"{s['optimizer_steps_total']} | FP32 {s['amp'] is False} | best epoch {s.get('best_epoch')} | best mean val "
          f"PSNR {s.get('best_val_psnr_mean')} (real {s.get('best_val_real_psnr')}, syn {s.get('best_val_synthetic_psnr')})")
    print('    label:', LABELS[m])
    return s

for m in ORDER:
    print(f'{m:14s} {state(m)}')
"""),
        _code("# 5. Retinexformer (first in the approved order; ~1.5 h on a T4 [estimate])\nrun_tierA('retinexformer');"),
        _code("# 6. HVI-CIDNet (~1.6 h [estimate]; not bit-reproducible on GPU, documented)\nrun_tierA('hvi_cidnet');"),
        _code("# 7. U-Net + exposure augmentation (~0.3 h [estimate])\nrun_tierA('unet_expaug');"),
        _code("# 8. DarkIR-m — training re-implemented from the paper (~1.6 h [estimate])\nrun_tierA('darkir_m');"),
        _code("""
# 9. Status of all Tier-A runs (read-only). Do NOT evaluate on test sets here: development export is a separate,
#    separately approved step.
for m in ORDER:
    print(f'{m:14s} {state(m)}')
    if state(m) == 'complete':
        report(m)
"""),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 0}


def main():
    out = ROOT / f"notebooks/colab_phase3_tierA_{time.strftime('%Y%m%d_%H%M')}.ipynb"
    if out.exists():
        sys.exit(f"{out} exists")
    out.write_text(json.dumps(notebook(), indent=1))
    print(out)


if __name__ == "__main__":
    main()
