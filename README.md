# Exposure-Aware Low-Light Image Enhancement

Research code for a B.Tech project (Amrita School of Computing, Bengaluru) on **low-light image enhancement (LLIE)**. The project:

- builds a leakage-controlled training and evaluation pipeline for LLIE;
- compares U-Net, Retinexformer and HVI-CIDNet under one protocol;
- diagnoses **exposure-dependent brightness miscalibration**: the output brightness of current LLIE models depends on how dark the input was, so mildly dark inputs are over-brightened and very dark inputs stay too dark.

> **Status (November 2026).** Stages 1–3 are complete. The Stage-4 exposure diagnostic is largely complete (preliminary). The proposed exposure-conditioned HVI-CIDNet (Stage 5) is **designed but not implemented or trained**. No results are claimed for it.

**Author:** Rupa Rasagna Kode ([@ruparasagna0-tech](https://github.com/ruparasagna0-tech))

---

## Key results (from the project's result files)

**U-Net (1.93 M params) trained on LOL-v2 Real + Synthetic, evaluated once:**

| Test set | PSNR | SSIM | LPIPS |
|---|---|---|---|
| LOL-v2 Real (seen) | 17.98 | 0.729 | 0.480 |
| LOL-v2 Synthetic (seen) | 18.91 | 0.740 | 0.420 |
| SICE (unseen) | 17.32 | 0.708 | 0.499 |
| LoLI-Street (unseen), overall | 15.77 | 0.678 | 0.491 |
| LoLI-Street, light level | 13.15 | — | — |

On the light level, the unprocessed input scores 13.92 dB, i.e. better than the model.

**Equal-budget comparison** (LOL-v2 scene-disjoint validation, 5,600 steps, seed 42, T4; mean PSNR):

| Model | Mean val PSNR |
|---|---|
| HVI-CIDNet | 22.58 dB |
| Retinexformer | 22.38 dB |
| U-Net | 18.39 dB |
| U-Net + exposure augmentation | 18.11 dB |

**Exposure diagnostic** (inference only; SICE Part 1, all 3,021 exposures; preliminary):
- For every learned checkpoint evaluated (ours and official), the brightness error grows with the input exposure (Spearman ρ = +0.41 to +0.97).
- On mildly under-exposed inputs, 36–96 % of outputs are worse than the input itself.

Details: `experiments/phase5_llie_direction/DIAGNOSTIC_FINDINGS.md`.

**Dataset audit:** all 500 LOL-v1 pairs are contained in LOL-v2 Real, so LOL-v1 is excluded from all generalised experiments.

---

## Repository layout

```
src/
  datasets/      LOL-v2, SICE, LoLI-Street loaders; scene split; balanced 2-domain sampler; exposure augmentation
  metrics/       PSNR, SSIM (incl. memory-bounded version for 24 MP), LPIPS, CIEDE2000 colour + structure metrics
  models/        U-Net baseline
  losses/        L1 reconstruction
  utils/         tiled inference (512 / 64 overlap)
  phase3/        common wrappers, official losses, schedules and trainer for Retinexformer / HVI-CIDNet / DarkIR-m
  phase5/        diagnostic model zoo (MD5-checked), minimal safetensors reader
scripts/         data preparation, training, evaluation, colour evaluation, diagnostic, Colab bundle/notebook builders
configs/         YAML configs (Stage 1–2 U-Net, Stage 3 Tier A models)
notebooks/       Colab notebooks (training, pilot, Tier A, Phase 5 diagnostic)
tests/           unittest suites
third_party/     vendored official model code (HVI-CIDNet, Retinexformer, DarkIR, IAT) with LICENSE and PATCHES.diff
third_party_src/ download manifest + original IAT files
experiments/     plans, results and findings (Markdown), scene-disjoint split files, dataset audit code
docs/            DATA.md (datasets), WEIGHTS.md (weights), weights_sources.json (checksums)
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional: pip install -r requirements-optional.txt   (see comments; pyiqa is used on Colab only)
```

**Software versions used.** These are the versions recorded in the run logs; the code may work with others, but they are not tested.

| Where | Python | PyTorch | Used for |
|---|---|---|---|
| Local Mac (CPU) | 3.12.5 | 2.14.0 | development, unit tests, CPU dry runs, data preparation |
| Colab T4, Stage 2 (Sep–Oct 2026) | 3.13.15 | 2.11.0+cu128 | U-Net training and one-time evaluation |
| Colab T4, Stage 3 (Oct 2026) | 3.13.15 | 2.11.0+cu130 | Tier-A training (torchvision 0.26.0, NumPy 2.1.3) |
| Colab T4, Stage 4 diagnostic (Nov 2026) | 3.13 | Colab default at run time; each session writes its exact versions to `session_records/` on Drive | inference only; `pyiqa==0.1.16` installed with core packages pinned |

No API keys or secrets are needed. `.env.example` documents the single optional environment variable.

### Google Drive layout assumed by the Colab notebooks

The notebooks mount Google Drive and expect the project at `MyDrive/LowLight-Generalization/`. To use another location, edit `DRIVE = ...` (Stage 3 and Stage 4 notebooks) or `PROJECT_DIR = ...` (Stage 1–2 notebooks) in the setup cell.

| Drive path | Contents |
|---|---|
| `MyDrive/LowLight-Generalization/data_prepared/phase2.zip` | Stages 2–3 data (from `prepare_phase2_data.py --zip`) |
| `MyDrive/LowLight-Generalization/experiments/...` | Run folders, checkpoints, logs and reports written by the notebooks |
| `MyDrive/LowLight-Generalization/phase5_inputs/` | Stage 4: code bundle, weights pack, data pack (from `make_phase5_diag_bundle.py` / `prepare_phase5_diag_data.py`) |

Each notebook checks the MD5 of its bundle and inputs before running. Rebuild the bundle with the matching `make_*` script after any code change; the notebook embeds the new checksum.

**Data:** see [docs/DATA.md](docs/DATA.md). Datasets are not included.
**Weights:** see [docs/WEIGHTS.md](docs/WEIGHTS.md). Official weights are downloaded at pinned revisions and MD5-checked; our checkpoints are reproducible from the configs.

---

## Usage

All commands are run from the repository root.

### Data preparation and splits

```bash
python scripts/prepare_phase2_data.py --zip          # writes data_prepared/phase2 (+ .zip for Colab)
python scripts/make_phase2_splits.py                 # regenerates / verifies the scene-disjoint splits
```

### Stage 2: generalised U-Net

```bash
python scripts/smoke_test_generalized.py --device cpu
python scripts/train_generalized.py --data-root data_prepared/phase2 --device cuda
python scripts/evaluate_generalized.py --data-root data_prepared/phase2 --device cuda   # runs once; refuses a second run
```

### Stage 3: equal-budget models

Colab, T4. Use `notebooks/colab_phase3_tierA_*.ipynb`, or:

```bash
python scripts/train_phase3.py --config configs/phase3/tierA_hvi_cidnet.yaml --data-root /content/phase2 \
    --run-dir <run dir> --device cuda --amp off
```

### Stage 4: exposure diagnostic (inference only)

```bash
python scripts/prepare_phase5_diag_data.py --tmp <scratch dir>   # SICE Part 1 + LoLI-Street pack
python scripts/make_phase5_diag_bundle.py                        # code + weights bundles and Colab notebook
# then run notebooks/colab_phase5_diag_*.ipynb on a T4, or directly:
python scripts/phase5_diagnostic.py run    --data <pack> --weights <weights dir> --out <out dir>
python scripts/phase5_diagnostic.py report --data <pack> --out <out dir>
```

---

## Tests

```bash
python -m unittest discover -s tests
```

In a fresh clone (no datasets, weights or upstream clones), the full suite runs 72 tests: **55 pass, 16 are skipped, 1 is an expected failure, 0 fail** (recorded 2026-11-09, Python 3.12.5, CPU). The result is the same whatever the test order.

**Skipped tests** each print the reason and what to provide:

| Required material | Skipped tests |
|---|---|
| Prepared data `data_prepared/phase2` (run `prepare_phase2_data.py`, see docs/DATA.md) | `test_stage0`: 3 colour-metric tests on real images and the prepared-manifest test; `test_phase3` resume test; `test_phase3_pilot_v2` full-integrity test |
| Upstream clones at the pinned commits in `third_party_src/` (see `third_party/README.md`) | `test_phase3.TestVendoring`: pinned commits, recorded patches, byte identity |
| Our checkpoints and official weights in `models/phase5_diag_weights` (see docs/WEIGHTS.md) | `test_phase5_diag`: weight loading, CPU dry run |
| Exported Stage-0 outputs, imported Stage-2 checkpoint | `test_stage0.TestExportedOutputs`, checkpoint test |
| Authors' local integrity snapshot (not distributed) | `test_stage0` existing-assets test |

**Expected failure (documented known limitation):** `test_phase3_interrupts.TestRunGuarded.test_process_sigterm_with_other_threads_still_saves_report`.
- In a multi-threaded process, blocking SIGTERM in the main thread does not defer a *process-directed* SIGTERM: it is delivered to another thread, and Python runs the handler right away.
- So the Stage-3 pilot guard (`scripts/phase3_pilot.py::run_guarded`) can raise `KeyboardInterrupt` before it saves the "interrupted" report.
- A real pilot always has extra threads (PyTorch). The pilot code is kept exactly as it was used for the recorded Stage-3 runs; this test documents the race instead of hiding it.
- Its companion test (`test_worker_error_while_sigterm_pending_is_interrupted`) now sends the signal to the main thread (`signal.pthread_kill`), the "pending, not yet handled" case it describes. Before this change it depended on test order: it failed whenever an earlier test had started PyTorch threads.

---

## Reproducibility and evaluation rules

- **Scene-disjoint validation** for LOL-v2. Split files are in `experiments/generalized_lol_v2/splits/`.
- **One-time evaluation** of each selected checkpoint. Test sets are never used for training or selection.
- **SICE Part 2 is sealed** for a single final evaluation (see docs/DATA.md).
- **Integrity checks:** code bundles, data packs and weights are checksummed; results are written atomically and once.
- **Single runs (seed 42):** differences below about 0.3 dB are not interpreted.

---

## Related work and novelty

- A related recent method, **InterLight** (arXiv 2605.19982), already conditions an HVI-based network on the input.
- The intended contribution of this project is narrow:
  - an exposure-stratified analysis of current LLIE checkpoints;
  - a minimal, identity-initialised exposure-conditioning plug-in for an existing HVI-CIDNet model, still to be implemented and validated.
- No state-of-the-art or "first" claims are made.

---

## Licences and credits

- **Project code:** no licence has been chosen yet. Until one is added, the author's code in this repository is not licensed for reuse (default copyright applies); please contact the author via GitHub for permission.
- **Vendored code** (`third_party/`):
  - HVI-CIDNet, Retinexformer, DarkIR: MIT;
  - IAT: Apache-2.0.
  - Each folder keeps its LICENSE and a PATCHES.diff of the small compatibility changes.
- **Datasets and weights** are subject to their own licences and are not redistributed here.

**Citations:**
- HVI-CIDNet: Yan et al., CVPR 2025.
- Retinexformer: Cai et al., ICCV 2023.
- DarkIR: Feijoo et al., CVPR 2025.
- IAT: Cui et al., BMVC 2022.
- LOL: Wei et al., BMVC 2018.
- LOL-v2: Yang et al., TIP 2021.
- SICE: Cai et al., TIP 2018.
- LoLI-Street: Islam et al., ACCV 2024.

Full references are in the project report, which is not included in this repository.

The original project also contained an exploratory study on low-light pedestrian detection. It is not part of this repository.
