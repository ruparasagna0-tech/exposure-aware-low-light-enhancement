# Phase 3 / Stage 1: pilot preparation (Mac) — status 2026-10-14

Approved scope: Mac code preparation + one Colab T4 feasibility pilot. **No Tier-A training. Sealed SICE Part2 untouched.**

## Downloads (as approved)
- Official code at pinned commits (`third_party_src/`, manifest `third_party_src/DOWNLOAD_MANIFEST.md`): Retinexformer
  1e9a0ef…, HVI-CIDNet eb43d7d…, DarkIR 0910291…; all MIT; no weight files in any repository.
- `einops==0.8.0` (approved), in `requirements-phase3.txt` (existing `requirements.txt` unchanged).
- Weights: none downloaded on the Mac. The Colab pilot fetches torchvision `VGG19_Weights.IMAGENET1K_V1`
  (hash-checked) for the HVI-CIDNet / DarkIR training losses only, and records its MD5 (`vgg19_weights_used.json`).

## New files
`third_party/` (vendored + LICENSE + PATCHES.diff + README), `src/phase3/{models,losses,schedules,engine}.py`,
`configs/phase3/tierA_{unet_expaug,retinexformer,hvi_cidnet,darkir_m}.yaml`, `scripts/train_phase3.py`,
`scripts/phase3_pilot.py`, `scripts/make_phase3_bundle.py`, `tests/test_phase3.py`,
`stage1/bundle/phase3_pilot_bundle_20261014_0422.zip` (MD5 4743f61a07ae6007e504e18adb6a0779),
`notebooks/colab_phase3_t4_pilot_20261014_0422.ipynb`.

## Verification on the Mac
- Tests: `tests.test_phase3` 17/17 and `tests.test_stage0` 16/16 pass (33 total; includes "existing assets unchanged").
- Findings recorded as upstream properties: HVI-CIDNet `I_LCA5` never receives gradients (output overwritten upstream);
  paper's 1.88 M = 1,975,569 / 2^20; DarkIR zero-initialised residual scales; DarkIR frequency loss is dead code upstream.
- CPU dry run of `scripts/phase3_pilot.py` (6 steps per model, 600x900 stand-in for the 20 MP tile test, random-init VGG;
  numbers are NOT T4 measurements and are not used):

| Model | Status | Params | Loss / eval range | FP32 islands | Val + tiled inference | Resume (CPU) |
|---|---|---|---|---|---|---|
| unet_expaug | complete | 1,925,667 ✓ | finite / [0,1] ✓; expaug: references identical, 60% augmented ✓ (after fixing a pilot-code bug) | — | ✓ | **pass** (bitwise identical) |
| retinexformer | complete | 1,605,701 ✓ | finite / [0,1] ✓ | — | ✓ | **pass** (bitwise identical) |
| hvi_cidnet | complete | 1,975,569 ✓ | finite (8 terms) / [0,1] ✓ | HVI transform ✓ (unit test) | ✓ | skipped on CPU (by approval) |
| darkir_m | complete | 3,321,638 ✓ | finite (4 terms) / [0,1] ✓ | 8 FreMLP blocks ✓ (unit test) | ✓ | skipped on CPU (by approval) |

P1 calibration "fails" on CPU by design (the gate is defined against the T4). Resume is **mandatory for every model on
the T4** (not skippable in the notebook).

Bug found and fixed by the dry run: the exposure-augmentation check interleaved two data loaders that share the global RNG
(num_workers=0), so all crops differed. Fixed by materialising each stream first; verified (identical references, 60/100
augmented).

## Pilot v1 run on Colab (`pilot_20261013_2309`, bundle `phase3_pilot_bundle_20261014_0422`) and v2 fixes
Observed (user-inspected, files on Drive): setup P0 full MD5 passed; U-Net + exposure augmentation pilot completed;
Retinexformer / HVI-CIDNet have no completed outputs in the notebook; DarkIR-m cell ended with `KeyboardInterrupt`
(resume-test folders saved, no `pilot_report.json`); P1 ratio 1.506 (projection 1,141.6 s vs 757.8 s), cause open.

Diagnosis and fixes (v2; the v1 bundle and notebook stay unchanged as the record of what ran):
1. **Notebook helper** `sh()` did not stop its child on interrupt, so the pilot (and its resume-test subprocesses /
   DataLoader workers) could keep running or block holding the GPU. v2 runs the child in its own process group and on
   interrupt sends SIGTERM to the group, then SIGKILL after a grace period; cell 1 refuses to start if a pilot process is
   left over. The pilot turns SIGTERM into an orderly stop and writes `pilot_report.json` with `status: interrupted`
   and the section in progress; reports are also written after every section.
2. **Integrity wording**: `integrity_md5_verified=false` in model reports was by design (`--p0-no-md5`) but was not
   documented and not enforced. v2 records `integrity_scope` (`full` / `structural`), a description,
   `integrity_md5_verified_in_this_run`, and `integrity_md5_reference` (path, time, boot id, number of files verified);
   the shortcut requires a passed full check from the same VM session.
3. **Calibration**: v1 compared 5,600 x (per-step time with explicit GPU syncs, warm-up excluded) + 50 x (ONE validation
   pass, the first full-resolution pass of the process) against Phase-2 totals that measure whole epochs. Precision,
   model, loss, batch, workers and determinism match; the timing method and the validation measurement do not. v2 times
   whole epochs exactly as Phase 2 did and splits train/validation; the gate stays at ±30%. The v1 numbers remain as
   recorded; the cause of 1.506 is to be established from the v2 measurement (environment differences remain possible).

## v2.1 (interruption reporting) — bundle `phase3_pilot_bundle_v2_1_20261014_0545` (MD5 a4d8add3d686db92168e223be5423231)
- Race fix: a failure is recorded as `interrupted` (not `error`) when an interrupt was handled OR a SIGTERM is still
  pending (`signal.sigpending`), e.g. when a DataLoader worker killed by the same process-group SIGTERM surfaces first.
  SIGTERM is blocked while the status is classified and written. Genuine failures (no interrupt in flight, e.g. a worker
  killed by the OOM killer) stay `error` and the pilot continues with the next model.
- Explicit `section_in_progress` (P2, P3_P4, max_batch, P5, P6, P7) instead of inferring it from missing keys.
- `pilot_run_status.json` per invocation: stages P0 / P1 / each model = running / complete / interrupted / failed /
  error, `stopped_in_stage` (including "after <stage>" for a stop between models), outcome.
- Notebook: summaries describe the LATEST attempt folder of a model (v2 could show an older attempt's report if the
  newest attempt stopped before writing one); new cell 9a lists every attempt; model cells warn if setup is incomplete.
- Tests: `tests/test_phase3_interrupts.py` (13; includes a real DataLoader + process-group SIGTERM test, no model).

## Calibration reference change (approved 2026-10-14) — see `CALIBRATION_DECISION_20261014.md`
- v2.1 pilot on Colab (`pilot_20261014_0026`): P0 full integrity passed (10,196 files MD5-verified). P1 vs the
  September reference: ratio **1.38 → FAIL** (kept as recorded).
- Approved timing check: the original, unmodified Phase-2 trainer in the same session ran at **20.55 s/epoch**
  (September: 15.1 s) and reproduced the September log exactly; the pilot P1 method agrees with it within ≈1.5%.
- **Explicitly approved reference change (Option 1):** calibration reference = original Phase-2 trainer measured in the
  current session; tolerance ±30% unchanged; re-anchored ratio 1.016 → pass. This is not a pass under the September
  reference. October/September slowdown ≈1.36×, root cause not proven. Accepted practical runtime ≈21 s/epoch;
  all T4 estimates use current-session measurements.
