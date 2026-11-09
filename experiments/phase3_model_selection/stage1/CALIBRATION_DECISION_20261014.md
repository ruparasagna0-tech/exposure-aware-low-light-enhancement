# Calibration reference change — explicitly approved by the project owner (2026-10-14)

**Status:** approved decision (Option 1). This is a change of the calibration **reference**, approved explicitly. It is
**not** a pass under the original September reference, which the pilot failed (ratio 1.38 vs ±30%).

## What stays unchanged
- September Phase-2 baseline (preserved as recorded): **15.1 s per epoch** (median of 50 epochs, range 14.2–16.2 s),
  **757.8 s** total for 50 epochs (5,600 steps + 50 validations), Colab Tesla T4, `torch 2.11.0+cu128`, Python 3.13.15.
  Source: `experiments/phase3_model_selection/stage0/source_phase2/history.json` (MD5 10debcea1c0fe82d41f4d8426dedfa4d).
- Calibration tolerance: **±30%** (`GATES["calibration_tolerance"] = 0.30` in `scripts/phase3_pilot.py`), unchanged.
- All integrity checks, datasets, splits, configs, model architectures, training settings and existing results.
- The original pilot result is kept as recorded: v2.1 pilot P1 vs September, ratio **1.38 → FAIL**
  (Drive `.../stage1/pilots/pilot_20261014_0026/setup/P1_calibration.json`).

## New reference (current Colab session, boot id 7fc505c0-c1c8-4088-abfa-20c19c8eebf0)
The **original, unmodified Phase-2 trainer** (`scripts/train_generalized.py`, MD5 41f8d805c9e5bfdc837cc9472de3f5c5),
Phase-2 config (MD5 7326edc089669cdd43fbfe92889769f8), same data / splits / FP32 / deterministic cuDNN / 2+2 batches /
2 workers / 112 steps per epoch, 3 epochs, throwaway weights (never evaluated), `torch 2.11.0+cu130` (CUDA 13.0,
cuDNN 9.27), Tesla T4. Drive: `.../stage1/pilots/phase2_trainer_timing_20261014_003632/summary.json`.

| Measurement (same session) | Steady epoch | Train | Validation | Projected 50 epochs |
|---|---|---|---|---|
| Original Phase-2 trainer (reference) | 20.55 s | ~13.39 s (derived from line timestamps) | ~7.15 s (derived) | 1,027.7 s |
| Pilot P1 (Phase-2-comparable method) | 20.87 s | 13.59 s | 7.28 s | 1,045.4 s |

**Re-anchored calibration:** pilot / reference = 20.87 / 20.55 = **1.016** on the steady epoch
(1,045.4 / 1,027.7 = 1.017 on the 50-epoch projection), i.e. agreement within **≈1.5–1.7%**, inside ±30% → **PASS
(re-anchored reference)**.

The reference run reproduced the September Phase-2 log exactly for all three epochs (train L1 0.1773 / 0.1548 / 0.1514;
validation PSNR real/syn 14.732/16.356, 14.594/16.494, 15.020/16.316; SSIM to 4 decimals), so the computation is
unchanged; only the speed differs.

## Environment finding (recorded, cause not proven)
- October / September slowdown of the original trainer: 20.55 / 15.1 ≈ **1.36×** per epoch (1,027.7 / 757.8 = 1.356 on
  50-epoch projections).
- Known difference: PyTorch build `+cu128` (September) vs `+cu130` (October); same PyTorch version 2.11.0 and Python.
  GPU clocks, driver and host CPU were not recorded in September, so **the root cause is not established**; the
  October values are now recorded in the reference `summary.json` for future comparison.

## Consequence for planning
- Practical runtime accepted by the project owner: **≈21 s per epoch** for the Phase-2 U-Net on the current Colab T4.
- All T4 time estimates (Tier A and later) use **current-session measurements** (pilot per-model rates), not the
  September figure. Earlier estimates derived from the September anchor (STAGE1_PLAN_T4.md section 5) are low by
  ≈1.36× and are superseded by the pilot's measured projections.
- Data-wait share of the original trainer was not measurable without modifying it; the pilot's same-session P1
  diagnostic was 1.7% (different code path, same data pipeline).

Machine-readable record: `CALIBRATION_DECISION_20261014.json` (same folder).

## Addendum 2026-10-18 — new Colab VM
- The session ended; the notebook now runs on a new VM (boot id f79b313b-6d4c-4100-a2a8-857faf42961d). The setup's
  same-VM MD5 reference no longer applies (model cells run the full MD5 check automatically), and the approved timing
  reference (20.55 s/epoch) was measured on the previous VM (boot id 7fc505c0-...).
- Decision (choice delegated by the project owner to the assistant): **Option 1** — repeat the same approved 3-epoch
  reference measurement with the original Phase-2 trainer on the new VM (`phase2_trainer_timing_cell_v2.py`; throwaway
  weights, no evaluation) before the remaining model checks, so every model's timing has a reference from its own VM.
  Tolerance ±30% and all earlier results unchanged.
- **Result (2026-10-18, new VM f79b313b-6d4c-4100-a2a8-857faf42961d):** original Phase-2 trainer, 3 epochs:
  22.9 / 20.7 / 21.4 s; steady median **21.05 s/epoch** (train ~13.68 s, validation ~7.37 s, derived); projected 50 epochs
  1,054.3 s (1.39x September). New-VM / previous-VM reference = 21.05 / 20.55 = **1.024**. This is the timing reference
  for the models measured on the new VM (Retinexformer onward); U-Net + exposure augmentation was measured on the
  previous VM with its reference 20.55 s.
- Note: the measurement was run with `phase2_trainer_timing_cell.py` (the v1 cell, pasted by mistake) instead of `_v2`.
  The measurement procedure is identical (same unmodified trainer, MD5 checks, timing method); consequences: the
  notebook's summary() function was overwritten (restored by re-running cell 4), and the field `pilot_P1_same_session`
  in that run's `summary.json` actually refers to the PREVIOUS VM's pilot P1 (20.87 s), so its `trainer_vs_pilot_epoch_ratio`
  (1.009) is a cross-VM ratio, not a same-session one. The Drive file is left unchanged; this note is the correction.
