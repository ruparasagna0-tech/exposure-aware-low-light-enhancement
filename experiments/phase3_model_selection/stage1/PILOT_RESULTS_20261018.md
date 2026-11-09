# Phase 3 / Stage 1 — T4 feasibility pilot results (v2.1)

Pilot folder (Drive): `experiments/phase3_model_selection/stage1/pilots/pilot_20261014_0026/` plus
`phase2_trainer_timing_20261014_003632/`, the 2026-10-18 timing folder (new VM) and
`hvi_cidnet_resume_control_20261018_192219/`. Throwaway weights only; nothing evaluated; no test set, SICE Part 2 sealed.
Environment: Colab Tesla T4 (14.56 GB), torch 2.11.0+cu130, CUDA 13.0, cuDNN 9.27, 2 vCPUs (Xeon @ 2.00 GHz).
Two VMs: 7fc505c0... (2026-10-14: setup, U-Net+aug, first Retinexformer run) and f79b313b... (2026-10-18: Retinexformer,
HVI-CIDNet, DarkIR-m). Timing references (original Phase-2 trainer): 20.55 s/epoch and 21.05 s/epoch respectively
(approved re-anchored calibration; see `CALIBRATION_DECISION_20261014.md`). Integrity: full per-file MD5 (10,196 files)
passed on both VMs. All interruptions were recorded cleanly (attempt log: 8 attempts, 2 interrupted, 0 ambiguous).

| Model | Feasible? | Precision (pilot gate) | s/step | Peak GB (batch 4) | Val pass | 20 MP tiled | Resume | Tier A est. | Dev export est. |
|---|---|---|---|---|---|---|---|---|---|
| U-Net + exposure aug | yes | FP16 passes (2.12x, 0.03%) | 0.056 (FP32 0.118) | 0.31 (FP32 1.01) | 7.3 s | 8.0 s | bitwise | 0.19 h (FP32 ~0.29 h) | 0.17 h |
| Retinexformer | yes | FP32 (FP16 loss diff 5.7% > 2%) | 0.705 | 5.84 | 21.0 s | 23.9 s | bitwise | 1.39 h | 0.51 h |
| HVI-CIDNet | yes, with documented resume exception | FP32 (FP16 loss diff 20.3%, 17 skipped) | 0.884 | 5.01 | 14.1 s | 14.9 s | gate FAIL (non-determinism); accepted by pre-registered control | 1.57 h | 0.33 h |
| DarkIR-m (optional) | yes (training loop rebuilt from the paper) | FP16 passes (2.12x, 0.02%) | 0.409 (FP32 0.865) | 3.43 (FP32 4.99) | 13.6 s | 14.6 s | 9e-7 (not bitwise) | 0.83 h (FP32 ~1.53 h) | 0.32 h |

Correctness (all models): parameter counts match the papers/conventions exactly; finite losses with all loss terms;
eval output in [0, 1]; HVI and FFT FP32 islands active; exposure augmentation verified (references unchanged, 60% of
inputs augmented). VGG-19 loss weights used: `vgg19-dcbb9e9d.pth`, 574,673,361 bytes, MD5 92881fe292bd7d2408ecff58a101fd03.

Estimates are projections from 300 measured steps (5,600 steps + 50 validations + checkpoint writes); dev-export
estimates exclude CPU-bound PNG encoding / LPIPS. They are not Tier-A results.

Open decisions before Tier A (none taken): precision policy (mixed vs FP32 for all), DarkIR-m inclusion, seeds.
