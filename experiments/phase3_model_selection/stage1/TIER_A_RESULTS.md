# Tier A results (reported from Colab by the project owner; validation only — not test results)

Rule: `best.pth` = max mean validation PSNR over 50 epochs (5,600 steps), FP32, seed 42. Validation = scene-disjoint
LOL-v2 val splits (64 real + 92 synthetic), the same data used for checkpoint selection (so slightly optimistic).
Reference: Phase-2 U-Net best validation (same protocol): mean 18.391 dB (real 17.455, synthetic 19.328), epoch 45.

| Model | Date | Status | Best epoch | Mean val PSNR | Real | Synthetic | Note |
|---|---|---|---|---|---|---|---|
| Retinexformer | 2026-10-19 | COMPLETE (50 ep, 5,600 steps, FP32) | 50 (last) | 22.380 | 21.498 | 23.262 | best at the final epoch: likely still improving (Tier-B relevance) |
| HVI-CIDNet | 2026-10-19 | COMPLETE (50 ep, 5,600 steps, FP32) | 46 (epoch 50: 22.539) | 22.578 | 22.042 | 23.115 | not bit-reproducible on GPU; +0.20 dB vs Retinexformer is within expected run-to-run variation (single seed): no ranking claimed |
| U-Net + exposure aug | 2026-10-19 | COMPLETE (50 ep, 5,600 steps, FP32) | 45 | 18.112 | 17.291 | 18.933 | -0.28 dB vs Phase-2 U-Net on in-distribution validation (expected cost of a broader exposure distribution); its target (exposure robustness on LoLI light/moderate, SICE) is untested until development evaluation |

Progress demo (CPU, 2026-10-19): `progress_demo_20261019_1310/` — validation metrics incl. colour (ΔE00), training
curves, fixed-rule validation examples and LoLI-Street darkness-level panels (development data). Checkpoints from the
downloaded Drive zips (`tierA_runs_local/SOURCE_ZIPS.json`); reload check passed for all models (|Δ| < 1e-4 dB).
