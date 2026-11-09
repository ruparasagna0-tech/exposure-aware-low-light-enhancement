# Phase 3 progress demo — validation metrics (NOT a final evaluation)

LOL-v2 validation (64 real + 92 synthetic images; also used for checkpoint selection, so slightly optimistic). One training run (seed 42) per model, 5,600 steps, FP32. Outputs quantized to 8 bit. ΔE00 = CIEDE2000 colour difference (lower is better); colour-only = with brightness equalised; ΔL* > 0 = brighter than the reference.

| Model | Subset | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | ΔE00 ↓ | ΔE00 colour-only ↓ | ΔL* | chroma ratio |
|---|---|---|---|---|---|---|---|---|
| U-Net (Phase 2) | real | 17.45 | 0.7443 | 0.4774 | 13.87 | 7.52 | -6.84 | 1.32 |
| U-Net (Phase 2) | synthetic | 19.33 | 0.7651 | 0.3832 | 10.90 | 6.63 | +3.02 | 0.80 |
| U-Net (Phase 2) | all 156 (image-weighted) | 18.56 | 0.7566 | 0.4219 | 12.12 | 7.00 | -1.02 | 1.01 |
| U-Net + exp. aug | real | 17.29 | 0.7453 | 0.4713 | 14.24 | 7.24 | -7.57 | 1.28 |
| U-Net + exp. aug | synthetic | 18.93 | 0.7649 | 0.3888 | 11.27 | 6.74 | +1.52 | 0.83 |
| U-Net + exp. aug | all 156 (image-weighted) | 18.26 | 0.7568 | 0.4226 | 12.49 | 6.94 | -2.21 | 1.02 |
| Retinexformer | real | 21.50 | 0.7577 | 0.3372 | 9.31 | 5.45 | -1.81 | 1.05 |
| Retinexformer | synthetic | 23.26 | 0.9012 | 0.1002 | 7.03 | 3.80 | +0.31 | 1.00 |
| Retinexformer | all 156 (image-weighted) | 22.54 | 0.8423 | 0.1974 | 7.96 | 4.48 | -0.56 | 1.02 |
| HVI-CIDNet | real | 22.04 | 0.7823 | 0.3499 | 8.36 | 4.71 | -0.27 | 0.85 |
| HVI-CIDNet | synthetic | 23.11 | 0.9017 | 0.1056 | 6.62 | 2.90 | +0.98 | 0.96 |
| HVI-CIDNet | all 156 (image-weighted) | 22.67 | 0.8527 | 0.2058 | 7.33 | 3.64 | +0.47 | 0.92 |

**Selection metric** (mean of the real and synthetic validation PSNR, as used for checkpoint selection):

- U-Net (Phase 2): 18.39 dB (best epoch 45)
- U-Net + exp. aug: 18.11 dB (best epoch 45)
- Retinexformer: 22.38 dB (best epoch 50)
- HVI-CIDNet: 22.58 dB (best epoch 46)

Checkpoint reload check (recomputed vs stored validation PSNR, tolerance 0.05 dB): PASS
Figures: validation_examples.png, training_curves_val_psnr.png, LoLI-Street darkness-level panels for scenes [30231, 30867] (development data)
CPU time: 694 s
