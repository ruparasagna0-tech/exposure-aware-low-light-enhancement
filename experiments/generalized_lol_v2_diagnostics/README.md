# Phase-2 failure-case diagnostic (generalized_lol_v2, best.pth epoch 45)

Read-only diagnostic. Nothing in `experiments/generalized_lol_v2/` or the evaluation results was changed.
Source metrics: `~/Downloads/phase2_evaluation_results.zip` (per_image CSVs, sice_evaluation_manifest.json).
Data: `data_prepared/phase2` (paths verified; every pair listed in `model_free/pair_statistics.csv`).

**Status:** model-free part done. Model outputs for the target cases are still pending because `best.pth` is on Google Drive only
(`/content/drive/MyDrive/LowLight-Generalization/experiments/generalized_lol_v2/checkpoints/best.pth`).

## Contents
- `model_free/pair_statistics.csv`: per test pair (3,255 pairs): input/reference mean luma, ref/input ratio, saturated and
  black-clipped fractions, colour ratios, identity PSNR (input as output), oracle-gain PSNR and oracle per-channel-LUT PSNR
  (both fit **with** the reference, so they are upper bounds and not methods), and the model's PSNR/SSIM/LPIPS from the CSVs.
  SICE is computed at 1/4 resolution.
- `model_free/fixed_target_scale_baseline.csv`: a non-oracle baseline that scales the input so its mean luma is 0.48 (the mean
  reference luma of the LOL-v2 **train** sets; no test reference is used).
- `model_free/panels/*.png`: input | input scaled to 0.48 | reference | error map | luma histograms.

## Domain-level evidence
| domain | model PSNR | identity | fixed-target scale | oracle gain | oracle LUT | input mean luma | ref/in ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| LOL-v2-real | 17.98 | 9.72 | 14.78 | 18.67 | 28.75 | 0.057 | 8.4 |
| LOL-v2-synthetic | 18.91 | 11.22 | 14.43 | 14.11 | 30.68 | 0.215 | 2.8 |
| SICE | 17.32 | 7.51 | 13.61 | 11.72 | 24.28 | 0.086 | 7.0 |
| LoLI dense | 18.20 | 6.30 | 18.96 | 22.05 | 27.05 | 0.125 | 4.6 |
| LoLI moderate | 15.96 | 8.80 | **19.93** | 25.21 | 28.66 | 0.235 | 2.4 |
| LoLI light | 13.15 | **13.92** | **20.67** | 29.43 | 31.21 | 0.385 | 1.5 |

- LoLI references are byte-identical across the three levels (999/999 ids), but PSNR falls light < moderate < dense in
  99.6% of ids. The **less dark** the input, the **worse** the model is. On "light" inputs the model scores below the
  unmodified input (13.15 vs 13.92 dB), and the fixed-target scaling beats it on 97% of images.
- LOL-v2-real: model PSNR correlates with ref/in ratio (r = +0.69). Images that need less brightening score worse.

## Target cases
| case | model PSNR | identity | fixed-target scale | finding |
|---|---:|---:|---:|---|
| `low00730.png` (real; also 00729) | 11.65 | 11.86 | 21.89 | Input luma 0.141 is about 3x the domain median (0.047) and needs only 2.7x gain (domain about 7x). The oracle LUT reaches 36 dB, so the mapping is a simple global curve. The model mis-sets global exposure. |
| `r01058910t.png` (synthetic) | 9.14 | **26.53** | 22.32 | The "low" image is **brighter than its reference** (0.453 vs 0.416, ratio 0.92). Doing nothing gives 26.5 dB. The model brightens unconditionally. Also `r1694427et` (identity 17.2 vs model 10.0) and `r15919202t` (20.6 vs 12.3). |
| SICE `seq46` (also seq39, 77, 60) | 8.27 | 6.50 | 6.98 | 92% of input pixels are black-clipped (max channel <= 3/255) at a ratio of 17.5x. The oracle LUT reaches only 18.7 dB, so the input is **information-limited**. Across SICE, PSNR vs black-clipped fraction is r = -0.71. The selection rule (darkest exposure with mean >= 5) picks 2.JPG at mean 7.0 while the -1 EV frame is at 44.7. |
| LoLI `light_30231.jpg` | 10.70 | 15.42 | 23.54 | Mildly dark input (0.289, ratio 1.5). The same scene's dense level scores 17.9 dB. This is the same exposure-adaptation failure as above, at its most extreme. |

Separate pattern (not exposure): `low00726-728` (grass texture) have the worst SSIM (0.52) and LPIPS (0.80) in LOL-v2-real
at ordinary exposure. The likely cause is high-frequency texture or noise loss. This needs model outputs to confirm.

## Open (needs best.pth, CPU inference)
Confirm the **direction** of the exposure error (over- vs under-exposure) and the colour shift on the target cases. Measure
output mean luma vs reference across LoLI levels. Check for tile-seam artefacts at 512/64 on SICE (a faint grid is visible
in the existing seq49 panel).
