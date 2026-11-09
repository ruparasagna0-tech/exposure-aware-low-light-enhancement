# Phase 3, Stage 0: baseline recovery, output export, colour evaluation

Approved scope (2026-10-13): recover the Phase-2 best checkpoint, export its enhanced images, reproduce the
recorded metrics, and run the colour / image-quality evaluation. **No training, no new model or loss, no
downloads of pretrained weights, no dataset changes.**

Evidence status: LOL-v2 Real/Synthetic Test are seen-domain test sets. **SICE and LoLI-Street Val are previously
evaluated and inspected development/evaluation evidence, not untouched final test sets.**

## Layout
```
stage0/
├── README.md                         this file
├── source_phase2/                    (pending) imported Drive files + provenance.json
│   ├── checkpoints/best.pth
│   ├── history.json, config.yaml, logs/
│   └── provenance.json               source paths, sizes, MD5/SHA-256, load + consistency checks
└── outputs/phase2_unet_best/         (pending)
    ├── <domain>/<stem>.png           lossless 8-bit RGB = exactly the tensors the metrics use
    ├── per_image/<domain>.csv        PSNR / SSIM / LPIPS recomputed
    ├── export_manifest.json          checkpoint MD5, code MD5s, versions, device, per-file MD5
    ├── reproduction_report.{json,md} recomputed vs recorded (tolerances fixed in advance)
    └── colour_eval/                  per_image CSVs, summary.json, summary.md
```

## New files (nothing existing was modified)
| File | Role |
|---|---|
| `scripts/stage0_import_phase2.py` | Copies best.pth / history / config / logs from a downloaded Drive folder or zip; never overwrites; MD5 + SHA-256; checks loading, parameter count, config, epoch, validation metrics vs the recorded results, best epoch in history |
| `scripts/export_outputs.py` | Same code path as `evaluate_generalized.py` (`build_test`, `evaluate_domain`, 8-bit quantization, tiled 512/64 SICE, separable SSIM, LPIPS-Alex) + saves every prediction as PNG; per-domain atomic folders; refuses to write in existing experiment folders; refuses to download LPIPS weights |
| `scripts/stage0_reproduction_report.py` | Compares recomputed vs recorded CSVs (read from the MD5-checked results zip). Tolerances: per image 0.05 dB / 5e-4 SSIM / 2e-3 LPIPS; per mean 0.005 dB / 5e-5 / 5e-4 |
| `src/metrics/colour_metrics.py` | sRGB→CIELAB (D65), CIEDE2000, pointwise colour stats (exact, row strips), edge/noise metrics |
| `scripts/evaluate_colour.py` | Colour/quality metrics for model, identity floor, fixed-target scaling floor; flags; summaries with bootstrap CIs (LoLI clustered by scene) |
| `tests/test_stage0.py` | 16 unittest tests (see below) |

## Metric definitions (colour_eval)
See the docstring of `src/metrics/colour_metrics.py`. Key choices: ΔE00 per pixel (mean, median, p95); ΔE00 with L*
equalised = colour-only error; ΔL* (+ = brighter than reference); chroma ratio (<1 desaturated); mean a*/b* offset
(cast); median hue error on pixels chromatic in both images (C*ab > 10); clipping fractions; Sobel edge correlation
and edge-energy ratio on the strongest 10% reference edges (<1 = blur); flat-region noise ratio on the flattest 30%
of the reference (>1 = added grain). Shares: over-exposed (ΔL* > 10), desaturated (< 0.8), over-saturated (> 1.25),
cast (|Δa*| or |Δb*| > 5); these thresholds are descriptive conventions, not perceptual standards.

## Limitations (recorded before results)
- References: SICE Labels are human-selected fusion results (Cai et al., TIP 2018); LoLI-Street lows are
  Photoshop-darkened (procedure undocumented). ΔE00 vs these partly measures reference style.
- SICE edge/noise metrics are computed after a 4x block-mean downscale (pointwise colour metrics are full resolution).
- Flagged pairs are kept and also summarised without the flag: reference darker than input (LOL-v2-synthetic
  anomalies) and information-limited inputs (>= 50% black-clipped, SICE).
- Single seed; LoLI-Street RLLT (real, unpaired) not part of Stage 0.
- Recorded metrics were computed on CUDA; this export runs on CPU, so tiny float differences are expected
  (handled by the pre-declared tolerances).

## Commands
```bash
# 1. import (after downloading the Drive folder experiments/generalized_lol_v2 as a folder or zip)
.venv/bin/python scripts/stage0_import_phase2.py --src ~/Downloads/<downloaded folder or zip>
# 2. export + recompute metrics (CPU; about 2.5-3 h, SICE dominates) [estimate]
.venv/bin/python scripts/export_outputs.py --checkpoint experiments/phase3_model_selection/stage0/source_phase2/checkpoints/best.pth
# 3. reproduction check (must pass before step 4 is interpreted)
.venv/bin/python scripts/stage0_reproduction_report.py
# 4. colour / quality evaluation
.venv/bin/python scripts/evaluate_colour.py
# tests
.venv/bin/python -m unittest tests.test_stage0 -v
```

## Stage 0 run (2026-10-14, Mac CPU) — COMPLETE
- Source: `~/Downloads/generalized_lol_v2-20261013T184517Z-1-001.zip` (single part; MD5 e75de8602d19a74241203f628be305e6,
  SHA-256 ea24f388…cf9279; `drive_zip_verification.json`). Imported best.pth MD5 73f71cf4d01c07b5f3176a6ed067b32f; all 8
  import checks passed (`source_phase2/provenance.json`). Drive `evaluation/` copy is byte-identical to the results zip; splits
  identical; run config differs only in `train.device`.
- Export (CPU, 8 threads): LOL-v2 real 54 s, synthetic 42 s, SICE 3,903 s, LoLI-Street 4,134 s (concurrent with SICE colour
  evaluation). 3,255 PNGs, 1.7 GB.
- Reproduction: **0 discrepancies** over 3,255 images / 7 subsets; max per-image |diff| 7.6e-6 dB PSNR, 3.9e-5 SSIM,
  1.0e-5 LPIPS (`outputs/phase2_unet_best/reproduction_report.md`).
- Colour evaluation: LOL-v2 2 min 06 s (2 workers), SICE 69 min (1 worker; peak swap 5.70 GB, min free memory 31%, watchdog
  thresholds never reached), LoLI-Street 11 min 48 s (6 workers). Results: `outputs/phase2_unet_best/colour_eval/summary.md`.
- Tests: 16/16 pass, 0 skipped.
- Incident: two helper wait-loops (watchdog, queue) matched their own command lines with `pgrep -f` and never exited; no
  colour process was running for about 6 min after SICE finished. Stopped manually; no data affected.
- Interpretation caveats: hue error is measured only where both images have C*ab > 10, so the identity floor (dark,
  low-chroma input) is evaluated on a different, smaller pixel set than the model; the fixed-scaling floor nearly inverts a
  global darkening, which favours it on synthetically darkened data (LoLI-Street).

## Verification before the checkpoint was available (2026-10-13)
- Tests: 16 run, 12 pass, 4 skipped (need the checkpoint / exports).
- End-to-end dry run in a scratch folder with a random-weight checkpoint (3 items per domain): import refuses
  re-import and correctly FAILS the fake validation metrics; export writes atomic per-domain folders, skips completed
  domains, refuses `experiments/generalized_lol_v2/`; the reproduction report flags every random-weight value;
  colour evaluation runs on all four domains.
- Cross-check: the identity-floor PSNR from `evaluate_colour.py` equals the independent Oct-10 diagnostic
  (`generalized_lol_v2_diagnostics/model_free/pair_statistics.csv`) to <= 0.0005 dB on full-resolution sets.
