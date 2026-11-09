# Phase 2 — `generalized_lol_v2`: first generalized low-light enhancement experiment

## 1. Research objective
Measure the effect of **multi-domain training alone** on cross-dataset generalization: the
*unchanged* LOL-v1 baseline U-Net and L1 loss are trained on two training domains with explicit
50/50 batches, then evaluated on seen domains and on two held-out unseen domains.

## 2. Datasets and 3. roles

| Dataset | Role | Data used | Count |
|---|---|---|---:|
| LOL-v2-real (Real_captured) | training domain | official Train minus val scenes | 625 pairs / 223 scenes |
| | validation (checkpoint selection) | 25 scene-disjoint scenes of official Train | 64 pairs |
| | **seen-domain test** | official Test | 100 pairs |
| LOL-v2-synthetic | training domain | official Train minus val scenes | 808 pairs / 802 scenes |
| | validation (checkpoint selection) | 89 scene-disjoint scenes of official Train | 92 pairs |
| | **seen-domain test** | official Test | 100 pairs |
| SICE (Dataset_Part1) | **unseen-domain test only** | 58 official test sequences (`testing index.txt`) | 58 sequences |
| LoLI-Street | **unseen-domain test only** | official Val, all 3 darkness levels, id 30727 excluded | 2,997 pairs (999 ids x 3) |
| LOL-v1 | **not used** (legacy benchmark of Phase 1 only) | — | — |

LoLI-Street Test (1,000 unpaired images) and the LoLI-Street YOLO annotations are not used in Phase 2.

## 4. Leakage considerations
- **LOL-v1 was excluded from generalized training because the dataset overlap audit showed substantial
  scene/image overlap with LOL-v2-real** (`experiments/dataset_audit/overlap_audit.txt`: all 500 LOL-v1
  pairs occur in LOL-v2-real, 12 of 15 eval15 inputs in LOL-v2-real Train). Consequently LOL-v1 eval15 is
  also not a valid test set for Phase-2 models.
- LOL-v2-real official Train/Test are scene-disjoint (audit). Split-time check: max val↔test thumbnail
  correlation 0.671 (real), 0.899 (synthetic); 0 identical files. One synthetic Train↔Test pair reaches
  0.952 (`r07ece355t` ↔ `r0905697bt`); visually different scenes (similar layout) — not leakage.
- SICE and LoLI-Street: no exact or near-duplicate overlap with LOL-v1/v2 (audits); they are never read
  by the training script. Pre-training integrity checks verify that no training/val file is identical to
  any SICE/LoLI-Street file and that all training paths lie under `LOL-v2/`.

## 5. Scene-level splitting
Filename ids are not assumed to be scenes. Scenes = connected components of: pixel-identical low **or**
reference image, or brightness-normalised grayscale thumbnail correlation ≥ threshold (low–low or
reference–reference). LOL-v2-real: threshold **0.85**, 96x64 thumbnails → 248 scenes. LOL-v2-synthetic:
threshold **0.95**, 64x64 → 891 scenes (lower thresholds chain unrelated images). Validation scenes are
`random.Random(42).sample` over scenes ordered by first member: **25** (real) and **89** (synthetic).
Exact membership: `splits/lol_v2_real_split.json`, `splits/lol_v2_synthetic_split.json`
(`scripts/make_phase2_splits.py` regenerates and verifies them bit-identically).

## 6. Balanced sampling
`src/datasets/balanced_sampler.py`: every batch of 4 = **2 LOL-v2-real + 2 LOL-v2-synthetic** (never a
concatenation with proportional sampling). Exactly **112 batches per epoch** → 224 samples per domain per
epoch. Within a domain, samples are drawn without replacement from seeded permutations
(`SeedSequence([42, domain, cycle])`), continuing across epochs (real: one full pass ≈ 2.8 epochs,
synthetic ≈ 3.6 epochs) — i.e. real data is intentionally oversampled relative to proportional sampling.
Batches are regenerated per epoch, so resuming reproduces the same stream. The training loop aborts if a
batch is not [real, real, synthetic, synthetic] or an epoch does not produce exactly 112 steps, and logs
per-domain sample counts and per-domain training L1 every epoch.

## 7. Model and 8. loss
Identical to the LOL-v1 baseline: `BaselineUNet` (base 32, depth 3, 1,925,667 parameters, sigmoid
output in [0,1]); L1 reconstruction loss only.

## 9. Preprocessing
RGB, [0,1]. Training: paired random 256x256 crop + paired horizontal flip (p=0.5), no vertical flip.
Validation/test: deterministic, full resolution, no resizing. SICE: native resolution (2–24 MP) with
**tiled inference**, tile 512, overlap 64, linear blending (`src/utils/tiling.py`).

## 10. Metrics
PSNR and SSIM as in the baseline (`src/metrics/image_metrics.py`: [0,1] RGB, data range 1, Gaussian
11x11 σ=1.5 SSIM; final test on 8-bit-quantized outputs). **LPIPS** (`lpips` 0.1.4, `net="alex"`,
`normalize=True` maps [0,1] inputs to [-1,1]), whole image at the same resolution. Reported **separately**
per domain: LOL-v2-real Test, LOL-v2-synthetic Test, SICE, LoLI-Street Val overall / dense / moderate / light
(n, mean PSNR, SSIM, LPIPS).

For SICE (20–24 MP), SSIM is computed with `src/metrics/large_image.py::ssim_separable` — the **same
definition** (same Gaussian window, constants, valid region, mean) evaluated separably in row strips to
bound memory; agreement with the baseline SSIM is < 1e-4 (float summation order only; checked in the
smoke test). Every metric value is sanity-checked during evaluation (finite PSNR, SSIM ∈ [−1, 1],
LPIPS ∈ [0, 2]); an impossible value stops the evaluation instead of being averaged.

SICE specifics: input = darkest exposure with mean grayscale brightness ≥ 5 (0–255), measured at full
resolution — exposure order is not taken from filenames; the testing-index "−1 EV" image index is recorded
for reference only (`splits/sice_test_selection.json`). Reference = the **dataset-provided Label**, not
independently verified ground truth. Label orientation is checked (none of the 58 test Labels is rotated).

## 11. Training configuration (`configs/generalized_lol_v2.yaml`)
Adam (β 0.9/0.999), lr 2e-4, weight decay 0, no scheduler, batch 4 (2+2), 112 steps x 50 epochs = **5,600
optimizer steps** (same step budget as the LOL-v1 baseline), seed 42, cuDNN deterministic.

## 12. Checkpoint selection
After every epoch: full-resolution validation on both val subsets. `best.pth` = max
**(real_val_PSNR + synthetic_val_PSNR) / 2**; both individual values and the mean are logged. No test set
(LOL-v2 Test, SICE, LoLI-Street) is ever used for selection.

## 13. Evaluation protocol
`best.pth` is evaluated **once** on all test domains (the script refuses to run again). Outputs in
`evaluation/`: `metrics_summary.{json,md}`, `per_image/*.csv`, `sice_evaluation_manifest.json`,
`qualitative/*.png` (Input | Output | Reference), `training_curves.png`.

## 14. Limitations
- Scene grouping is heuristic (thumbnail correlation); thresholds were chosen from audit evidence.
- LoLI-Street low images appear synthetically darkened; its unpaired real-looking Test set is not scored.
- SICE Labels are dataset-provided (fused) references; SICE inputs are darker than the "−1 EV" protocol
  of the original paper, so numbers are not comparable with papers using −1 EV.
- The model is trained on 256 crops of ≤0.24 MP images and applied to 20 MP SICE images (tiled).
- 5,600 steps is a short, comparability-driven budget; the model is likely under-trained.
- Apple-silicon MPS produced invalid metric values on 20 MP images during local testing; the evaluation
  script refuses MPS. Local checks use CPU; the real run uses CUDA (Colab), where the smoke test
  cross-checks SICE metrics on CUDA against a CPU reference before training.

## Directory layout
```
experiments/generalized_lol_v2/
├── README.md             this file
├── config.yaml           resolved config written by the training run
├── history.json          per-epoch log
├── checkpoints/          best.pth, last.pth, final.pth
├── splits/               split + SICE selection files (fixed before training)
├── evaluation/           test results (written once)
└── logs/                 integrity checks, run_info, train_summary, smoke tests, pre-Phase-2 file MD5 snapshot
```

## Exact commands
Local (Mac), already done:
```bash
.venv/bin/python scripts/prepare_phase2_data.py --zip      # data_prepared/phase2 + phase2.zip (sources untouched)
.venv/bin/python scripts/make_phase2_splits.py             # writes / re-verifies splits/
.venv/bin/python scripts/smoke_test_generalized.py --device cpu   # smoke test (no training)
```
Colab (`notebooks/colab_train_generalized_lol_v2.ipynb`), from the project root on Drive:
```bash
unzip -q data_prepared/phase2.zip -d /content
python scripts/make_phase2_splits.py --data-root /content/phase2
python scripts/smoke_test_generalized.py --data-root /content/phase2 --skip-baseline-check --device cuda
python scripts/train_generalized.py --data-root /content/phase2 --device cuda
python scripts/evaluate_generalized.py --data-root /content/phase2 --device cuda
python scripts/plot_generalized_curves.py
```
