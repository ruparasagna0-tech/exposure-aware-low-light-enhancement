# Phase 5 diagnostic: findings (interim, 2026-11-09)

**Source:** `experiments/phase5_llie_direction/diagnostic/DIAG_REPORT.md` on Drive, interim copies `interim_*`. All numbers below are copied from that report.

**Status of the data:**

| Dataset | Status |
|---|---|
| LoLI-Street Val | Complete: 12 conditions × 2,997 pairs |
| SICE Part 1 | 9 of 12 conditions × 3,021 exposures. Missing: `hvi_off_gen`, `iat_lol`, `iat_exp` (GPU quota ran out) |
| LoLI-Street real test | Not run yet |

**Inference only.** SICE Part 2 was not accessed.

**Binning:** SICE images are grouped by r = input mean luma / Label mean luma.
- r < 1 means darker than the dataset Label.
- r > 1 means brighter (over-exposed). Those bins are **outside the intended scope of LLIE** and are reported only as a stress test.

---

## 1. Facts

### 1.1 Real exposures (SICE)

**Output-brightness error grows with input exposure for every learned checkpoint.**

Spearman correlation between r and (output − reference luma):

| Checkpoint | ρ |
|---|---|
| U-Nets | +0.96 / +0.97 |
| Retinexformer (ours) | +0.92 |
| HVI-CIDNet (ours) | +0.91 |
| HVI-CIDNet official LOLv2-real | +0.84 |
| HVI-CIDNet official LOLv2-syn | +0.96 |
| Retinexformer official LOLv2-real | +0.41 |

**Mildly under-exposed inputs** (r = 0.75–1.0, n = 370; the untouched input already scores 18.82 dB):

| Condition | PSNR | Harm (output worse than input) | Over-brightened |
|---|---|---|---|
| identity | 18.82 | — | 0 % |
| gain048 | 18.75 | 44 % | 6 % |
| U-Net (P2) | 12.49 | 89 % | 100 % |
| U-Net + aug | 13.02 | 87 % | 99 % |
| Retinexformer (ours) | 17.86 | 50 % | 74 % |
| Retinexformer official real | 13.22 | 96 % | 15 % |
| HVI-CIDNet (ours) | 17.58 | 54 % | 76 % |
| HVI-CIDNet official real | 18.63 | 44 % | 23 % |
| HVI-CIDNet official syn | **19.47** | 36 % | 57 % |

**Very dark inputs** (r < 0.25, n = 551): every learned model is **too dark**.
- Output luma is 0.23–0.40 against a reference of 0.48.
- The best is about 16.6 dB (HVI-CIDNet ours 16.60, Retinexformer ours 16.45).
- The learned models are still 3–4 dB better than the global-gain rule (12.93 dB).

**r = 0.25–0.75:** the learned models clearly help.
- Best 19.02 dB (Retinexformer ours, r 0.25–0.5) and 20.98 dB (HVI-CIDNet official syn, r 0.5–0.75), against an input of 9.9 / 13.7 dB.
- Our checkpoints already over-brighten 28–55 % of these images.

**Over-exposed inputs** (r > 1, n = 1,164): every LLIE model brightens further, harming 49–100 % of images. Only the global-gain rule darkens them.

**Retinexformer official (LOLv2-real) degrades badly on bright inputs:**
- r > 1.5: SSIM 0.107, NIQE 9.87, harm 90 %.
- r 0.75–1.0: harm 96 %.
- This needs the visual panels to confirm (artefacts vs colour).

### 1.2 Synthetic darkening (LoLI-Street; levels share one reference)

- Our LOL-v2-trained checkpoints and the official "Generalization" checkpoint over-brighten the light level 58–100 % of the time.
- The official LOLv2-real HVI-CIDNet and Retinexformer **under-brighten** at every level.
- A single global gain to luma 0.48 gives the best overall PSNR (19.84), SSIM (0.853) and LPIPS (0.077). LoLI-Street's Photoshop darkening is close to a global brightness change, so this dataset favours global brightness correction and is weak evidence on its own.

### 1.3 No-reference metrics

- NIQE barely separates conditions (the untouched input scores 3.01 on LoLI).
- BRISQUE flags the U-Nets (39–54).
- Both remain supplementary.

---

## 2. Is over-enhancement a consistent problem?

**On real exposures (SICE): yes.**
- For every learned checkpoint tested (9 of 12 conditions so far; 3 still pending), including the official HVI-CIDNet weights, the brightness error rises monotonically with input exposure (ρ = +0.84 to +0.97; Retinexformer official +0.41).
- 36–96 % of mildly under-exposed inputs come out worse than the input itself.

**More precisely, the consistent problem is *brightness miscalibration across exposure levels*:**
- inputs that are already close to normal are over-brightened;
- very dark inputs are under-brightened.

On synthetic LoLI-Street, the direction depends on the checkpoint. The miscalibration is still present, but its sign varies.

---

## 3. Interpretation (hypotheses)

- **[H]** Models trained on LOL-type pairs learn a roughly fixed brightening. Their output brightness tracks the input brightness instead of a target exposure.
- **[H]** An input-exposure-aware correction should help most at r ≥ 0.5, where the networks already preserve structure but get brightness wrong, and somewhat at r < 0.25.
- **[H]** The global-gain rule's strength (best on SICE overall: 16.79 dB vs 15.50) shows that a large part of PSNR here is global brightness. The learned models' advantage is structure, noise and colour in dark inputs (3–4 dB at r < 0.5).
- The goal is to combine both. Beating the gain rule and the identity on mild inputs, while keeping the dark-input advantage, is the bar for the proposed module.

---

## 4. Caveats

- **SICE references** are dataset Labels selected from fusion results. Their brightness is a subjective choice, so "over-brightened" means "brighter than the Label by > 0.05 mean luma", not "wrong" in an absolute sense.
- **Resizing:** SICE was resized to a 900 px long side (area resampling, JPEG q95) for every condition alike. Absolute numbers differ from full-resolution protocols.
- **Single run** per checkpoint; no seed variance. Bins have n = 370–706, and Wilson 95 % intervals are in the report.
- **"Harm" compares against the input's own PSNR.** For very dark inputs harm is ~0 for everyone by construction.

---

## 5. Still to run (≈ 1.5 T4 h; original notebook, cells 1–5)

- SICE: `hvi_off_gen`, `iat_lol`, `iat_exp`.
- LoLI-Street real test (no reference): NIQE and BRISQUE, output brightness.
- Then the final report, with the fixed plot and panel code (a new small bundle).
