# Exposure-conditioned HVI-CIDNet: module design, verification plan, novelty status and revised schedule

Date: 2026-11-08. Status: **design only. Not implemented, not trained.** The HVI-CIDNet architecture is unchanged; this file describes what would be added after approval.

Labels:
- **[V]** read in the vendored code or a paper
- **[H]** hypothesis
- **[U]** unverified

---

## 1. Where brightness is decided in HVI-CIDNet (from `third_party/hvi_cidnet/net/CIDNet.py`) [V]

1. **Input transform.**
   - `hvi = HVIT(x)`.
   - The intensity channel is **I = max(R, G, B)** (`HVI_transform.py`, line 21), i.e. the HSV value.
   - H and V are the colour plane, collapsed by a learnable `density_k`.
2. **Two encoder–decoders.**
   - The I-branch (`IE_block*`, `ID_block*`) processes I.
   - The HV-branch processes H and V.
   - They exchange information through LCA cross-attention at every scale.
3. **Output is a residual in HVI space:**

   `output_hvi = cat([hv_0, i_dec0]) + hvi`, then `PHVIT(output_hvi)` maps it back to RGB.

   - So the **change in intensity is exactly `i_dec0`**, the single-channel output of the I-decoder. That is the natural control point for "how much to brighten".
4. **Upstream properties to keep in mind:**
   - `I_LCA5` is computed but its output is overwritten, so it never receives gradients (documented in Phase 3).
   - The official code has a **user-set** brightness knob at test time: `gated2` / `alpha` in `PHVIT` multiplies the RGB output by a constant. It is fixed, not learned and not input-dependent. This matters for the novelty wording (§5).

---

## 2. Proposed module (adds a few thousand parameters; nothing existing is changed)

**Exposure encoder E_x.** Input: the I channel of the *input* (`hvi[:, 2]`, before any processing).
- **Global statistics (20 values):**
  - mean;
  - p05, p25, p50, p75, p95;
  - fraction of pixels below 0.05 and above 0.95;
  - a 12-bin histogram of log(I + 1e-3).
- **Spatial context:** a 1/16-resolution average-pooled I map, followed by 2 conv layers (8 channels) and global average and max pooling (16 values).
- **Head:** concatenate (36 values), then MLP 36 → 64 → K.

The head outputs:

| Output | Size | Role |
|---|---|---|
| Δγ_s, β_s | per channel, for the I-decoder stages `i_dec4` (144 ch), `i_dec3` (72), `i_dec1` (36) | FiLM: f ← f · (1 + Δγ_s) + β_s |
| Δs | one scalar | gate on the intensity residual: `I_out = I_in + (1 + Δs) · i_dec0` |

**Identity at initialisation (exact, not approximate).**
- The last linear layer of the MLP is **zero-initialised** (weights and bias).
- So Δγ = 0, β = 0 and Δs = 0 for every input. Then f·1 + 0 = f, and (1 + 0)·i_dec0 = i_dec0, so the forward pass equals the original network **bit for bit**.
- Planned unit test: for the official weights, outputs with and without the module are identical (`torch.equal`) on random inputs and on LOL-v2 val. The gradient into the zero layer is non-zero, so the module can still learn.

**How it controls brightness.**
- Δs scales the whole intensity correction. Δs < 0 means "brighten less"; Δs > 0 means "brighten more".
- The FiLM terms reshape the decoder's intensity features per channel, so tone shape can change, not only global strength (for example, protecting highlights while lifting shadows).
- The HV (colour) branch is untouched. However, it sees I-features through LCA, so colour can stay consistent with the new intensity [H].

**Ablation alternatives:**
- A5: the gate Δs only (no FiLM).
- FiLM only (no gate).
- E_x without the spatial map (global statistics only).

---

## 3. How we verify that it learns *meaningful* exposure adaptation (not only a PSNR change)

1. **Identity test** (before training): bit-exact equality at initialisation, as in §2.
2. **Response curves** (after training, validation only):
   - Take a fixed set of validation scenes and re-expose each input over a grid, k ∈ {0.25, 0.5, 1, 1.5, 2, 3}, with the reference fixed.
   - Plot (a) the learned gate 1 + Δs, (b) the output mean luma, and (c) the output − reference luma against the input mean luma.
   - **Meaningful adaptation** means:
     - 1 + Δs decreases monotonically as the input gets brighter;
     - output brightness stays close to the reference across the grid (a flat output − reference curve).
   - The baseline (C0) is expected to show output brightness rising with input brightness, i.e. over-brightening.
3. **Counterfactual swap.** Feed the encoder statistics from a *different* exposure of the same scene while the backbone sees the real input.
   - If the output brightness follows the swapped statistics, the module really controls brightness.
   - If nothing changes, the module is ignored.
4. **Real multi-exposure check** (no synthetic re-exposure): SICE Part 1 sequences (development only). Measure the output-brightness spread across the exposures of one scene: a lower std is better, as long as PSNR does not drop.
5. **Not only PSNR:** harm rate, |ΔY|, ΔE00 (colour), edge-energy and flat-noise ratios (detail, noise), and NIQE / BRISQUE (supplementary).
   - A configuration that gains PSNR but raises noise or colour error is reported as such.
6. **Controls:**
   - C0 (same fine-tuning, no module, original data);
   - A1 (exposure data only);
   - A2 (module only).
   - The claim rests on A3 − C0, with scene-level bootstrap confidence intervals.

---

## 4. Re-exposure augmentation (training only; reference unchanged)

- Pick a target brightness ratio t = input mean luma / reference mean luma. Sample log-uniformly in [0.1, 1.0], plus with probability 0.1 the original input (t unchanged).
- Scale the linearised input (sRGB → linear → × factor → sRGB), with the existing `soft_shoulder` highlight protection. This reuses `src/datasets/exposure_augment.py` with a new range (that file itself is not changed).
- **Noise caveat:** brightening a dark input by scaling also scales its noise. So "mildly dark" synthetic inputs are noisier than real mildly dark captures [H]. This is why SICE (real exposures) is the main check.

---

## 5. Novelty status after reading the related papers (2026-11-08)

**Read:**
- full method text: InterLight, arXiv 2504.00400, URWKV, ControlLight, MambaLIE;
- abstract only: IAT, UBLLIE;
- **not readable** (HTTP 403 / empty): ENC (CVPR 2022) and the "Two-Stage Exposure Controllable Network" (IEEE 11194160). These still need a manual read, e.g. through the college library.

| Rank | Work | Overlap with our plan | What it does (as read) |
|---|---|---|---|
| 1 | **InterLight** (arXiv 2605.19982, May 2026; comment: accepted to IJCAI 2026) | **High** | HVI dual-branch network derived from CIDNet. A global prompt from the input drives FiLM-style γ/β in the **HV** branch, plus a learned luminance gate and a brightness bias in the **I** branch. Trained on LOL, SICE-Mix/Grad and others. Only mild photometric augmentation (gamma 0.95–1.05). No per-darkness evaluation |
| 2 | Adaptive LLIE via Joint Global-Local Illumination Adjustment (arXiv 2504.00400) | Medium–high (mechanism) | Global guidance gives FiLM-like modulation of local features; brightness-level-dependent depth; no HVI; no re-exposure augmentation; no stratified evaluation |
| 3 | ControlLight (arXiv 2605.25569) | Medium | **User-set** strength s (LoRA scale) on a 9B generative model; several target levels from one reference |
| 4 | URWKV (CVPR 2025) | Medium–low | "Luminance-adaptive normalisation": scale-only modulation from features. **Avoid this name** |
| 5 | IAT (BMVC 2022) | Low–medium (abstract only) | Input-predicted global ISP parameters (gamma, colour matrix) |
| 6 | MambaLIE (arXiv 2607.03013) | Low–medium | Input-side illumination gate |
| 7 | UBLLIE (arXiv 2608.04429) | Low | CLIP prompts, backlit images |
| — | HVI-CIDNet official code [V] | Low | User-set `alpha` brightness multiplier at test time |

**Consequences:**
- "Input-adaptive modulation inside an HVI network" is **not new**; InterLight does it. We must cite it and must **not** claim it.
- What we did not find in any paper we could read:
  1. an **identity-initialised plug-in** that turns an *existing, released* HVI-CIDNet checkpoint into an exposure-conditioned one without retraining from scratch;
  2. conditioning on **explicit exposure statistics** of the input I channel;
  3. **re-exposure training** (very dark to near-normal, same reference) aimed at *preventing over-enhancement*;
  4. a **stratified evaluation** by darkness level (LoLI-Street levels, SICE exposure sweep), with harm and over-brightening rates, comparing official checkpoints.
- **Defensible claim if the results hold:** "an analysis showing that current LLIE checkpoints over-enhance mildly dark inputs, and a minimal, identity-initialised exposure-conditioning plug-in for HVI-CIDNet that reduces this, evaluated by darkness level."
  - This is an analysis plus plug-in contribution, not a new architecture.
- **InterLight should be a baseline** if its code and weights are released (not yet checked; downloading would need your approval).
- ENC and paper 4 must be read before the final wording.

---

## 6. Revised short plan (target ≈ 18–22 T4-hours over 10 days)

Prioritised by information per GPU-hour; nothing starts without your approval after the diagnostic.

| Day | Work | T4 h |
|---|---|---|
| 1 | Diagnostic (this stage) | 1.5–3 |
| 2 | Implement the module + augmentation in new Phase-5 files; identity test; CPU unit tests; read ENC / paper 4 | — |
| 3 | Smoke test (300 steps); memory and speed | 0.3 |
| 4 | **C0** and **A3**, fine-tuned from official LOLv2-real weights, 6,000 steps, batch 4, 256² | ≈ 3.0 |
| 5 | **A1** (data only), **A2** (module only) | ≈ 3.0 |
| 6 | A3 seeds 43 and 44 | ≈ 3.0 |
| 7 | C0 seeds 43 and 44 | ≈ 3.0 |
| 8 | A5 (gate only) **or** R (Retinexformer + plug-in), whichever the day-5 results make more informative | ≈ 1.5 |
| 9 | Development evaluation on SICE Part 1 sweep, LoLI levels and LOL-v2 val; response curves; selection record written | ≈ 1.5 |
| 10 | Final test once: LOL-v2 test; **SICE Part 2 only if you approve unsealing then**; write-up | ≈ 1 |
| | **Total** | **≈ 18–20** |

**Dropped from the longer plan:**
- A4 (consistency loss) and A6 (over-exposure penalty);
- the extra seeds for A1 and A2.

These can be added back if time remains.

**Fine-tuning rate:** the step time (≈ 0.88 s/step at batch 4, 256²) is from Tier A. With the module the cost increase is estimated at < 5 % [H].

---

## 7. Addendum after the interim diagnostic (2026-11-09; see `DIAGNOSTIC_FINDINGS.md`)

**Problem statement, revised.**
- The measured failure is *brightness miscalibration across exposure levels*:
  - over-brightening of inputs at r ≥ 0.5 of the reference brightness;
  - under-brightening at r < 0.25.
- On real SICE exposures the brightness error rises with input exposure for every learned checkpoint tested (ρ = +0.84 to +0.97; Retinexformer official +0.41).
- So the gate (1 + Δs) and the FiLM terms must be able to move brightness in **both** directions. The design already allows this, because Δs is unbounded at initialisation, not clamped to ≤ 0.

**Augmentation range.**
- Re-expose to target ratios r ∈ [0.1, 1.0], so the network sees near-normal inputs paired with the same reference.
- r > 1 (over-exposure) stays out of scope for training. It is reported as a stress test only.

**Success criteria, pre-registered before any training.** Evaluated on SICE Part 1 (development) by r-bin, against C0:
1. r 0.75–1.0: harm rate lower than C0 and |output − reference luma| lower than C0. PSNR at least that of the untouched input (18.82 dB on this bin).
2. r 0.25–0.75: PSNR within 0.3 dB of C0 or better.
3. r < 0.25: PSNR not below C0 by more than 0.3 dB.
4. LOL-v2 val (P-B) or LOL-v2 test (P-A, once): within ±0.3 dB of C0.
5. Reported alongside the simple global-gain rule. Beating it on mild bins is required for a strong claim, but it is not required for a "reduces harm" claim.

**Compute, revised.**
- The diagnostic so far took about 4.5 T4-hours of session time, plus about 1.5 still pending.
- That leaves about 12–14 h of the 18–22 h budget for training and evaluation:

| Item | Estimate |
|---|---|
| Smoke test | 0.3 h |
| C0, A1, A2, A3 at 6,000 steps (≈ 1.5 h each at 0.88 s/step) | 6 h |
| Seeds 43 for C0 and A3 | 3 h |
| Development evaluation | 1.5 h |
| Final test | 1 h |
| **Total** | **≈ 12 h** |

- A5 (gate only) and R (Retinexformer) are dropped unless time remains.
