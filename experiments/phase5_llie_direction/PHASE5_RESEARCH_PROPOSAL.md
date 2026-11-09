# Phase 5 research proposal: competitive, exposure-robust low-light image enhancement

Date: 2026-11-08. Status: **investigation and recommendation only.** No code was changed, nothing was trained, installed or downloaded, and no file was overwritten. This file extends `RESEARCH_DIRECTION_REPORT.md` (same folder), which stays unchanged.

**Labels**

| Label | Meaning |
|---|---|
| **[V]** | Verified on the official paper page or the official repository README, read on 2026-11-08 |
| **[P]** | Measured in this project |
| **[H]** | My hypothesis or interpretation |
| **[U]** | Not verified; check before citing |

---

## 1. LLIE research landscape (2023–2026)

| Family | Representative verified works | What they buy | Practical cost on a T4 |
|---|---|---|---|
| Colour-space / dual-branch CNN–attention | HVI-CIDNet (CVPR 2025) [V] | Colour fidelity, low noise in dark regions; 1.88 M params | Low |
| Retinex-guided transformers | Retinexformer (ICCV 2023) [V] | Illumination-guided attention; 1.61 M params [P] | Low |
| Efficient multi-degradation CNN | DarkIR (CVPR 2025; NTIRE 2025 best method per README) [V] | Low light + noise + blur | Low–medium (m: 3.31 M) |
| RWKV / state-space | URWKV (CVPR 2025) [V]; MambaLLIE (NeurIPS 2024) [V] | Long-range modelling at low cost | Medium; Mamba needs custom CUDA kernels [H] |
| Generative priors | GLARE (ECCV 2024, codebook + flow) [V]; Reti-Diff (ICLR 2025, latent diffusion) [V]; Diff-Retinex (ICCV 2023), GSAD (NeurIPS 2023), LightenDiffusion (ECCV 2024) [V]; QuadPrior (CVPR 2024, zero-reference) [V] | Perceptual quality, hallucinated detail | High (GLARE ≈ 59 M params per the HVI paper [U]) |
| Zero-reference / unsupervised | Zero-DCE++ (TPAMI 2022), SCI (CVPR 2022), PairLIE (CVPR 2023), CLIP-LIT (ICCV 2023) [V] | No paired data; tiny | Very low |
| Exposure correction (under **and** over) | MSEC (CVPR 2021), ENC (CVPR 2022), LCDPNet (ECCV 2022), IAT (BMVC 2022), CoTF (CVPR 2024), UEC (ECCV 2024) [V] | Exposure-range robustness | Low |
| Controllable brightness | KinD (ACM MM 2019), KinD++ (IJCV 2021) [V] | User-set light level | Low |

**Two facts that shape everything below:**

1. **Protocols differ a lot**, so published numbers are often not comparable.
   - **GT-mean testing** rescales the output's brightness to match the ground truth's. That uses the reference at test time.
     - HVI-CIDNet LOLv2-Real: 23.43 dB without GT-mean, **27.76 dB** with it (same "wo perc" weights).
     - Retinexformer's README reports GT-mean numbers (LOL-v2-real 27.71) and says the author "does not suggest this test setting".
   - **Checkpoint choice.** HVI-CIDNet releases `best_PSNR`, `best_SSIM` and `best_gt_mean` weights for LOLv2-Real. The README says metrics are computed on "the validation set", but LOL-v2 has no official validation split. It is plausible that the official test set served as validation [H; not stated].
   - **Training data.** DarkIR's LOLv2 numbers come from **multi-task** training on LOLBlur, LOLv2 and LSRW together [V].
2. **Mixed and varied exposure is still weak, even for the strongest model.**
   - HVI-CIDNet reports only **13.4 dB** on SICE-Mix and SICE-Grad (GT-mean) [V, README table], against 24.1 dB on LOLv2-Real.
   - Our own measurements [P] show over-brightening of mildly dark inputs (LoLI-Street light level).
   - Both point to the same weakness: brightness is not adapted to the input's exposure [H].

---

## 2. Literature review and comparison (verified items only)

| Method | Venue | Authors | Architecture | Params | Code / weights (README) | Strengths | Limitations for us |
|---|---|---|---|---|---|---|---|
| **HVI-CIDNet** | CVPR 2025 | Q. Yan, Y. Feng, C. Zhang, G. Pang, K. Shi, P. Wu, W. Dong, J. Sun, Y. Zhang | HVI colour space (learnable k) + dual I/HV branches with LCA cross-attention; loss L1 + SSIM + edge + VGG in RGB and HVI | 1.88 M (paper), 7.57 G FLOPs at 256² | Yes; LOLv1, LOLv2-Real (best_PSNR / best_SSIM / w_perc), LOLv2-Syn, SICE, SID, LOL-Blur, FiveK, "generalization" | Best no-GT-mean numbers among T4-feasible models; colour; we already vendor it | Training schedule is long (1,500 epochs on LOL); weak on SICE mixed exposure |
| Retinexformer | ICCV 2023 | Y. Cai, H. Bian, J. Lin, H. Wang, R. Timofte, Y. Zhang | One-stage Retinex illumination estimator + illumination-guided transformer | 1.61 M [P] | Yes; per dataset (LOL_v1, LOL_v2_real, …) | Simple and stable; second backbone | README gives no-GT-mean numbers only as an image (not verified) |
| DarkIR | CVPR 2025 | D. Feijoo, J. C. Benito, A. Garcia, M. V. Conde | Efficient CNN, frequency + spatial blocks | m: 3.31 M / 7.25 GMACs; l: 12.96 M | Yes (Hugging Face) | Noise and blur robustness | LOLv2 numbers from multi-task training; no official training code in our Phase 3 (we re-implemented it) [P] |
| URWKV | CVPR 2025 | R. Xu, Y. Niu, Y. Li, H. Xu, W. Liu, Y. Chen | RWKV encoder–decoder, luminance-adaptive normalisation, state-aware fusion | not stated numerically | Weights via Baidu Pan; some sets missing | Luminance-adaptive normalisation (related to our idea) | Weights hard to access; RWKV kernels [H] |
| GLARE | ECCV 2024 | H. Zhou, W. Dong, et al. | VQ codebook of normal-light features + invertible flow + adaptive feature transform | ≈ 59.5 M / 508 G FLOPs (from the HVI paper [U]) | Yes (LOL, LOL-v2-real) | Perceptual detail | Heavy; multi-stage training with VQGAN |
| Reti-Diff | ICLR 2025 | C. He, C. Fang, Y. Zhang, et al. | Retinex latent diffusion prior + Retinex-guided transformer | not read | Yes | Strong priors | Diffusion training cost |
| MambaLLIE | NeurIPS 2024 | J. Weng et al. | Global-then-local state-space blocks | not stated | Yes (Drive) | Recent | Custom CUDA kernels; numbers only as images |
| IAT | BMVC 2022 | Z. Cui et al. | Local branch + global ISP queries (gamma, colour matrix) | ≈ 90 K | Demo only per README | **Both LLIE and exposure correction**: the key baseline for exposure robustness | Lower fidelity than modern models |
| Zero-DCE++ | TPAMI 2022 | C. Li, C. Guo, C. C. Loy | Curve estimation, zero-reference | ≈ 10 K | Yes | Tiny | Fidelity |
| KinD++ | IJCV 2021 | Y. Zhang, X. Guo, J. Ma, W. Liu, J. Zhang | Retinex decomposition + multi-scale illumination attention; adjustable light level | — | TensorFlow | Controllable brightness | Old backbone; level set by hand |
| SCI, PairLIE, CLIP-LIT, QuadPrior | CVPR 2022 / CVPR 2023 / ICCV 2023 / CVPR 2024 | see §17 | Unsupervised / zero-reference | small | Yes | No paired data needed | Fidelity on paired benchmarks |
| SNR-Aware | CVPR 2022 | X. Xu, R. Wang, C.-W. Fu, J. Jia | SNR-guided attention / convolution | — | Yes | Noise awareness | Not exposure-adaptive |
| LLFormer | AAAI 2023 | T. Wang, K. Zhang, T. Shen, W. Luo, B. Stenger, T. Lu | Axis-based attention; UHD-LOL benchmark | [U] | Yes | High resolution | UHD training cost |

---

## 3. Benchmark performance (only numbers printed by the authors; never compared across protocols)

**LOL-v2-Real test (100 pairs), no GT-mean, authors' own tables:**

| Method | PSNR | SSIM | LPIPS | Notes |
|---|---|---|---|---|
| HVI-CIDNet | 24.111 | 0.871 | 0.108 | Paper Table 1 |
| HVI-CIDNet | 24.11 / 23.90 / 23.90 / 23.43 | 0.868 / 0.866 / 0.871 / 0.862 | 0.116 / 0.122 / 0.119 / 0.169 | README rows, in order: "best Normal", best_PSNR, best_SSIM, wo_perc |
| DarkIR-m | 23.87 | 0.880 | 0.186 | Multi-task training (LOLBlur + LOLv2 + LSRW) |
| URWKV | 23.11 | 0.874 | — | README: no GT-mean |
| IAT | 23.50 | 0.824 | — | README news entry (arXiv version) |

**LOL-v2-Synthetic test (100 pairs), no GT-mean:**

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| HVI-CIDNet | 25.705 | 0.942 | 0.045 |
| URWKV | 26.36 | 0.944 | — |
| DarkIR-m (multi-task) | 25.54 | 0.934 | 0.058 |

**LOL-v1 test (15 pairs):** not used in this project. LOL-v1 is contained in LOL-v2-Real [P: overlap audit].
- HVI-CIDNet no-GT-mean 23.81 / 23.50; GT-mean 27.71–28.20.
- URWKV 24.27 / 0.856.
- IAT 23.38 / 0.809.

**GT-mean protocol (reported separately; reference-dependent, not our primary metric):**
- HVI-CIDNet LOLv2-Real 27.76–28.39.
- Retinexformer LOL-v2-real 27.71 / 0.856; LOL-v2-syn 29.04 / 0.939.

**SICE (mixed / gradual exposure, GT-mean):** HVI-CIDNet SICE-Mix 13.42 / 0.636 / 0.362 and SICE-Grad 13.45 / 0.648 / 0.318.
- These splits differ from our SICE protocol. In ours the input is the darkest frame with mean ≥ 5/255 against the Label, and our Phase-2 U-Net reached 17.32 dB [P].
- **Not comparable.**

**Our own protocol** (scene-disjoint LOL-v2 val, 5,600 steps): HVI-CIDNet 22.58 dB, Retinexformer 22.38 dB [P]. This is short training on our splits, so it is not comparable with the tables above.

**Reproducibility on a T4:**

| Model | Feasible? | Notes |
|---|---|---|
| HVI-CIDNet | Yes | Inference and fine-tuning ✓. The full official schedule (1,500 epochs, 400² crops, batch 8) is **not** realistic on a free T4 (estimated tens of GPU-hours [H]) |
| Retinexformer | Yes | ✓ |
| DarkIR-m | Yes | ✓ |
| IAT, Zero-DCE++ | Yes | Inference ✓ |
| GLARE, Reti-Diff, diffusion models | Inference only, at best | Training not realistic in our timeline |
| MambaLLIE | Risky | Custom CUDA kernels |

---

## 4. Candidate baseline comparison and choice

| Criterion | HVI-CIDNet | Retinexformer | DarkIR-m | URWKV | GLARE |
|---|---|---|---|---|---|
| Best no-GT-mean LOLv2-Real (authors) | **24.11** | not in text | 23.87 (multi-task) | 23.11 | [U] |
| Official weights we can get | **many datasets** | per dataset | yes | partly (Baidu) | LOL, v2-real |
| Params | 1.88 M | 1.61 M | 3.31 M | not stated | ≈ 59 M [U] |
| Already vendored and tested here | **yes** | yes | yes (re-implemented training) | no | no |
| Our protocol result | **22.58** | 22.38 | none | — | — |
| T4 fine-tuning | **easy** | easy | easy | unknown | hard |

**Selected baseline: HVI-CIDNet.** This is chosen on evidence, not by default:
- the strongest no-GT-mean results among T4-feasible models;
- official weights for LOLv2-Real, LOLv2-Syn **and SICE**;
- the smallest training risk, because it is already integrated and tested in our code.

**Second backbone, for generality:** Retinexformer.

---

## 5. The research gap

**Gap.** Strong supervised LLIE models enhance well on strongly under-exposed benchmarks. However, their brightening is not adapted to the input's exposure:
- the strongest verified model scores 13.4 dB on mixed-exposure SICE [V];
- in our measurements, every model over-brightens mildly dark inputs, and a fixed-gain rule beats the U-Net by 7.5 dB on the LoLI-Street light level [P].

Exposure-correction methods handle a range of exposures, but they are separate, weaker-fidelity models (IAT 23.50 dB on LOL-v2-Real [V]).

**Open question:** can a strong LLIE model be made exposure-adaptive with a *minimal* change, while keeping its benchmark fidelity?
- Related ideas exist: luminance-adaptive normalisation in URWKV [V]; exposure-invariant features in ENC [V]; user-set light level in KinD/KinD++ [V].
- I found no verified work applying a learned exposure conditioning to HVI-CIDNet's intensity branch and evaluating it stratified by darkness level.
- This is a search result, **not a novelty claim**. §13 lists what must still be read.

---

## 6. Three candidate directions

### D1: Exposure-conditioned HVI-CIDNet (recommended)
1. **Problem:** fixed brightening; over-exposure of mildly dark inputs; weak robustness across exposure levels.
2. **Motivation:** §5. Surveillance frames span very dark to near-normal.
3. **Limitations of existing work:**
   - LLIE models are trained on a narrow exposure range.
   - Exposure-correction models trade away fidelity.
   - KinD needs manual control.
4. **Modification:**
   - (a) Exposure-range training data.
   - (b) A tiny exposure encoder: global statistics of the input I channel (mean, percentiles, log-histogram) plus a low-resolution illumination map. It produces **FiLM (scale and shift) for the I-branch stages** of CIDNet, so the amount of brightening depends on the input.
   - (c) An optional exposure-consistency loss.
5. **Architecture:** HVI-CIDNet unchanged plus one small conditioning MLP / conv (a few thousand parameters [H]).
6. **Training:**
   - Fine-tune from the official weights (benchmark protocol) and from D1 (our generalisation protocol).
   - A fixed number of steps; no test-based selection.
7. **Losses:** official loss + λ_c · exposure-consistency (optional) + a one-sided over-exposure penalty (optional ablation).
8. **Data:**
   - Train on LOL-v2 Real/Syn, with re-exposed inputs; the reference stays unchanged.
   - Evaluate on the LOL-v2 tests, the SICE exposure sweep, LoLI-Street levels, and SICE Part 2 (sealed).
9. **Metrics:**
   - PSNR, SSIM, LPIPS;
   - ΔE00 and colour-only ΔE00;
   - signed ΔL*;
   - **harm rate** (% of images with output PSNR below input PSNR);
   - edge-energy and flat-noise ratios;
   - NIQE and BRISQUE where no reference exists.
10. **Ablations:** §12.
11. **Expected:**
    - Clearly better metrics on light and moderate inputs, and lower |ΔL*|.
    - LOL-v2 fidelity within a small tolerance.
    - No promise of a state-of-the-art benchmark number.
12. **Risks:**
    - A small PSNR loss on the darkest inputs.
    - Synthetic re-exposure is not the same as real exposure; SICE real multi-exposure mitigates this.
13. **Novelty:** moderate and narrow (§13).

### D2: Noise- and detail-aware fine-tuning of HVI-CIDNet
1. **Problem:** noise amplification and texture loss in the darkest regions. Our grass-texture LOL-v2 cases have SSIM 0.52 [P].
2. **Motivation:** LPIPS and SSIM gains without new modules.
3. **Limitations:** SNR-Aware needs a new architecture; diffusion is heavy.
4. **Modification:** an SNR-weighted reconstruction loss (estimate the noise from the input), plus a frequency (FFT-amplitude) loss and a flat-region noise penalty.
5. **Architecture:** unchanged.
6. **Training:** fine-tune from the official weights.
7. **Losses:** as in (4), added to the official loss.
8. **Data and metrics:** LOL-v2 and SICE; LPIPS, SSIM, flat-noise and edge-energy ratios, NIQE.
9. **Ablations:** each loss term alone and combined.
10. **Expected:** small LPIPS and SSIM gains [H].
11. **Risks:** over-smoothing; small effects within seed noise.
12. **Novelty:** **low.** SNR and frequency losses are established.

### D3: Hybrid HVI-CIDNet with a Retinexformer-style illumination prior
1. **Problem:** the I-branch has no explicit illumination estimate.
2. **Modification:** add Retinexformer's light-up estimator (illumination prior) as guidance for HVI-CIDNet's I-branch attention.
3. **Expected:** possible fidelity gains [H].
4. **Risks:** more complexity, harder attribution, and more GPU time.
5. **Novelty:** low–moderate.

The user's brief warns against adding modules for complexity, and D3 is the closest to that.

---

## 7. Final recommendation: D1

**Title:** *Exposure-Conditioned HVI-CIDNet: Learning How Much to Brighten for Robust Low-Light Enhancement across Darkness Levels*

**Problem statement.** Strong LLIE models such as HVI-CIDNet apply a near-fixed brightening learned from strongly under-exposed training pairs. Mildly dark or mixed-exposure inputs are then over-enhanced. We make the enhancement explicitly depend on the input's exposure, with a minimal conditioning module and exposure-range training, while keeping standard-benchmark fidelity.

**Why D1 and not the others:**
- It targets the one failure we have measured ourselves.
- It is backed by a verified weakness of the strongest baseline (SICE-Mix/Grad 13.4 dB).
- It adds almost no compute.
- Its effect can be cleanly isolated with controls.

---

## 8. Proposed architecture and modifications

```
input RGB ─► HVI transform ─► I channel ─┬─► exposure encoder E_x (global stats + 1/16-res map) ─► (γ_s, β_s) per stage
                                          │                                                        │ FiLM
                                          └─► I-branch encoder/decoder ◄──────────────────────────┘
                              HV channels ─► HV-branch (unchanged) ◄─ LCA ─► (unchanged)
                                         ─► inverse HVI ─► output
```

- **E_x:** global average and percentile features of I (p10, p50, p90), a 16-bin log histogram and a 1/16-resolution I map. Two small MLP / conv layers output γ and β for each I-branch stage.
  - **Initialised so that γ = 1 and β = 0**, so the network starts exactly at the baseline.
- **Unchanged:** the HV branch, the LCA blocks, the HVI transform and the output head.
- **Gate alternative (ablation):** output = x + α(x)·(CIDNet(x) − x), with α from E_x.

---

## 9. Loss-function design

L = L_HVI (official: L1 + SSIM + edge + VGG in RGB and HVI) + λ_c·L_cons + λ_o·L_over

| Term | Definition | Status |
|---|---|---|
| L_HVI | Official HVI-CIDNet loss | Unchanged, always on |
| L_cons | ‖f(x_k1) − f(x_k2)‖₁ for two re-exposures of the same input (both share one reference) | Optional (ablation A4). Related to ENC and to Huang et al. 2023, so cited as such |
| L_over | Mean of max(0, Y(f(x)) − Y(y) − δ) on luminance (one-sided over-brightening penalty, δ = 0.05) | Optional (ablation A6) |

λ values are chosen on validation only, from {0.1, 0.5}, at most two values each to save compute. PSNR is not the only selection criterion (§11).

---

## 10. Dataset and evaluation plan

| Role | Data | Notes |
|---|---|---|
| Train (benchmark protocol P-A) | Official LOL-v2 Real train (689) + Synthetic train (900), with exposure-range re-exposure of the inputs | Comparable with published LOL-v2 test numbers |
| Train (generalisation protocol P-B) | Our scene-disjoint LOL-v2 train split | As in Phases 2–3 |
| Selection | P-A: **none**; a fixed number of steps and the last checkpoint (avoids test-set selection). P-B: our scene-disjoint val + its exposure-swept copy | |
| Development evaluation | SICE Part 1, all exposures per sequence (real multi-exposure); LoLI-Street Val levels | Used before; development only |
| Final test, once | LOL-v2-Real and Syn test (P-A); **SICE Part 2** (sealed; needs your approval); LoLI-Street real unpaired test (NIQE / BRISQUE only) | |
| Not used | LOL-v1 (overlap), LLVIP (detection project) | |

**Metrics:**
- **Reference-based**, only where ground truth exists: PSNR, SSIM, LPIPS, ΔE00, signed ΔL*, edge-energy ratio. Reported **without** GT-mean as primary; GT-mean in a separate column for comparison with papers.
- **Exposure consistency:** PSNR as a function of input mean luma; harm rate per bin; std of output brightness across the SICE exposures of one scene.
- **No-reference:** NIQE and BRISQUE (they need `pyiqa` or an equivalent, which is an install needing approval).
- **Cost:** parameters, FLOPs at 256² and 600×400, T4 ms per image.
- **Visuals:** the same scenes at dense / moderate / light levels, and a SICE exposure strip.

---

## 11. Training configuration (planned)

- **Initialisation:** official HVI-CIDNet weights (P-A: LOLv2_real `w_perc`, LOLv2_syn `w_perc`); D1 Tier-A (P-B).
- **Optimiser:** Adam(0.9, 0.999), lr 2e-5 → 1e-7 cosine (fine-tuning).
- **Data:** 256² crops (400² if memory allows), batch 8 (or 4 + gradient accumulation); 10,000 steps per run.
- **Exposure sampler:** with probability 0.6, re-expose the input by k ∈ [0.5, r_ref], log-uniform, where r_ref is the reference / input brightness ratio, so inputs range up to nearly normal light. A soft highlight shoulder (existing `soft_shoulder` policy) avoids clipping.
- **Precision and seeds:** FP32 (as in Tier A); seed 42; C0 and A3 also with seeds 43 and 44.
- **Selection:** P-A, the last checkpoint. P-B, a pre-registered rule: mean val PSNR, subject to harm rate ≤ that of C0.

---

## 12. Ablation study

| ID | Variant | Question |
|---|---|---|
| B0 | Official weights, no fine-tuning | Reference |
| **C0** | Fine-tune, same steps, no changes | Is the gain just extra training? |
| A1 | + exposure-range data | Effect of data alone |
| A2 | + FiLM conditioning (original data) | Effect of the module alone |
| **A3** | A1 + A2 (proposed) | Combined |
| A4 | A3 + L_cons | Consistency loss |
| A5 | A3 with the residual gate instead of FiLM | Form of the conditioning |
| A6 | A3 + L_over | Over-exposure penalty |
| R | A3 on Retinexformer | Does it generalise across backbones? |

**External baselines:**
- identity (no processing), gamma (tuned on val), fixed-target gain;
- Zero-DCE++ and IAT (pretrained weights; downloads need approval);
- DarkIR-m (official weights);
- Retinexformer (official and Tier A).

---

## 13. Novelty check still required before any claim

The following must be read in full: IAT (BMVC 2022), ENC (CVPR 2022), URWKV (CVPR 2025, luminance-adaptive normalisation), and the 2025–2026 exposure-control papers seen only as abstracts (a two-stage exposure-controllable network, IEEE 11194160; ControlLight, arXiv 2605.25569; UBLLIE, arXiv 2608.04429).

**Safe claim:** "a minimal exposure-conditioning of HVI-CIDNet that reduces over-enhancement across darkness levels while preserving LOL-v2 fidelity, with controlled ablations."

**Not claimed:** state of the art; "first exposure-adaptive LLIE".

---

## 14. Expected results and realistic targets (targets, not promises)

| Item | Target |
|---|---|
| LOL-v2-Real / Syn test, P-A, no GT-mean | A3 within ±0.3 dB of C0 and of the official weights (23.4–24.1 dB depending on checkpoint) |
| LoLI-Street light / moderate and the SICE bright half of the exposure sweep | A3 better than C0 in PSNR and |ΔL*|, with a lower harm rate |
| Bootstrap | 95 % CI of A3 − C0 excludes 0 on the exposure-stratified metrics |
| Cost | Parameters +< 1 %; runtime +< 5 % [H] |

Beating published state of the art on LOL-v2 is **not** a realistic target with our compute, and it is not the claim.

---

## 15. Computational requirements (T4 estimates [H], from the Tier-A rate of ≈ 0.88 s/step at batch 4, 256²)

| Item | Estimate |
|---|---|
| One fine-tune (10,000 steps, batch 4) | ≈ 2.5 h (batch 8 ≈ 4–5 h) |
| Runs: C0, A1–A6, R ≈ 8, plus 4 extra seeds ≈ 12 runs | ≈ 30–40 T4-hours |
| Stage 0 inference (existing checkpoints) | ≈ 1–2 h |
| Final evaluation | ≈ 2 h |
| **Total** | **≈ 35–45 T4-hours** |

That is more than free Colab gives reliably. Kaggle (≈ 30 h per week) and/or paid units are needed.

**Cheaper plan if needed:** 6,000 steps per run, and drop A5, A6 and two seeds, for ≈ 18–22 h.

---

## 16. Day-by-day plan (12 working days; compressible to 10 with the cheaper plan)

| Day | Work | GPU |
|---|---|---|
| 1 | Read the §13 papers; pre-register the protocol, metrics, bins and success criteria | — |
| 2 | **Stage 0 (go / no-go):** official HVI-CIDNet + D1 + Retinexformer + DarkIR-m inference on the SICE sweep and LoLI levels. Is over-enhancement present at scale for the strong models? | 1–2 h |
| 3 | Implement the exposure sampler, E_x + FiLM and the gate in **new** Phase-5 files; CPU unit tests (identity at init, shapes, parameter count) | — |
| 4 | Colab / Kaggle smoke test (200 steps), memory and time; B0 evaluation | 1 h |
| 5–6 | C0, A1, A2, A3 (P-A) | ≈ 10 h |
| 7 | A4, A6 | ≈ 5 h |
| 8 | A5, R (Retinexformer) | ≈ 5 h |
| 9 | Seeds 43 and 44 for C0 and A3 | ≈ 10 h |
| 10 | External baselines (IAT, Zero-DCE++, DarkIR-m); development evaluation of all runs; **write the selection record** | 2–3 h |
| 11 | Final test once (LOL-v2 tests, SICE Part 2, LoLI RLLT no-reference) | 2 h |
| 12 | Analysis, figures, report chapter | — |

---

## 17. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Stage 0 shows no over-enhancement for HVI-CIDNet at scale | Switch to D2 (noise / detail). Stage 0 is still a reportable analysis |
| A §13 paper already does D1 on an LLIE backbone | Narrow the claim to an evaluation + plug-in study, or switch to D2 |
| Gains equal C0 (training alone) | Report as a negative result; claim nothing |
| LOL-v2 drop larger than 0.3 dB | Report the trade-off; tune the sampler probability on val only |
| Compute shortfall | Cheaper plan (§15); Kaggle |
| Synthetic re-exposure differs from real | SICE real multi-exposure is the main exposure-sweep evidence |
| Packages for NIQE / BRISQUE | Request approval for `pyiqa`, or omit no-reference metrics |

---

## 18. References (official pages; verified 2026-11-08 unless marked)

1. Q. Yan et al., "HVI: A New Color Space for Low-light Image Enhancement," CVPR 2025, pp. 5678–5687. https://openaccess.thecvf.com/content/CVPR2025/html/Yan_HVI_A_New_Color_Space_for_Low-light_Image_Enhancement_CVPR_2025_paper.html · code https://github.com/Fediory/HVI-CIDNet
2. Y. Cai et al., "Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement," ICCV 2023. https://arxiv.org/abs/2303.06705 · https://github.com/caiyuanhao1998/Retinexformer
3. D. Feijoo, J. C. Benito, A. Garcia, M. V. Conde, "DarkIR: Robust Low-Light Image Restoration," CVPR 2025. https://cvpr.thecvf.com/virtual/2025/poster/34876 · https://github.com/cidautai/DarkIR
4. R. Xu, Y. Niu, Y. Li, H. Xu, W. Liu, Y. Chen, "URWKV: Unified RWKV Model with Multi-state Perspective for Low-light Image Restoration," CVPR 2025, pp. 21267–21276. https://openaccess.thecvf.com/content/CVPR2025/html/Xu_URWKV_Unified_RWKV_Model_with_Multi-state_Perspective_for_Low-light_Image_CVPR_2025_paper.html · https://github.com/FZU-N/URWKV
5. H. Zhou, W. Dong, et al., "GLARE: Low Light Image Enhancement via Generative Latent Feature based Codebook Retrieval," ECCV 2024. https://arxiv.org/abs/2407.12431 · https://github.com/LowLevelAI/GLARE
6. C. He, C. Fang, Y. Zhang, et al., "Reti-Diff: Illumination Degradation Image Restoration with Retinex-based Latent Diffusion Model," ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/6b9980876e7335e359df01911b5107d5-Abstract-Conference.html
7. J. Weng et al., "MambaLLIE: Implicit Retinex-Aware Low Light Enhancement with Global-then-Local State Space," NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/30699996ff411d48903c9752b782a5c1-Abstract.html · https://github.com/wengjiangwei/MambaLLIE
8. Z. Cui et al., "You Only Need 90K Parameters to Adapt Light," BMVC 2022. https://arxiv.org/abs/2205.14871 · https://github.com/cuiziteng/Illumination-Adaptive-Transformer
9. C. Li, C. Guo, C. C. Loy, "Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation," TPAMI 44(8), 2022. DOI 10.1109/TPAMI.2021.3063604
10. L. Ma et al., "Toward Fast, Flexible, and Robust Low-Light Image Enhancement," CVPR 2022. https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_Toward_Fast_Flexible_and_Robust_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf
11. Z. Fu et al., "Learning a Simple Low-Light Image Enhancer From Paired Low-Light Instances," CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/html/Fu_Learning_a_Simple_Low-Light_Image_Enhancer_From_Paired_Low-Light_Instances_CVPR_2023_paper.html
12. Y. Zhang, X. Guo, J. Ma, W. Liu, J. Zhang, "Beyond Brightening Low-light Images" (KinD++), IJCV 129, 2021. DOI 10.1007/s11263-020-01407-x
13. Y. Zhang, J. Zhang, X. Guo, "Kindling the Darkness: A Practical Low-light Image Enhancer," ACM MM 2019. https://arxiv.org/abs/1905.04161
14. X. Yi, H. Xu, H. Zhang, L. Tang, J. Ma, "Diff-Retinex," ICCV 2023. https://openaccess.thecvf.com/content/ICCV2023/html/Yi_Diff-Retinex_Rethinking_Low-light_Image_Enhancement_with_A_Generative_Diffusion_Model_ICCV_2023_paper.html
15. J. Hou et al., "Global Structure-Aware Diffusion Process for Low-Light Image Enhancement," NeurIPS 2023. https://arxiv.org/abs/2310.17577
16. H. Jiang et al., "LightenDiffusion," ECCV 2024. https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6440_ECCV_2024_paper.php
17. W. Wang, H. Yang, J. Fu, J. Liu, "Zero-Reference Low-Light Enhancement via Physical Quadruple Priors," CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Zero-Reference_Low-Light_Enhancement_via_Physical_Quadruple_Priors_CVPR_2024_paper.html
18. Z. Liang et al., "Iterative Prompt Learning for Unsupervised Backlit Image Enhancement," ICCV 2023. https://arxiv.org/abs/2303.17569
19. X. Xu, R. Wang, C.-W. Fu, J. Jia, "SNR-Aware Low-Light Image Enhancement," CVPR 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Xu_SNR-Aware_Low-Light_Image_Enhancement_CVPR_2022_paper.html
20. T. Wang et al., "Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and Transformer-Based Method," AAAI 2023. https://ojs.aaai.org/index.php/AAAI/article/view/25364
21. M. Afifi et al., "Learning Multi-Scale Photo Exposure Correction," CVPR 2021. https://openaccess.thecvf.com/content/CVPR2021/html/Afifi_Learning_Multi-Scale_Photo_Exposure_Correction_CVPR_2021_paper.html
22. J. Huang et al., "Exposure Normalization and Compensation for Multiple-Exposure Correction," CVPR 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Huang_Exposure_Normalization_and_Compensation_for_Multiple-Exposure_Correction_CVPR_2022_paper.html
23. J. Huang et al., "Learning Sample Relationship for Exposure Correction," CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Learning_Sample_Relationship_for_Exposure_Correction_CVPR_2023_paper.html
24. H. Wang, K. Xu, R. W. H. Lau, "Local Color Distributions Prior for Image Enhancement," ECCV 2022. DOI 10.1007/978-3-031-19797-0_20
25. Z. Li et al., "Real-Time Exposure Correction via Collaborative Transformations and Adaptive Sampling," CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Li_Real-Time_Exposure_Correction_via_Collaborative_Transformations_and_Adaptive_Sampling_CVPR_2024_paper.html
26. R. Cui, L. Niu, G. Hu, "Unsupervised Exposure Correction," ECCV 2024. https://arxiv.org/abs/2507.17252
27. M. T. Islam et al., "LoLI-Street: Benchmarking Low-light Image Enhancement and Beyond," ACCV 2024. https://openaccess.thecvf.com/content/ACCV2024/html/Islam_LoLI-Street_Benchmarking_Low-light_Image_Enhancement_and_Beyond_ACCV_2024_paper.html
28. C. Li, C. Guo, et al., "Low-Light Image and Video Enhancement Using Deep Learning: A Survey," TPAMI 44(12), 2022. https://arxiv.org/abs/2104.10729
29. W. Yang et al., "Sparse Gradient Regularized Deep Retinex Network…," IEEE TIP 30, 2021 (LOL-v2) [U: recheck]
30. J. Cai, S. Gu, L. Zhang, "Learning a Deep Single Image Contrast Enhancer from Multi-Exposure Images," IEEE TIP 27(4), 2018 (SICE) [U: recheck]

---

**STOP: awaiting approval.** Not yet approved:
- Stage 0 inference;
- new Phase-5 code;
- downloads (official HVI-CIDNet, DarkIR, IAT and Zero-DCE++ weights);
- `pyiqa` (NIQE / BRISQUE);
- unsealing SICE Part 2;
- any training.
