# Research-gap verification and novelty review — LLIE B.Tech project

Date: 2026-11-01. Status: **review only; nothing implemented, trained or downloaded for it.**
Labels used throughout:

- **[F]** Verified fact: from our repo or from a paper page read during this review.
- **[I]** My interpretation.
- **[H]** Unverified hypothesis, still to be tested.
- **[M]** Background from memory, not re-verified in this review. Treat [M] items as "check before citing".

---

## 1. Audit of completed work (to "Phase 3")

| Item | Status | Evidence |
|---|---|---|
| Phase 2 U-Net baseline (LOL-v2 real+syn, scene-disjoint val, 5,600 steps) | Complete. Reproduced exactly on the Mac (0 discrepancies / 3,255 images) | [F] `stage0/README.md`, `stage0/outputs/phase2_unet_best/reproduction_report.md` |
| Stage 0 colour / failure diagnosis | Complete | [F] `stage0/outputs/phase2_unet_best/colour_eval/summary.md` |
| Phase 3 = **model selection** (Tier A, 1 seed, FP32) | 3 of 4 models trained. DarkIR-m not trained (GPU quota) | [F] `stage1/TIER_A_RESULTS.md` |
| Tier A val PSNR (mean real/syn) | HVI-CIDNet 22.58, Retinexformer 22.38, U-Net+expaug 18.11 (U-Net Phase 2: 18.39) | [F] `stage1/progress_demo_20261019_1310/SUMMARY.md` |
| Development evaluation of Tier A on SICE / LoLI-Street | **Not done.** Only 2 LoLI scenes shown in the demo | [F] |
| Test-set evaluation | Not done (correct: not allowed yet). SICE Part 2 sealed | [F] |
| **Uncertainty estimation** | **Does not exist in this repo**: no code, configs, notebooks or results (grep for uncertainty/NLL/Laplace/conformal returned nothing) | [F] |
| MILL dataset | **Not in `data/`** (present: LOL, LOL-v2, LoLI-Street, SICE; DarkVision and SID are empty) | [F] |

**Discrepancy to resolve:** the prompt describes Phase 3 as "initial uncertainty estimation". The repository's Phase 3 is model selection. If uncertainty work exists outside this repo (a notebook or a teammate's code), it must be added before it can be reported. Until then, the honest project status is: **baselines and model selection done; no uncertainty result exists.**

**Measured problem that the evidence supports most strongly** ([F]; Phase-2 U-Net, Stage 0):

| LoLI-Street level | over-exposed (ΔL*>10) | PSNR: model / unprocessed input / fixed-target scaling |
|---|---|---|
| dense (darkest) | 0.0 % | — |
| moderate | 31.8 % | — |
| light (mildest) | **100.0 %** | **13.15** / 13.92 / 20.74 dB |

On the mildest inputs, the U-Net is **worse than doing nothing**. A trivial global scaling beats it in ΔE00 on 93.5 % of light images ([F]: "beats scaled" = 6.5 %). Tier A models still over-brighten LoLI scene 30231 "light" by ΔL* +19 to +26 ([F], one scene only). Whether this holds for every image of Retinexformer/HVI-CIDNet is **[H]** and still unmeasured. This, not uncertainty, is the problem our data actually shows.

---

## 2. Literature review (2024–2026 emphasis), grouped by theme

Status key: **PR** = peer-reviewed venue confirmed. **Pre** = preprint (no acceptance found). **?** = not verified.

### A. Backbones we already use
- **Retinexformer**, ICCV 2023, PR [M]. **HVI-CIDNet**, CVPR 2025, PR [M]. **HVI-CIDNet+**, arXiv 2507.06814, Pre [F: search result]. **DarkIR**, CVPR 2025, PR [M].
- Other recent backbones found: SAIGFormer (arXiv 2507.15520) and BSMamba (arXiv 2506.18346), Pre [F: listing]. The NTIRE 2026 LLIE challenge report is arXiv 2604.17669 [F: listing].

### B. Uncertainty *inside* LLIE networks (the named competitors)
- **U2CLLIE**, arXiv 2508.04176 (6 Aug 2025), Pre [F, read]. Its "uncertainty" is the **normalized entropy of feature activations**, used as the *value* term of a cross-attention. **No loss supervises it and no calibration is measured.** Metrics: PSNR/SSIM/LPIPS/NIQE only. LOL-v2-real PSNR 22.63.
- **MFLab**: only a ResearchGate listing was found. CIELab decoupling + "uncertainty-aware constrained refinement". The year and venue are inconsistent and its loss is not verified. **Status ?. Do not cite until the PDF is obtained.**
- **"UQuadCGAN"**: **not found under this name.** The closest match is **UCAMNet**, a Springer book chapter (doi prefix 10.1007/978-981-95-6957-1_22): an unsupervised HVI-space GAN with an "uncertainty constraint" on the intensity channel, PR (book chapter) [F: listing]. Please send the exact citation if UQuadCGAN is a different paper.
- **UGMamba**, SSRN abstract 6947161, Pre (SSRN, no arXiv) [F: listing]. Uncertainty-guided restoration in a state-space block. Calibration not mentioned in the abstract.
- **GSAD** (global structure-aware diffusion), NeurIPS 2023 [M]. Learns an uncertainty map used **as a loss weight**, not calibrated [F: search snippet].
- **Data-relativistic uncertainty** for anime-scene LLIE, arXiv 2512.21944, Pre [F: snippet]. Data uncertainty is used to re-weight the objective.
- **LLFlow**, AAAI 2022 [M]. A normalizing flow trained with NLL. It is a *distribution* model, but not evaluated for calibration.
- **BlindPSNR**, arXiv 2607.27628 (30 Jul 2026), Pre, "submitted to IEEE" [F, read]. Predicts the PSNR of an enhanced image without GT and uses it to **select among enhancement candidates** (top-1 89.5 % in-domain, 12.5–38 % cross-dataset). It has a heteroscedastic log-variance head, but the variance is **only used inside the training loss; no calibration metric is reported.**
- **Finding [F, within the searches made]:** **no LLIE paper found that evaluates calibration** (coverage, ECE for regression, sparsification, PIT). Uncertainty in LLIE is used *heuristically* (attention, loss weighting). This is a real but **narrow** gap, and the search was not exhaustive (Google Scholar, IEEE Xplore and ScienceDirect could not be searched systematically).

### C. Calibrated / conformal uncertainty for image-to-image problems (adjacent field)
- **Angelopoulos et al.**, "Image-to-Image Regression with Distribution-Free UQ", arXiv 2202.05265, ICML 2022 [F: listing; venue M]. Per-pixel calibrated intervals with risk control (RCPS).
- **Kutiel et al.**, conformal prediction masks, arXiv 2211.15211 [F]; ICLR 2023 [M].
- **QUTCC** (quantile training + conformal calibration), arXiv 2507.14760, Pre [F]. **Self-supervised conformal (SURE)**, arXiv 2502.05127, and the **Poisson** version, arXiv 2502.19194 [F]. Metric-guided conformal bounds, arXiv 2404.15274 [F]. Belhasin et al., principal uncertainty quantification [M].
- **[I]** Calibrating *per-pixel* uncertainty for LLIE with these tools would be an **application** of existing methods, not new methodology, and it **does not change the enhanced image**. It therefore fails the requirement that the contribution be in enhancement itself.

### D. Brightness mismatch / global photometric alignment
- **GT-Mean Loss**, Liao et al., ICCV 2025, PR [F, read]: L = W·L(f,y) + (1−W)·L((E[y]/E[f])·f, y), with W from a Bhattacharyya distance (σ = 0.1). It is a *training loss*. It does not study varying input darkness.
- **PAL** (photometric alignment loss), arXiv 2604.08172 (9 Apr 2026), Pre [F, read]. A closed-form 3×3 affine alignment inside the loss, applied across 6 tasks. It describes GT-Mean as the "scalar family".
- **[I]** Both papers show that a large part of supervised LLIE error is **global photometry**. Both fix *training*. At test time the right brightness is still unknown, and "GT-mean" evaluation uses the GT, so it is an **oracle**.

### E. Multi-illumination-level LLIE — the closest competitors to our measured problem
- **MILL**, Pilligua, …, M. S. Brown, Vazquez-Corral, arXiv 2511.15496 (v2 9 Apr 2026), Pre [F, read].
  - Dataset: 50 scenes × 11 light levels, one GT per scene (the brightest), DSLR + phone, RAW + lux. Split 30/12/8 scenes. Project page color.cvc.uab.cat/mill; **download and licence unverified**.
  - Finding: methods enhance very dark inputs but **fail on moderately dark ones**, and oversaturation grows with input brightness. Retinexformer and CIDNet degrade when LOL inputs are blended toward the GT.
  - Method: auxiliary losses on Retinexformer's bottleneck. An intensity-prediction loss on one latent channel plus a scene triplet loss.
  - **[I] This is an independent confirmation of our Stage 0 finding, and it pre-empts "level-invariant training" as a novelty.**
- **CLE-RWKV + Light100**, arXiv 2603.25296 (26 Mar 2026), Pre [F, read].
  - Enhancement **conditioned on a target luminance β** (FiLM) in HVI space. 275 scenes × 100 light levels. Noise-decoupled targets.
  - **β is supplied by the user at inference.** The paper does not say how to choose β automatically or reliably. Code/data release not stated.
- **CLE Diffusion**, ACM MM 2023, PR [F]. A diffusion model with a user-chosen brightness embedding (mean intensity) and SAM regions.
- **Automatic amplification estimation:** LLPackNet (arXiv 2011.14133; BMVC 2020 [M]) predicts the gain from the input histogram. L3Fnet (arXiv 2003.02438) and KinD (user-set ratio α) are related [F: snippets]. **These are point estimates with no uncertainty or calibration.**
- **InterLight**, IJCAI 2026 / arXiv 2605.19982, PR (IJCAI) [F: snippet]. Perturbation-invariant consistency with physics-guided illumination augmentation.

### F. Exposure correction (multi-exposure, same GT)
- MSEC (Afifi, CVPR 2021) [M]; ENC (Huang, CVPR 2022, PR) [F]; Ntumba consistency modelling (BMVC 2021) [F]; unsupervised exposure correction (Cui, ECCV 2024, PR) [F]; VECNet dynamic scenes (arXiv 2402.17296) [F].
- **[I]** Under/over-exposure invariance is well studied. Repeating it in LLIE is not novel.

### G. Colour fidelity in LLIE
- "Towards colour-faithful LLIE via adaptive colour debiasing and saturation rectification", arXiv 2608.10512 (2026), Pre [F: listing]. Our measured HVI-CIDNet desaturation (chroma 0.85 on real) is relevant, but **this direction is being actively published**.

### H. Frequency-domain, self-supervised and zero-reference LLIE (context only) [M]
- FourLLIE (ACM MM 2023), UHDFour (ICLR 2023), Zero-DCE / Zero-DCE++ (CVPR 2020 / TPAMI 2021), SCI (CVPR 2022), PairLIE (CVPR 2023), CLIP-LIT (ICCV 2023).
- Relevant only as baselines or for generalization arguments. **Do not use pretrained weights** (leakage, see Stage 1 notes).

### I. Adaptive-brightness LLIE (2025, mostly journals)
- Adaptive enhancement matrix (Multimedia Systems 2025), joint global-local illumination with early stopping (arXiv 2504.00400), and others [F: listings].
- They adapt strength heuristically. **None found reports per-darkness-level evaluation or a calibrated brightness target.**

### J. Downstream / video (excluded from the primary contribution as required)
- Not reviewed further. Future extension only.

---

## 3. Three-pass gap verification

**Pass 1 — direct competitors (uncertainty in LLIE).**
- U2CLLIE, UGMamba, UCAMNet, GSAD, BlindPSNR and (unverified) MFLab all already claim "uncertainty-aware LLIE".
- **Not found in any of them:** (a) calibration measured or guaranteed; (b) uncertainty that is illumination-level-conditional; (c) uncertainty that *controls* the enhanced output with a stated reliability property.
- **Verdict:** "uncertainty-aware LLIE" is **taken**. "Calibrated" is unclaimed in what we could read.

**Pass 2 — adjacent fields.**
- Conformal image-to-image regression exists (Angelopoulos 2022, QUTCC, SURE-conformal), so **"calibrated per-pixel intervals" is not methodologically new**. It would only be "first applied to LLIE", which is weak.
- Brightness-target conditioning exists (CLE Diffusion, CLE-RWKV), but the target is user-chosen.
- Automatic gain prediction exists (LLPackNet, L3Fnet), but as point estimates.
- Multi-level failure analysis exists (MILL).

**Pass 3 — novelty challenge (acting as a hostile reviewer).**
- *"Illumination-conditioned uncertainty calibration"* as stated by us: a reviewer will ask *"what does calibration improve in the enhanced image?"* If the answer is "nothing, it is a side output", **the primary-contribution requirement fails.** → **Reject as stated.**
- *"Level-adaptive enhancement"* alone: MILL and CLE-RWKV already cover it. → **Reject as the headline.**
- **What survives:** the **target exposure** of an enhanced image is genuinely ambiguous. The GT brightness differs across datasets and scenes; this is the reason GT-mean / PAL exist.
  - CLE-RWKV makes it controllable but leaves the choice to the user.
  - LLPackNet predicts it, but with no uncertainty.
  - **Nobody found predicts the target exposure as a *calibrated distribution*, conditioned on the input darkness level, and uses it to choose the output brightness with a stated over-exposure risk.**
  - This joins Pass 1's gap (calibration) and our measured failure (over-exposure of mildly dark inputs), and it changes the enhanced image itself.
- **Honest strength:** this is an **integration-level** novelty: a known conditioning mechanism + known conformal machinery + a new target and decision rule. It is not a new building block. It is adequate for a B.Tech thesis and a workshop or mid-tier venue. Novelty at the level of a top-tier main conference is **not** established.

---

## 4. Rejected or weak directions (and why)

| Direction | Reason for rejection |
|---|---|
| Per-pixel calibrated uncertainty maps as the main contribution | Not an enhancement contribution. The conformal methods already exist (C). |
| Yet another "uncertainty-aware" attention / loss weighting | Directly overlaps U2CLLIE, UGMamba, UCAMNet and GSAD. No new claim. |
| Level-invariance / consistency training on synthetic re-exposure | MILL (aux losses), InterLight (PIC) and exposure-correction work (F) cover it. Our own U-Net + exposure augmentation **did not help** (18.11 vs 18.39 val; [F]). |
| A GT-Mean / PAL-style loss variant | Two recent papers. Marginal. |
| A new backbone (Mamba / RWKV / diffusion) | Compute (free T4, quota-limited). Crowded. Diffusion is infeasible on our budget. |
| Colour-fidelity module only | Measurable for us, but arXiv 2608.10512 and others publish it right now. Keep it as a secondary metric. |
| YOLO, video or surveillance | Excluded by the brief. Future work. |

---

## 5. Alternative proposals

- **P1 — Calibrated per-pixel uncertainty for LLIE.** Heteroscedastic head on HVI-CIDNet + split-conformal per-pixel intervals, with coverage by darkness level.
- **P2 — Illumination-level-consistent training.** Synthetic multi-level re-exposure of LOL-v2 pairs + feature-consistency loss (MILL/InterLight-like, without real multi-level data).
- **P3 — Calibrated exposure-target prediction + exposure-conditioned enhancement with risk-controlled over-exposure (RECOMMENDED).** Details in §6–§12.
- **P4 — Uncertainty-guided per-pixel fusion of Retinexformer and HVI-CIDNet.** Fuse the two outputs pixel-wise using predicted error.
- **P5 — Colour-faithful HVI enhancement.** A chroma-preservation term or module for HVI-CIDNet's desaturation on real data.
- **P6 — No-reference output selection.** Pick the best of N brightness variants with a BlindPSNR-like predictor plus calibrated confidence.

**Ranked comparison** (1–5, 5 = best; "Enh." = contribution to the enhanced image itself):

| | Gap strength | Feasibility | Enh. | Demonstrability | Compute | Data | Publication | **Total** |
|---|---|---|---|---|---|---|---|---|
| **P3** | 4 | 5 | 4 | 5 | 5 | 4 | 3 | **30** |
| P6 | 3 | 4 | 3 | 4 | 4 | 4 | 3 | 25 |
| P5 | 2 | 4 | 4 | 4 | 4 | 5 | 2 | 25 |
| P4 | 2 | 4 | 3 | 4 | 4 | 5 | 2 | 24 |
| P1 | 3 | 4 | 1 | 4 | 4 | 5 | 2 | 23 |
| P2 | 1 | 4 | 3 | 3 | 3 | 4 | 1 | 19 |

P3 scores 5 on compute because its core needs **no new enhancer training**. A tiny predictor is trained and the enhancer is frozen. P6 overlaps BlindPSNR (July 2026) heavily. P5 is a good *secondary* add-on if time allows.

---

## 6. Final direction

**Working title:** *Know How Bright: Calibrated, Illumination-Conditioned Exposure Targets for Reliable Low-Light Image Enhancement*.
Academic form: *"Illumination-Conditioned Calibrated Exposure Prediction for Over-Enhancement-Safe Low-Light Image Enhancement"*.

**Problem.**
- Supervised LLIE networks learn one fixed brightness mapping.
- On mildly dark inputs they over-expose: 100 % of LoLI "light" images for our U-Net, worse than the input itself [F]. MILL reports the same pattern independently [F].
- The correct output brightness is ambiguous (dataset/scene-dependent GT exposure). Current remedies either use the GT at test time (GT-mean evaluation, an oracle) or ask the user (CLE-RWKV, CLE Diffusion).

**Gap.**
- No verified LLIE method predicts the target exposure **as a calibrated distribution conditioned on input illumination level**.
- None uses that distribution to set the output brightness with a **stated, testable bound on over-exposure frequency**.
- Existing gain predictors (LLPackNet) are point estimates. Existing "uncertainty-aware LLIE" (U2CLLIE etc.) is uncalibrated and does not control exposure.

**Objectives.**
1. Quantify how much of each Tier A model's error is global exposure: PSNR vs GT-mean PSNR, and ΔL* by darkness level.
2. Build a lightweight predictor of the target exposure distribution p(β* | x).
3. Calibrate it per illumination level (Mondrian conformal), with exact finite-sample coverage *under exchangeability*.
4. Apply the chosen β to a frozen enhancer (global HVI-intensity/gain re-targeting). Fine-tune a β-conditioned enhancer only if the simpler route is insufficient.
5. Show reduced over-enhancement on mild inputs at no or small loss on dark inputs, across LOL-v2, LoLI-Street levels, SICE exposures and (if approved) MILL levels.

**Contributions** (claimed only if the experiments support them):
- C1. A per-darkness-level diagnosis of exposure error for Retinexformer / HVI-CIDNet / U-Net, separating global-exposure error from content error.
- C2. An illumination-conditioned, calibrated exposure-target predictor (≈10⁴–10⁵ params). This is the first calibrated one that we could find in LLIE.
- C3. A risk-controlled exposure rule. Under exchangeability, it bounds the probability that the output is brighter than the reference by a user-set α *within each darkness bin*.
- C4. Plug-in use on existing enhancers without retraining them, plus the evaluation protocol (per-level metrics, coverage under shift).

---

## 7. Methodology and architecture

```
input x ──► [frozen enhancer E (HVI-CIDNet / Retinexformer)] ──► ŷ0 (native output)
   │
   ├──► darkness bin k(x)  = bin of mean input luminance (bins fixed on the TRAIN set)
   │
   └──► [exposure head g]:  features = 64-bin log-luminance histogram of x
                             + 16 global stats of ŷ0 (mean/percentiles of L*)
                             + pooled 1/16-scale tiny CNN features (optional)
                             ─► (μ(x), log σ(x))  = predictive distribution of β*
          │
          └──► Mondrian conformal calibration per bin k ─► β_med, [β_lo, β_hi], β_safe(α)
                                                         │
ŷ0 ──► [exposure re-targeting R(ŷ0, β)]  ◄──────────────┘
          (scale the HVI intensity / L* of ŷ0 so that mean luminance = β; chroma kept)
          ─► ŷ(β)   final enhanced image (+ optional reported interval of exposures)
```

- **β\*** is the mean CIE L* (or linear luminance; fixed once, before experiments) of the reference image.
- **Re-targeting R**, minimal variant: a global gain on linearised RGB with a soft shoulder, the same family as our Stage 0 "fixed-target scaling", which already beat the U-Net on mild inputs [F]. It is applied to ŷ0 (not to x), so the enhancer's denoising and colour are kept.
- **Optional variant R′** (only if R leaves a significant gap to the oracle): FiLM-condition HVI-CIDNet's intensity branch on β, as in CLE-RWKV. Fine-tune from our Tier A best.pth with targets re-exposed in linear space. This costs GPU time and needs approval.
- **Why no bigger model:** the oracle experiment (§10, E1) bounds what any exposure fix can gain. If R with the oracle β already gains most of it, R′ is unnecessary complexity.

---

## 8. Mathematics

**Target.** β\*(y) = mean_p L\*(y_p), where L\* is the CIELAB lightness of the reference y.

**Predictive model.** g outputs μ(x), σ(x). Trained on the TRAIN split only with Gaussian NLL:
  L_NLL = ½ log σ² + (β\* − μ)² / (2σ²)
(Alternatives for the ablation: Laplace NLL; pinball loss for quantiles τ ∈ {0.05, 0.5, 0.95}.)

**Illumination bins.** k(x) ∈ {1..K}, K = 3. Tertiles of the input mean luminance on the TRAIN set. Observable at test time, so Mondrian conformal is valid.

**One-sided risk control against over-exposure.**
- Score on calibration pair i: r_i = (μ(x_i) − β\*_i) / σ(x_i). Large r means the prediction is too bright.
- For bin k with n_k calibration points: q_k = the ⌈(n_k+1)(1−α)⌉-th smallest r_i in bin k.
- Safe target: β_safe(x) = μ(x) − q_{k(x)} σ(x).
- **Claim (standard split-conformal result):** if the calibration and test pairs of bin k are exchangeable, then P(β_safe(x) > β\*(y) | k) ≤ α. Equivalently, P(r_test > q_k) ≤ α.
- **This is not a guarantee under distribution shift** (LoLI-Street, SICE, MILL have other GT conventions). There we only *measure* the coverage.

**Two-sided interval** (reported to the user): r_i = |β\*_i − μ_i| / σ_i, giving [μ − q_k σ, μ + q_k σ] with coverage ≥ 1 − α per bin under exchangeability.

**From target to image.**
- The bound concerns β\* vs β_safe.
- The output satisfies mean L\*(ŷ(β)) = β + e, where e is the re-targeting error.
- Hence P(mean L\*(ŷ) > β\* + τ) ≤ α + P(|e| > τ). We **measure e**; we do not assume e = 0.
- No guarantee on per-pixel error, PSNR or perceptual quality is claimed.

**Operating points.** β_med = μ (expected-quality point) and β_safe(α) (conservative point). We report the PSNR / over-exposure trade-off curve over α ∈ {0.05, 0.1, 0.2, 0.3, 0.5}.

---

## 9. Pseudocode

```python
# --- training (TRAIN split only; CPU-sized model, but still needs approval) ---
for x, y in train_pairs:
    y0   = E_frozen(x)                        # cached once
    feat = features(x, y0)                    # histogram + global stats
    mu, log_sigma = g(feat)
    loss = gaussian_nll(beta_star(y), mu, log_sigma)
# model selection for g: 5-fold CV on TRAIN by scene (validation set not touched)

# --- calibration (held-out calibration split; never test) ---
for k in bins:
    r = [(mu(x) - beta_star(y)) / sigma(x) for x, y in calib if bin(x) == k]
    q[k] = sorted(r)[ceil((len(r) + 1) * (1 - alpha)) - 1]   # needs len(r) >= 1/alpha - 1

# --- inference ---
def enhance(x, alpha=None):
    y0 = E_frozen(x); mu, sigma = g(features(x, y0)); k = bin(x)
    beta = mu if alpha is None else mu - q[k] * sigma
    return retarget(y0, beta), (mu - qq[k] * sigma, mu + qq[k] * sigma)   # image + 2-sided interval
```

---

## 10. Datasets, splits, training/testing and experiments

**Splits (no change to existing splits):**
- **Train g:** LOL-v2 real + synthetic train (625 + 808).
- **Calibration:** the scene-disjoint LOL-v2 validation (64 + 92). It was already used to select the *enhancer* checkpoint. That choice does not touch g or β\*, and the scores r depend only on g and the reference. This reuse is still recorded as a limitation. If needed, split val by scene into a selection half and a calibration half.
- **Development (distribution shift, coverage only measured):** LoLI-Street Val by level (dense / moderate / light), SICE Part 1 test by exposure.
- **MILL** test scenes (8 scenes × 10 levels, one GT): the most direct per-level benchmark. **Download needs your approval and a licence check.**
- **Final:** SICE Part 2, sealed until the very end. LOL-v2 test only at the end.

**Experiments** (E0–E2 are inference only; they decide whether to continue):

| ID | Experiment | Decides |
|---|---|---|
| E0 | Tier A models on LoLI levels + SICE: ΔL* and over-exposure rate per level | Is the failure general beyond the U-Net? If over-exposure on "light" is < 20 % for HVI-CIDNet/Retinexformer → **the direction weakens; stop and re-plan** |
| E1 | Oracle: re-target ŷ0 to the true β\* (GT-mean-style) | Upper bound of the method. If the oracle PSNR gain is < 0.5 dB on val and < 1 dB on LoLI light → **stop** |
| E2 | Constant-β and input-only heuristics (gain from input histogram, LLPackNet-like point estimate) | The baselines our predictor must beat |
| E3 | Train g; calibration plots (PIT histogram, coverage per bin, interval width); NLL/CRPS | C2 |
| E4 | β_med and β_safe(α) on all development sets; trade-off curves | C3, C4 |
| E5 | Ablations: no bins vs Mondrian bins; Gaussian vs Laplace vs quantile; histogram only vs + ŷ0 stats; K = 2/3/4; R vs R′ (only if E1–E4 justify it) | Design choices |
| E6 | Shift: coverage on LoLI/SICE/MILL; recalibration with m = 10/20/50 target-domain pairs | Honest limits of the guarantee |
| E7 | Seeds: ≥3 seeds for the finalist enhancer (already the agreed Tier B rule) | Variance |

**Baselines:**
- Each frozen enhancer as is.
- Enhancer + GT-Mean-loss fine-tune (optional, GPU).
- Enhancer + point gain predictor (LLPackNet-style).
- Enhancer + constant target.
- Oracle (upper bound).
- Unprocessed input and fixed-target scaling (Stage 0 floors).

**Metrics:**
- **Fidelity:** PSNR, SSIM, LPIPS; GT-mean PSNR (reported separately, clearly labelled as an oracle-aligned metric).
- **No-reference:** NIQE (implementation still to be pinned, e.g. pyiqa; needs approval).
- **Colour:** ΔE00 and colour-only ΔE, chroma ratio (existing `colour_metrics.py`).
- **Exposure:** ΔL\* and the over-exposure rate (ΔL\* > 10) per darkness level.
- **Calibration:** per-bin coverage, mean interval width, PIT, NLL, CRPS.
- **Control accuracy:** re-targeting error e.
- **Cost:** params, ms/image on T4 and Mac CPU.

All metrics are per illumination level and cross-dataset, with bootstrap CIs (LoLI clustered by scene, as in Stage 0).

---

## 11. Roadmap (each GPU / download step needs your separate approval)

| Week | Work | Compute |
|---|---|---|
| 1 | Approve the plan. E0 + E1 + E2 for Tier A models on val / LoLI / SICE (inference + existing metrics) | Mac CPU (several hours) or a short T4 session |
| 2 | Kill/continue decision on E0/E1. Implement g, features, conformal, re-targeting + unit tests (conformal coverage on synthetic data, monotonic re-targeting) | Mac (code only) |
| 3 | Train g (seconds to minutes; tiny), E3–E4 | Needs approval (CPU-sized) |
| 4 | E5 ablations, E6 shift study; decide on MILL | — |
| 5 | Optional R′ (FiLM fine-tune) only if E1 vs E4 leaves a gap > 0.5 dB | T4, ~2 h per run, approval |
| 6 | Seeds (Tier B), final SICE Part 2 + LOL-v2 test once, write-up | T4 |

**DarkIR-m:** the Tier A run is still pending. It is no longer critical for this direction, which works on any frozen enhancer.

---

## 12. Risks and mitigations

| Risk | Mitigation / honest consequence |
|---|---|
| E1 oracle gain is small for Retinexformer/HVI-CIDNet (they may already adapt better than the U-Net) | Pre-registered stop rule. Fall back to P5 (colour fidelity) or P6 |
| The reference exposure differs across datasets → coverage breaks under shift | Expected. Report it, and show few-shot recalibration (E6). Never claim a guarantee there |
| Global re-targeting cannot fix *local* over-exposure (e.g. lamps, sky) | Measure clipped-highlight fraction. A local variant is future work |
| 156 calibration images / 3 bins → coarse quantiles (n_k ≈ 50 gives valid but loose bounds) | Report n_k. Use α ≥ 0.1. Merge bins if n_k < 20 |
| Reviewer: "this is just post-processing" | Correct as a description. The contribution is the *calibrated, level-conditional target and risk rule*, validated against the oracle and point-estimate baselines. Sell it as reliability, not as a new architecture |
| MFLab / UQuadCGAN / UGMamba full texts not read | Read their PDFs before writing related work. If one of them already predicts a calibrated exposure target, re-scope |

---

## 13. Final title (proposed)

**"Illumination-Conditioned Calibrated Exposure Prediction for Over-Enhancement-Safe Low-Light Image Enhancement"**

---

## 14. Strict-reviewer verdict

- **Weak accept for a workshop / mid-tier venue, *conditional* on E0–E1 showing a real exposure gap for strong backbones. Reject for a top-tier main track.**
- **Strengths:** the problem is real and measured; the claim is precise and falsifiable; the method is cheap; the evaluation covers darkness levels and calibration properly.
- **Weaknesses:**
  - The novelty is integrative.
  - The guarantee holds only under exchangeability.
  - A global target cannot address local errors.
  - Only one train domain (LOL-v2).
  - MILL is preprint-level competition that a reviewer will know.

---

## 15. Explicit answers

1. **Is our direction sufficiently different?**
   - *As originally stated* ("illumination-conditioned uncertainty calibration"): **no**. It overlaps the uncertainty-aware LLIE papers, and calibration alone is not an enhancement contribution.
   - *As re-scoped* (a calibrated, level-conditional exposure target that sets the output with an over-exposure risk bound): **yes, as far as the literature we could verify shows.**
2. **Exact novelty:** predicting the *target exposure* of the enhanced image as a calibrated, illumination-level-conditional distribution, and the risk-controlled rule that turns it into the output brightness. Not the conditioning mechanism, not conformal prediction, and not "uncertainty-aware LLIE" in general.
3. **Closest competitors:** CLE-RWKV (2603.25296, user-given β), MILL (2511.15496, level analysis + intensity loss), GT-Mean (ICCV 2025) / PAL (2604.08172), LLPackNet (point gain prediction), CLE Diffusion (MM 2023). On the uncertainty side: U2CLLIE, UGMamba, UCAMNet, BlindPSNR.
4. **Evidence for the gap:**
   - [F] Our over-exposure numbers by level.
   - [F] MILL's independent finding.
   - [F] CLE-RWKV requires a user β.
   - [F] U2CLLIE and BlindPSNR measure no calibration.
   - [F] GT-mean evaluation needs the GT.
   - The absence of a calibrated exposure predictor is **[F within our searches]** only. It is not proof of absence.
5. **What a reviewer would criticise:** "post-processing, not a network"; "guarantee breaks under shift"; "global only"; "only LOL-v2 for training"; "is the over-exposure problem still present in strong backbones?" (E0 answers the last one).
6. **What to implement next:** nothing until you approve this. After approval, **E0 + E1 only** (inference on existing Tier A checkpoints). These two decide whether the direction lives.
7. **Retain or abandon uncertainty calibration?** **Retain, re-scoped.** Calibrate the *exposure target* that controls the image. **Abandon** per-pixel calibrated uncertainty maps as the main contribution; at most they are an optional extension.

---

## 16. Confidence assessment (qualitative)

**Moderate.**
- **For:** the over-exposure failure is measured in our data and independently in MILL. The method's core is cheap and testable within our compute. Clear stop rules exist.
- **Against:**
  - The literature search could not cover IEEE Xplore, ScienceDirect or Scholar systematically.
  - Three competitor papers (MFLab, the paper behind "UQuadCGAN", UGMamba) were not read in full.
  - The 2026 preprint pace in this area is high, so a near-identical paper may appear.
  - E0 may show that strong backbones over-expose much less than our U-Net, which would shrink the gap.

## Sources read or listed in this review
- [GT-Mean Loss, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Liao_GT-Mean_Loss_A_Simple_Yet_Effective_Solution_for_Brightness_Mismatch_ICCV_2025_paper.html) · [arXiv 2507.20148](https://arxiv.org/abs/2507.20148)
- [PAL — Global Photometric Alignment, arXiv 2604.08172](https://arxiv.org/abs/2604.08172)
- [MILL — Evaluating LLIE Across Multiple Intensity Levels, arXiv 2511.15496](https://arxiv.org/abs/2511.15496)
- [CLE-RWKV / Light100, arXiv 2603.25296](https://arxiv.org/abs/2603.25296)
- [CLE Diffusion, arXiv 2308.06725 (ACM MM 2023)](https://arxiv.org/abs/2308.06725)
- [U2CLLIE, arXiv 2508.04176](https://arxiv.org/abs/2508.04176)
- [BlindPSNR, arXiv 2607.27628](https://arxiv.org/abs/2607.27628)
- [UGMamba, SSRN 6947161](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6947161)
- [UCAMNet, Springer chapter](https://link.springer.com/chapter/10.1007/978-981-95-6957-1_22)
- [Data relativistic uncertainty, arXiv 2512.21944](https://arxiv.org/abs/2512.21944)
- [GSAD, arXiv 2310.17577](https://arxiv.org/abs/2310.17577)
- [Angelopoulos et al. 2022, arXiv 2202.05265](https://arxiv.org/abs/2202.05265)
- [Kutiel et al., arXiv 2211.15211](https://arxiv.org/abs/2211.15211)
- [QUTCC, arXiv 2507.14760](https://arxiv.org/abs/2507.14760)
- [SURE conformal, arXiv 2502.05127](https://arxiv.org/abs/2502.05127)
- [Poisson conformal, arXiv 2502.19194](https://arxiv.org/abs/2502.19194)
- [Metric-guided bounds, arXiv 2404.15274](https://arxiv.org/abs/2404.15274)
- [LLPackNet, arXiv 2011.14133](https://arxiv.org/abs/2011.14133)
- [L3Fnet, arXiv 2003.02438](https://arxiv.org/abs/2003.02438)
- [KinD, arXiv 1905.04161](https://arxiv.org/abs/1905.04161)
- [InterLight, arXiv 2605.19982](https://arxiv.org/abs/2605.19982)
- [ENC, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Exposure_Normalization_and_Compensation_for_Multiple-Exposure_Correction_CVPR_2022_paper.pdf)
- [Unsupervised Exposure Correction, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00986.pdf)
- [Exposure consistency modelling, BMVC 2021](https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/0038.pdf)
- [VECNet, arXiv 2402.17296](https://arxiv.org/abs/2402.17296)
- [Colour-faithful LLIE, arXiv 2608.10512](https://arxiv.org/abs/2608.10512)
- [HVI-CIDNet+, arXiv 2507.06814](https://arxiv.org/abs/2507.06814)
- [Adaptive global-local illumination, arXiv 2504.00400](https://arxiv.org/abs/2504.00400)
