# LLIE research direction: investigation report (planning only)

Date: 2026-11-08. Status: **research planning only.** Nothing has been implemented, trained, installed or downloaded. No existing file was changed.

Labels used in this report:

| Label | Meaning |
|---|---|
| **[V]** | Verified: the paper's title, authors and venue were checked on an official or proceedings page during this investigation (links in §10) |
| **[P]** | Measured in this project (file paths given) |
| **[H]** | My hypothesis or interpretation, not yet tested |
| **[U]** | Unverified detail. Check it in the paper before citing |

Published numbers from different datasets, splits or protocols are **not** compared with each other in this report.

---

## 0. What the project already has (inspected 2026-11-08)

| Asset | Location | Status |
|---|---|---|
| HVI-CIDNet Tier-A `best.pth` (epoch 46) | `experiments/phase3_model_selection/stage1/tierA_runs_local/tierA_hvi_cidnet_s42/` | Local [P] |
| Retinexformer Tier-A `best.pth` (epoch 50) | `.../tierA_retinexformer_s42/` | Local [P] |
| U-Net + exposure augmentation `best.pth` | `.../tierA_unet_expaug_s42/` | Local [P] |
| Phase-2 U-Net `best.pth` (epoch 45) | `experiments/phase3_model_selection/stage0/` | Local [P] |
| Vendored code | `third_party/` (HVI-CIDNet eb43d7d, Retinexformer 1e9a0ef, DarkIR 0910291) | Local [P] |
| Exposure augmentation (`hard_cap`, `soft_shoulder`) | `src/datasets/exposure_augment.py` | Implemented and smoke-tested [P] |
| Colour and quality metrics (ΔE00, ΔL*, chroma ratio, edge and noise ratios) | `src/metrics/colour_metrics.py` | Implemented [P] |
| Datasets | `data/LOL-v2`, `data/SICE` (8.8 GB, Part 2 **sealed**), `data/LoLI-Street` (2.6 GB) | Local [P] |
| Scene-disjoint LOL-v2 splits | `experiments/generalized_lol_v2/splits/` | Fixed [P] |

Two practical consequences:
- New experiments can start from existing checkpoints and code. **No new architecture is needed.**
- SICE Part 1 and LoLI-Street Val were already used as development evidence in Phases 2–3. They are therefore no longer untouched test sets.
  - **SICE Part 2 is the only clean final test set we have locally.** Unsealing it needs your approval.

---

## 1. Current LLIE research landscape

1. **Paired supervised enhancers.** These lead LOL-type benchmarks.
   - CNN, transformer and Retinex-guided designs: SNR-Aware (CVPR 2022), Retinexformer (ICCV 2023), LLFormer (AAAI 2023), HVI-CIDNet (CVPR 2025), DarkIR (CVPR 2025) [V].
   - Main recent trends: colour-space redesign (HVI), Retinex priors inside attention, and efficiency.
2. **State-space (Mamba) enhancers.** MambaLLIE (NeurIPS 2024) [V] uses global-then-local state-space blocks.
3. **Diffusion enhancers.** Diff-Retinex (ICCV 2023), GSAD (NeurIPS 2023) and LightenDiffusion (ECCV 2024) [V].
   - They give strong perceptual quality, but training and sampling cost much more. They are hard to fit on a T4 within 10 days [H].
4. **Zero-reference and unsupervised methods.** Zero-DCE++ (TPAMI 2022, about 10K parameters), SCI (CVPR 2022), PairLIE (CVPR 2023) and CLIP-LIT (ICCV 2023, backlit) [V].
   - They are cheap and flexible, but usually below supervised methods on paired metrics [H; common observation].
5. **Exposure correction, a separate line that handles both under- and over-exposure.**
   - MSEC dataset and network (CVPR 2021), ENC (CVPR 2022), LCDPNet (ECCV 2022, mixed exposure within one image), IAT (BMVC 2022, about 90K parameters, both LLIE and exposure correction), sample-relationship learning (CVPR 2023), CoTF (CVPR 2024, real-time) and unsupervised exposure correction (ECCV 2024) [V].
6. **Brightness-controllable enhancement.**
   - KinD (ACM MM 2019) lets the user set the light level [V]. Newer controllable work also exists: a two-stage exposure-controllable network (IEEE, 2025) and ControlLight (arXiv 2026) [V title only; details U].

**Key observation [H, supported by §3]:** LLIE methods (families 1–4) are trained on datasets whose inputs are almost all strongly under-exposed, such as LOL. Exposure-correction methods (family 5) are trained on datasets with a range of exposure values (MSEC). The two lines are mostly evaluated on different benchmarks. As a result, how LLIE models behave on *mildly* dark inputs is rarely reported, even though surveillance frames range from dark to near-normal.

---

## 2. Literature comparison table

Only **verified** papers are listed. The "Data / metrics" column gives what the papers state, as far as verified. Numbers are deliberately omitted, because the protocols differ.

| # | Paper (venue, year) | Authors | Architecture / idea | Data / metrics (as stated) | Strengths | Limitations relative to our problem |
|---|---|---|---|---|---|---|
| 1 | HVI: A New Color Space for Low-light Image Enhancement (CVPR 2025) | Q. Yan, Y. Feng, C. Zhang, G. Pang, K. Shi, P. Wu, W. Dong, J. Sun, Y. Zhang | HVI colour space (learnable intensity collapse) and a dual-branch CIDNet with LCA cross-attention; about 1.98 M params [P] | LOL-v1/v2 and others; PSNR, SSIM, LPIPS [U: full list] | Best validation PSNR and lowest colour error in our Tier A [P]; official code and weights | Trained on LOL-like dark inputs; no exposure-adaptive mechanism [H] |
| 2 | Retinexformer (ICCV 2023) | Y. Cai, H. Bian, J. Lin, H. Wang, R. Timofte, Y. Zhang | One-stage Retinex estimator plus illumination-guided transformer; 1.61 M params [P] | LOL-v1/v2, SID, SMID, SDSD, etc. [U] | Strong and simple; our Tier-A run is still improving at epoch 50 [P] | Same training-distribution limitation [H] |
| 3 | LLFormer: UHD Low-Light Image Enhancement (AAAI 2023) | T. Wang, K. Zhang, T. Shen, W. Luo, B. Stenger, T. Lu | Axis-based multi-head attention, cross-layer attention fusion | UHD-LOL4K/8K (4K: 5,999 train / 2,100 test pairs) | Designed for high resolution | Parameter count larger than #1–2 [U]; UHD training is heavy on a T4 [H] |
| 4 | Zero-DCE++ (TPAMI 2022, 44(8):4225–4238) | C. Li, C. Guo, C. C. Loy | Pixel-wise high-order curves; about 10K params; no paired data | Non-reference losses | Extremely light; real time | Lower fidelity than supervised methods [H]; exposure-control loss uses a fixed target level [U] |
| 5 | SNR-Aware LLIE (CVPR 2022) | X. Xu, R. Wang, C.-W. Fu, J. Jia | SNR map routes low-SNR regions to convolution and high-SNR regions to attention | Seven benchmarks; user study | Explicit noise awareness | Not exposure-adaptive |
| 6 | Kindling the Darkness / KinD (ACM MM 2019) | Y. Zhang, J. Zhang, X. Guo | Retinex decomposition with an adjustable light level | LOL | **User-set** brightness | The level is chosen by the user, not inferred; older backbone |
| 7 | Learning Multi-Scale Photo Exposure Correction (CVPR 2021) | M. Afifi, K. G. Derpanis, B. Ommer, M. S. Brown | Laplacian-pyramid network | **MSEC: over 24,000 images at several EVs** rendered from FiveK raw | Under and over exposure in one benchmark | Not low-light / noise specific |
| 8 | Exposure Normalization and Compensation (ENC, CVPR 2022) | J. Huang et al. | Plug-in module mapping features to an exposure-invariant space, plus compensation | Multi-exposure correction | **Plug-in**, exposure-invariant features | Built for exposure-correction backbones, not LLIE |
| 9 | Local Color Distributions Prior (LCDPNet, ECCV 2022) | H. Wang, K. Xu, R. W. H. Lau | Local colour-distribution embedding plus dual illumination | New LCDP dataset with mixed exposure | Over- and under-exposed regions in one image | Not trained for extreme low light |
| 10 | IAT: You Only Need 90K Parameters to Adapt Light (BMVC 2022) | Z. Cui et al. | Local branch plus global ISP branch (gamma / colour-matrix queries); about 90K params | LOL and exposure correction | **Closest prior art**: a single light model for both LLIE and exposure correction; pretrained weights | Separate weights per task [U]; not evaluated stratified by darkness level [U] |
| 11 | Learning Sample Relationship for Exposure Correction (CVPR 2023) | J. Huang et al. | Batch-level exposure-relationship regulariser | Exposure correction | No inference cost | For exposure-correction training |
| 12 | Real-Time Exposure Correction / CoTF (CVPR 2024) | Z. Li, F. Zhang, M. Cao, J. Zhang, Y. Shao, Y. Wang, N. Sang | Global 3D LUT plus pixel transforms, adaptive sampling | Exposure correction, high resolution | Real time | Not LLIE |
| 13 | Unsupervised Exposure Correction (ECCV 2024) | R. Cui, L. Niu, G. Hu | Emulated-ISP pairs; tiny transformation function | Radiometry Correction Dataset | Very small | Not LLIE |
| 14 | MambaLLIE (NeurIPS 2024) | J. Weng et al. | Global-then-local state-space blocks, implicit Retinex | LOL and others [U] | Recent state of the art | Custom CUDA kernels; risky on Colab [H] |
| 15 | Diff-Retinex (ICCV 2023) | X. Yi, H. Xu, H. Zhang, L. Tang, J. Ma | Transformer decomposition plus diffusion per component | Real low-light sets | Handles noise and colour | Heavy compute |
| 16 | GSAD (NeurIPS 2023) | J. Hou, Z. Zhu, J. Hou, H. Liu, H. Zeng, H. Yuan | Diffusion with curvature (global structure) and uncertainty regularisation | LOL-v1/v2 | Detail and contrast | Heavy compute |
| 17 | LightenDiffusion (ECCV 2024) | H. Jiang, A. Luo, X. Liu, S. Han, S. Liu | Latent-Retinex diffusion, unsupervised | Real benchmarks | Generalisation claim | Heavy compute |
| 18 | PairLIE (CVPR 2023) | Z. Fu, Y. Yang, X. Tu, Y. Huang, X. Ding, K.-K. Ma | Retinex learned from pairs of low-light images | Public benchmarks | No normal-light references needed | Lower fidelity [H] |
| 19 | SCI (CVPR 2022) | L. Ma et al. | Self-calibrated illumination; single block at inference | Unsupervised | Fast | Fidelity |
| 20 | CLIP-LIT (ICCV 2023) | Z. Liang, C. Li, S. Zhou, R. Feng, C. C. Loy | CLIP prompt learning for backlit images | Unpaired | Handles mixed lighting | CLIP dependency; backlit focus |
| 21 | DarkIR (CVPR 2025) | D. Feijoo, J. C. Benito, A. Garcia, M. V. Conde | Efficient CNN with frequency blocks for low light, noise and blur | LOLBlur, LOL [U] | Multi-degradation | We have no official training code (re-implemented) [P] |
| 22 | LoLI-Street (ACCV 2024) | M. T. Islam, I. Alam, S. S. Woo, S. Anwar, I. H. Lee, K. Muhammad | Dataset with 3 darkness levels, plus the TriFuse model | About 33K pairs plus 1,000 real unpaired test images | **Levels share one reference**, which allows a controlled darkness study | Synthetic darkening [P: audit] |
| 23 | Low-Light Image and Video Enhancement Using Deep Learning: A Survey (TPAMI 2022) | C. Li, C. Guo, et al. | Survey; LLIV-Phone dataset | — | Context | — |

---

## 3. Existing architecture comparison (practical, T4)

Our numbers come from **one protocol**: LOL-v2 scene-disjoint validation, 5,600 steps, FP32, seed 42. Published numbers are not shown, because they use other protocols.

| Model | Params | Our mean val PSNR | Training on T4 (5,600 steps) | Checkpoints | Fit for a 10-day project |
|---|---|---|---|---|---|
| **HVI-CIDNet** | 1.98 M [P] | **22.58 dB** [P] | ≈ 1.7 h (pilot estimate) [P] | Ours (local) and official LOL-v2 weights [P] | **High.** Small, colour-aware, known code |
| **Retinexformer** | 1.61 M [P] | 22.38 dB [P] | ≈ 1.5 h [P] | Ours (local) | **High.** Second backbone for a generality check |
| U-Net (+ exposure aug.) | 1.93 M [P] | 18.39 / 18.11 dB [P] | < 1 h [P] | Ours | Low as the main model (≈ 4 dB behind); useful as an ablation backbone |
| LLFormer | [U] | — | Not measured; UHD-oriented | Official | Medium-low |
| IAT | ≈ 90K [V] | — | Light | Official LLIE and exposure weights [V] | **Needed as a baseline** (closest prior art); a download needs your approval |
| Zero-DCE++ | ≈ 10K [V] | — | Light | Official [V] | Baseline only |
| DarkIR-m | 3.32 M [P] | — (no Tier-A result) | ≈ 1.6 h est. [P] | Re-implemented training | Medium-low |
| MambaLLIE / diffusion models | — | — | Custom kernels / heavy | — | Low in 10 days [H] |

**GPU memory [P]:** all Tier-A models trained on a T4 at batch 4 with 256×256 crops in FP32. Full-resolution validation on LOL-v2 fits in memory. SICE needs tiling (512 px tiles, 64 px overlap, already implemented).

---

## 4. Research gaps (supported by literature + our measurements)

### G1: LLIE models harm inputs that are only mildly dark (exposure non-adaptivity)
- **[P] Phase-2 U-Net, LoLI-Street (same reference at every level):**

  | Level | Model PSNR | Input PSNR (no processing) |
  |---|---|---|
  | dense | 18.20 dB | 6.30 dB |
  | moderate | 15.96 dB | 8.80 dB |
  | light | 13.15 dB | **13.92 dB** |

  - On the light level the model is worse than doing nothing.
  - The order dense > moderate > light holds in 99.6 % of scenes.
  - A non-learned rule (scale the input so its mean brightness is 0.48) reaches 20.67 dB on the light level.
- **[P] All four Phase-3 models do the same.** On LoLI scene 30231 (one scene, so illustrative only), ΔL* rises from about +3…+6 (dense) to about +19…+26 (light) for the U-Net, the U-Net with exposure augmentation, Retinexformer and HVI-CIDNet. So stronger models still over-brighten mildly dark inputs.
- **[P] LOL-v2 Synthetic:** some "low" inputs are brighter than their reference. For example, `r01058910t` gives 26.5 dB if left unchanged and 9.1 dB after the model.
- **Literature:**
  - Exposure-correction work (MSEC, ENC, LCDPNet, IAT, CoTF) addresses under- and over-exposure, but on exposure-correction benchmarks.
  - KinD offers a user-set brightness level.
  - Surveys and journal papers report over-exposed results for some LLIE methods [V: the arXiv 2212.10772 survey snippet; details U].
  - I found **no verified paper** that (a) reports exposure-stratified "harm" (output worse than input) for recent supervised LLIE models, and (b) fixes it with a minimal change to such a model while keeping LOL fidelity. This is a search result, **not** a novelty claim. A focused check of IAT's and ENC's full texts is still needed (§9).

### G2: Noise amplification in the darkest regions
- SNR-Aware (CVPR 2022) and Diff-Retinex address it, but with new architectures or diffusion.
- A cheap, loss-level SNR weighting for small enhancers is less explored [H]. Our flat-region noise metric exists [P].
- The novelty margin is smaller, because SNR guidance is established.

### G3: Input-exposure normalisation as a plug-in for LLIE models
- ENC normalises exposure **in feature space** for exposure correction [V].
- [P] Our non-learned "scale to a fixed mean" rule beats every trained model on the LoLI light level (20.67 dB). That suggests the input's exposure, not the network's capacity, drives the failure [H].
- A simple, explicit exposure normalisation before an LLIE network, or a tiny predicted gain, does not appear to be evaluated as a plug-in for HVI-CIDNet or Retinexformer [search result; not a claim].

---

## 5. Three candidate directions

### D1 (recommended): exposure-adaptive HVI-CIDNet, learning *how much* to brighten
1. **Problem:** LLIE models apply a nearly fixed brightening, so mildly dark inputs become over-exposed (G1).
2. **Why existing work doesn't solve it:**
   - LLIE training sets lack mildly dark inputs.
   - Exposure-correction models are separate models, trained on separate data.
   - KinD needs a user to set the level.
   - Whether the issue can be fixed *inside* a strong LLIE model without losing LOL fidelity is not shown in the verified literature [H].
3. **Proposed modification:** two small, separable changes.
   - **(a) Exposure-range training (data):** re-expose LOL-v2 inputs over a wide range, including near-reference brightness, with the target unchanged. This reuses `exposure_augment.py`, extended so that mildly dark samples are common.
   - **(b) Exposure-conditioned modulation (module):** a tiny global estimator on input intensity statistics (mean and percentiles of the I channel) produces FiLM scale and shift for HVI-CIDNet's intensity branch (a few thousand parameters [H]).
     - Alternative ablation: a learned residual gate, output = x + α(x)·(E(x) − x).
   - **(c) Optional exposure-consistency loss:** outputs for two exposures of the same scene should match. This is related to ENC and to sample-relationship learning, so it is an ablation only.
4. **Architecture:** HVI-CIDNet, unchanged except the small conditioning module. Retinexformer is a second backbone, to show the method is general.
5. **Losses:** the official HVI-CIDNet loss stays unchanged; the consistency term is optional.
6. **Expected:**
   - (+) Less harm on light inputs; better behaviour across the exposure range; negligible extra cost.
   - (−) A possible small PSNR loss on very dark LOL inputs; our U-Net with exposure augmentation lost 0.28 dB in-distribution [P].
7. **Data and metrics:**
   - Train: LOL-v2 Real and Synthetic (our splits).
   - Exposure-sweep evaluation: SICE Part 1, all exposures of each sequence (same Label); LoLI-Street levels.
   - Final test: SICE Part 2 (sealed) and LOL-v2 test, once.
   - Metrics: PSNR, SSIM, LPIPS and ΔE00; signed ΔL*; **harm rate** = % of images where output PSNR < input PSNR; PSNR as a function of input brightness (stratified curve).
8. **GPU:** fine-tuning from D1 takes about 1–1.7 h per run on a T4 [P estimate]. About 8–10 runs in total, so about 10–15 T4 hours.
9. **Experiments:** see §7 (controls, ablations, second backbone, simple-rule and IAT baselines).
10. **Defensible if:**
    - a same-steps fine-tuning control is included;
    - harm is measured per exposure bin;
    - LOL-v2 fidelity is reported;
    - IAT and the simple scaling rule are compared;
    - the claim is kept narrow: "a minimal exposure-adaptive modification to a strong LLIE model", not "the first exposure-adaptive LLIE".

### D2: noise-aware loss weighting for HVI-CIDNet
1. **Problem:** noise amplified in the darkest regions (G2).
2. SNR-Aware solves it with a new architecture; a loss-only version for small models is less studied [H].
3. **Change:** weight the per-pixel loss by an SNR estimate of the input, or add a flat-region noise penalty.
4. **Backbones:** HVI-CIDNet; Retinexformer.
5. **Losses:** the official loss, SNR-weighted, plus a noise term.
6. (+) Cleaner dark regions. (−) Risk of over-smoothing texture; the gains may be small.
7. **Data and metrics:** LOL-v2 and SICE; PSNR, SSIM, LPIPS, the flat-region noise ratio and the edge-energy ratio [P metrics].
8. **GPU:** similar to D1.
9. **Experiments:** loss ablations; a comparison with SNR-Aware where compute allows.
10. **Weaker novelty**, because SNR guidance is established.

### D3: exposure-normalisation front end for LLIE models (lightest; also D1's fallback)
1. **Problem:** the same as G1, approached from the input side (G3).
2. ENC is feature-level and built for exposure correction; a plug-in for LLIE models is not evaluated [search result].
3. **Change:** before the enhancer, normalise the input's global exposure, either with a closed-form gain to a target statistic or with a tiny predicted gain. Then fine-tune the enhancer on normalised inputs.
4. **Backbones:** HVI-CIDNet; Retinexformer; U-Net.
5. **Losses:** unchanged.
6. (+) Simple; can be tested inference-only first. (−) A single global gain can't fix locally mixed exposure, and it may amplify noise.
7. **Data and metrics:** as in D1.
8. **GPU:** inference first (minutes), then 1–2 fine-tunes.
9. **Experiments:** no front end vs closed-form vs learned gain; with and without fine-tuning.
10. **Defensible** as an analysis contribution. Thinner as a method contribution.

---

## 6. Final recommendation: D1

**Title:** *Learning How Much to Brighten: Exposure-Adaptive Fine-Tuning of HVI-CIDNet for Robust Low-Light Enhancement across Darkness Levels*

**Problem statement:** Supervised LLIE models trained on strongly under-exposed data apply a near-constant brightening. On mildly dark inputs they over-expose, and they can score below the unprocessed image. We aim to make a strong existing LLIE model adapt its enhancement to the input exposure, without losing fidelity on standard low-light data.

**Motivation:** this is supported by our own measurements (§4 G1), and the failure persists in HVI-CIDNet and Retinexformer. Night surveillance frames range from very dark to near-normal, so an enhancer must "do no harm".

**Methodology:**
- (a) Exposure-range training.
- (b) A tiny exposure-conditioned FiLM module on the intensity branch.
- (c) Optional consistency loss.
- Initialisation from our D1 checkpoint. The architecture is otherwise unchanged.

**Components modified:** the training data pipeline (exposure sampler), one small conditioning module, and optionally one loss term.

**Expected contribution (no novelty claim before the §9 check):**
1. An exposure-stratified evaluation of current LLIE models (harm rate and brightness error by darkness level).
2. A minimal exposure-adaptive modification of HVI-CIDNet, tested on two backbones.
3. Controlled evidence of what each part contributes.

**Dataset plan:**

| Use | Data |
|---|---|
| Training | LOL-v2 Real and Synthetic, existing scene-disjoint train split |
| Selection | LOL-v2 val plus an exposure-swept version of LOL-v2 val (fixed EV grid, created once) |
| Exposure-sweep evaluation (development) | SICE Part 1, all exposures; LoLI-Street Val levels |
| Final test, once | LOL-v2 test; **SICE Part 2** (needs your approval to unseal) |

**Baselines:**
- identity (no processing), gamma, and fixed-target scaling;
- Zero-DCE++ and IAT, using pretrained weights (download needs approval);
- U-Net + exposure augmentation, Retinexformer and HVI-CIDNet D1 (all local).

**Ablations:**

| ID | Variant |
|---|---|
| C0 | D1 fine-tuned with the same steps, no changes (controls for extra training) |
| A1 | + exposure-range data only |
| A2 | + FiLM module only |
| A3 | A1 + A2 (proposed) |
| A4 | A3 + consistency loss |
| A5 | A3 with a residual gate instead of FiLM |
| B | A3 on Retinexformer (generality) |

**Metrics:** PSNR, SSIM, LPIPS, ΔE00, signed ΔL*, harm rate, and per-exposure-bin curves. Each dataset is reported separately and never averaged across datasets.

---

## 7. Experimental methodology

1. **Pre-registration (day 1).** Before any training, write down: the exposure grid, the bins, the metrics, the selection rule (mean LOL-v2 val PSNR **and** harm rate on exposure-swept val, both fixed in advance) and the success criteria.
   - **Success:** A3 lowers the harm rate and |ΔL*| on the light and moderate bins versus C0, with ≤ 0.3 dB loss on LOL-v2 val.
2. **Stage 0, inference only (go / no-go).** Run the existing checkpoints on the SICE exposure sweep and the LoLI levels. This measures the gap for HVI-CIDNet and Retinexformer at scale; the current evidence is the U-Net at scale plus one scene for the stronger models.
   - If HVI-CIDNet shows **no** harm at scale, switch to D2 or D3.
3. **Training.** Fine-tune from D1 with the same steps for every run, seed 42. Use 2 extra seeds for C0 and A3 if time allows, because single-seed differences under about 0.3 dB are not meaningful [P: Tier-A note].
4. **Evaluation.** Development sets during the work. The final test sets once, after the selection record is written.
5. **Statistics.** Scene-level bootstrap confidence intervals for the differences A3 − C0.

---

## 8. Ten-day plan (T4 on Colab; Mac for code and checks only)

| Day | Work | GPU |
|---|---|---|
| 1 | Pre-registration; exposure-sweep evaluation script; check the IAT and ENC full texts (§9) | — |
| 2 | **Stage 0** inference of existing checkpoints on the SICE sweep and LoLI levels → go / no-go | ≈ 1–2 h |
| 3 | Implement the exposure sampler, the FiLM module and the gate (small, additive, tested on Mac CPU); smoke test | ≈ 0.5 h |
| 4 | C0 and A1 | ≈ 3 h |
| 5 | A2 and A3 | ≈ 3 h |
| 6 | A4, A5, and B (Retinexformer) | ≈ 4–5 h |
| 7 | Extra seeds for C0 and A3; baselines (IAT, Zero-DCE++) inference | ≈ 3–4 h |
| 8 | Write the selection record; final test once (LOL-v2 test, SICE Part 2) | ≈ 1–2 h |
| 9 | Analysis: per-bin curves, bootstrap, qualitative panels, failure cases | — |
| 10 | Write-up (report chapter), slides | — |

The total is about 15–20 T4 hours, more than free Colab reliably gives in 10 days. **Kaggle (about 30 GPU-h per week) or paid units** are recommended. Every run is resumable.

---

## 9. Risks and fallbacks

| Risk | Mitigation / fallback |
|---|---|
| Stage 0 shows HVI-CIDNet is not harmed at scale | Switch to D3 (analysis plus front end) or D2. Stage 0 itself is a reportable finding |
| The IAT or ENC full text already does exactly D1 | Narrow the claim to the stratified evaluation plus the plug-in on HVI-CIDNet, with IAT as a baseline; or move to D3 |
| Gains only from extra training | C0 control; claim only A3 − C0 |
| LOL-v2 PSNR drops | Pre-registered tolerance; report the trade-off honestly |
| Single-seed noise | 3 seeds for C0 and A3; bootstrap confidence intervals |
| Compute limits | Fine-tune rather than train from scratch; Kaggle; drop A5 or B first |
| Synthetic re-exposure ≠ real exposure | SICE (real multi-exposure captures) is the main exposure-sweep evaluation |

---

## 10. References (verified pages)

1. Q. Yan et al., "HVI: A New Color Space for Low-light Image Enhancement," CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Yan_HVI_A_New_Color_Space_for_Low-light_Image_Enhancement_CVPR_2025_paper.html · arXiv 2502.20272 · code https://github.com/Fediory/HVI-CIDNet
2. Y. Cai, H. Bian, J. Lin, H. Wang, R. Timofte, Y. Zhang, "Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement," ICCV 2023, pp. 12470–12479. https://arxiv.org/abs/2303.06705
3. T. Wang, K. Zhang, T. Shen, W. Luo, B. Stenger, T. Lu, "Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and Transformer-Based Method," AAAI 2023, 37(3):2654–2662. https://ojs.aaai.org/index.php/AAAI/article/view/25364 · code https://github.com/TaoWangzj/LLFormer
4. C. Li, C. Guo, C. C. Loy, "Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation," IEEE TPAMI 44(8):4225–4238, 2022. DOI 10.1109/TPAMI.2021.3063604 · https://arxiv.org/abs/2103.00860
5. X. Xu, R. Wang, C.-W. Fu, J. Jia, "SNR-Aware Low-Light Image Enhancement," CVPR 2022, pp. 17714–17724. https://openaccess.thecvf.com/content/CVPR2022/html/Xu_SNR-Aware_Low-Light_Image_Enhancement_CVPR_2022_paper.html
6. Y. Zhang, J. Zhang, X. Guo, "Kindling the Darkness: A Practical Low-light Image Enhancer," ACM MM 2019, pp. 1632–1640. https://arxiv.org/abs/1905.04161
7. M. Afifi, K. G. Derpanis, B. Ommer, M. S. Brown, "Learning Multi-Scale Photo Exposure Correction," CVPR 2021. https://openaccess.thecvf.com/content/CVPR2021/html/Afifi_Learning_Multi-Scale_Photo_Exposure_Correction_CVPR_2021_paper.html
8. J. Huang et al., "Exposure Normalization and Compensation for Multiple-Exposure Correction," CVPR 2022, pp. 6043–6052. https://openaccess.thecvf.com/content/CVPR2022/html/Huang_Exposure_Normalization_and_Compensation_for_Multiple-Exposure_Correction_CVPR_2022_paper.html
9. H. Wang, K. Xu, R. W. H. Lau, "Local Color Distributions Prior for Image Enhancement," ECCV 2022. DOI 10.1007/978-3-031-19797-0_20 · https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2257_ECCV_2022_paper.php
10. Z. Cui et al., "You Only Need 90K Parameters to Adapt Light: A Light Weight Transformer for Image Enhancement and Exposure Correction," BMVC 2022. https://arxiv.org/abs/2205.14871 · https://bmvc2022.mpi-inf.mpg.de/0238.pdf · code https://github.com/cuiziteng/Illumination-Adaptive-Transformer
11. J. Huang et al., "Learning Sample Relationship for Exposure Correction," CVPR 2023, pp. 9904–9913. https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Learning_Sample_Relationship_for_Exposure_Correction_CVPR_2023_paper.html
12. Z. Li et al., "Real-Time Exposure Correction via Collaborative Transformations and Adaptive Sampling," CVPR 2024, pp. 2984–2994. https://openaccess.thecvf.com/content/CVPR2024/html/Li_Real-Time_Exposure_Correction_via_Collaborative_Transformations_and_Adaptive_Sampling_CVPR_2024_paper.html
13. R. Cui, L. Niu, G. Hu, "Unsupervised Exposure Correction," ECCV 2024, pp. 252–268. DOI 10.1007/978-3-031-72658-3_15 · https://arxiv.org/abs/2507.17252
14. J. Weng et al., "MambaLLIE: Implicit Retinex-Aware Low Light Enhancement with Global-then-Local State Space," NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/30699996ff411d48903c9752b782a5c1-Abstract.html
15. X. Yi, H. Xu, H. Zhang, L. Tang, J. Ma, "Diff-Retinex: Rethinking Low-light Image Enhancement with A Generative Diffusion Model," ICCV 2023, pp. 12302–12311. https://openaccess.thecvf.com/content/ICCV2023/html/Yi_Diff-Retinex_Rethinking_Low-light_Image_Enhancement_with_A_Generative_Diffusion_Model_ICCV_2023_paper.html
16. J. Hou, Z. Zhu, J. Hou, H. Liu, H. Zeng, H. Yuan, "Global Structure-Aware Diffusion Process for Low-Light Image Enhancement," NeurIPS 2023. https://arxiv.org/abs/2310.17577
17. H. Jiang, A. Luo, X. Liu, S. Han, S. Liu, "LightenDiffusion: Unsupervised Low-Light Image Enhancement with Latent-Retinex Diffusion Models," ECCV 2024. DOI 10.1007/978-3-031-73195-2_10 · https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6440_ECCV_2024_paper.php
18. Z. Fu, Y. Yang, X. Tu, Y. Huang, X. Ding, K.-K. Ma, "Learning a Simple Low-Light Image Enhancer From Paired Low-Light Instances," CVPR 2023, pp. 22252–22261. https://openaccess.thecvf.com/content/CVPR2023/html/Fu_Learning_a_Simple_Low-Light_Image_Enhancer_From_Paired_Low-Light_Instances_CVPR_2023_paper.html
19. L. Ma et al., "Toward Fast, Flexible, and Robust Low-Light Image Enhancement," CVPR 2022. https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_Toward_Fast_Flexible_and_Robust_Low-Light_Image_Enhancement_CVPR_2022_paper.pdf
20. Z. Liang, C. Li, S. Zhou, R. Feng, C. C. Loy, "Iterative Prompt Learning for Unsupervised Backlit Image Enhancement," ICCV 2023. https://arxiv.org/abs/2303.17569
21. D. Feijoo, J. C. Benito, A. Garcia, M. V. Conde, "DarkIR: Robust Low-Light Image Restoration," CVPR 2025 (vendored code; venue as recorded in Phase 3 [U: recheck])
22. M. T. Islam, I. Alam, S. S. Woo, S. Anwar, I. H. Lee, K. Muhammad, "LoLI-Street: Benchmarking Low-light Image Enhancement and Beyond," ACCV 2024, pp. 1250–1267. https://openaccess.thecvf.com/content/ACCV2024/html/Islam_LoLI-Street_Benchmarking_Low-light_Image_Enhancement_and_Beyond_ACCV_2024_paper.html
23. C. Li, C. Guo, et al., "Low-Light Image and Video Enhancement Using Deep Learning: A Survey," IEEE TPAMI 44(12):9396–9416, 2022. DOI 10.1109/TPAMI.2021.3126387 · https://arxiv.org/abs/2104.10729
24. J. Cai, S. Gu, L. Zhang, "Learning a Deep Single Image Contrast Enhancer from Multi-Exposure Images," IEEE TIP 27(4):2049–2062, 2018 (SICE; from the Phase-2 reference list [U: recheck])
25. W. Yang et al., "Sparse Gradient Regularized Deep Retinex Network for Robust Low-Light Image Enhancement," IEEE TIP 30:2072–2086, 2021 (LOL-v2; from the Phase-2 reference list [U: recheck])

Searched but **not** used as evidence (titles only; venue or details unverified): UBLLIE (arXiv 2608.04429), ControlLight (arXiv 2605.25569), "A Two-Stage Exposure Controllable Network…" (IEEE, document 11194160), "Adaptive Low Light Enhancement via Joint Global-Local Illumination Adjustment" (arXiv 2504.00400). **These four must be read before any novelty statement**, because they touch exposure adaptivity.

---

**STOP. Awaiting approval.** Nothing below is approved yet:
- Stage 0 inference;
- any code change;
- any download (IAT or Zero-DCE++ weights, MSEC);
- unsealing SICE Part 2;
- any training.
