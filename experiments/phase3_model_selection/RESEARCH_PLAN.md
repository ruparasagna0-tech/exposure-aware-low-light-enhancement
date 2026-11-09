# Phase 3: research reset and model-selection plan

> **Update 2026-10-14:** Stage 0 is complete (`stage0/README.md`). Sections C, D and F are superseded by
> [`STAGE1_PLAN.md`](STAGE1_PLAN.md): HVI-CIDNet, DarkIR and Retinexformer are the priority candidates, with optional
> diffusion (DiffLL) and Colab GPU training. Sections A, B and E remain valid.

Project: *Dark Image/Video Enhancement of Night-Time Surveillance Dataset*
Status: **plan only. Nothing is implemented or trained. Every item in section I needs approval.**
Prepared 2026-10-13 from a read-only audit of `<project-root>`.
(A note about the local project path was removed for the public repository.)

Labels used below: **[verified]** means checked in files or primary sources during this audit. **[hypothesis]** means
plausible but not yet tested. **[estimate]** means a resource guess.

---

## A. Existing project audit

### A.1 Inventory
| Area | What exists | Status |
|---|---|---|
| Models | `src/models/baseline_unet.py`: `BaselineUNet` (base 32, depth 3, LeakyReLU, no norm, sigmoid head, reflect-pad to a multiple of 8). This is the only registered model. | [verified] 1,925,667 params; 9.22 GMACs at 256x256; about 0.48 s per 600x400 on 4 CPU threads |
| Losses | `src/losses/reconstruction.py`: `l1` only | [verified] |
| Data | `src/datasets/{lol_v2,sice,loli_street,phase2,paired_list,paired_transforms,balanced_sampler,scene_split}.py`. PIL `convert("RGB")` then `to_tensor` gives sRGB in [0,1]. Training: 256 crop + hflip. Val/test: full resolution. | [verified] |
| Exposure aug. | `src/datasets/exposure_augment.py` with two policies (`hard_cap`, `soft_shoulder`). Configs `configs/generalized_lol_v2_expaug{,_soft}.yaml`; notebook `notebooks/colab_train_generalized_lol_v2_expaug.ipynb`; `scripts/smoke_test_expaug.py` | [verified] implemented. All three smoke-test reports `passed: true`. **No training run, no checkpoint, no results.** |
| Training | `scripts/train_generalized.py` (`--config --data-root --device --num-workers --resume`; test-only: `--output-dir --max-steps-per-epoch --epochs`) | [verified] |
| Evaluation | `scripts/evaluate_generalized.py`: refuses to run twice. PSNR/SSIM on 8-bit-rounded outputs; LPIPS-Alex; SICE tiled 512/64. | [verified] **Does not save enhanced images.** It only saves four 4-image montages. |
| Metrics | `src/metrics/{image_metrics,large_image,lpips_metric,domain_evaluation}.py` | [verified] no colour metrics |
| Checkpoints | None on this Mac. `best.pth` (epoch 45) is recorded at `/content/drive/MyDrive/.../generalized_lol_v2/checkpoints/` | **cannot be checked locally** |
| Phase-2 results | `~/Downloads/phase2_evaluation_results.zip` (MD5 `e7302a3b…`, unchanged; recorded in `generalized_lol_v2_diagnostics/source_results_md5.json`) | [verified] |
| Phase-1 (LOL-v1) | Code, config, notebook, and visualisations of the data only. No Phase-1 experiment folder, history or checkpoint locally. | 17.9618 dB is **not locally verifiable** |
| Diagnostics | `experiments/generalized_lol_v2_diagnostics/` (model-free pair statistics, panels; new `colour_montage/`) | [verified] |
| Presentation | `presentation/` (deck builders, notes). The notes correctly call the exposure augmentation untrained. | [verified] |
| Unused data | `data/DarkVision`, `data/SID` hold only `.DS_Store` (empty) | [verified] |

### A.2 What was verified
- The recorded Phase-2 table reproduces **exactly** (all 7 rows, 3 metrics) from the per-image CSVs. The checkpoint metadata
  in `metrics_summary.json` gives epoch 45 and validation PSNR 17.455 (real), 19.328 (synthetic) and 18.391 (mean), matching the brief.
- Integrity: of the 10,276 files in the pre-augmentation MD5 snapshot, only `src/datasets/phase2.py` has changed. That change
  is the documented off-by-default augmentation switch. Everything else is byte-identical.
- No OpenCV, BGR conversion, ICC profiles, EXIF rotations or gamma conversions anywhere in the pipeline. This was checked
  in the code and on samples of every dataset, plus all 116 SICE input/label files. Images are 8-bit sRGB end to end.

### A.3 Incomplete or needing correction
1. **Checkpoint not available locally**, so model loading and metric reproduction cannot be checked. This blocks all
   output-level analysis.
2. **No saved full-resolution outputs.** A colour study needs them. The evaluation script must not be changed (it guards
   the one-time test). A *separate* export script that writes to a new folder is needed (G, stage 0).
3. **Single seed.** Seed-to-seed variation is unknown, so small differences between models cannot yet be called real.
4. **No colour metrics** in the codebase.
5. `build_model` (`src/models/__init__.py`) only knows `baseline_unet`, and checkpoints rebuild through it. New
   architectures need either an additive registry entry (a change to an existing file, which needs approval) or a separate registry.
6. **No dataset is night-time surveillance footage.** LoLI-Street low images were made in Photoshop from well-exposed street
   photos (the paper's Sec. 3.1 says only "Photoshop v25.0"; the procedure is not documented). The closest real material we
   have is the **LoLI-Street real low-light test set (RLLT, 1,000 images, no references)**, which is currently unused.

---

## B. Colour-loss diagnosis

Evidence base: (i) model-free statistics on all 3,255 test pairs (earlier diagnostic); (ii) the **16 model outputs that
exist** (montage panels, display resolution: `generalized_lol_v2_diagnostics/colour_montage/`). n = 16 is small. Treat (ii)
as a description of failure modes, not population estimates.

### B.1 Observed failure modes
| Failure mode | Evidence | Strength |
|---|---|---|
| **Over-brightening of moderately dark inputs** (exposure not adapted to the input) | LoLI: same reference for all levels, but PSNR is ordered light < moderate < dense in 99.6% of 999 scenes. On *light* inputs the model is below the unchanged input (13.15 vs 13.92 dB). A train-derived fixed-brightness scaling beats it on 97% of light images. Panels: light_30489 has L* 75 vs 57 in the reference; r13837601t has +19 L*. | **Strong** [verified] |
| **Hue shifts on saturated objects where the input still holds the colour** | r082626aft: the input keeps a pink door and teal roof. The output turns them orange-yellow and pale blue, and the bench turns grey. ΔE00 on chromatic pixels after equalising L*: 13.6. low00695: coloured bows lose their magenta and green. | Strong per case (visual + metric), n small |
| **Saturation errors in both directions** (not uniform "desaturation") | Chroma ratio output/reference: synthetic 0.54, 0.60, 0.67, 1.28. Real low00743: **2.27x over-saturated** with a warm cast (b* +19.4). SICE 1.13 to 1.19. | Moderate (n = 16) |
| **Colour error beyond brightness** | After equalising L*, mean ΔE00 is still 4.2 to 11.9 on LOL-v2 and SICE panels. On LoLI and over-bright cases, 51 to 76% of ΔE00 is lightness. | Moderate |
| **Loss of global contrast / haze** on LoLI | L* std: output 16.2, 16.6, 17.6, 11.8 vs reference 23.5, 23.5, 26.3, 21.0 | Moderate (4/4 LoLI panels) |
| **Noise amplification / grain** | low00739: heavy grain on flat walls and floor (visual) | Weak (visual, 1 case) |
| **Texture smoothing / blur, banding** | r141665bdt: blurred crowd and streaked sky. low00726 to 728 (grass): worst SSIM 0.52 and LPIPS 0.80 in LOL-v2-real at normal exposure. | Weak to moderate |
| **Failure on extremely dark inputs** | SICE seq46: 92% of input pixels black-clipped; even a best-case per-channel tone curve fitted to the reference reaches only 18.7 dB. Across SICE, PSNR vs black fraction r = -0.71. | **Strong**, mostly a **data/protocol** limit |
| Highlight clipping in outputs | Output pixels >= 250: at most 1.7%, never above the reference | **Not observed** |

### B.2 Where the problems come from
- **Ruled out** [verified]: RGB/BGR swaps, ICC/colour-profile handling, EXIF orientation, image saving (none happens),
  quantisation (plain rounding).
- **Partly the data / reference** [verified]: reference "style" differs by dataset. Mean chroma C*ab is 12.9 (LOL-v2-real
  train), 17.2 (synthetic train), 9.5 (real test), 17.8 (synthetic test), 16.2 (SICE labels) and 7.0 (LoLI refs). SICE labels
  are the best of 13 fusion/HDR results picked by human judges (Cai et al., 2018), not camera ground truth. So ΔE00 against a
  reference partly measures style mismatch, not lost information. This is why the protocol pairs reference metrics with
  visual review.
- **Model-side** [verified for r082626aft-type cases]: when the input still holds the hue and the output changes it, the
  information was lost or changed by the model.
- **Candidate mechanisms** [hypothesis, untested]:
  1. *L1 in sRGB* rewards the per-pixel median of plausible colours. Where dark pixels are ambiguous, this pulls hue and
     chroma toward averages, which shows up as desaturation or shifts. L1 also gives exposure and colour errors the same
     weight with no separate term for colour direction.
  2. *Limited global context*: with 3 downsamplings the network decides brightness and colour from local evidence. On SICE,
     512-pixel tiles of a 20 MP image each see about 1% of the scene, so exposure and colour are decided per tile.
  3. *Mixed training references* (real is less saturated than synthetic) give an inconsistent colour target.
  4. *Training only on 0.24 MP crops* and a short 5,600-step budget (under-training).

  Each of these is a testable claim, and section D tests 1 and (indirectly) 2.

### B.3 Colour-preservation approaches compared
| Approach | What it does | Benefit | Risk / trade-off |
|---|---|---|---|
| L1 + **angular colour loss** (per-pixel angle between output and reference RGB vectors) | Penalises colour *direction* (hue and saturation) and ignores brightness | Does **not** resist the needed brightening, which is our biggest error | Unstable near black (needs an epsilon / masking). Original source to verify before citing (often attributed to DeepUPE, Wang et al., CVPR 2019) |
| L1 + **CIELAB loss** (L1 on a*, b*, or ΔE76) | Penalises colour in a perceptually more uniform space | Directly matches the ΔE evaluation | Lab needs gamma linearisation and cube roots (care with gradients). Partly duplicates L1. Risk of metric overfitting if ΔE is both loss and selection metric. |
| CIEDE2000 as a loss | Exact evaluation metric | — | Piecewise, with hue discontinuities. Gradient behaviour is poorly suited; **not recommended** |
| Gray-world colour constancy (Zero-DCE) | Pushes channel means equal | Needs no reference | Assumes neutral scenes; wrong for sodium-lit night streets. **Not recommended** for supervised training |
| Perceptual (VGG) loss | Matches deep features | Sharper texture | Can invent texture. Correlates with LPIPS (an evaluation metric). Not a colour fix. |
| Retinex decomposition (RetinexNet, Retinexformer) | Splits brightening (illumination) from content (reflectance) | Brightening is modelled explicitly | Reflectance noise; colour still learned by pixel loss |
| Colour-decoupled space (HVI-CIDNet) | Learns intensity separately from hue/saturation planes | Designed for the red-hue discontinuity and black-noise problems of HSV | A newer method; its benefit in our protocol is unproven |

**Smallest well-motivated set**: one colour term (angular loss, chosen because it does not fight exposure) tested against
L1 alone, **crossed with** the already-built exposure augmentation, because exposure is the largest error component and a
colour loss cannot fix it. That is a 2x2 design with 3 new runs. Lab/ΔE losses are kept as a fallback only if the angular
loss does not reduce ΔE00 on validation.

---

## C. Candidate model research
Published scores are **not** comparable across papers (different splits, resolutions, GT-mean tricks). They are not used here.

| | Compact U-Net (ours) | Retinexformer | RetinexNet | Zero-DCE++ | **HVI-CIDNet** (added) |
|---|---|---|---|---|---|
| Paper | — | Cai et al., *Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement*, ICCV 2023 (arXiv 2303.06705) | Wei et al., *Deep Retinex Decomposition for Low-Light Enhancement*, BMVC 2018 (arXiv 1808.04560) | Li, Guo, Loy, *Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation*, TPAMI 2021 (arXiv 2103.00860); Zero-DCE: CVPR 2020 (arXiv 2001.06826) | Yan et al., *HVI: A New Color Space for Low-light Image Enhancement*, CVPR 2025 (arXiv 2502.20272) |
| Code / licence | in repo | github.com/caiyuanhao1998/Retinexformer, **MIT**, PyTorch (BasicSR) | github.com/weichen582/RetinexNet, **MIT**, **TensorFlow 1.x (>= 1.5)** | github.com/Li-Chongyi/Zero-DCE_extension, **CC BY-NC 4.0** (academic only), PyTorch 1.0 era | github.com/Fediory/HVI-CIDNet, **MIT**, PyTorch 1.13 |
| Mechanism | Encoder-decoder, direct RGB regression | Estimates a 3-channel light-up map (input x map), then an illumination-guided transformer restores corruption | Decom-Net splits reflectance and illumination; Enhance-Net adjusts illumination; BM3D denoises the reflectance | Tiny CNN predicts per-pixel high-order brightening curves (iterated) | Maps sRGB into HVI (polarised hue/saturation plus learnable intensity collapse); dual-branch network decouples colour and intensity |
| Supervision | Paired (L1) | Paired (L1) | Paired (reconstruction + reflectance-consistency + smoothness) | **Zero-reference** (spatial, exposure E = 0.6, gray-world colour, smoothness); needs no references | Paired (losses in HVI and sRGB) |
| Models illumination / colour? | No / no | Illumination: yes. Colour: implicit | Illumination + reflectance: yes. **No explicit colour-restoration module** (colour sits in the shared 3-channel reflectance) | Illumination via curves; colour via gray-world prior | **Colour: explicit** (purpose of the method) |
| Size | 1.93 M, 9.22 GMACs at 256 | 1.61 M, 15.57 G "FLOPs" at 256 (paper) | small (not verified) | about 10 K params (paper) | 1.88 M, 7.57 G "FLOPs" at 256 (paper). **Capacity-matched to our U-Net.** |
| Official training recipe | 5,600 steps, batch 4, 256 crops | 250 k iters, batch 8, 128 crops, cosine LR 2e-4 to 1e-6 | 96 crops, batch 16, SGD, staged | 512x512 resized SICE inputs | LOL-v2-real: 400 crops, batch 8, 1,500 epochs; Adam 1e-4, cosine |
| Full-res / large images | Tiled 512/64 | Has split-and-test for up to 4000x6000 | — | Downsampling option for speed (factor not verified) | Not verified |
| Known weaknesses (sources) | Our diagnosis (B) | Authors advise against the GT-mean test setting their tables also report | Paper: noise is amplified and remains in reflectance, hence the BM3D step | Paper: noise not handled; over-enhances lit regions without over-exposed training data | Paper Table 5: cross-dataset LPIPS worse than Zero-DCE (0.193 vs 0.169) |
| Retrain under our protocol? | Done | **Yes** (PyTorch, L1, the same loss as ours, so a clean architecture comparison) | **Hard**: official TF1 does not run on current Colab; PyTorch ports are unofficial | **Yes**, cheaply (needs only our training *inputs*) | **Yes** (needs its HVI transform + loss) |
| Surveillance fit [interpretation] | Fast, small | Strongest restoration prior; heavier (attention) | Dated; noise and colour handling weak | Very fast, which suits video; ignores noise; gray-world is wrong under coloured street lights | Targets our main issue (colour); fast |

**Pretrained-weight leakage** [verified unless noted]:
| Weights | Trained on | Unsafe for our | Usable for |
|---|---|---|---|
| Any **LOL-v1-trained** weights (Retinexformer `LOL_v1`, CIDNet LOLv1, RetinexNet) | LOL-v1 `our485` | **LOL-v2-real Test**: our audit found 86/100 Test low images (and 91/100 references) are exact copies of `our485` images | SICE, LoLI only |
| Retinexformer / CIDNet **LOL-v2-real** | official LOL-v2-real Train (689) | none for test. It includes our 64 val pairs, so it **cannot be used with our validation-based selection** | all four test sets |
| Retinexformer / CIDNet **LOL-v2-syn** | official synthetic Train | none for test | all four test sets |
| CIDNet **SICE.pth** | SICE, 7:1:2 split (split membership not verified) | **SICE** (likely overlap) | LOL-v2, LoLI |
| Zero-DCE released weights | Zero-DCE paper: all **360** SICE Part1 sequences (inputs only). Part1 has exactly 360 sequences [verified], so **all 58 of our test sequences were seen**. Zero-DCE++ training data assumed the same (not verified). | **SICE** | LOL-v2, LoLI |
| RetinexNet | `our485` + 1,000 RAISE synthetic pairs. **Likely the same images as LOL-v2-synthetic** (needs a hash check) | LOL-v2-real Test; probably LOL-v2-syn Test | SICE, LoLI |

**Recommendation on candidates**: keep **Retinexformer** (Retinex family, same L1 loss, MIT), **HVI-CIDNet** (the only
candidate built for colour; same size as our U-Net), and **Zero-DCE++** (curve family; zero-reference; fastest for video).
Keep **RetinexNet as supplementary/optional**: it is dated, TensorFlow 1 makes faithful retraining risky, and Retinexformer
already represents the Retinex idea. Add two **non-learned reference floors** to every table: *identity* (the input itself)
and *fixed-target brightness scaling* (target from the training references only). Any model must beat these.

---

## D. Experiment matrix

**Group 1: directly comparable.** Same training data, splits, crops, sampler, step budget, seed, checkpoint rule and evaluation.
| ID | Model | Training | Status | Colour strategy | Main hypothesis | Key risk | Decision criterion | Work needed | Cost per run [estimate] |
|---|---|---|---|---|---|---|---|---|---|
| E0 | U-Net, L1 | existing | **done** (needs checkpoint for outputs) | none | baseline | — | reference | export outputs | — |
| E1 | U-Net, L1 + exposure aug (soft-shoulder) | new, matched budget | **ready**, awaiting approval | none | exposure errors (light/moderate) shrink | hurts dense / very dark | val PSNR; then test rule (E.3) | none (built) | 0.5–1 h T4 |
| E2a–c | U-Net, L1 + λ·angular colour loss, λ ∈ {0.1, 0.3, 1.0} | new | design, **awaiting approval** | colour-direction loss | hue/chroma error falls without hurting exposure | λ too high slows brightening; instability near black | λ chosen on **validation** only (E.3) | loss + tests | 3 x 0.5–1 h |
| E3 | U-Net, L1 + aug + chosen colour loss | new | awaiting approval | both | effects add up | interaction | 2x2 factorial reading | config only | 0.5–1 h |
| E4 | Retinexformer, L1 (same pipeline, matched budget) | new | awaiting approval | illumination-guided | Retinex prior + attention reduces colour and exposure error | **under-trained** at 5,600 steps (official recipe is 250 k) | same rule | vendor model (MIT), wrapper, tests | 1–2 h T4 |
| E5 | HVI-CIDNet, own loss (matched budget) | new | awaiting approval | HVI colour decoupling | explicit colour space lowers ΔE00 at equal capacity | official recipe much longer; extra loss terms | same rule | vendor model + HVI + loss | 1–2 h T4 |
| E6 | Zero-DCE++, zero-reference on our training **inputs** | new | awaiting approval | gray-world | adaptive curves fix over-brightening cheaply | cannot learn the reference look, so PSNR is structurally lower; noise | evaluated mainly on exposure/colour/visual criteria | port model + 4 losses | < 1 h |
| F0/F1 | Identity; fixed-target scaling | no training | ready | — | floors | — | must be beaten | metric script | minutes (CPU) |

**Group 2: supplementary only.** Not trained under our protocol; leakage rules from C apply.
| ID | Weights | Evaluated on | Status |
|---|---|---|---|
| P1 | Retinexformer LOL-v2-real and LOL-v2-syn (official) | all test sets (seen-domain numbers noted as same-domain official training) | awaiting approval (download) |
| P2 | HVI-CIDNet LOL-v2-real, LOL-v2-syn, `generalization.pth` | all; `SICE.pth` **excluded** | awaiting approval |
| P3 | Zero-DCE++ released snapshot | LOL-v2, LoLI only (**SICE excluded**) | awaiting approval |
| P4 | RetinexNet official | SICE, LoLI only, if TF1 runs | optional, low priority |

**Fairness note.** The matched budget (5,600 steps) favours small, fast-converging models. If a model looks under-trained
(validation curve still rising at the end), we will report that fact, and optionally run an **extended-budget tier**
(e.g. 4x steps) **for every Group-1 model including the U-Net**. We never extend only one model.

---

## E. Evaluation protocol

**Dataset roles (unchanged)**: training is LOL-v2-real Train minus validation scenes (625) plus LOL-v2-synthetic Train minus
validation scenes (808), with 2+2 balanced batches. Validation is the scene-disjoint 64 + 92 pairs, used for checkpoint, λ
and configuration choices only. Seen-domain test is the official LOL-v2 Test sets (100 + 100). Unseen test is SICE (58
sequences) and LoLI-Street Val (2,997; dense/moderate/light reported separately). LOL-v1 stays excluded.

**Leakage protections**: the existing integrity checks run before every training run. Pretrained weights follow the table in C.
Test sets are evaluated **once per final model**; the script's refuse-to-rerun guard is kept for every new experiment
folder. λ, budgets and checkpoints are never chosen from test numbers. GT-mean rescaling is **never** used.

**Resolution policy**: native resolution for every model. SICE uses tiled 512/64 linear blending for **all** models, so they
share the same handicap. A pre-registered second SICE track uses the **-1 EV input** of the testing index, alongside the
current darkest-usable-exposure input (which is information-limited, see seq46).

**Metrics** (mean, SD, median, per-image CSV, worst cases listed):
- Fidelity: PSNR, SSIM (current definitions), LPIPS-Alex.
- **Colour** (sRGB to CIELAB D65): mean / median / 95th-percentile **ΔE00**; ΔE00 with L* equalised (colour-only error);
  mean ΔL*, chroma ratio C*out/C*ref, hue error on chromatic pixels (C* > 10).
- **Exposure**: mean L* error and the share of images over-exposed by > 10 L*; highlight clipping (% pixels >= 250 in
  any channel) relative to the reference.
- **Noise / detail**: noise in flat regions (residual std in low-gradient areas); gradient-magnitude correlation with the
  reference (edge preservation).
- **Cost**: parameters, MACs at 256x256, seconds per 600x400 and per 20 MP image (same GPU type), peak memory.
- **No-reference track**: LoLI-Street RLLT (1,000 real low-light street images, the closest to surveillance) gets only
  brightness, clipping and colour statistics plus a **blind side-by-side visual review** of a fixed random 50 images
  (model names hidden, order randomised, seed fixed now).
- Reference caveats are reported with the numbers: SICE labels are human-selected fusion results; LoLI lows are
  Photoshop-darkened with an undocumented procedure.

**Statistics**: paired per-image differences against E0 with 95% bootstrap confidence intervals (LoLI resampled by
scene id, so the three levels stay together). Where seeds allow, a difference smaller than the seed-to-seed spread is reported as "no evidence of a difference".

### E.3 Model-selection rule (fixed **before** any new test result is seen)
1. **Inside a family** (validation only): checkpoint = max of the mean of real and synthetic validation PSNR (as before).
   Colour-loss λ = the value with the lowest mean validation ΔE00, provided mean validation PSNR is no more than 0.3 dB
   below E0's. If none qualifies, the colour loss is reported as not helpful.
2. **Eligibility gates** (test, per model): no leakage; and not worse than the *fixed-target scaling* floor on any unseen
   subset's PSNR.
3. **Primary comparison against E0**, over the macro-average of the 4 unseen subsets (SICE, LoLI dense, moderate,
   light) and separately the 2 seen subsets. A model **improves on** E0 if it lowers mean ΔE00 with the 95% CI excluding 0,
   **and** is non-inferior on LPIPS (no more than +0.01), SSIM (no more than -0.01) and PSNR (no more than -0.3 dB), **and** does not increase the over-exposed
   share. No weighted score; no single-metric winner.
4. If several models improve on E0: keep those not dominated on (ΔE00, LPIPS, PSNR, unseen-subset worst case). Among
   those, the blind visual review decides, then cost (faster wins a tie, because video is next).
5. If none is clearly better, **report the trade-off** and keep the simplest model that passes the gates. A higher-PSNR
   model with worse colour is reported as a trade-off, never as the winner.

---

## F. Resource estimate (all [estimate] unless noted)
- **Hardware**: Colab GPU (T4 assumed) for training and evaluation. The Mac CPU is for exports, colour metrics and
  smoke tests. MPS stays refused (it produced invalid values before) [verified in code].
- **Per training run** (5,600 steps): U-Net about 0.5–1 h on T4. Retinexformer and CIDNet about 1–2 h (more compute
  per pixel; attention memory). Zero-DCE++ under 1 h. Full test evaluation about 0.5–1 h per model, dominated by 20 MP SICE I/O and LPIPS.
- **Totals**: Group 1 needs about 8 new runs, roughly 8–14 GPU-hours with evaluations. Three seeds for the final 2–3
  contenders plus E0 add about 6–10 h. Official-recipe retraining of Retinexformer/CIDNet (250 k iterations / 1,500 epochs)
  would take **days** on a T4, so it is not planned.
- **Storage**: lossless outputs per model are about 0.2 GB (LOL-v2), about 1 GB (LoLI) and **2–3 GB (SICE full-res PNG)**.
  Proposal: full-res PNG for LOL/LoLI; for SICE, compute metrics in-flight and save a 1/4-scale PNG plus 20 full-res
  crops. *(Decision needed.)*
- **Dependency risks**: Retinexformer needs `einops`; the architecture file can be used without all of BasicSR. CIDNet
  code targets PyTorch 1.13 (minor API risk on 2.x). Zero-DCE++ code is PyTorch 1.0 era (model is trivial to port).
  RetinexNet is TF1, **high risk**. Zero-DCE++ is non-commercial (fine for academic use; must be stated).
- **Difficulty**: colour loss is low. Zero-DCE++ is low. Retinexformer is medium. CIDNet is medium (HVI transform + loss
  must match the paper). RetinexNet is high.

---

## G. Implementation plan (staged; **nothing created yet**)
**Stage 0: unblock and measure (no training)**
- [ ] Copy `best.pth` from Drive into `experiments/generalized_lol_v2/checkpoints/` (read-only use); record its MD5.
- [ ] `scripts/export_outputs.py` (new): runs a checkpoint and writes outputs to `experiments/phase3_model_selection/outputs/<model>/`.
      It never touches `evaluation/`. Acceptance: reproduced PSNR/SSIM/LPIPS match the saved CSVs (tolerance about 1e-3 dB; CPU vs CUDA).
- [ ] `src/metrics/colour_metrics.py` (new): ΔE00 (Sharma test pairs as unit test), L*/C*/h statistics, clipping, noise, edge metrics.
- [ ] `scripts/evaluate_colour.py` (new): reads saved outputs and writes colour CSVs and summaries.
- [ ] Full colour diagnosis of E0 on all 3,255 outputs (replaces the 16-panel evidence); floors F0/F1.

**Stage 1: supplementary pretrained (after download approval)**: vendored model files under
`third_party/<name>/` with licence and commit hash; P1–P3 evaluated with the leakage exclusions.

**Stage 2: U-Net factorial**: `src/losses/colour.py` (new: angular loss, epsilon/masking near black); new configs
`configs/phase3/unet_{expaug,colour_l*,expaug_colour}.yaml`; runs E1, E2a–c (validation choice of λ), E3.

**Stage 3: other architectures**: `src/models/{retinexformer,cidnet,zero_dce_pp}.py`, `src/losses/{hvi,zero_reference}.py`,
registry entry (approval needed: touches `src/models/__init__.py`, or a separate `phase3` registry), `scripts/train_phase3.py`
(new; reuses the data, sampler and integrity checks unchanged); runs E4–E6.

**Stage 4: seeds**: 3 seeds for E0 and the 2–3 best by validation.

**Stage 5: final test** (once per model) and the selection rule E.3. Then video (temporal consistency) and YOLO.

**Tests (every stage)**: references byte-identical before/after loading; outputs in [0,1] and finite; no BGR (a
known-colour patch round-trips); train/val/test isolation (existing integrity checks); losses finite and gradients non-zero
for every parameter group; checkpoint save → load → identical outputs; same seed gives identical first N steps; MD5 snapshot
of all existing assets before and after each stage.

---

## H. Faculty explanation (short)
We had trained one network, a compact U-Net, and tested it carefully on data it never saw. That shows *how well* it works,
not whether it is the *right* model. The diagnosis shows two separate problems. First, the network brightens every image by
roughly the same amount, so already-bright images come out over-exposed. Second, on some images it changes or weakens colours
that are still visible in the dark input. The image-reading pipeline was checked and is not the cause.

We will therefore compare four families of method on exactly the same data, splits and metrics: our U-Net, a Retinex
transformer (Retinexformer), a colour-space method designed against colour shifts (HVI-CIDNet), and a lightweight curve
method (Zero-DCE++). We will also test two targeted fixes on the U-Net, an exposure augmentation and a colour-direction loss,
one at a time, so we know which change helped. Colour is measured with the standard CIEDE2000 difference, not only PSNR.
The rule for choosing the final model was written down before we see any new test results. A model wins only if it improves
colour without losing structure or brightness accuracy. If no model clearly wins, we report the trade-off honestly.

---

## I. Approval checklist
1. **Checkpoint access**: copy `best.pth` (and `history.json`, `config.yaml`, logs) from Drive to the local project, read-only use.
2. **Saved outputs**: approve `export_outputs.py` (new file, new output folder) and the SICE storage policy (F).
3. **Colour metrics**: approve the metric set in E (ΔE00 + L*/C*/h + exposure + noise/edge + cost).
4. **Candidates**: Retinexformer, HVI-CIDNet, Zero-DCE++ in Group 1; RetinexNet supplementary/optional (or drop).
5. **Downloads**: official pretrained weights for P1–P3 (small files) under the leakage rules in C.
6. **Exposure augmentation**: which policy to train (`soft_shoulder`, as presented, or `hard_cap`), as E1.
7. **Colour loss**: angular loss with λ ∈ {0.1, 0.3, 1.0} and the validation rule in E.3 (Lab-a*b* as fallback only).
8. **Budget**: matched 5,600-step budget for all Group-1 runs; optional extended tier for *all* models only.
9. **Seeds**: 3 seeds for E0 and the final contenders.
10. **Registry change**: additive entry in `src/models/__init__.py`, or a separate Phase-3 registry (no edit to existing files).
11. **Selection rule E.3**: confirm it now. It is frozen afterwards.
12. **No-reference track**: use LoLI-Street RLLT (1,000 real images) for blind visual review (50 images, fixed seed).
13. **SICE -1 EV track**: add as a pre-registered second SICE protocol.

### Sources
- Retinexformer: arXiv 2303.06705; github.com/caiyuanhao1998/Retinexformer
- RetinexNet: arXiv 1808.04560; github.com/weichen582/RetinexNet
- Zero-DCE: arXiv 2001.06826. Zero-DCE++: arXiv 2103.00860; github.com/Li-Chongyi/Zero-DCE_extension
- HVI-CIDNet: arXiv 2502.20272; github.com/Fediory/HVI-CIDNet
- SICE: Cai, Gu, Zhang, *Learning a Deep Single Image Contrast Enhancer from Multi-Exposure Images*, IEEE TIP 2018 (ieeexplore 8259342)
- LoLI-Street: Islam et al., ACCV 2024, arXiv 2410.09831
- CIEDE2000 test data: Sharma, Wu, Dalal, *Color Research & Application* 30(1), 2005
