# Stage 1 plan (revised 2026-10-14): recent models, Colab GPU training

> **T4-only revision:** sections 3, 4, 5 and 8 are superseded by [`STAGE1_PLAN_T4.md`](STAGE1_PLAN_T4.md) (single Colab
> T4; pilot first; DarkIR-m optional; DarkIR-l, DiffLL and extra seeds deferred).

Status: **plan only. Nothing downloaded, no code or data changed, no Colab runtime launched.**
This document supersedes sections C, D and F of `RESEARCH_PLAN.md` (2026-10-13). Sections A, B and E there still apply,
updated by the Stage 0 results (`stage0/outputs/phase2_unet_best/`).
Labels: **[verified]** = checked in a paper or official repository on 2026-10-14; **[estimate]**; **[decision]** = needs approval.

---

## 1. What Stage 0 established (the evidence this plan builds on)
- The Phase-2 U-Net (`best.pth`, epoch 45) is recovered. Its metrics reproduce exactly on 3,255 images, and its enhanced
  outputs are saved. **It stays the verified baseline (E0).**
- Measured weaknesses (reference-based):
  1. **Brightness not adapted to input darkness**: over-exposed in 100% of LoLI-light and 46% of LOL-v2-real images.
  2. **Colour error beyond brightness**: L*-equalised ΔE00 is 4.4–7.4. On synthetic and LoLI data, plain scaling preserves
     colour better. Median hue error 14–27°; 57% of synthetic outputs desaturated; colour casts in 29–45% of LOL-v2/SICE.
  3. **Detail loss**: edge energy only 41–70% of the reference.
  4. **Noise**: added grain on SICE (flat-noise ratio 1.54).
- Phase-2 U-Net training took **12.8 min on a Colab T4** (5,600 steps) [verified, `train_summary.json`]. This is the timing
  anchor for all GPU estimates below. (The earlier plan's 0.5–1 h guess was too high.)

## 2. Candidate verification (primary sources)

| | **HVI-CIDNet** | **DarkIR** | **Retinexformer** | DiffLL (optional diffusion) | U-Net (E0) |
|---|---|---|---|---|---|
| Paper | Yan et al., *HVI: A New Color Space for Low-light Image Enhancement*, **CVPR 2025**, arXiv 2502.20272 | Feijoo, Benito, Garcia, Conde, *DarkIR: Robust Low-Light Image Restoration*, **CVPR 2025**, arXiv 2412.13443 | Cai et al., *Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement*, **ICCV 2023**, arXiv 2303.06705 | Jiang et al., *Low-Light Image Enhancement with Wavelet-based Diffusion Models*, **SIGGRAPH Asia 2023 (ACM TOG 42(6))**, arXiv 2306.00306 | ours |
| Official repo | github.com/Fediory/HVI-CIDNet | github.com/cidautai/DarkIR | github.com/caiyuanhao1998/Retinexformer | github.com/JianghaiSCU/Diffusion-Low-Light | — |
| Licence [verified] | MIT | MIT (GitHub API) | MIT (GitHub API) | MIT (GitHub API) | — |
| Size [paper] | 1.88 M params, 7.57 G at 256² | DarkIR-m 3.31 M / 7.25 GMACs; DarkIR-l 12.96 M / 27.19 GMACs (256²) | 1.61 M, 15.57 G at 256² | not verified | 1.93 M, 9.22 GMACs at 256² |
| Core idea | Learnable HVI colour space (polarised hue/saturation + intensity collapse) decouples colour from intensity | Efficient CNN for joint low light + noise + blur: frequency MLP on FFT amplitude (encoder), dilated spatial attention (decoder), 8x-downsampled illumination guidance | One-stage Retinex: light-up map × input, then illumination-guided transformer | Conditional diffusion on wavelet low-frequency band + high-frequency restoration module | encoder-decoder |
| What it targets for us | **colour shifts** (our failure 2) | **noise + detail** (3, 4), robustness | **brightness/illumination** (1) | detail realism | — |
| Official training recipe [verified] | `train.py` + `data/options.py`: Adam 1e-4, cosine restart + warmup, batch 8, crop 256, **1,000 epochs**; loss = (L1 + 0.5·SSIM + 50·edge + 0.01·VGG-perceptual) in **both RGB and HVI**; optional random-gamma input augmentation (0.60–1.20) used for `generalization.pth` | **Training script not public**: README training section commented out, no `train.py`, no `options/train/`. `archs/`, `losses/` released. Paper: AdamW 5e-4 cosine to 1e-6, batch 32, crops 384, **H100**; loss L1 + 0.01·LPIPS-VGG19 + 50·edge + low-light guide L1 | BasicSR YAML (LOLv2-real): Adam 2e-4, cyclic cosine restarts, batch 8, patch 256 (comment says 128), L1, mixup; `total_iter: 150000` but header says 300k (**internally inconsistent**) | `configs/LOLv1.yml`: batch 8, patch 256, 500 epochs, lr 1e-4, 200 diffusion timesteps; two-part model | Adam 2e-4, batch 4 (2+2), crop 256, 5,600 steps, L1 |
| **Official validation uses the test set** [verified] | **Yes**: `data_val_lolv2_real = …/Test/Low`, evaluated every 10 epochs; released `best_PSNR.pth` / `best_SSIM.pth` imply **test-set checkpoint selection** | not inspectable (no training code) | **Yes**: val dataroot = `…/Real_captured/Test`, every 1,000 iters | `val_dataset: "LOLv1"` (validation on LOL-v1 eval data) | no (scene-disjoint val) |
| GT-mean in reported numbers | some rows ("w GT mean") | not stated | reported separately; authors advise against | GSAD-style adjustment not stated for DiffLL | never |
| Original environment | Python 3.7, PyTorch 1.13.1 | Python 3.10, PyTorch 2.5.1, CUDA 12.4 | PyTorch 1.11 (paper) / 2.x (NTIRE'24); BasicSR | requirements not shown; derived from WeatherDiff | Python 3.12, PyTorch 2.14 |
| Pretrained-weight exposure (for later supplementary use only) | LOL-v1 weights: saw 86/100 LOL-v2-real Test inputs (our audit). LOL-v2 weights: test-selected. `SICE.pth`: SICE split unverified | multitask weights trained on LOLBlur + LOLv2 + LSRW; paper appendix also says "LOL datasets", so **LOL-v1 exposure cannot be excluded** | LOL-v1 weights: same LOL-v1 exposure. LOL-v2 weights: test-validated | training data of released weights **not stated** | none |
| Feasibility under our protocol | **High**: training code public; small model; integrate arch + HVI + loss | **Medium**: arch + losses public, training loop must be written to the paper's description (risk of deviating from the authors' recipe) | **High**: arch file is self-contained (needs `einops`); loss is L1 | **Low/optional**: diffusion training is long; sampling × tiled 20 MP SICE is slow; adapting the data pipeline is needed | done |

Not recommended:
- **GSAD** (NeurIPS 2023): the repository has **no licence** (GitHub API `license: null`), and its README describes GT-mean
  brightness adjustment.
- **PyDiff**: trained on LOL-v1 only.
- **RetinexNet**: TF1, dated (unchanged from the earlier plan).

Zero-DCE++ moves to an optional **efficiency reference for the video track** (about 10 K params, zero-reference, CC BY-NC).

**Interpretation (not verified fact)**: the three priority models map onto our three measured weaknesses (colour →
HVI-CIDNet, noise/detail → DarkIR, illumination → Retinexformer). That makes the comparison informative whichever wins.
Published leaderboard numbers are not used: they come from test-validated or GT-mean protocols.

## 3. Image-enhancement experiment design (Track I)

**Data protocol (unchanged, leakage-controlled)**
- Train: LOL-v2-real Train minus val scenes (625) + LOL-v2-synthetic Train minus val scenes (808); balanced 2+2 batches.
- **Validation (all checkpoint, schedule and λ choices): our scene-disjoint 64 + 92 pairs. This replaces the official
  test-set validation of every candidate.**
- Seen-domain test: LOL-v2 Test (100 + 100), evaluated once per final model.
- SICE Part1 (58) and LoLI-Street Val (2,997): **development evidence** (already inspected). Used for analysis, not as untouched final tests.
- **[decision] New sealed final test**: SICE **Part2** (229 sequences with Labels, never used or inspected in any experiment).
  It needs a short leakage audit first, because thumbnail correlation flagged Part1/351 ~ Part2/12 (0.94) as a candidate.
  It stays sealed until the selection rule is applied.
- No-reference: LoLI-Street RLLT (1,000 real night street images; brightness/colour statistics + blind visual review).
- LOL-v1 stays excluded. No pretrained weights are used in the main comparison.

**Fairness tiers**
- **Tier A: matched protocol** (primary): every model trained from scratch on the same data, sampler, crops (256), seed,
  validation rule, **5,600 steps, batch 4**. Each model keeps **its own published loss** (the method as published).
  Learning curves are reported so under-training is visible.
- **Tier B: extended equal budget**: the top 2 by validation plus the U-Net, each at the same larger budget (proposed 50,000
  steps, cosine LR). This is the "given enough training" comparison.
- **Loss control** (only for the Tier-B winner): retrain with L1 only, to separate architecture from loss.
- **Seeds**: 3 seeds for Tier-B finalists (decides whether differences exceed seed noise).
- U-Net variants from the earlier plan (exposure augmentation, colour-angle loss) run in Tier A as E1–E3.

**Evaluation** (Stage 0 tooling, unchanged definitions): PSNR/SSIM/LPIPS; ΔE00 (mean/p95, L*-equalised); ΔL*,
over/under-exposure shares; chroma ratio, hue error, cast; edge correlation / energy ratio; flat-region noise; clipping;
params, MACs, s/image at 600x400 and 20 MP (same GPU). Per darkness level (LoLI dense/moderate/light). Plus a new
**exposure-response probe** on the validation set: inputs scaled by {0.25, 0.5, 1, 1.5, 2}; a brightness-adaptive model keeps
output L* nearly constant. Selection rule = `RESEARCH_PLAN.md` E.3 (frozen before the sealed test is opened).

## 4. Colab execution requirements
- **Mac**: code, unit tests, CPU smoke tests on 2–3 images, result inspection only. **No training on the Mac.**
- **Colab GPU**: T4 is the minimum (Tier A). For Tier B, DarkIR-l and diffusion, an **L4 or A100 (Colab Pro compute units)**
  is strongly advised. Session length and idle disconnects vary by plan, so every job must be resumable.
- **Drive layout** (new, nothing existing touched):
  `experiments/phase3_model_selection/stage1/<run_id>/{config.yaml, checkpoints/{last.pth,best.pth}, history.json, logs/, eval/}`.
  The `<run_id>` folder must not exist at start, or `--resume` is given explicitly.
- **Safety mechanisms (reuse Phase-2 patterns)**:
  - atomic checkpoint writes (`save_checkpoint` writes a temp file then renames);
  - `last.pth` every epoch + `best.pth` on validation improvement;
  - a deterministic batch stream so resume reproduces the same data order;
  - integrity checks before training;
  - one-time test evaluation guard;
  - MD5 of the data zip verified each session;
  - `pip freeze` and GPU name logged per run.
- **Code plan** (after approval; all new files):
  - vendored model files under `third_party/<model>/` with licence + commit hash;
  - a model registry for Phase 3 (no edit to `src/models/__init__.py` unless approved);
  - `scripts/train_phase3.py`, reusing the existing datasets, sampler and integrity checks;
  - loss modules per method;
  - a Colab notebook per tier;
  - tests: forward/backward on CPU, parameter count matches the paper (±1%), loss finite, checkpoint round trip,
    RGB/range checks, resume determinism.
- **Dependencies / compatibility risks**:
  - Colab ships a recent PyTorch, while the original code targets 1.11–1.13 (CIDNet, GSAD) or 2.5 (DarkIR). Vendored
    files may need small API fixes (each documented).
  - Retinexformer needs `einops` (BasicSR is not needed for the architecture alone).
  - DarkIR's FFT blocks are sensitive to mixed precision (we train in fp32 unless validated).
  - VGG/LPIPS backbones for the CIDNet/DarkIR losses need ImageNet weights: a download at training time that needs approval.
  - The DarkIR recipe is reconstructed from the paper, so deviations are a methodological risk.

## 5. GPU-time estimates [estimates; anchor: U-Net 5,600 steps = 12.8 min on T4]
| Job | T4 | A100 (about 3–4x faster) |
|---|---|---|
| Tier A, per model, 5,600 steps: U-Net variants (E1–E3) | 13 min each | 4 min |
| Retinexformer | 35–60 min | 10–20 min |
| HVI-CIDNet (VGG perceptual loss on RGB + HVI dominates) | 40–80 min | 12–25 min |
| DarkIR-m | 40–80 min | 12–25 min |
| Export + metrics per model (all domains; SICE 20 MP tiled) | 15–45 min | 5–15 min |
| **Tier A total** (U-Net ×3 variants + 3 models + evals) | **about 5–8 h** | **about 1.5–3 h** |
| Tier B (50,000 steps): U-Net + 2 finalists | 15–25 h | 4–8 h |
| Seeds (2 extra × 3 Tier-B models) | 30–50 h | 8–16 h |
| Optional DiffLL (training + slow sampling eval) | 1–2 days | 6–12 h |
Colour evaluation runs on the Colab CPU (high-RAM) or the Mac. Mac timing measured in Stage 0: about 1.5 h for all domains.

## 6. Roadmap: separate tracks, run in order
| Track | Question | Inputs | Starts after |
|---|---|---|---|
| **I: Image enhancement** (Stages 1–3) | Which method best preserves colour/detail and adapts brightness on unseen data? | as section 3 | now (on approval) |
| **V: Video temporal consistency** | Does the chosen model flicker frame to frame; does a temporal method help? | paired low-light video data (not yet obtained; candidates need a licence/leakage audit), plus real night clips | Track I model frozen |
| **D: Downstream detection** | Does enhancement improve detection on dark images? | LoLI-Street includes YOLO labels for Train/Val (**verified in our audit, unused so far**); a real night detection set (to be audited) | Track I model frozen |
Video metrics (planned): flow-warped frame difference, temporal ΔE00 / flicker index, per-frame image metrics.
Detection metrics: mAP of a fixed, pre-trained detector on raw vs enhanced vs reference images (detector not fine-tuned on test data).

## 7. Main risks
1. **Under-training in Tier A** makes the larger models look worse than they are. Mitigation: learning curves + Tier B.
2. **DarkIR recipe reconstruction**: no official training code. Mitigation: follow the paper exactly; label "re-implemented training".
3. **Loss confound**: methods differ in architecture and loss. Mitigation: L1-only control for the winner.
4. **Test reuse**: SICE Part1/LoLI already inspected. Mitigation: SICE Part2 sealed test.
5. **Colab disconnects and quota**: resumable jobs, checkpoints on Drive, budget tiers.
6. **Perceptual losses vs LPIPS metric**: CIDNet/DarkIR train with VGG-based perceptual losses, so LPIPS (Alex) may favour them.
   Mitigation: report ΔE00 / edge / noise alongside, and the selection rule is not LPIPS-only.
7. **Single-seed noise**: the seeds in Tier B.

## 8. Decisions needed before any implementation
1. Candidate set for Tier A: HVI-CIDNet, DarkIR-m, Retinexformer (+ U-Net E0–E3). DarkIR-l and DiffLL optional.
2. Method-faithful losses in Tier A (vs L1 for all).
3. Tier B budget (50,000 steps) and GPU type (T4 vs Colab Pro L4/A100).
4. SICE Part2 as the sealed final test (after a leakage audit).
5. Approval for ImageNet VGG weight downloads needed by the CIDNet/DarkIR training losses (metric/loss backbones, not
   enhancement weights).
6. Vendoring MIT-licensed model code under `third_party/` with attribution.
7. Video and detection tracks stay deferred until Track I's model is frozen.
