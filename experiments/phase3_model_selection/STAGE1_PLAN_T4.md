# Stage 1 plan: T4-only revision (2026-10-14)

Status: **plan only.** Nothing downloaded, no project changes, no Colab runtime, no training.
Constraint: **one Google Colab T4 GPU** (no L4/A100). This revision supersedes sections 3, 4, 5 and 8 of
`STAGE1_PLAN.md`. Its sections 1 (Stage 0 evidence), 2 (verified candidate facts), 6 (roadmap) and 7 (risks) still apply.
Labels: **[measured]** = observed in this project; **[spec]** = vendor/paper figure; **[estimate]** = to be replaced by
pilot measurements; **[decision]** = needs approval.

---

## 1. Scope under the T4 constraint
| Model | Role now | Reason |
|---|---|---|
| U-Net E0 (Phase-2 `best.pth`) | **verified baseline**: no retraining; Stage 0 outputs and metrics reused | reproduced exactly in Stage 0 |
| **HVI-CIDNet** | **priority** | targets our largest model-side colour errors; public training code; 1.88 M params |
| **Retinexformer** | **priority** | targets brightness adaptation; public code; loss = L1 (same as U-Net); 1.61 M params |
| DarkIR-m | **optional**: enters only if its pilot passes the gates in section 3 | no public training script (recipe rebuilt from the paper); FFT + depthwise convs; LPIPS-VGG19 loss |
| U-Net E1 (exposure augmentation, already built) | optional, cheap (about 15 min) | tests the brightness hypothesis on the baseline architecture |
| DarkIR-l, DiffLL, extra seeds, U-Net colour-loss variants | **deferred** | revisit only if Tier-A results justify them |

## 2. Hardware facts and precision policy
- T4 [spec]: 16 GB GDDR6, Turing tensor cores (FP16), **no bfloat16**. Mixed precision on T4 therefore means **FP16 autocast +
  GradScaler**.
- Colab sessions [not guaranteed]: runtime length, idle timeouts and GPU availability vary. Every job must survive a disconnect.
- Colab VM: 2 vCPUs (data loading, PNG encoding and colour metrics are CPU-bound), about 12.7 GB RAM.
- **AMP policy** (decided per model by the pilot, not assumed):
  - training may use FP16 autocast only if the pilot shows (a) no inf/NaN and fewer than 1% of steps skipped by GradScaler,
    (b) the loss over the same 300 steps stays within 2% of FP32 (same seed and batches), and (c) a speed-up of at least 1.3x;
  - colour-space code (HVI transform, sRGB to Lab), FFT blocks (DarkIR) and loss reductions are wrapped to run in **FP32**;
  - **evaluation, export and all reported metrics: always FP32**, matching E0;
  - AMP use is recorded per run as a documented protocol difference from E0 (which was FP32).
- **Batch size**: protocol batch 4 (2 real + 2 synthetic) at 256x256 crops. The pilot measures the maximum feasible batch, but
  the protocol batch is kept for comparability.
- **Gradient accumulation**: not planned (batch 4 is expected to fit). Fallback if a model cannot fit batch 4: 2 micro-batches
  of (1 real + 1 synthetic) with averaged gradients. All three candidates use layer-type normalisation, not BatchNorm, so this is
  equivalent to batch 4 (to be confirmed per model in the pilot).

## 3. GPU feasibility pilot (one Colab session, before any full training)
Pilot weights are throwaway and never evaluated. Pilots read **training data only** (plus validation images for timing; no
validation metric is used for any decision). Results go to `Drive: experiments/phase3_model_selection/stage1/pilots/<model>/pilot_report.json`.

| Step | What is measured | Pass gate |
|---|---|---|
| P0 environment | GPU name, torch/CUDA/cuDNN versions; data zip MD5; prepared-data integrity. **Setup run: `full` scope** (structural checks + per-file MD5 of every prepared file). **Model runs: `structural` scope** (same checks without re-hashing), allowed only with `--p0-md5-reference` to a passed full check from the same VM session (boot id); otherwise the model run does the full check | all pass |
| P1 calibration | U-Net, **Phase-2-comparable**: 3 full epochs timed exactly like `train_generalized.py` (112 steps without extra syncs + 1 validation pass per epoch, train/val split recorded); projection = epoch 1 + 49 × median(epochs 2–3), compared with the sum of the 50 recorded Phase-2 epochs (757.8 s). The synced per-step rate (v1 method) is kept as a diagnostic only | within ±30% (unchanged) |
| P2 correctness | per model: parameter count vs paper (±1%); forward/backward on a real 2+2 batch; output in [0,1]; loss finite; RGB order (known-colour patch) | all pass |
| P3 speed & memory | 50 warm-up + 200 timed steps, batch 4 @256, **FP32 and AMP**; `torch.cuda.max_memory_allocated`; max batch that fits | batch 4 fits in at most 12 GB (3 GB headroom) |
| P4 AMP equivalence | 300 steps FP32 vs AMP from the same init and batches: loss trajectories, skipped steps | AMP gates in section 2 |
| P5 inference | validation: 1 warm-up pass + median of 3 timed passes (156 images, 600x400 / 384x384); one 20 MP input with 512/64 tiles: time and peak memory, FP32 | projected full export of all dev sets at most 1.5 h |
| P6 resume | 100 steps, checkpoint to Drive, simulated restart, resume: next batch identical; loss continues | identical batch order; no step lost |
| P7 projection | projected Tier-A wall time = steps × s/step + 50 validation passes + checkpoint I/O | **at most 3 h** (fits one session with margin) |
| DarkIR-m only | rebuilt training loop matches the paper's loss terms/weights; FFT under AMP (FP16 FFT on CUDA is limited to power-of-two sizes; else FP32) | loss decreases over 300 steps; FFT path validated |

> **2026-10-14, approved reference change:** the P1 reference is now the original Phase-2 trainer measured in the same Colab session (20.55 s/epoch), not the September record (15.1 s/epoch, kept). Tolerance ±30% unchanged; re-anchored ratio 1.016 (pass); vs September 1.38 (fail, kept). Slowdown ≈1.36×, cause not proven. See `stage1/CALIBRATION_DECISION_20261014.md`.

**Pilot session cost [estimate]**: P0–P1 about 20 min; P2–P7 about 15–25 min per model, so **1.5–2 T4-hours** for 3 models.
If a model fails a gate, the options are listed in its report (e.g. FP32 instead of AMP, fallback accumulation). It does not
proceed without approval.

## 4. Experiment sequence (T4)
| Step | Work | Where | Uses | T4 cost [estimate] |
|---|---|---|---|---|
| S1.0 | Vendor MIT model code + write trainer/registry/tests; CPU unit tests (2–3 images) | **Mac** (no training) | — | 0 |
| S1.1 | **Pilot session** (section 3) | Colab T4 | train data (+ val timing only) | **1.5–2 h** |
| S1.2 | Decision point: which models and precision go to Tier A (from measured pilots) | you | pilot reports | — |
| S1.3 | **Tier A training**: Retinexformer, HVI-CIDNet (+ DarkIR-m, + U-Net E1 if approved). 5,600 steps, batch 4, own published loss, validation every epoch, `best.pth` = max mean val PSNR (Phase-2 rule) | Colab T4, **one model per session** | train + val | see section 5 |
| S1.4 | **Development export**: FP32 outputs + PSNR/SSIM/LPIPS on **validation**, SICE Part1 and LoLI-Street Val (dev evidence) | Colab T4 | val + dev | see section 5 |
| S1.5 | Colour / quality metrics on the exported outputs (Stage 0 tooling) | **Colab CPU-only runtime** (no GPU quota) or Mac [decision] | outputs | 0 GPU |
| S1.6 | Decision point B: Tier-B finalists (at most 2) + whether Tier B is justified. Rule: a model qualifies only if its val PSNR slope over the last 10 epochs is > 0.05 dB/epoch (still improving) **or** it is within 0.5 dB of the best on val; ranking by validation ΔE00 and PSNR, dev-set evidence reported alongside | you | val + dev | — |
| S1.7 | **Tier B** (only if justified): finalists + U-Net at 3x budget (16,800 steps), same rules, resumable across sessions | Colab T4 | train + val | see section 5 |
| S1.8 | Sealed-test preparation: SICE Part2 leakage audit (vs Part1 test, LOL, LoLI) | **Mac CPU** | Part2 metadata/thumbnails only, no model outputs | 0 |
| S1.9 | **Final evaluation, once per finalist**: LOL-v2 Test + sealed SICE Part2 + colour metrics; selection rule (RESEARCH_PLAN E.3) frozen before this step | Colab T4 | test sets (first and only use) | 1–2 h |

The sealed SICE Part2 is never used in S1.1–S1.7. LOL-v2 Test is also deferred to S1.9 for new models (E0's existing test numbers
remain as recorded). Checkpoint, λ and Tier-B choices use **validation only**. SICE Part1 / LoLI-Street are reported as dev
evidence, never as final results.

## 5. Expected T4 costs [estimates, to be replaced by pilot measurements]
Anchor [measured]: U-Net Phase-2 training = 757.8 s of epochs for 5,600 steps (about 0.135 s/step **including** 50
full-resolution validation passes), FP32, T4.

| Model | Train memory, batch 4 @256 | s/step FP32 | Tier A (5,600 steps + 50 val passes) | Dev export (val + SICE P1 + LoLI, FP32) |
|---|---|---|---|---|
| U-Net E1 | about 2 GB | about 0.13 | **about 15 min** | 15–20 min |
| Retinexformer (MACs 1.7x U-Net; attention, LayerNorm and many small kernels) | 4–7 GB | 0.35–0.60 | **35–60 min** (AMP: maybe 25–45) | 30–60 min |
| HVI-CIDNet (VGG-19 perceptual loss on RGB **and** HVI = 4 VGG passes per sample, which dominates) | 5–9 GB | 0.45–0.80 | **45–80 min** (AMP: maybe 30–55) | 25–45 min |
| DarkIR-m (depthwise + FFT; LPIPS-VGG19 loss) | 5–9 GB | 0.45–0.80 | **45–80 min** | 30–60 min |

Why dev export is slow: SICE Part1 is 58 images × about 126 tiles of 512² each, about 7,300 tile forwards, plus LPIPS on
20 MP images and PNG encoding on 2 vCPUs. [measured] The original U-Net T4 evaluation of all test domains took about 10 min.

| Stage | T4 hours [estimate] | Sessions |
|---|---|---|
| Pilot (S1.1) | 1.5–2 | 1 |
| Tier A, priority (Retinexformer + HVI-CIDNet: train + dev export) | 2.5–4 | 2 |
| Tier A, optional (DarkIR-m; U-Net E1) | 1.3–2.3 | 1–2 |
| Tier B (2 finalists + U-Net at 16,800 steps), if justified | 4.5–8 | 3–4 (resumable) |
| Final evaluation (S1.9) | 1–2 | 1 |
| **Total** | **about 6–10 h without Tier B; about 11–18 h with Tier B** | **about 5–9 sessions** |
Colour metrics (S1.5) add about 1.5–2 h of CPU per model on the Mac [measured in Stage 0], or longer on a 2-vCPU Colab CPU
runtime. No GPU time is used.

## 6. Resumable, safe Drive layout
```
Drive: LowLight-Generalization/experiments/phase3_model_selection/stage1/
  pilots/<model>/pilot_report.json            (throwaway weights not kept)
  runs/<run_id>/                              e.g. tierA_retinexformer_s42
    config.yaml  run_info.json (GPU, versions, AMP flag, pip freeze)
    checkpoints/last.pth  (every epoch; model + optimizer + GradScaler + scheduler + RNG + sampler cycle)
    checkpoints/best.pth  (on validation improvement only)
    history.json  logs/  dev_eval/  (written once)  final_eval/ (written once, S1.9 only)
```
- Atomic writes (temp file + rename) for checkpoints, history and manifests. A new run refuses an existing `<run_id>`
  unless `--resume` is given. Resume verifies the config MD5 and continues the deterministic batch stream.
- Data: `phase2.zip` copied to `/content`, MD5-checked, unzipped each session. **Never** written back to Drive.
- One device per output folder. Mac and Colab never write the same folder.
- On disconnect: rerun setup cells, then the training cell with `--resume runs/<run_id>/checkpoints/last.pth`. At most one
  epoch (112 steps, about 1–2 min) is lost.

## 7. Risks specific to the T4
1. **Session interruptions**: mitigated by per-epoch checkpoints and one model per session.
2. **AMP changing results**: pilot gate P4; FP32 for colour transforms/FFT and all evaluation.
3. **Under-training** at 5,600 steps (the official recipes use 150k–300k iterations or 1,000 epochs): reported via learning
   curves. Tier B is only partly a remedy at 16,800 steps. Conclusions are stated as "at a matched, limited budget".
4. **DarkIR-m reconstruction risk**: no official training code. Optional, and labelled as re-implemented.
5. **CPU bottlenecks on Colab** (2 vCPUs: data loading, PNG export): pilot P1/P5 measure the data-wait share. If it is high, the
   dev export can skip PNGs for SICE (metrics in-flight) on request.
6. **Quota limits**: total about 6–18 T4-hours over several days. Order of work = priority order, so stopping early still
   leaves a complete Retinexformer vs HVI-CIDNet vs U-Net comparison.

## 8. Decisions needed
1. Approve the **pilot** (S1.0 Mac code work + S1.1 one T4 session), including:
   - downloading the official HVI-CIDNet and Retinexformer code (and DarkIR code if the DarkIR-m pilot is approved);
   - the ImageNet VGG-19 weights their training losses need (loss/metric backbones, not enhancement weights).
2. Include **DarkIR-m** in the pilot (yes / no).
3. Include **U-Net E1** (exposure augmentation) in Tier A (yes / no).
4. **AMP**: pilot-gated, as in section 2 (or FP32 everywhere for strict comparability, at higher T4 cost).
5. **Colour metrics location**: Colab CPU-only runtime or Mac.
6. Tier-B qualification rule and 3x budget (section 4, S1.6).
7. SICE Part2 as the sealed final test (audit on the Mac first).
