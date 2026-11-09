# HVI-CIDNet resume control — pre-registered rule (written 2026-10-19, BEFORE the control runs)

**Context.** Pilot P6 (attempt `hvi_cidnet_20261018_184508`, VM f79b313b...) failed its resume gate: epoch-2
train_loss A (uninterrupted) 2.7467193603515625 vs B (interrupted + resumed) 2.743663191795349, relative difference
1.1127e-3 > tolerance 1e-4, not bitwise identical. Batch order identical. Epoch 1 (no resume in either run) already
differed: 2.4412569284439085 vs 2.4407834649085998 (1.94e-4), indicating GPU non-determinism. U-Net and Retinexformer
resumed bitwise identically through the same code path. **The P6 result stays recorded as FAIL.**

**Approved diagnostic (option 1).** Two additional uninterrupted runs C1, C2 of exactly the pilot's resume-test command
(`scripts/train_phase3.py --config configs/phase3/tierA_hvi_cidnet.yaml --amp off --epochs 2 --steps-per-epoch 10
--no-val --skip-integrity-md5`), same VM, throwaway weights, no evaluation. Cell: `cidnet_resume_control_cell.py`.

**Rule (fixed now):**
- spread = largest pairwise relative difference of the epoch-2 train_loss among the uninterrupted runs {A, C1, C2};
- resume_diff = |A − B| / |A| at epoch 2 = 1.1127e-3;
- **resume accepted ⇔ identical batch order in all runs AND resume_diff ≤ 3 × spread**, i.e. spread ≥ 3.709e-4.
- Otherwise resume is NOT accepted and HVI-CIDNet stays "not yet feasible" pending further diagnosis.

The rule factor (3×) and this threshold are not changed after the result is known.

## Result (control run 2026-10-18 19:22 UTC, same VM; Drive `.../stage1/pilots/hvi_cidnet_resume_control_20261018_192219/`)
| Run | Epoch-1 loss | Epoch-2 loss | Batches |
|---|---|---|---|
| A (uninterrupted, pilot) | 2.441256928 | 2.746719360 | a10f7f15, 6c249c2f |
| B (interrupted + resumed, pilot) | 2.440783465 | 2.743663192 | same |
| C1 (uninterrupted, control) | 2.445152020 | 2.765099335 | same |
| C2 (uninterrupted, control) | 2.439806390 | 2.741907549 | same |

Spread among uninterrupted runs: epoch 1 2.19e-3, epoch 2 **8.39e-3**; resume difference **1.11e-3** ≤ 3 × 8.39e-3 =
2.52e-2; identical batch order → **resume ACCEPTED by the pre-registered rule** (resume difference = 0.13 × the
non-resume spread). The P6 gate result stays recorded as FAIL (tolerance 1e-4 assumes bitwise determinism, which does
not hold for HVI-CIDNet on this GPU stack). Consequence: HVI-CIDNet training is not bit-reproducible here; identical runs
differ by ~0.8% in loss after 20 steps, so model comparisons involving HVI-CIDNet need repeated seeds.
