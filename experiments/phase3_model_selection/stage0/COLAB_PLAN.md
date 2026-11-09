# Stage 0 on Colab GPU: execution plan (NOT executed; awaiting approval)

Written 2026-10-14 00:35 while the Mac CPU export was running (not interrupted, not modified).

## 0. Mac export: resume / overwrite inspection (read-only)
| Question | Finding (from `scripts/export_outputs.py`) |
|---|---|
| Can completed work be lost? | No. Completed domains are final folders (`lol_v2_real_test/`, `lol_v2_synthetic_test/`) plus their CSVs and manifest entries. A rerun **skips** them. |
| What happens to the domain in progress if stopped? | `sice.partial/` (about 8/58 images at 00:33) is **deleted and redone** on the next run. There is no per-image resume inside a domain, so up to about 75 min of SICE work is lost. |
| Could anything existing be overwritten? | No. The script refuses `experiments/generalized_lol_v2*`, never writes over a final domain folder, refuses a different checkpoint (MD5), and `save_png` raises if a file exists. The only deletion is its own `<domain>.partial/`. |
| Weak points found | (a) `export_manifest.json` is rewritten non-atomically after each domain. A kill at that exact moment could truncate it (tiny window). (b) Run-level fields (`device`, `versions`, `torch_threads`) are rewritten on every invocation, and per-domain entries do not record the device. **Resuming into the same folder on a different device would mislabel provenance.** Rule below: one device per output folder. (c) Two exports must never target the same folder at once. |
| Mac progress / ETA [estimate] | LOL-v2 done and reproduced. SICE about 1.3 min/image, done about 01:40. LoLI about 50 min, done about **02:30**. |

**There is no need to stop the Mac run for Colab.** The two runs write to different folders on different machines.

## 1. Exact paths
| Item | Path | Must match |
|---|---|---|
| Project (Drive) | `/content/drive/MyDrive/LowLight-Generalization` | — |
| Checkpoint (original, read-only) | `$P/experiments/generalized_lol_v2/checkpoints/best.pth` | MD5 `73f71cf4d01c07b5f3176a6ed067b32f` (= the imported copy) |
| Config | `$P/configs/generalized_lol_v2.yaml` | MD5 `7326edc089669cdd43fbfe92889769f8` |
| Data archive | `$P/data_prepared/phase2.zip` → unzip to `/content/phase2` | MD5 `3afbe90bb3026faa39591bf820ee3916`; then the prepared-data MD5 integrity checks |
| Recorded results (for comparison) | `$P/experiments/phase3_model_selection/stage0/phase2_evaluation_results.zip` (**upload**) | MD5 `e7302a3b660629702bc19bf6af2f5e0c` |
| LPIPS backbone | copy into `/root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth` from Drive (**upload** from the Mac cache) | MD5 `cc6360b220eb7cfb64d83dc81b1963b1` |
| Working output (VM disk) | `/content/stage0_colab/phase2_unet_best_colab_cuda/` | — |
| Final output (Drive, new) | `$P/experiments/phase3_model_selection/stage0/outputs/phase2_unet_best_colab_cuda/` | must **not** exist before copying |

New files to upload to the Drive project (verified by MD5 in the notebook):
`scripts/export_outputs.py` 506babe3c786450a56caff639d104d34 · `scripts/stage0_reproduction_report.py`
339befb70a4c3c1b62a1ad9467d7f571 · `scripts/evaluate_colour.py` 7ed24bb9eaf2f351e0acad5c81f05da8 ·
`src/metrics/colour_metrics.py` 7b0ae3f31804dd09f92e783c29ba3fc1. Plus the Mac LOL-v2 CPU outputs
(`outputs/phase2_unet_best/lol_v2_*`, 44 MB) for the CPU-vs-GPU comparison.
The Drive copy of `src/datasets/phase2.py` may be the pre-augmentation version. Both versions build identical test sets
(the change only affects training transforms); the export manifest records whichever MD5 is used.

## 2. Notebook steps (each step stops on failure)
1. **GPU**: assert CUDA; print the GPU name, torch, torchvision, Python.
2. **Drive + MD5 gate**: mount; assert every MD5 in section 1.
3. **Environment**: `pip install lpips==0.1.4` (the original used `>=0.1.4`, which resolved to 0.1.4). Copy the AlexNet file into the hub
   cache and assert its MD5. **No internet weight download**: `export_outputs.py` refuses if the cache is missing.
4. **Data**: unzip to `/content/phase2`; run `integrity_checks(cfg, '/content/phase2')` (MD5 of every prepared file).
5. **Gate A: GPU correctness on LOL-v2 (about 3 min)**
   - `python scripts/export_outputs.py --checkpoint $CKPT --data-root /content/phase2 --device cuda --domains lol_v2_real_test lol_v2_synthetic_test --out-dir /content/stage0_colab/phase2_unet_best_colab_cuda`
   - Reproduction: copy the two CSVs to `/content/gateA/per_image/` and run
     `stage0_reproduction_report.py --export-dir /content/gateA --results-zip <uploaded zip> --allow-partial`. The report is
     written in the copy, never in the output folder. **Pass = zero LOL discrepancies** (existing tolerances: per image 0.05 dB /
     5e-4 / 2e-3; means 0.005 dB / 5e-5 / 5e-4).
   - **RGB order on GPU**: compare every CUDA PNG with the Mac CPU PNG of the same image. Pass = max |diff| <= 1 level, per-channel
     mean diff < 0.05 level, and the diff after any channel permutation of the CUDA image is much larger (sanity). Combined with
     reproduced metrics against RGB references, a swap would fail both checks.
   - **Determinism**: export LOL-v2-real a second time into `/content/stage0_colab/determinism_rerun/` and compare PNG MD5s.
     Report the number of differing files and the max diff (cuDNN defaults are kept, as in the original evaluation, so 1-level
     flips are possible; >1 level = STOP).
   - **Any failure: STOP** and report. Nothing has been written to Drive at this point.
6. **Full export**: same command with `--domains sice loli_street_val` (completed LOL domains are skipped).
7. **Gate B**: `stage0_reproduction_report.py --export-dir /content/stage0_colab/phase2_unet_best_colab_cuda --results-zip …`
   (all domains, no `--allow-partial`). **Any discrepancy: STOP.**
8. **Copy to Drive**: `rsync -a` to `<final>.partial/`, verify every file's MD5 against `export_manifest.json`, then rename to the final
   name (asserted absent beforehand).
9. **Colour evaluation and tests**: on the **Mac**, after downloading the folder (about 2.5 GB). `tests/test_stage0.py` assumes the
   Mac layout (`data_prepared/phase2` extracted, results zip in `~/Downloads`), and the Mac has 8 cores against Colab's 2.
   Alternatively run `evaluate_colour.py --export-dir <Drive folder> --data-root /content/phase2` on Colab CPU (slower).

## 3. Disconnects
- Steps 1–7 write only to the VM disk (`/content`), so a disconnect loses nothing on Drive. **Restart from step 1.** The export is
  short (about 15–25 min), so a clean restart is safer than resuming on a new VM.
- During step 8: only `<final>.partial/` can exist on Drive. On restart, delete that `.partial` folder only and redo steps 1–8.
- Never point a Colab run at the Mac output folder or the reverse (one device per folder; see 0b).

## 4. Runtime [estimates]
| Step | T4 estimate | Basis |
|---|---|---|
| Setup, MD5 checks, unzip, integrity | 5–10 min | phase2.zip size; MD5 of about 10k files |
| Gate A (incl. determinism rerun) | 3–5 min | Mac CPU took 1.6 min for LOL; original T4 eval similar |
| SICE + LoLI export | 12–25 min | original T4 evaluation of all domains took about 10 min (results timestamps 03:30 to 03:40) plus PNG encoding (about 20 MB/SICE image, 2 vCPUs) |
| Copy to Drive (about 2.5 GB) + MD5 | 5–15 min | Drive FUSE throughput varies |
| Download to Mac + colour eval + tests | 1–1.5 h | download speed; colour eval dry run (SICE about 45 s/image with 2 workers) |
Total about 1.5–2.5 h wall time including manual uploads and downloads. **The Mac export alone finishes around 02:30**, so Colab saves
little time. Its main value is an **independent CUDA reproduction** on the same device type as the original evaluation.

## 5. Dependency / compatibility risks
- Colab's current torch/torchvision/Pillow versions are unknown until step 1. The Mac uses torch 2.14.0, torchvision 0.29.0,
  Pillow 12.3.0, numpy 2.5.3, Python 3.12.5. Small numeric differences are expected and covered by the tolerances.
- `lpips 0.1.4` calls torchvision AlexNet with the deprecated `pretrained=` argument (warning today; could break in a future
  torchvision). Fallback: `pip install torchvision==<Mac version>`. This needs your approval because it changes the environment.
- `Image.fromarray(..., mode="RGB")` is deprecated in recent Pillow (warning only, behaviour unchanged).
- Drive quota: about 2.5 GB is needed for the final folder (plus 0.3 GB uploads).
- Colab RAM (about 12.7 GB) / T4 VRAM (15 GB): the original SICE evaluation (tiling + LPIPS on 20–24 MP on CUDA) ran on a T4, so this is expected to fit.
- Optional small fix (approval needed, after the Mac run ends): record device/versions **per domain** in the manifest and
  refuse to resume a folder on a different device; write the manifest atomically.

## 6. Decisions needed
1. Run Colab at all? Recommendation: **let the Mac run finish**, and use Colab only as an independent CUDA cross-check (or as fallback
   if the Mac run fails).
2. Approve uploading the AlexNet cache file, the results zip, the 4 new files, and the 44 MB of Mac LOL-v2 outputs to Drive (your own
   Drive; nothing shared externally).
3. Colour evaluation location: Mac (recommended) or Colab CPU.
4. The optional manifest fix in 5.
