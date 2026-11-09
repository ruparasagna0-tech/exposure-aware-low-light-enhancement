# Model weights

No weights or checkpoints are included in this repository. `models/`, `*.pth` and `*.safetensors` are git-ignored.

## Official pretrained weights (used in the Phase 5 diagnostic)

All are downloaded at pinned revisions. The MD5 and SHA-256 of every file are listed in [`weights_sources.json`](weights_sources.json), and `src/phase5/zoo.py` refuses a file whose MD5 differs. Every file loads into the vendored code in `third_party/` with all parameter names matching.

| Condition | Source (revision) | File | Licence |
|---|---|---|---|
| `hvi_off_real` | huggingface.co/Fediory/HVI-CIDNet-LOLv2-real-bestPSNR (`c317f57`) | model.safetensors | MIT |
| `hvi_off_syn` | huggingface.co/Fediory/HVI-CIDNet-LOLv2-syn-wperc (`001d921`) | model.safetensors | MIT |
| `hvi_off_gen` | huggingface.co/Fediory/HVI-CIDNet-Generalization (`51481ef`) | model.safetensors | MIT |
| `rf_off_real` | huggingface.co/CaiYuanhao/Retinexformer (`51d9f18`) | LOL_v2_real.pth | MIT (GitHub repository) |
| (second backbone) | huggingface.co/CaiYuanhao/Retinexformer (`51d9f18`) | LOL_v2_synthetic.pth | MIT |
| `iat_lol` | github.com/cuiziteng/Illumination-Adaptive-Transformer (`c764722`), IAT_enhance/ | best_Epoch_lol.pth | Apache-2.0 |
| `iat_exp` | same repository and commit | best_Epoch_exposure.pth | Apache-2.0 |

### Example download

```bash
mkdir -p models/phase5_official/hvi_cidnet_lolv2_real_bestPSNR
curl -L -o models/phase5_official/hvi_cidnet_lolv2_real_bestPSNR/model.safetensors \
  https://huggingface.co/Fediory/HVI-CIDNet-LOLv2-real-bestPSNR/resolve/c317f57dec7dd66704ac1b13789a2fa5f20746de/model.safetensors
md5sum models/phase5_official/hvi_cidnet_lolv2_real_bestPSNR/model.safetensors   # must match weights_sources.json
```

`.pth` files are loaded with `torch.load(..., weights_only=True)`.

## Our own checkpoints

Our checkpoints are produced by the training scripts. They are not distributed.

| Checkpoint | How to reproduce |
|---|---|
| Stage-2 U-Net (`best.pth`, epoch 45) | `scripts/train_generalized.py --config configs/generalized_lol_v2.yaml` |
| Tier-A U-Net + exposure aug., Retinexformer, HVI-CIDNet, DarkIR-m | `scripts/train_phase3.py --config configs/phase3/tierA_<model>.yaml --amp off` (5,600 steps, seed 42, T4) |

For the Phase 5 diagnostic, `scripts/make_phase5_diag_bundle.py` expects them under `models/phase5_diag_weights/ours/`, with the official files under `models/phase5_diag_weights/official/`. The expected file names and MD5s are listed in `src/phase5/zoo.py`.

**Re-trained checkpoints will not reproduce our MD5s:**
- A checkpoint file also stores optimiser state and metadata.
- GPU training is not bitwise identical across GPU models, drivers and library versions. In our T4 resume test (`experiments/phase3_model_selection/stage1/PILOT_RESULTS_20261018.md`), U-Net and Retinexformer continued bitwise identically; HVI-CIDNet and DarkIR-m did not, even on the same GPU.
- Re-trained models should match the reported validation PSNR to within normal run-to-run variation (single-seed differences below about 0.3 dB are not interpreted), not byte for byte.

After re-training, update `WEIGHTS` in `zoo.py` with the new MD5s.

## Downloaded automatically at run time

| Item | Source |
|---|---|
| LPIPS AlexNet backbone | torchvision |
| VGG19 (HVI-CIDNet and DarkIR perceptual losses, training only) | torchvision |
| NIQE / BRISQUE parameter files | pyiqa (from huggingface.co/chaofengc/IQA-PyTorch-Weights) |
