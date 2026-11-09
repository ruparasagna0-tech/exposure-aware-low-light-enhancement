# Stage 1 pilot: download confirmation manifest (2026-10-14)

Pristine shallow copies of the official repositories, fetched at pinned commits (`git fetch --depth 1 origin <sha>`).
They are **not** imported directly by project code. Only the files listed under "used" are later copied (vendored) into
`third_party/<model>/` with documented, minimal patches.

| Model | Repository | Pinned commit | Licence | Size on disk | Files used |
|---|---|---|---|---|---|
| Retinexformer | https://github.com/caiyuanhao1998/Retinexformer | 1e9a0efce4b306b6701b824768370ff26066c32a | MIT (`LICENSE.txt`, © 2023 Yuanhao Cai) | 13 MB | `basicsr/models/archs/RetinexFormer_arch.py`; config values from `Options/RetinexFormer_LOL_v2_{real,synthetic}.yml` (n_feat 40, stage 1, num_blocks [1,2,2]); test-time padding/clamp from `Enhancement/test_from_dataset.py` (reflect pad to multiple of 4, clamp [0,1]) |
| HVI-CIDNet | https://github.com/Fediory/HVI-CIDNet | eb43d7d91e9a336c66856824ff9e4603ae41f408 | MIT (`LICENSE`) | 64 MB | `net/{CIDNet,HVI_transform,LCA,transformer_utils}.py`; `loss/{losses,loss_utils,vgg_arch}.py`; loss composition from `train.py` and `data/options.py` |
| DarkIR | https://github.com/cidautai/DarkIR | 0910291157ad04f210481a002cf6bb480f6a7f18 | MIT (`LICENSE`) | 69 MB | `archs/{DarkIR,arch_model,arch_util}.py`; `losses/loss.py` (classes `L1Loss`, `VGG19`, `VGGLoss`, `EdgeLoss`, `EnhanceLoss` only); architecture config from `options/test/*.yml` (width 32 = DarkIR-m). **No training script or training config exists in the repository.** |

No weight files (`.pth/.pt/.ckpt/.bin/.safetensors/.onnx/.npy`) are present in any of the three copies (checked).

## Weight files required (none downloaded yet)
| File | Source | Purpose | Needed by | Where / when |
|---|---|---|---|---|
| `vgg19-dcbb9e9d.pth` (torchvision `VGG19_Weights.IMAGENET1K_V1`, about 548 MB) | `download.pytorch.org` via torchvision (SHA-256 prefix `dcbb9e9d` is checked by torchvision on download) | **training-loss backbone only**: the perceptual term of HVI-CIDNet (conv1_2/2_2/3_4/4_4 features, MSE, weight 0.01) and of DarkIR (5 VGG-19 slices, L1, weight 0.01; also inside its low-light guide loss). Never an enhancement model; never used for evaluation | HVI-CIDNet, DarkIR-m | **Colab only**, at pilot time. Mac unit tests use the same VGG-19 architecture with random weights, so the Mac downloads nothing. |
| — | — | Retinexformer and the U-Net use L1 only | — | — |
| `alexnet-owt-7be5be79.pth` (LPIPS evaluation) | already cached on the Mac since 2026-09-27 | evaluation metric only | all | not needed in the pilot (no LPIPS evaluation in the pilot) |

## Python dependencies
| Package | Needed by | Status |
|---|---|---|
| `einops` | Retinexformer (`IG_MSA`), HVI-CIDNet (`CAB`): `rearrange` only | **not installed on the Mac; needs approval** (pure-Python, about 60 KB; upstream pins 0.6.1 / 0.8.0) |
| `huggingface_hub` | HVI-CIDNet `CIDNet` class (`PyTorchModelHubMixin`, a convenience for Hub upload/download) | **avoided**: the mixin is removed in the vendored copy (no effect on the network) |
| `pytorch_msssim`, `ptflops`, `kornia`, `wandb` | DarkIR SSIM class, FLOP printing, other utilities | **avoided**: not part of the vendored files |
| torch / torchvision | all | already installed (Mac: torch 2.14.0, torchvision 0.29.0); Colab version recorded at pilot time |

## Known deviations required for vendoring (to be documented per file)
- HVI-CIDNet `EdgeLoss` and DarkIR `VGGLoss` hard-code `.cuda()`: changed to follow the input device (needed for CPU tests).
- HVI-CIDNet `CIDNet`: `PyTorchModelHubMixin` base class removed.
- HVI-CIDNet `vgg_arch.py`: loads `pretrained=True` via the deprecated torchvision argument; replaced with the equivalent
  `weights=VGG19_Weights.IMAGENET1K_V1` (same file), plus a `weights=None` switch for CPU tests.
- No changes to any layer, initialisation or forward computation.
