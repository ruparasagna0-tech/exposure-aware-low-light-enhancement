# Vendored third-party code (Phase 3 / Stage 1)

Copied from pinned commits of the official repositories (see `third_party_src/DOWNLOAD_MANIFEST.md`).

**Public repository note:** only the IAT originals are included (`third_party_src/IAT/`, about 44 KB). The pristine
Retinexformer, HVI-CIDNet and DarkIR clones (about 146 MB with `.git`) are not redistributed. Fetch them at the commits
below (`git clone <repo>; git -C <dir> checkout <commit>`) into `third_party_src/{Retinexformer,HVI-CIDNet,DarkIR}` to run
the vendoring tests in `tests/test_phase3.py` (they are skipped otherwise).

Each package keeps its upstream licence file. `PATCHES.diff` is the complete
diff against upstream, and `tests/test_phase3.py::test_only_recorded_patches` fails if any unrecorded change appears.

| Package | Upstream | Commit | Licence | Patches |
|---|---|---|---|---|
| `retinexformer/` | caiyuanhao1998/Retinexformer, `basicsr/models/archs/RetinexFormer_arch.py` | 1e9a0efce4b306b6701b824768370ff26066c32a | MIT (© 2023 Yuanhao Cai) | none (byte-identical) |
| `hvi_cidnet/` | Fediory/HVI-CIDNet, `net/*.py` (4 files), `loss/{losses,loss_utils,vgg_arch}.py` | eb43d7d91e9a336c66856824ff9e4603ae41f408 | MIT | package-relative imports; `PyTorchModelHubMixin` base removed; `.cuda()` calls removed (device follows the input); VGG-19 loaded via `weights=VGG19_Weights.IMAGENET1K_V1` (same file as upstream `pretrained=True`); test-only `PHASE3_VGG_RANDOM_INIT=1` |
| `darkir/` | cidautai/DarkIR, `archs/{DarkIR,arch_model,arch_util}.py`, `losses/{loss,loss_utils}.py` | 0910291157ad04f210481a002cf6bb480f6a7f18 | MIT | package-relative imports; unused `pytorch_msssim` import commented out; `.cuda()` removed; test-only `PHASE3_VGG_RANDOM_INIT=1` |
| `iat/` (added Phase 5, inference only) | cuiziteng/Illumination-Adaptive-Transformer, `IAT_enhance/model/{IAT_main,blocks,global_net}.py` | c76472265247f47cea57649af28b15018bb64cb1 | Apache-2.0 (© 2022 The University of Tokyo; `LICENSE` kept) | package-relative imports; unused `import imp` removed (module removed in Python 3.12); the three `timm` helpers replaced by `_timm_compat.py` (`trunc_normal_` = `torch.nn.init.trunc_normal_`, `DropPath`, `to_2tuple`); changed lines marked `[phase5 patch]` (the plain relative-import rewrites `from .blocks import ...` are unmarked); full diff in `iat/PATCHES.diff`; unmodified originals in `third_party_src/IAT/` |

No layer, initialisation or forward computation is changed. Mixed-precision FP32 islands (HVI transform, DarkIR FreMLP)
are applied by wrappers in `src/phase3/models.py`, not by editing these files.

Upstream properties that are kept as published (documented, not "fixed"):
- HVI-CIDNet: `I_LCA5` is computed and its output overwritten (`i_dec2 = self.ID_block2(...)`), so this block (70,960
  parameters) never receives gradients. The paper's "1.88 M" equals 1,975,569 / 2^20.
- DarkIR: residual scales `beta`/`gamma` are initialised to 0 (block bodies receive gradients from step 2 onwards). The
  repository's frequency loss is stored under a misspelled key and never applied. No training script/config is public:
  our loss follows the paper (`src/phase3/losses.py`).
- Retinexformer: the LOL-v2 cyclic schedule's first cycle anneals toward 3e-4 (above the 2e-4 base LR). Reproduced,
  rescaled to our step budget (`src/phase3/schedules.py`).
