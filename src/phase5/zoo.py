"""Phase 5 diagnostic: every enhancement condition, loaded read-only with MD5 checks (inference only).

Each condition maps an RGB float tensor [1, 3, H, W] in [0, 1] to an output in [0, 1].
Weights live in a weights directory (``WEIGHTS[name]`` = relative path, expected MD5). Nothing is trained here.

Official-test-time handling reproduced per model:
  - HVI-CIDNet / Retinexformer / U-Net: the Phase-3 wrappers (padding, clamping) used for Tier A.
  - IAT (BMVC 2022, commit c764722): LOL weights take (x - 0.5) / 0.5 input (evaluation_lol_v2.py: pre_norm=True);
    exposure weights take raw [0, 1] input and IAT(type='exp') (evaluation_exposure.py: pre_norm=False).
"""

import hashlib
import os
from collections import OrderedDict

import torch

from src.phase5.safetensors_min import load_file

# relative path inside the weights directory, MD5 (local files verified 2026-11-08)
WEIGHTS = {
    "unet_p2": ("ours/unet_phase2_best.pth", "73f71cf4d01c07b5f3176a6ed067b32f"),
    "unet_expaug": ("ours/unet_expaug_tierA_best.pth", "0688d5ec2e2348bbee52414f1380827f"),
    "rf_tierA": ("ours/retinexformer_tierA_best.pth", "01079637c2175636b17616bd9fe8ad02"),
    "hvi_tierA": ("ours/hvi_cidnet_tierA_best.pth", "ec66d19ab8c9719fd58381520cfc97c8"),
    "hvi_off_real": ("official/hvi_cidnet_lolv2_real_bestPSNR/model.safetensors", "f5ee3e155015f07434c7850c3d2e20f9"),
    "hvi_off_syn": ("official/hvi_cidnet_lolv2_syn_wperc/model.safetensors", "b43c7af3a29099977eadb9b417f02988"),
    "hvi_off_gen": ("official/hvi_cidnet_generalization/model.safetensors", "8653111a12efa0e2ea86e731282ceabf"),
    "rf_off_real": ("official/retinexformer_lolv2_real/LOL_v2_real.pth", "6738bd02171824d9843b6dfa5f0d0f0f"),
    "iat_lol": ("official/iat_lol/best_Epoch_lol.pth", "23d508f28a9756a0a00581aef802fcdc"),
    "iat_exp": ("official/iat_exposure/best_Epoch_exposure.pth", "df92a0ab8dcea657cdc83b419d74c846"),
}

# display order; identity and gain048 need no weights
CONDITIONS = ["identity", "gain048", "unet_p2", "unet_expaug", "rf_tierA", "rf_off_real", "hvi_tierA",
              "hvi_off_real", "hvi_off_syn", "hvi_off_gen", "iat_lol", "iat_exp"]

DESCRIPTION = OrderedDict([
    ("identity", "input unchanged"),
    ("gain048", "global gain so that mean Rec.601 luma = 0.48 (constant from the Phase-2 diagnostic, LOL-v2 train references), clipped"),
    ("unet_p2", "Phase-2 U-Net (ours, LOL-v2 real+syn, epoch 45)"),
    ("unet_expaug", "U-Net + exposure augmentation (ours, Tier A)"),
    ("rf_tierA", "Retinexformer (ours, Tier A, 5,600 steps)"),
    ("rf_off_real", "Retinexformer official LOL_v2_real.pth (HF CaiYuanhao/Retinexformer @51d9f18)"),
    ("hvi_tierA", "HVI-CIDNet (ours, Tier A = D1)"),
    ("hvi_off_real", "HVI-CIDNet official LOLv2-real bestPSNR (HF @c317f57)"),
    ("hvi_off_syn", "HVI-CIDNet official LOLv2-syn w_perc (HF @001d921)"),
    ("hvi_off_gen", "HVI-CIDNet official 'Generalization' (HF @51481ef)"),
    ("iat_lol", "IAT official LOL weights (GitHub @c764722)"),
    ("iat_exp", "IAT official exposure-correction weights (MSEC; GitHub @c764722)"),
])

GAIN_TARGET = 0.48


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def luma(x):
    """Rec.601 luma of [B, 3, H, W] in [0, 1] -> [B, H, W]."""
    return 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]


def check_weights(wdir):
    bad = {}
    for k, (rel, h) in WEIGHTS.items():
        p = os.path.join(wdir, rel)
        got = md5(p) if os.path.exists(p) else "missing"
        if got != h:
            bad[k] = got
    return bad


def build(name, wdir, device):
    """Return a callable f(x) for condition ``name`` (x: [1, 3, H, W] in [0, 1] on ``device``)."""
    if name == "identity":
        return lambda x: x
    if name == "gain048":
        return lambda x: (x * (GAIN_TARGET / luma(x).mean().clamp_min(1e-4))).clamp(0, 1)
    rel, h = WEIGHTS[name]
    path = os.path.join(wdir, rel)
    if md5(path) != h:
        raise RuntimeError(f"{name}: weights MD5 mismatch ({path}): STOP")
    if name.startswith("iat"):
        from third_party.iat.IAT_main import IAT
        net = IAT(type="exp") if name == "iat_exp" else IAT()
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        net = net.to(device).eval()
        norm = name == "iat_lol"

        def f(x):
            return net((x - 0.5) / 0.5 if norm else x)[2].clamp(0, 1)
        return f
    from src.phase3.models import build_phase3_model
    if name in ("unet_p2", "unet_expaug"):
        m = build_phase3_model("unet")
        sd = torch.load(path, map_location="cpu", weights_only=False)["model_state"]
        (m.net if name == "unet_p2" else m).load_state_dict(sd, strict=True)
    elif name.startswith("rf_"):
        m = build_phase3_model("retinexformer")
        if name == "rf_tierA":
            m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state"], strict=True)
        else:
            m.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["params"], strict=True)
    else:
        m = build_phase3_model("hvi_cidnet")
        if name == "hvi_tierA":
            m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state"], strict=True)
        else:
            m.net.load_state_dict(load_file(path), strict=True)
    m = m.to(device).eval()
    return lambda x: m(x).clamp(0, 1)
