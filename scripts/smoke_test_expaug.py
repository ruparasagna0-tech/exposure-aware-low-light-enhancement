"""Smoke test for the Phase-2 exposure-augmentation experiment (NO training run).

Checks
  [1] config: experiment config differs from configs/generalized_lol_v2.yaml ONLY in experiment_name,
      output.dir and data.input_exposure_aug (approved values); splits read from the Phase-2 dir
  [2] integrity checks (data, splits, leakage) with the experiment config
  [3] placement: aug is the last train step after crop -> hflip -> ToTensor; val/test unchanged
  [4] Phase-2 config unchanged: transforms and sampled tensors bit-identical to the original code path
  [5] factor sampler: P(apply), log-uniform raw factors in [0.5, 3.0], determinism, no global-RNG
      consumption, finite outputs; soft_shoulder: formula, continuity, monotonicity, < 1, and identical
      raw draws to hard_cap
  [6] DataLoader (per --workers): references bit-identical to Phase 2, inputs = Phase-2 crop or
      f(s * same crop) for the configured policy, finite and in [0, 1], reruns bit-identical, new stream per epoch
  [7] measurement on real/synthetic training crops: factor distribution, cap activation (hard_cap),
      clipping / destroyed reference detail, shoulder compression (soft_shoulder), needed gain
      (mean reference / mean augmented input) vs. no augmentation; per-crop reference identity + finiteness
  [8] pipeline test: 2 optimizer steps of scripts/train_generalized.py for BOTH configs (temp dirs)
  [9] file snapshot: nothing changed except src/datasets/phase2.py (intended switch)
Works for both policies (configs/generalized_lol_v2_expaug.yaml = hard_cap,
configs/generalized_lol_v2_expaug_soft.yaml = soft_shoulder).
Writes only <experiment output.dir>/logs/smoke_test_expaug_report_<policy>_<device>.json

Usage (project root):
    .venv/bin/python scripts/smoke_test_expaug.py --workers 0 2
    .venv/bin/python scripts/smoke_test_expaug.py --config configs/generalized_lol_v2_expaug_soft.yaml --workers 0 2
    python scripts/smoke_test_expaug.py --data-root /content/phase2 --device cuda --skip-snapshot-check   # Colab
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader  # noqa: E402

from src.datasets.balanced_sampler import BalancedDomainBatchSampler  # noqa: E402
from src.datasets.exposure_augment import InputExposureAug, build_input_exposure_aug, soft_shoulder  # noqa: E402
from src.datasets.lol_v2 import LOLv2Dataset  # noqa: E402
from src.datasets.paired_transforms import build_paired_transforms  # noqa: E402
from src.datasets.phase2 import SUBSETS, build_test, build_train_val, integrity_checks, load_json, paths, split_file  # noqa: E402
from src.utils import load_config, make_generator, resolve_path, seed_worker  # noqa: E402

APPROVED_BY_POLICY = {
    "hard_cap": {"enabled": True, "p": 0.5, "s_min": 0.5, "s_max": 3.0, "cap_quantile": 0.999, "q_floor": 1e-6},
    "soft_shoulder": {"enabled": True, "policy": "soft_shoulder", "p": 0.5, "s_min": 0.5, "s_max": 3.0, "shoulder_knee": 0.8},
}
REPORT, MEASURE = [], {}


def check(name, ok, detail=""):
    REPORT.append({"check": name, "passed": bool(ok), "detail": str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail != "" else ""), flush=True)
    return ok


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def loader_batches(tr, nw, epoch, nb, seed):
    smp = BalancedDomainBatchSampler([len(tr[0]), len(tr[1])], 2, 112, seed)
    smp.set_epoch(epoch - 1)
    torch.manual_seed(seed * 1000 + epoch)  # exactly as scripts/train_generalized.py
    dl = DataLoader(ConcatDataset(tr), batch_sampler=smp, num_workers=nw, worker_init_fn=seed_worker,
                    generator=make_generator(seed * 1000 + epoch))
    return [(b["low"].clone(), b["high"].clone()) for _, b in zip(range(nb), dl)]


def same_crop(x, y, s_min, s_max, aug):
    """y == x, or y == aug.apply_factor(x, s) (policy mapping) for a single s in [s_min, s_max]."""
    if torch.equal(x, y):
        return True
    lin_top = 0.79 if aug.policy == "soft_shoulder" else 0.999  # estimate s from the linear part of the mapping
    m = (x > 0.05) & (y < lin_top)
    if m.sum() < 100:
        return False
    s_hat = (y[m] / x[m]).median().item()
    return s_min - 1e-3 <= s_hat <= s_max + 1e-3 and (y - aug.apply_factor(x, s_hat)).abs().max().item() < 2e-3


def finite01(t):
    return bool(torch.isfinite(t).all()) and float(t.min()) >= 0.0 and float(t.max()) <= 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/generalized_lol_v2_expaug.yaml")
    ap.add_argument("--phase2-config", default="configs/generalized_lol_v2.yaml")
    ap.add_argument("--data-root")
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--device", default="cpu", help="device for the 2-step pipeline test (cpu/cuda; not mps)")
    ap.add_argument("--crops-per-image", type=int, default=2)
    ap.add_argument("--skip-md5", action="store_true", help="skip data MD5 verification inside integrity checks")
    ap.add_argument("--skip-pipeline-test", action="store_true")
    ap.add_argument("--skip-snapshot-check", action="store_true", help="e.g. on Colab (data_prepared not on Drive)")
    args = ap.parse_args()
    t0 = time.time()
    cfg, cfg2 = load_config(args.config), load_config(args.phase2_config)
    seed = cfg["seed"]
    policy = cfg["data"].get("input_exposure_aug", {}).get("policy", "hard_cap")
    APPROVED = APPROVED_BY_POLICY[policy]
    print(f"config {args.config} | policy {policy}")

    print(f"[1] configuration  (t={time.time() - t0:.0f}s)")
    f1, f2 = flatten(cfg), flatten(cfg2)
    diff = sorted(k for k in set(f1) | set(f2) if f1.get(k) != f2.get(k))
    allowed = {"experiment_name", "output.dir"} | {f"data.input_exposure_aug.{k}" for k in APPROVED}
    check("only experiment_name, output.dir, data.input_exposure_aug differ from Phase-2 config",
          set(diff) <= allowed, diff)
    check(f"augmentation parameters = approved for policy {policy}",
          cfg["data"]["input_exposure_aug"] == APPROVED, cfg["data"]["input_exposure_aug"])
    out_dir, out2 = resolve_path(cfg["output"]["dir"]), resolve_path(cfg2["output"]["dir"])
    check("separate output dir", out_dir != out2 and out_dir.name == cfg["experiment_name"], out_dir)
    check("splits read from the Phase-2 split dir (unchanged)", cfg["splits"] == cfg2["splits"], cfg["splits"]["dir"])
    check("Phase-2 config has no augmentation (switch off by default)",
          build_input_exposure_aug(cfg2["data"].get("input_exposure_aug")) is None)

    print(f"[2] integrity checks (experiment config)  (t={time.time() - t0:.0f}s)")
    res = integrity_checks(cfg, args.data_root, verify_md5=not args.skip_md5)
    bad = [c for c, ok, _ in res if not ok]
    check(f"all {len(res)} integrity checks pass", not bad, bad)

    print(f"[3] placement and val/test invariance  (t={time.time() - t0:.0f}s)")
    tr, va = build_train_val(cfg, args.data_root)
    tr2, va2 = build_train_val(cfg2, args.data_root)
    names = [type(t).__name__ for t in tr[0].transform.transforms]
    check("train (experiment): crop -> hflip -> ToTensor -> InputExposureAug (last)",
          names == ["PairedRandomCrop", "PairedRandomHorizontalFlip", "PairedToTensor", "InputExposureAug"], names)
    check("both training domains use the augmentation", all(type(d.transform.transforms[-1]).__name__ == "InputExposureAug" for d in tr))
    check("validation transforms identical in both configs, no random op",
          all(repr(a.transform) == repr(b.transform) == "PairedCompose([PairedToTensor()])" for a, b in zip(va, va2)))
    te, te2 = build_test(cfg, args.data_root), build_test(cfg2, args.data_root)
    check("test transforms identical in both configs, no random op",
          all(repr(te[k].transform) == repr(te2[k].transform) == "PairedCompose([PairedToTensor()])" for k in te))
    check("validation tensors identical in both configs (first item per domain)",
          all(torch.equal(a[0]["low"], b[0]["low"]) and torch.equal(a[0]["high"], b[0]["high"]) for a, b in zip(va, va2)))

    print(f"[4] Phase-2 configuration unchanged  (t={time.time() - t0:.0f}s)")
    names2 = [type(t).__name__ for t in tr2[0].transform.transforms]
    check("Phase-2 train transform = crop -> hflip -> ToTensor", names2 == ["PairedRandomCrop", "PairedRandomHorizontalFlip", "PairedToTensor"])
    d = cfg2["data"]
    p = paths(cfg2, args.data_root)
    orig_tf = build_paired_transforms("train", crop_size=d["crop_size"], hflip=d["hflip"], vflip=d["vflip"])
    same = True
    for k, s in enumerate(SUBSETS):
        ref_ds = LOLv2Dataset(p["lol_v2"], s, "Train", names=load_json(split_file(cfg2, s))["train_names"], transform=orig_tf)
        for i in range(0, len(ref_ds), max(1, len(ref_ds) // 10)):
            torch.manual_seed(1000 + i)
            a = tr2[k][i]
            torch.manual_seed(1000 + i)
            b = ref_ds[i]
            same &= torch.equal(a["low"], b["low"]) and torch.equal(a["high"], b["high"])
    check("Phase-2 config samples bit-identical to the original (pre-switch) code path", same)

    print(f"[5] factor sampler  (t={time.time() - t0:.0f}s)")
    torch.manual_seed(seed)
    aug = build_input_exposure_aug(APPROVED)
    dummy = torch.full((3, 8, 8), 0.01)  # dark input -> cap = 3.0, so the raw distribution is observed
    draws = []
    for _ in range(100_000):
        torch.rand(3)  # stands in for the crop/flip draws that precede the augmentation in the pipeline
        aug(dummy, dummy)
        draws.append((aug.last["applied"], aug.last["s_raw"], aug.last["s_final"]))
    a = np.array(draws)
    app = a[:, 0] == 1
    u = (np.log(a[app, 2]) - math.log(0.5)) / math.log(6.0)
    qs = np.quantile(u, [0.1, 0.25, 0.5, 0.75, 0.9])
    check("P(apply) = 0.5", abs(app.mean() - 0.5) < 0.006, f"{app.mean():.4f}")
    check("factors within [0.5, 3.0], log-uniform", a[app, 2].min() >= 0.5 and a[app, 2].max() <= 3.0
          and np.abs(qs - [.1, .25, .5, .75, .9]).max() < 0.006, f"normalised log quantiles {np.round(qs, 4).tolist()}")
    def stream(n, reseed_between=False):
        torch.manual_seed(seed)
        A_ = build_input_exposure_aug(APPROVED)
        out_ = []
        for _ in range(n):
            torch.rand(3)  # stands in for crop/flip draws that advance the global RNG in the pipeline
            A_(dummy, dummy)
            out_.append(A_.last["s_final"])
        return out_
    r1, r2 = stream(1000), stream(1000)
    check("same seed -> identical factor stream (also after re-seeding with the same value in one process)", r1 == r2)
    check("factor stream does not consume the global RNG", (lambda: (torch.manual_seed(5), torch.rand(1).item())[1])() ==
          (lambda: (torch.manual_seed(5), InputExposureAug()(dummy, dummy), torch.rand(1).item())[2])())
    hi = torch.full((3, 8, 8), 0.9)
    check("reference returned unchanged (same object, every call)", all(aug(hi, hi)[1] is hi for _ in range(50)))
    xr = torch.rand(3, 64, 64)
    outs = [aug.apply_factor(xr, f) for f in (0.5, 0.9, 1.0, 1.3, 2.0, 3.0)] + [aug.apply_factor(torch.ones(3, 4, 4), 3.0)]
    check("outputs finite and within [0, 1] for factors 0.5..3.0 (incl. all-white input)", all(finite01(o) for o in outs))
    if policy == "soft_shoulder":
        k_ = APPROVED["shoulder_knee"]
        z = torch.linspace(0, 3, 300_001, dtype=torch.float64)
        f = soft_shoulder(z, k_)
        expected = torch.where(z <= k_, z, k_ + (1 - k_) * (1 - torch.exp(-(z - k_) / (1 - k_))))
        check("shoulder = approved formula (knee 0.8)", torch.allclose(f, expected, atol=0, rtol=0))
        e = 1e-9
        fk = soft_shoulder(torch.tensor([k_ - e, k_, k_ + e], dtype=torch.float64), k_)
        check("shoulder continuous with slope 1 at the knee", abs(fk[1] - k_) < 1e-15 and abs((fk[2] - fk[0]).item() / (2 * e) - 1) < 1e-6)
        check("shoulder strictly increasing on [0, 3] and < 1", bool((f[1:] > f[:-1]).all()) and float(f.max()) < 1.0,
              f"max {float(f.max()):.8f}")
        f32 = soft_shoulder(torch.linspace(0, 3, 30_001), k_)
        check("float32 shoulder non-decreasing and < 1", bool((f32[1:] >= f32[:-1]).all()) and float(f32.max()) < 1.0)
        check("darkening (s <= 1) is exactly linear", torch.equal(aug.apply_factor(xr, 0.7), xr * 0.7))
        hc = build_input_exposure_aug(APPROVED_BY_POLICY["hard_cap"])
        same_draws = True
        for t in range(2000):
            torch.manual_seed(t)
            aug(dummy, dummy)
            torch.manual_seed(t)
            hc(dummy, dummy)
            same_draws &= aug.last["applied"] == hc.last["applied"] and aug.last["s_raw"] == hc.last["s_raw"]
        check("identical raw draws (apply decision, factor) as hard_cap for the same RNG state", same_draws)

    print(f"[6] DataLoader invariance / reproducibility  (t={time.time() - t0:.0f}s)")
    for nw in args.workers:
        base = loader_batches(tr2, nw, 1, 6, seed)
        a1 = loader_batches(tr, nw, 1, 6, seed)
        a2 = loader_batches(tr, nw, 1, 6, seed)
        e2 = loader_batches(tr, nw, 2, 6, seed)
        n = sum(len(b[0]) for b in base)
        changed = sum(int(not torch.equal(b[0][i], x[0][i])) for b, x in zip(base, a1) for i in range(len(b[0])))
        check(f"workers={nw}: references bit-identical to Phase-2 pipeline", all(torch.equal(b[1], x[1]) for b, x in zip(base, a1)))
        aug_t = tr[0].transform.transforms[-1]
        check(f"workers={nw}: each input = Phase-2 crop, or {policy} mapping of the same crop, s in [0.5, 3]",
              all(same_crop(b[0][i], x[0][i], 0.5, 3.0, aug_t) for b, x in zip(base, a1) for i in range(len(b[0]))),
              f"{changed}/{n} inputs augmented")
        check(f"workers={nw}: all inputs and references finite and in [0, 1]", all(finite01(x[0]) and finite01(x[1]) for x in a1))
        check(f"workers={nw}: rerun bit-identical", all(torch.equal(x[0], y[0]) for x, y in zip(a1, a2)))
        check(f"workers={nw}: epoch 2 uses a different stream", not all(torch.equal(x[0], y[0]) for x, y in zip(a1, e2)))

    print(f"[7] measurement on training crops ({args.crops_per_image} crops per training pair)  (t={time.time() - t0:.0f}s)")
    for k, sub in enumerate(SUBSETS):
        ds = tr2[k]  # Phase-2 (un-augmented) crops; the augmentation is applied explicitly below
        torch.manual_seed(seed * 7 + k)
        A = build_input_exposure_aug(APPROVED)
        rows, ok_ref, ok_fin = [], True, True
        knee = APPROVED.get("shoulder_knee", 0.8)
        for i in range(len(ds)):
            for c in range(args.crops_per_image):
                torch.manual_seed(10_000 * (k + 1) + 10 * i + c)
                it = ds[i]
                x, y = it["low"], it["high"]
                y_before = y.clone()
                out, y_out = A(x, y)
                ok_ref &= (y_out is y) and torch.equal(y, y_before)
                ok_fin &= finite01(out)
                L = A.last
                pre = x * L["s_final"]
                pre_raw = x * L["s_raw"] if L["applied"] else x
                in_shoulder = (pre.amax(0) > knee) if (policy == "soft_shoulder" and L["s_final"] > 1) else torch.zeros_like(pre[0], dtype=torch.bool)
                slope = torch.exp(-(pre.amax(0)[in_shoulder] - knee) / (1 - knee)).mean().item() if in_shoulder.any() else float("nan")
                ym = y.mean().item()
                if policy == "soft_shoulder":
                    beyond = 0.0                                                           # mapping never exceeds 1
                    destroyed = ((out >= 1 - 1e-7) & (x < 1) & (y < 0.98)).float().mean().item()
                else:
                    beyond = (pre.amax(0) > 1 + 1e-6).float().mean().item()               # pixels pushed beyond 1.0
                    destroyed = ((pre > 1 + 1e-6) & (x < 1) & (y < 0.98)).float().mean().item()
                rows.append([L["applied"], L["s_raw"], L["s_final"], L["capped"],
                             (out.amax(0) >= 1).float().mean().item(),                     # pixels at 1.0 after aug (any channel)
                             beyond, destroyed,
                             (pre_raw.amax(0) > 1 + 1e-6).float().mean().item(),          # uncapped/linear counterfactual: beyond 1.0
                             ((pre_raw > 1 + 1e-6) & (x < 1) & (y < 0.98)).float().mean().item(),
                             in_shoulder.float().mean().item(), slope,
                             ym / max(out.mean().item(), 1e-6), ym / max(x.mean().item(), 1e-6)])
        R = np.array(rows, dtype=np.float64)
        app, br = R[:, 0] == 1, (R[:, 0] == 1) & (R[:, 1] > 1)
        sf = R[app, 2]
        g_aug, g_none = R[:, 11], R[:, 12]
        m = {
            "policy": policy, "crops": len(R), "brightening_draws": int(br.sum()), "applied_rate": app.mean(),
            "cap_activation_rate_of_brightening_draws": R[br, 3].mean() if br.any() else 0.0,
            "final_factor_quantiles_applied_p5_25_50_75_95": np.quantile(sf, [.05, .25, .5, .75, .95]).round(3).tolist(),
            "share_unchanged_darkened_brightened": [float((R[:, 2] == 1).mean()), float((R[:, 2] < 1).mean()), float((R[:, 2] > 1).mean())],
            "mean_final_factor_when_brightening_draw": float(R[br, 2].mean()),
            "mean_raw_factor_when_brightening_draw": float(R[br, 1].mean()),
            "pixels_at_1_after_aug_mean_pct": 100 * R[:, 4].mean(),
            "pixels_beyond_1_mean_pct": 100 * R[:, 5].mean(),
            "pixels_beyond_1_max_pct": 100 * R[:, 5].max(),
            "ref_detail_destroyed_mean_pct": 100 * R[:, 6].mean(),
            "crops_beyond_1_gt1_5_10_pct": [100 * (R[:, 5] > .01).mean(), 100 * (R[:, 5] > .05).mean(), 100 * (R[:, 5] > .10).mean()],
            "linear_uncapped_counterfactual_beyond_1_mean_pct": 100 * R[:, 7].mean(),
            "linear_uncapped_counterfactual_detail_destroyed_mean_pct": 100 * R[:, 8].mean(),
            "shoulder_pixels_mean_pct": 100 * R[:, 9].mean(),
            "shoulder_mean_slope_where_compressed": float(np.nanmean(R[:, 10])) if np.isfinite(R[:, 10]).any() else None,
            "needed_gain_median_p10_aug_vs_none": [float(np.median(g_aug)), float(np.percentile(g_aug, 10)),
                                                   float(np.median(g_none)), float(np.percentile(g_none, 10))],
            "needed_gain_share_le_1.5_2.0_2.5_aug": [100 * (g_aug <= 1.5).mean(), 100 * (g_aug <= 2.0).mean(), 100 * (g_aug <= 2.5).mean()],
            "needed_gain_share_le_1.5_2.0_2.5_none": [100 * (g_none <= 1.5).mean(), 100 * (g_none <= 2.0).mean(), 100 * (g_none <= 2.5).mean()],
        }
        MEASURE[sub] = {kk: (round(float(v), 4) if isinstance(v, (float, np.floating)) else
                             ([round(float(e), 4) for e in v] if isinstance(v, list) else v)) for kk, v in m.items()}
        print(f"  {sub}: " + json.dumps(MEASURE[sub]))
        check(f"{sub}: every reference returned unchanged (same object, identical values)", ok_ref)
        check(f"{sub}: every augmented input finite and within [0, 1]", ok_fin)
        check(f"{sub}: no NaN in needed-gain measurement", bool(np.isfinite(g_aug).all() and np.isfinite(g_none).all()))
        if policy == "hard_cap":
            n_px = 256 * 256  # training crop; pixels strictly above the linearly interpolated q99.9:
            allowed = (n_px - 1 - math.floor(APPROVED["cap_quantile"] * (n_px - 1))) / n_px  # = 66 / 65536 = 0.1007%
            check(f"{sub}: capped augmentation pushes at most the pixels above q99.9 beyond 1.0 (<= {100 * allowed:.4f}%) in every crop",
                  R[:, 5].max() <= allowed + 1e-12, f"max {100 * R[:, 5].max():.4f}%")
        else:
            check(f"{sub}: soft shoulder destroys no reference detail (no channel value reaches 1.0 where input < 1)",
                  R[:, 6].max() == 0.0, f"max {100 * R[:, 6].max():.4f}%")
            check(f"{sub}: no cap applied (final factor = raw factor for every augmented crop)", bool(np.all(R[app, 2] == R[app, 1])))

    if not args.skip_pipeline_test:
        print(f"[8] 2-step training pipeline test for both configs (temp dirs)  (t={time.time() - t0:.0f}s)")
        for c in (args.config, args.phase2_config):
            with tempfile.TemporaryDirectory() as td:
                cmd = [sys.executable, "scripts/train_generalized.py", "--config", c, "--output-dir", td,
                       "--max-steps-per-epoch", "2", "--epochs", "1", "--num-workers", "0", "--device", args.device]
                if args.data_root:
                    cmd += ["--data-root", args.data_root]
                r = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
                saved = load_config(os.path.join(td, "config.yaml")) if os.path.exists(os.path.join(td, "config.yaml")) else {}
                ok = r.returncode == 0 and os.path.exists(os.path.join(td, "checkpoints", "best.pth"))
                aug_saved = saved.get("data", {}).get("input_exposure_aug")
                last = [ln for ln in r.stdout.splitlines() if ln.startswith(("Epoch", "Done"))]
                check(f"train_generalized.py runs with {os.path.basename(c)} (2 steps, temp dir)", ok,
                      (last[-2] if len(last) > 1 else r.stderr[-300:]) + f" | saved aug block: {aug_saved}")

    if not args.skip_snapshot_check:
        print(f"[9] project file snapshot  (t={time.time() - t0:.0f}s)")
        snap = load_json(resolve_path("experiments/generalized_lol_v2_expaug/logs/pre_expaug_file_md5.json"))["files"]
        changed, missing = [], []
        for f, md5 in snap.items():
            fp = os.path.join(PROJECT_ROOT, f)
            if not os.path.exists(fp):
                missing.append(f)
                continue
            h = hashlib.md5()
            with open(fp, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != md5:
                changed.append(f)
        check("no pre-existing file deleted", not missing, missing[:5])
        check("only src/datasets/phase2.py changed (intended switch)", changed == ["src/datasets/phase2.py"], changed)
        p2 = [f for f in snap if f.startswith("experiments/generalized_lol_v2/")]
        added = sorted(os.path.relpath(os.path.join(r, f), PROJECT_ROOT) for r, _, fs in os.walk(os.path.join(PROJECT_ROOT, "experiments/generalized_lol_v2"))
                       for f in fs if f != ".DS_Store" and os.path.relpath(os.path.join(r, f), PROJECT_ROOT) not in snap)
        check(f"experiments/generalized_lol_v2/ untouched ({len(p2)} files, none added)", not added and not [f for f in changed if f in p2], added[:5])

    failed = [r for r in REPORT if not r["passed"]]
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    rpt = logs / f"smoke_test_expaug_report_{policy}_{args.device}.json"
    with open(rpt, "w") as f:
        json.dump({"config": args.config, "policy": policy, "runtime_s": round(time.time() - t0, 1), "workers": args.workers, "passed": not failed,
                   "checks": REPORT, "measurement": MEASURE}, f, indent=1)
    print("=" * 70)
    print(f"EXPAUG SMOKE TEST: {'ALL ' + str(len(REPORT)) + ' CHECKS PASSED' if not failed else str(len(failed)) + ' FAILED'} "
          f"({time.time() - t0:.0f}s); report: {rpt}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
