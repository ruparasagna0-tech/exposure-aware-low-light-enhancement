"""Phase 3 / Stage 0: colour and image-quality evaluation of exported outputs (reference-based).

NEW script. Reads the PNGs written by scripts/export_outputs.py (MD5-checked against export_manifest.json),
pairs each with its reference through the SAME dataset loaders as the Phase-2 evaluation, and computes
src/metrics/colour_metrics.py metrics for three "methods":
    model     the exported enhanced image
    identity  the low-light input itself                       (floor F0, no learning)
    scaled    input scaled so its mean Rec.601 luma equals the mean luma of the LOL-v2 TRAINING references
              (floor F1, no learning; no test reference is used to set the target)
Pointwise colour metrics use full resolution (row strips). Edge/noise metrics use full resolution for
LOL-v2 / LoLI-Street and a 4x block-mean downscale (output and reference alike) for SICE (20-24 MP).

Validity / labelling (no pair is silently dropped):
  - LoLI-Street id 30727 is already excluded by the loader (black images).
  - flag_ref_darker_than_input: reference mean luma < input mean luma (e.g. LOL-v2-synthetic r01058910t)
  - flag_input_info_limited: >= 50% of input pixels have max(RGB) <= 3 (e.g. SICE seq46)
  - SICE references are dataset-provided fused Labels; LoLI-Street lows are Photoshop-darkened.
  - SICE and LoLI-Street are previously inspected development/evaluation evidence, not untouched test sets.

Writes <export dir>/colour_eval/{per_image/<domain>.csv, summary.json, summary.md}. Refuses to overwrite.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.datasets.phase2 import build_test, load_json, split_file  # noqa: E402
from src.metrics.colour_metrics import downscale_area, pointwise_metrics, structure_metrics  # noqa: E402
from src.utils import load_config  # noqa: E402

DOMAINS = ("lol_v2_real_test", "lol_v2_synthetic_test", "sice", "loli_street_val")
METHODS = ("model", "identity", "scaled")
SICE_STRUCT_DOWNSCALE = 4
EVIDENCE = {"lol_v2_real_test": "seen-domain test", "lol_v2_synthetic_test": "seen-domain test",
            "sice": "unseen domain; previously evaluated/inspected development evidence (fused Label reference)",
            "loli_street_val": "unseen domain; previously evaluated/inspected development evidence (synthetic darkening)"}
_G = {}


def luma(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    return float((0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).mean() / 255.0)


def to_uint8(t) -> np.ndarray:
    """Dataset tensor [3, H, W] (PIL -> to_tensor, i.e. k/255) -> uint8 [H, W, 3]; exact."""
    a = np.round(t.numpy().astype(np.float64) * 255.0)
    return a.clip(0, 255).astype(np.uint8).transpose(1, 2, 0)


def psnr8(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2) / 255.0 ** 2
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def train_reference_luma(cfg, data_root) -> float:
    from src.datasets.phase2 import paths
    p = paths(cfg, data_root)
    vals = []
    for subset, folder in (("real", "Real_captured"), ("synthetic", "Synthetic")):
        for n in load_json(split_file(cfg, subset))["train_names"]:
            ref = n.replace("low", "normal", 1) if subset == "real" else n
            vals.append(luma(np.asarray(Image.open(os.path.join(p["lol_v2"], folder, "Train", "Normal", ref)).convert("RGB"))))
    return float(np.mean(vals))


def _init(cfg, data_root, export_dir, target):
    _G.update(ds=build_test(cfg, data_root), export=Path(export_dir), target=target)


def _work(job):
    domain, i, fn, want_md5 = job
    item = _G["ds"][domain][i]
    path = _G["export"] / domain / fn
    data = path.read_bytes()
    if hashlib.md5(data).hexdigest() != want_md5:
        raise RuntimeError(f"{path}: MD5 differs from export_manifest.json")
    out = np.asarray(Image.open(path).convert("RGB"))
    ref, low = to_uint8(item["high"]), to_uint8(item["low"])
    if out.shape != ref.shape:
        raise RuntimeError(f"{path}: size {out.shape} != reference {ref.shape}")
    k = _G["target"] / max(luma(low), 1e-4)
    scaled = np.clip(np.round(low.astype(np.float64) * k), 0, 255).astype(np.uint8)
    row = {"name": item["name"], "output_file": f"{domain}/{fn}"}
    for key in ("level", "scene_id", "seq_id"):
        if key in item:
            row[key] = item[key] if not hasattr(item[key], "item") else item[key].item()
    row.update(flag_ref_darker_than_input=int(luma(ref) < luma(low)),
               flag_input_info_limited=int((low.max(-1) <= 3).mean() >= 0.5),
               input_luma=round(luma(low), 5), ref_luma=round(luma(ref), 5), scale_factor=round(k, 4))
    f = SICE_STRUCT_DOWNSCALE if domain == "sice" else 1
    for m, img in (("model", out), ("identity", low), ("scaled", scaled)):
        r = {"psnr8": psnr8(img, ref), **pointwise_metrics(img, ref),
             **structure_metrics(downscale_area(img, f), downscale_area(ref, f))}
        row.update({f"{m}_{k2}": (round(v, 6) if isinstance(v, float) else v) for k2, v in r.items()})
    return row


# ----------------------------------------------------------------------------------------- summaries
KEYS = ["psnr8", "de00_mean", "de00_median", "de00_p95", "de00_leq_mean", "dL_mean", "chroma_ratio", "da_mean",
        "db_mean", "hue_err_median_deg", "clip_hi_out", "clip_lo_out", "lstd_out", "edge_corr", "edge_energy_ratio",
        "flat_noise_ratio"]


def stats(vals):
    v = np.array([x for x in vals if x == x], dtype=np.float64)  # drop NaN (e.g. hue on achromatic images)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "median": float(np.median(v)), "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}


def boot_ci(rows, key, cluster=None, n=2000, seed=0):
    """95% bootstrap CI of the mean (resampling clusters, e.g. LoLI scene ids, when given)."""
    rng = random.Random(seed)
    groups = {}
    for r in rows:
        groups.setdefault(r.get(cluster) if cluster else id(r), []).append(float(r[key]))
    g = list(groups.values())
    means = []
    for _ in range(n):
        s = [x for _ in g for x in rng.choice(g)]
        means.append(sum(s) / len(s))
    means.sort()
    return [means[int(0.025 * n)], means[int(0.975 * n) - 1]]


def summarize(rows, cluster=None):
    out = {"n": len(rows), "methods": {}}
    for m in METHODS:
        out["methods"][m] = {k: stats([r[f"{m}_{k}"] for r in rows]) for k in KEYS}
    mo = "model"
    out["model_shares"] = {
        "overexposed_dL_gt_10": float(np.mean([r[f"{mo}_dL_mean"] > 10 for r in rows])),
        "underexposed_dL_lt_-10": float(np.mean([r[f"{mo}_dL_mean"] < -10 for r in rows])),
        "desaturated_chroma_ratio_lt_0.8": float(np.mean([r[f"{mo}_chroma_ratio"] < 0.8 for r in rows])),
        "oversaturated_chroma_ratio_gt_1.25": float(np.mean([r[f"{mo}_chroma_ratio"] > 1.25 for r in rows])),
        "cast_abs_da_or_db_gt_5": float(np.mean([max(abs(r[f"{mo}_da_mean"]), abs(r[f"{mo}_db_mean"])) > 5 for r in rows])),
        "model_dE00_better_than_identity": float(np.mean([r["model_de00_mean"] < r["identity_de00_mean"] for r in rows])),
        "model_dE00_better_than_scaled": float(np.mean([r["model_de00_mean"] < r["scaled_de00_mean"] for r in rows])),
        "flag_ref_darker_than_input": int(sum(r["flag_ref_darker_than_input"] for r in rows)),
        "flag_input_info_limited": int(sum(r["flag_input_info_limited"] for r in rows)),
    }
    for r in rows:
        r["_d_scaled"] = r["model_de00_mean"] - r["scaled_de00_mean"]
    out["ci95"] = {"model_de00_mean": boot_ci(rows, "model_de00_mean", cluster),
                   "model_minus_scaled_de00_mean": boot_ci(rows, "_d_scaled", cluster)}
    out["worst10_model_de00"] = [(r["name"], round(r["model_de00_mean"], 2)) for r in
                                 sorted(rows, key=lambda r: -r["model_de00_mean"])[:10]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", default="experiments/phase3_model_selection/stage0/outputs/phase2_unet_best")
    ap.add_argument("--config", default="configs/generalized_lol_v2.yaml")
    ap.add_argument("--data-root")
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=DOMAINS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--sice-workers", type=int, default=2)
    args = ap.parse_args()

    exp = (PROJECT_ROOT / args.export_dir).resolve()
    manifest = json.load(open(exp / "export_manifest.json"))
    ev = exp / "colour_eval"
    (ev / "per_image").mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    target = train_reference_luma(cfg, args.data_root)
    print(f"fixed-target luma from LOL-v2 training references: {target:.4f}")
    summary_path = ev / "summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    summary.setdefault("meta", {}).update({
        "export_dir": str(exp), "checkpoint_md5": manifest["checkpoint_md5"], "scaled_floor_target_luma": target,
        "sice_structure_metrics_downscale": SICE_STRUCT_DOWNSCALE, "evidence_status": EVIDENCE})
    summary.setdefault("subsets", {})
    ds_all = build_test(cfg, args.data_root)
    for d in args.domains:
        csv_path = ev / "per_image" / f"{d}.csv"
        if csv_path.exists():
            print(f"{d}: colour evaluation exists, skipping (never overwritten)")
            continue
        if d not in manifest["domains"]:
            print(f"{d}: not exported, skipping")
            continue
        files = manifest["domains"][d]["files"]
        by_item = {v["item"]: (fn, v["md5"]) for fn, v in files.items()}
        names = ds_all[d].names
        jobs = [(d, i, *by_item[n]) for i, n in enumerate(names) if n in by_item]
        if len(jobs) != len(files):
            sys.exit(f"{d}: {len(files)} exported files but {len(jobs)} matched dataset items")
        t0 = time.time()
        with Pool(args.sice_workers if d == "sice" else args.workers, initializer=_init,
                  initargs=(cfg, args.data_root, exp, target)) as pool:
            rows = pool.map(_work, jobs, chunksize=1 if d == "sice" else 8)
        with open(str(csv_path) + ".partial", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        os.replace(str(csv_path) + ".partial", csv_path)
        groups = {d: rows}
        if d == "loli_street_val":
            groups.update({f"{d}_{l}": [r for r in rows if r["level"] == l] for l in ("dense", "moderate", "light")})
        for g, rs in groups.items():
            summary["subsets"][g] = summarize(rs, cluster="scene_id" if d == "loli_street_val" else None)
            for key in ("flag_ref_darker_than_input", "flag_input_info_limited"):
                clean = [r for r in rs if not r[key]]
                if len(clean) != len(rs) and clean:
                    summary["subsets"][g][f"without_{key}"] = {
                        "n": len(clean), "model_de00_mean": stats([r["model_de00_mean"] for r in clean]),
                        "model_psnr8": stats([r["model_psnr8"] for r in clean])}
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=1)
        print(f"{d}: {len(rows)} images in {time.time() - t0:.0f}s", flush=True)
    write_md(summary, ev / "summary.md")


def write_md(summary, path):
    order = ["lol_v2_real_test", "lol_v2_synthetic_test", "sice", "loli_street_val", "loli_street_val_dense",
             "loli_street_val_moderate", "loli_street_val_light"]
    L = ["# Stage 0 colour / image-quality evaluation (Phase-2 U-Net, best.pth)", "",
         "Reference-based. SICE and LoLI-Street are previously inspected development/evaluation evidence, not untouched "
         "final test sets. Floors: identity = input unchanged; scaled = input scaled to the mean luma of the LOL-v2 "
         f"training references ({summary['meta']['scaled_floor_target_luma']:.4f}).", "",
         "## Means (model / identity / scaled)", "",
         "| Subset | n | PSNR (8-bit) | ΔE00 mean | ΔE00 p95 | ΔE00 L*-equalised | ΔL* | chroma ratio | hue err ° (median) |",
         "|---|---:|---|---|---|---|---|---|---|"]
    f3 = lambda s, k, fmt: " / ".join(f"{s['methods'][m][k].get('mean', float('nan')):{fmt}}" for m in METHODS)  # noqa: E731
    for g in [g for g in order if g in summary["subsets"]]:
        s = summary["subsets"][g]
        L.append(f"| {g} | {s['n']} | {f3(s, 'psnr8', '.2f')} | {f3(s, 'de00_mean', '.2f')} | {f3(s, 'de00_p95', '.1f')} | "
                 f"{f3(s, 'de00_leq_mean', '.2f')} | {f3(s, 'dL_mean', '+.1f')} | {f3(s, 'chroma_ratio', '.2f')} | "
                 f"{f3(s, 'hue_err_median_deg', '.1f')} |")
    L += ["", "## Model: structure, noise, clipping (mean ± sd)", "",
          "| Subset | edge corr | edge energy ratio | flat-noise ratio | clip ≥250 out | L* std out | ΔE00 95% CI | model − scaled ΔE00 95% CI |",
          "|---|---|---|---|---|---|---|---|"]
    ms = lambda s, k, fmt: f"{s['methods']['model'][k]['mean']:{fmt}} ± {s['methods']['model'][k]['sd']:{fmt}}"  # noqa: E731
    for g in [g for g in order if g in summary["subsets"]]:
        s = summary["subsets"][g]
        ci, cd = s["ci95"]["model_de00_mean"], s["ci95"]["model_minus_scaled_de00_mean"]
        L.append(f"| {g} | {ms(s, 'edge_corr', '.3f')} | {ms(s, 'edge_energy_ratio', '.2f')} | {ms(s, 'flat_noise_ratio', '.2f')} | "
                 f"{ms(s, 'clip_hi_out', '.4f')} | {ms(s, 'lstd_out', '.1f')} | [{ci[0]:.2f}, {ci[1]:.2f}] | [{cd[0]:+.2f}, {cd[1]:+.2f}] |")
    L += ["", "## Model failure-mode shares", "", "| Subset | over-exposed (ΔL*>10) | under-exposed (ΔL*<−10) | desaturated (<0.8) | "
          "over-saturated (>1.25) | cast (|Δa*| or |Δb*| > 5) | beats identity (ΔE00) | beats scaled (ΔE00) | flagged ref<input | flagged info-limited |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for g in [g for g in order if g in summary["subsets"]]:
        sh = summary["subsets"][g]["model_shares"]
        L.append(f"| {g} | {sh['overexposed_dL_gt_10']:.1%} | {sh['underexposed_dL_lt_-10']:.1%} | {sh['desaturated_chroma_ratio_lt_0.8']:.1%} | "
                 f"{sh['oversaturated_chroma_ratio_gt_1.25']:.1%} | {sh['cast_abs_da_or_db_gt_5']:.1%} | "
                 f"{sh['model_dE00_better_than_identity']:.1%} | {sh['model_dE00_better_than_scaled']:.1%} | "
                 f"{sh['flag_ref_darker_than_input']} | {sh['flag_input_info_limited']} |")
    L += ["", "## Worst 10 by model ΔE00", ""]
    for g in [g for g in order if g in summary["subsets"]]:
        L.append(f"- **{g}**: " + ", ".join(f"{n} ({v})" for n, v in summary["subsets"][g]["worst10_model_de00"]))
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
