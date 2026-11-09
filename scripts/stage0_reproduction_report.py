"""Phase 3 / Stage 0: compare re-computed Phase-2 metrics (export_outputs.py) with the recorded results.

Recorded results are read directly from ~/Downloads/phase2_evaluation_results.zip (MD5-checked, never
extracted into the project). Tolerances are fixed here BEFORE comparison:
  per image : |dPSNR| <= 0.05 dB, |dSSIM| <= 5e-4, |dLPIPS| <= 2e-3
  per domain: |d mean PSNR| <= 0.005 dB, |d mean SSIM| <= 5e-5, |d mean LPIPS| <= 5e-4
Rationale: the recorded run used CUDA; this one may use CPU. Float differences of order 1e-6 can flip
a few pixels by one 8-bit level after quantization, which moves PSNR by ~1e-3 dB on affected images.
Any value outside tolerance is listed as a discrepancy (not hidden, not "fixed").

Writes <export dir>/reproduction_report.{json,md}.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ZIP_MD5 = "e7302a3b660629702bc19bf6af2f5e0c"
TOL_IMG = {"psnr": 0.05, "ssim": 5e-4, "lpips": 2e-3}
TOL_MEAN = {"psnr": 0.005, "ssim": 5e-5, "lpips": 5e-4}
DOMAINS = ("lol_v2_real_test", "lol_v2_synthetic_test", "sice", "loli_street_val")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", default="experiments/phase3_model_selection/stage0/outputs/phase2_unet_best")
    ap.add_argument("--results-zip", default="~/Downloads/phase2_evaluation_results.zip")
    ap.add_argument("--allow-partial", action="store_true", help="PIPELINE TEST ONLY (--limit exports)")
    args = ap.parse_args()
    exp = (PROJECT_ROOT / args.export_dir).resolve()
    zp = Path(args.results_zip).expanduser()
    if hashlib.md5(zp.read_bytes()).hexdigest() != RESULTS_ZIP_MD5:
        sys.exit(f"{zp}: MD5 differs from the recorded {RESULTS_ZIP_MD5}")
    z = zipfile.ZipFile(zp)
    summary = json.loads(z.read("metrics_summary.json"))["domains"]

    report = {"tolerances_per_image": TOL_IMG, "tolerances_domain_mean": TOL_MEAN, "domains": {}, "discrepancies": []}
    for d in DOMAINS:
        p = exp / "per_image" / f"{d}.csv"
        if not p.is_file():
            report["domains"][d] = {"status": "not exported"}
            continue
        new = {r["name"]: r for r in csv.DictReader(open(p))}
        old = {r["name"]: r for r in csv.DictReader(io.TextIOWrapper(z.open(f"per_image/{d}.csv")))}
        if set(new) != set(old) and not args.allow_partial:
            report["discrepancies"].append({"domain": d, "issue": "item sets differ",
                                            "only_new": sorted(set(new) - set(old))[:10],
                                            "only_recorded": sorted(set(old) - set(new))[:10]})
        common = sorted(set(new) & set(old))
        groups = {d: common}
        if d == "loli_street_val":
            for lvl in ("dense", "moderate", "light"):
                groups[f"{d}_{lvl}"] = [n for n in common if old[n]["level"] == lvl]
        maxdiff = {}
        for k in TOL_IMG:
            diffs = [(abs(float(new[n][k]) - float(old[n][k])), n) for n in common]
            maxdiff[k] = max(diffs) if diffs else (0.0, None)
            bad = [(n, float(new[n][k]), float(old[n][k])) for dv, n in diffs if dv > TOL_IMG[k] or math.isnan(dv)]
            for n, a, b in bad:
                report["discrepancies"].append({"domain": d, "item": n, "metric": k, "reproduced": a, "recorded": b})
        for g, names in groups.items():
            if not names:
                continue
            entry = {"n_compared": len(names), "n_recorded": summary[g]["n"]}
            for k in TOL_IMG:
                m_new = sum(float(new[n][k]) for n in names) / len(names)
                m_old_csv = sum(float(old[n][k]) for n in names) / len(names)
                entry[k] = {"reproduced": m_new, "recorded_summary": summary[g][k], "recorded_csv_mean": m_old_csv,
                            "abs_diff": abs(m_new - m_old_csv), "within_tol": abs(m_new - m_old_csv) <= TOL_MEAN[k]}
                if not entry[k]["within_tol"]:
                    report["discrepancies"].append({"domain": g, "metric": f"mean {k}", "reproduced": m_new,
                                                    "recorded": m_old_csv})
            report["domains"][g] = entry
        report["domains"][d]["max_abs_per_image_diff"] = {k: {"value": v[0], "item": v[1]} for k, v in maxdiff.items()}

    report["reproduced"] = not report["discrepancies"] and all("status" not in v for v in report["domains"].values())
    with open(exp / "reproduction_report.json", "w") as f:
        json.dump(report, f, indent=1)
    md = ["| Subset | n | PSNR reproduced / recorded | SSIM reproduced / recorded | LPIPS reproduced / recorded |",
          "|---|---:|---|---|---|"]
    for g, e in report["domains"].items():
        if "status" in e:
            md.append(f"| {g} | — | {e['status']} | | |")
            continue
        cell = lambda k, f: f"{e[k]['reproduced']:{f}} / {e[k]['recorded_summary']:{f}}" + ("" if e[k]["within_tol"] else " **!**")  # noqa: E731
        md.append(f"| {g} | {e['n_compared']} | {cell('psnr', '.4f')} | {cell('ssim', '.5f')} | {cell('lpips', '.5f')} |")
    md.append("")
    for d in DOMAINS:
        e = report["domains"].get(d, {})
        if "max_abs_per_image_diff" in e:
            md.append(f"- {d}: max per-image |diff| " + ", ".join(
                f"{k} {v['value']:.2e} ({v['item']})" for k, v in e["max_abs_per_image_diff"].items()))
    md.append(f"\nTolerances per image {TOL_IMG}, per mean {TOL_MEAN}. Discrepancies: {len(report['discrepancies'])}."
              f" Reproduced: **{report['reproduced']}**.")
    with open(exp / "reproduction_report.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    if report["discrepancies"]:
        print(json.dumps(report["discrepancies"][:20], indent=1))
    sys.exit(0 if report["reproduced"] or args.allow_partial else 1)


if __name__ == "__main__":
    main()
