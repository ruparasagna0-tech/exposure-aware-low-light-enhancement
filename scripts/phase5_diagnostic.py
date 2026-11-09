"""Phase 5 inference-only diagnostic: does enhancement over-brighten mildly dark inputs, across models and exposures?

    python scripts/phase5_diagnostic.py run    --data <pack dir> --weights <weights dir> --out <out dir> [--device cuda]
    python scripts/phase5_diagnostic.py report --data <pack dir> --out <out dir>

Data (pack from scripts/prepare_phase5_diag_data.py):
  sice      SICE Part 1, every sequence x every exposure, long side 900, reference = dataset Label (Part 2 never used)
  loli_val  LoLI-Street Val, 999 ids x {dense, moderate, light}, shared reference per id
  loli_test LoLI-Street real low-light Test (1,000, no reference): no-reference metrics only
Conditions: src/phase5/zoo.py (identity, a fixed-gain rule, our checkpoints, official HVI-CIDNet / Retinexformer / IAT).

Per image: input / output / reference mean luma, PSNR, SSIM, LPIPS (outputs rounded to 8 bit, as saved images).
On a fixed seeded subset: CIEDE2000 colour error, signed dL*, edge-energy and flat-noise ratios, NIQE and BRISQUE (pyiqa,
if installed). Harm = output PSNR below the input's own PSNR (identity). No parameter updates anywhere.
Every (dataset, condition) result is written once, atomically; reruns skip finished parts.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.image_metrics import psnr, ssim  # noqa: E402
from src.phase5 import zoo  # noqa: E402

DATASETS = ["loli_val", "sice", "loli_test"]
SUBSET = {"sice": 600, "loli_val": 300, "loli_test": 500}     # seeded subsets for the slower metrics (loli_val: per level)
SEED = 0
RATIO_BINS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, float("inf")]
RATIO_LABELS = ["<0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0", "1.0-1.5", ">1.5"]
OVER = 0.05          # output mean luma more than 0.05 above the reference = over-brightened
PANEL_N = {"loli_val": 4, "sice": 3, "loli_test": 4}
PANEL_CONDS = ["identity", "gain048", "unet_p2", "rf_off_real", "hvi_tierA", "hvi_off_real", "hvi_off_gen", "iat_lol", "iat_exp"]


# ------------------------------------------------------------------------------------------------ data
def items(data, ds):
    """List of dicts: id, inp, ref (or None), group (scene id), level (LoLI level or SICE exposure index)."""
    data = Path(data)
    out = []
    if ds == "sice":
        for d in sorted((data / "SICE_Part1_900").iterdir(), key=lambda p: int(p.name)):
            for p in sorted(d.glob("*.jpg")):
                if p.stem == "label":
                    continue
                out.append({"id": f"{d.name}/{p.stem}", "inp": str(p), "ref": str(d / "label.jpg"),
                            "group": d.name, "level": int(p.stem)})
    elif ds == "loli_val":
        for p in sorted((data / "LoLI-Street/Val/low").glob("*.jpg")):
            lvl, sid = p.stem.split("_")
            out.append({"id": p.stem, "inp": str(p), "ref": str(data / "LoLI-Street/Val/high" / p.name),
                        "group": sid, "level": lvl})
    elif ds == "loli_test":
        for p in sorted((data / "LoLI-Street/Test").glob("*.jpg"), key=lambda q: int(q.stem.split("_")[-1])):
            out.append({"id": p.stem, "inp": str(p), "ref": None, "group": p.stem, "level": "real"})
    else:
        raise ValueError(ds)
    return out


def subset_ids(its, ds):
    rng = random.Random(SEED)
    if ds == "loli_val":                      # same scene ids at every level
        groups = sorted({i["group"] for i in its})
        keep = set(rng.sample(groups, min(SUBSET[ds], len(groups))))
        return {i["id"] for i in its if i["group"] in keep}
    ids = sorted(i["id"] for i in its)
    return set(rng.sample(ids, min(SUBSET[ds], len(ids))))


def panel_ids(its, ds):
    rng = random.Random(SEED + 1)
    groups = sorted({i["group"] for i in its})
    keep = set(rng.sample(groups, min(PANEL_N[ds], len(groups))))
    return {i["id"] for i in its if i["group"] in keep}


def load(p):
    from PIL import Image
    with Image.open(p) as im:
        a = np.asarray(im.convert("RGB"))
    return torch.from_numpy(a.copy()).permute(2, 0, 1)[None].float().div(255)


def q8(x):
    return x.clamp(0, 1).mul(255).round().div(255)


def save_png(x, p):
    from PIL import Image
    os.makedirs(os.path.dirname(p), exist_ok=True)
    Image.fromarray(x[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy()).save(p)


# ------------------------------------------------------------------------------------------------ run
def nr_metrics(device):
    try:
        import pyiqa
        return {"niqe": pyiqa.create_metric("niqe", device=device), "brisque": pyiqa.create_metric("brisque", device=device)}
    except Exception as e:  # pyiqa missing or its parameter download failed: recorded, not fatal
        print("no-reference metrics unavailable:", repr(e), flush=True)
        return {}


def stage_run(a):
    from src.metrics.colour_metrics import pointwise_metrics, structure_metrics
    from src.metrics.lpips_metric import LPIPSMetric
    dev = torch.device(a.device)
    bad = zoo.check_weights(a.weights)
    if bad:
        sys.exit(f"weights MD5 mismatch: {bad}: STOP")
    lp = LPIPSMetric(device=dev)
    nr = nr_metrics(dev)
    for ds in a.datasets:
        its = items(a.data, ds)
        sub, pan = subset_ids(its, ds), panel_ids(its, ds)
        for cond in a.conditions:
            p = Path(a.out) / "per_image" / f"{ds}__{cond}.csv"
            if p.exists():
                print("done:", p.name, flush=True)
                continue
            f = zoo.build(cond, a.weights, dev)
            rows, t0, tnet = [], time.time(), 0.0
            for k, it in enumerate(its):
                x = load(it["inp"]).to(dev)
                t1 = time.time()
                with torch.no_grad():
                    y = q8(f(x))
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                tnet += time.time() - t1
                r = {"id": it["id"], "group": it["group"], "level": it["level"], "sub": int(it["id"] in sub),
                     "in_luma": float(zoo.luma(x).mean()), "out_luma": float(zoo.luma(y).mean())}
                if it["ref"] is not None:
                    g = load(it["ref"]).to(dev)
                    r["ref_luma"] = float(zoo.luma(g).mean())
                    r["psnr"] = float(psnr(y, g)[0])
                    r["ssim"] = float(ssim(y, g)[0])
                    r["lpips"] = float(lp(y, g)[0])
                if it["id"] in sub:
                    yo = y[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
                    if it["ref"] is not None:
                        go = g[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
                        pm, sm = pointwise_metrics(yo, go), structure_metrics(yo, go)
                        r.update({"de00": pm["de00_mean"], "de00_colour": pm["de00_leq_mean"], "dL": pm["dL_mean"],
                                  "chroma_ratio": pm["chroma_ratio"], "edge_energy_ratio": sm["edge_energy_ratio"],
                                  "flat_noise_ratio": sm["flat_noise_ratio"]})
                    with torch.no_grad():
                        for mname, m in nr.items():
                            r[mname] = float(m(y).item())
                if it["id"] in pan and cond in PANEL_CONDS:
                    save_png(y, str(Path(a.out) / "panels_src" / ds / cond / (it["id"].replace("/", "_") + ".png")))
                rows.append(r)
                if k % 1000 == 0:
                    print(f"  {ds} {cond}: {k}/{len(its)}  {time.time() - t0:.0f}s", flush=True)
            keys = sorted({k for r in rows for k in r}, key=lambda c: (c not in ("id", "group", "level", "sub"), c))
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(p) + ".tmp"
            with open(tmp, "w", newline="") as fh:
                w = csv.DictWriter(fh, keys)
                w.writeheader()
                w.writerows(rows)
            os.replace(tmp, p)
            timing = {"dataset": ds, "condition": cond, "n": len(rows), "net_seconds": round(tnet, 1),
                      "ms_per_image": round(1000 * tnet / max(len(rows), 1), 2), "device": str(dev)}
            with open(Path(a.out) / "timing.jsonl", "a") as fh:
                fh.write(json.dumps(timing) + "\n")
            print("wrote", p.name, timing, flush=True)
            del f
            if dev.type == "cuda":
                torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------ report
def read(p):
    with open(p) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k not in ("id", "group", "level"):
                r[k] = float(v) if v not in ("", None) else float("nan")
    return rows


def mean(v):
    v = [x for x in v if not math.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def ratio_bin(r):
    for i in range(len(RATIO_LABELS)):
        if RATIO_BINS[i] <= r < RATIO_BINS[i + 1]:
            return RATIO_LABELS[i]
    return RATIO_LABELS[-1]


def summarise(rows, ident):
    n = len(rows)
    harm = [r for r in rows if not math.isnan(r.get("psnr", float("nan"))) and r["psnr"] < ident[r["id"]]["psnr"]]
    over = [r for r in rows if "ref_luma" in r and r["out_luma"] - r["ref_luma"] > OVER]
    has_ref = any("psnr" in r for r in rows)
    s = {"n": n, "in_luma": mean([r["in_luma"] for r in rows]), "out_luma": mean([r["out_luma"] for r in rows])}
    if has_ref:
        s.update({"ref_luma": mean([r["ref_luma"] for r in rows]), "psnr": mean([r["psnr"] for r in rows]),
                  "ssim": mean([r["ssim"] for r in rows]), "lpips": mean([r["lpips"] for r in rows]),
                  "dY": mean([r["out_luma"] - r["ref_luma"] for r in rows]),
                  "harm": len(harm) / n, "harm_ci": wilson(len(harm), n),
                  "over": len(over) / n, "over_ci": wilson(len(over), n)})
    for k in ("de00", "de00_colour", "dL", "chroma_ratio", "edge_energy_ratio", "flat_noise_ratio", "niqe", "brisque"):
        s[k] = mean([r.get(k, float("nan")) for r in rows if r.get("sub") == 1.0])
    return s


def stage_report(a):
    out = Path(a.out)
    conds = [c for c in zoo.CONDITIONS if all((out / "per_image" / f"{ds}__{c}.csv").exists() for ds in a.datasets)]
    data = {(ds, c): read(out / "per_image" / f"{ds}__{c}.csv") for ds in a.datasets for c in conds}
    tables = {}
    for ds in a.datasets:
        ident = {r["id"]: r for r in data[(ds, "identity")]}
        if ds == "loli_val":
            groups = {"all": lambda r: True, **{l: (lambda r, l=l: r["level"] == l) for l in ("dense", "moderate", "light")}}
        elif ds == "sice":
            groups = {"all": lambda r: True, **{b: (lambda r, b=b: ratio_bin(ident[r["id"]]["in_luma"] / max(ident[r["id"]]["ref_luma"], 1e-6)) == b)
                                               for b in RATIO_LABELS}}
        else:
            groups = {"all": lambda r: True}
        for gname, fn in groups.items():
            for c in conds:
                rows = [r for r in data[(ds, c)] if fn(r)]
                if rows:
                    tables[(ds, gname, c)] = summarise(rows, ident)
    # consistency of over-brightening: Spearman(input/ref ratio, output - ref luma) per model on SICE + LoLI val
    spear = {}
    for ds in ("sice", "loli_val"):
        if ds not in a.datasets:
            continue
        for c in conds:
            rows = data[(ds, c)]
            x = np.array([r["in_luma"] / max(r["ref_luma"], 1e-6) for r in rows])
            y = np.array([r["out_luma"] - r["ref_luma"] for r in rows])
            rx, ry = x.argsort().argsort(), y.argsort().argsort()
            spear[(ds, c)] = float(np.corrcoef(rx, ry)[0, 1])
    js = {"|".join(map(str, k)): v for k, v in tables.items()}
    json.dump({"tables": js, "spearman_ratio_vs_dY": {"|".join(k): v for k, v in spear.items()},
               "conditions": {c: zoo.DESCRIPTION[c] for c in conds}, "over_threshold": OVER,
               "subset_sizes": SUBSET, "seed": SEED}, open(out / "diag_summary.json", "w"), indent=1, default=str)
    L = ["# Phase 5 diagnostic: inference only, existing and official checkpoints", "",
         "PSNR / SSIM / LPIPS on 8-bit outputs vs the dataset reference. harm = share of images whose output PSNR is "
         "below the unprocessed input's PSNR; over = share with output mean luma > reference + 0.05. "
         "dE00, dL*, NIQE, BRISQUE on a fixed seeded subset. 95% Wilson intervals in brackets.", ""]
    for ds in a.datasets:
        L += [f"## {ds}", ""]
        gnames = sorted({g for (d, g, _) in tables if d == ds}, key=lambda g: (g != "all", g))
        for g in gnames:
            L += [f"### {ds} / {g}", "",
                  "| condition | n | in Y | out Y | ref Y | PSNR | SSIM | LPIPS | harm | over | dE00 | dL* | NIQE | BRISQUE |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for c in conds:
                s = tables.get((ds, g, c))
                if not s:
                    continue

                def f(k, d=3):
                    v = s.get(k, float("nan"))
                    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"
                hv = "" if "harm" not in s else f"{100 * s['harm']:.0f}% [{100 * s['harm_ci'][0]:.0f}-{100 * s['harm_ci'][1]:.0f}]"
                ov = "" if "over" not in s else f"{100 * s['over']:.0f}% [{100 * s['over_ci'][0]:.0f}-{100 * s['over_ci'][1]:.0f}]"
                L.append(f"| {c} | {s['n']} | {f('in_luma')} | {f('out_luma')} | {f('ref_luma')} | {f('psnr', 2)} | "
                         f"{f('ssim')} | {f('lpips')} | {hv} | {ov} | {f('de00', 2)} | {f('dL', 2)} | {f('niqe', 2)} | {f('brisque', 2)} |")
            L.append("")
    if spear:
        L += ["## Spearman correlation: (input / reference luma) vs (output - reference luma)", "",
              "| condition | " + " | ".join(d for d in ("sice", "loli_val") if d in a.datasets) + " |",
              "|---|" + "---|" * len([d for d in ("sice", "loli_val") if d in a.datasets])]
        for c in conds:
            L.append(f"| {c} | " + " | ".join(f"{spear[(d, c)]:+.2f}" for d in ("sice", "loli_val") if (d, c) in spear) + " |")
    if (out / "timing.jsonl").exists():
        L += ["", "## Timing (network forward, per image)", "", "| dataset | condition | ms/image |", "|---|---|---|"]
        for line in open(out / "timing.jsonl"):
            t = json.loads(line)
            L.append(f"| {t['dataset']} | {t['condition']} | {t['ms_per_image']} |")
    (out / "DIAG_REPORT.md").write_text("\n".join(L) + "\n")
    plots(a, tables, conds)
    panels(a, conds)
    print("\n".join(L))


def plots(a, tables, conds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(a.out)
    show = list(conds)
    cmap = plt.get_cmap("tab20")
    colour = {c: ("black" if c == "identity" else cmap(i % 20)) for i, c in enumerate(show)}
    for ds, gl, xl in (("sice", RATIO_LABELS, "input / reference mean luma"), ("loli_val", ["dense", "moderate", "light"], "LoLI-Street level")):
        if ds not in a.datasets:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for c in show:
            xs = [g for g in gl if (ds, g, c) in tables]
            st = dict(marker="o", label=c, color=colour[c], lw=2.2 if c == "identity" else 1.2,
                      ls="--" if c == "gain048" else "-")
            ax[0].plot(xs, [tables[(ds, g, c)]["psnr"] for g in xs], **st)
            ax[1].plot(xs, [tables[(ds, g, c)]["dY"] for g in xs], **st)
        ax[0].set_ylabel("PSNR (dB)")
        ax[1].set_ylabel("output - reference mean luma")
        ax[1].axhline(0, color="k", lw=0.6)
        for x in ax:
            x.set_xlabel(xl)
            x.grid(alpha=0.3)
        ax[1].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out / f"plot_{ds}.png", dpi=130)
        plt.close(fig)


def panels(a, conds):
    from PIL import Image, ImageDraw
    out, src = Path(a.out), Path(a.out) / "panels_src"
    if not src.exists():
        return
    for ds in a.datasets:
        its = {i["id"]: i for i in items(a.data, ds)}
        cs = [c for c in PANEL_CONDS if c in conds and (src / ds / c).exists()]
        if not cs:
            continue
        ids = sorted(p.stem for p in (src / ds / cs[0]).glob("*.png"))
        if ds == "sice":                       # keep at most 5 exposures per sequence, spread over the sweep
            by = {}
            for i in ids:
                by.setdefault(i.split("_")[0], []).append(i)
            ids = []
            for s, v in by.items():
                v = sorted(v, key=lambda i: int(i.split("_")[1]))
                ids += [v[j] for j in sorted({round(t * (len(v) - 1) / 4) for t in range(5)})]
        tw = 220
        for i in ids:
            it = its[i.replace("_", "/") if ds == "sice" else i]
            tiles = [("input", Image.open(it["inp"]).convert("RGB"))]
            tiles += [(c, Image.open(src / ds / c / f"{i}.png").convert("RGB")) for c in cs
                      if c != "identity" and (src / ds / c / f"{i}.png").exists()]
            if it["ref"]:
                tiles.append(("reference", Image.open(it["ref"]).convert("RGB")))
            th = round(tiles[0][1].height * tw / tiles[0][1].width)
            canvas = Image.new("RGB", (tw * len(tiles), th + 16), "white")
            d = ImageDraw.Draw(canvas)
            for j, (name, im) in enumerate(tiles):
                canvas.paste(im.resize((tw, th)), (j * tw, 16))
                d.text((j * tw + 3, 2), name, fill="black")
            (out / "panels" / ds).mkdir(parents=True, exist_ok=True)
            canvas.save(out / "panels" / ds / f"{i}.jpg", quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "report"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--conditions", nargs="+", default=zoo.CONDITIONS)
    a = ap.parse_args()
    if "sice" in a.datasets and "Part2" in str(a.data):
        sys.exit("SICE Part 2 is sealed: STOP")
    torch.manual_seed(0)
    {"run": stage_run, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
