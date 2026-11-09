"""Phase 3 / Stage 0: import the Phase-2 best checkpoint, history and logs into a separate working location.

Source: a local copy of the Drive folder ``LowLight-Generalization/experiments/generalized_lol_v2``
(downloaded folder or .zip). Nothing in the source or in experiments/generalized_lol_v2/ is modified.

Copies (never overwrites; refuses if the destination already exists):
    checkpoints/best.pth          (required)
    history.json, config.yaml     (optional, reported if missing)
    logs/**                       (optional)
into experiments/phase3_model_selection/stage0/source_phase2/ and writes provenance.json with the
source paths, sizes, MD5 and SHA-256 of every file, plus these verifications:
    - checkpoint loads with torch.load(weights_only=True) and rebuilds BaselineUNet (1,925,667 params)
    - checkpoint model config == configs/generalized_lol_v2.yaml model block
    - checkpoint epoch / validation metrics == the values recorded in the evaluation results
      (metrics_summary.json inside ~/Downloads/phase2_evaluation_results.zip, MD5-checked)
    - history.json (if present): best epoch by mean val PSNR == checkpoint epoch

Usage:
    python scripts/stage0_import_phase2.py --src ~/Downloads/generalized_lol_v2[.zip] \
        --drive-path /content/drive/MyDrive/LowLight-Generalization/experiments/generalized_lol_v2
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import count_parameters, load_config, load_model_from_checkpoint  # noqa: E402

DEST = PROJECT_ROOT / "experiments/phase3_model_selection/stage0/source_phase2"
RESULTS_ZIP = Path("~/Downloads/phase2_evaluation_results.zip").expanduser()
RESULTS_ZIP_MD5 = "e7302a3b660629702bc19bf6af2f5e0c"  # recorded in generalized_lol_v2_diagnostics/source_results_md5.json
EXPECTED_PARAMS = 1_925_667


def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_root(src: Path) -> Path:
    """The folder that contains checkpoints/best.pth (src itself or one level below)."""
    for cand in [src] + sorted(p for p in src.iterdir() if p.is_dir()):
        if (cand / "checkpoints" / "best.pth").is_file():
            return cand
    sys.exit(f"checkpoints/best.pth not found under {src}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="downloaded generalized_lol_v2 folder or .zip")
    ap.add_argument("--drive-path", default="/content/drive/MyDrive/LowLight-Generalization/experiments/generalized_lol_v2",
                    help="original Google Drive location (recorded for provenance)")
    ap.add_argument("--dest", default=str(DEST))
    args = ap.parse_args()

    dest = Path(args.dest)
    if dest.exists():
        sys.exit(f"{dest} already exists: refusing to overwrite an earlier import.")
    src = Path(args.src).expanduser().resolve()
    tmp = None
    src_md5 = None
    if src.is_file() and zipfile.is_zipfile(src):
        src_md5 = digest(src, "md5")
        with zipfile.ZipFile(src) as z:
            bad = z.testzip()
            if bad:
                sys.exit(f"corrupt member in {src}: {bad}")
            tmp = Path(tempfile.mkdtemp(prefix="stage0_import_"))
            z.extractall(tmp)
        root = find_root(tmp)
    elif src.is_dir():
        root = find_root(src)
    else:
        sys.exit(f"--src must be a folder or a .zip: {src}")

    wanted = ["checkpoints/best.pth", "history.json", "config.yaml"]
    wanted += sorted(str(p.relative_to(root)) for p in (root / "logs").rglob("*") if p.is_file()) if (root / "logs").is_dir() else []
    files, missing = {}, []
    staging = dest.with_name(dest.name + ".partial")
    if staging.exists():
        sys.exit(f"{staging} exists (interrupted import?): inspect and remove it manually.")
    for rel in wanted:
        s = root / rel
        if not s.is_file():
            missing.append(rel)
            continue
        d = staging / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        md5_src, md5_dst = digest(s, "md5"), digest(d, "md5")
        if md5_src != md5_dst:
            sys.exit(f"copy mismatch for {rel}")
        files[rel] = {"source": (f"{args.drive_path}/{rel}"), "local_source": str(s) if tmp is None else f"{src}!{rel}",
                      "bytes": d.stat().st_size, "md5": md5_dst, "sha256": digest(d, "sha256")}

    checks = []
    add = lambda name, ok, detail="": checks.append({"check": name, "passed": bool(ok), "detail": detail})  # noqa: E731

    model, ckpt = load_model_from_checkpoint(staging / "checkpoints/best.pth")
    n = count_parameters(model) if callable(count_parameters) else sum(p.numel() for p in model.parameters())
    add("checkpoint loads (weights_only) and rebuilds the model", True, type(model).__name__)
    add("parameter count", n == EXPECTED_PARAMS, n)
    cfg = load_config(PROJECT_ROOT / "configs/generalized_lol_v2.yaml")
    add("checkpoint model config == configs/generalized_lol_v2.yaml", ckpt["config"]["model"] == cfg["model"],
        ckpt["config"]["model"])
    add("checkpoint experiment_name", ckpt["config"].get("experiment_name") == "generalized_lol_v2",
        ckpt["config"].get("experiment_name"))

    rec = None
    if RESULTS_ZIP.is_file() and digest(RESULTS_ZIP, "md5") == RESULTS_ZIP_MD5:
        with zipfile.ZipFile(RESULTS_ZIP) as z:
            rec = json.loads(z.read("metrics_summary.json"))["meta"]
    add("recorded evaluation results available (zip MD5 matches)", rec is not None, str(RESULTS_ZIP))
    if rec:
        add("checkpoint epoch == recorded checkpoint_epoch", ckpt.get("epoch") == rec["checkpoint_epoch"],
            f"{ckpt.get('epoch')} vs {rec['checkpoint_epoch']}")
        cm, rm = ckpt.get("metrics") or {}, rec["checkpoint_val_metrics"]
        diffs = {k: abs(cm.get(k, float("nan")) - v) for k, v in rm.items()}
        add("checkpoint val metrics == recorded checkpoint_val_metrics", all(d < 1e-9 for d in diffs.values()), diffs)

    hist_path = staging / "history.json"
    if hist_path.is_file():
        h = json.load(open(hist_path))
        epochs = h if isinstance(h, list) else h.get("epochs", h.get("history", []))
        try:
            best = max(epochs, key=lambda e: e["val_psnr_mean"])
            add("history: best epoch by val_psnr_mean == checkpoint epoch", best["epoch"] == ckpt.get("epoch"),
                f"history best {best['epoch']} ({best['val_psnr_mean']:.4f})")
        except (KeyError, TypeError, ValueError) as e:
            add("history: best epoch by val_psnr_mean == checkpoint epoch", False, f"unrecognised history format: {e!r}")

    prov = {"imported": time.strftime("%Y-%m-%dT%H:%M:%S"), "source_argument": str(src), "source_zip_md5": src_md5,
            "drive_path": args.drive_path, "files": files, "missing_optional": missing, "checks": checks,
            "all_checks_passed": all(c["passed"] for c in checks)}
    with open(staging / "provenance.json", "w") as f:
        json.dump(prov, f, indent=1, default=str)
    os.replace(staging, dest)
    if tmp is not None:
        shutil.rmtree(tmp)
    for c in checks:
        print(("PASS " if c["passed"] else "FAIL ") + c["check"], "|", c["detail"])
    print(f"missing (optional): {missing}\nwritten: {dest}")
    sys.exit(0 if prov["all_checks_passed"] else 1)


if __name__ == "__main__":
    main()
