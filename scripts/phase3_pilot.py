"""Phase 3 / Stage 1: GPU feasibility pilot (STAGE1_PLAN_T4.md section 3). Throwaway weights; no evaluation metric
is used for any decision; test sets, SICE and LoLI-Street are never read.

P0 environment + data integrity (full = structural checks + per-file MD5; structural = MD5 skipped, only allowed with a
   verified full-MD5 P0 from the same machine session, given by --p0-md5-reference)
P1 U-Net calibration, Phase-2-comparable: full epochs timed exactly like scripts/train_generalized.py (112 steps +
   one validation pass per epoch), compared with the recorded Phase-2 epoch times; the per-step instrumented rate is
   kept as a diagnostic only
Per model:
P2 correctness (params, fwd/bwd on a real 2+2 batch, eval output range, finite loss; exposure-aug activity for unet_expaug)
P3/P4 FP32 and AMP runs from the same init on the same batches (300 steps; first 50 = warm-up for timing):
      s/step, data-wait share, peak memory, loss trajectories, GradScaler skipped steps; max batch that fits
P5 inference (FP32): validation = 1 warm-up pass + median of 3 timed passes; one 20 MP input (random tensor) tiled 512/64
P6 resume (subprocess, scripts/train_phase3.py): uninterrupted 2x10 steps vs 1 epoch + simulated disconnect + resume
P7 projection: Tier-A wall time (5,600 steps + 50 validation passes + 50 checkpoint writes)
Gates are fixed below (and in the plan); the report lists pass/fail and the measured values.

Usage (Colab): python scripts/phase3_pilot.py --data-root /content/phase2 --out-dir <Drive>/stage1/pilots \
                   --models unet_expaug retinexformer hvi_cidnet darkir_m
"""

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.datasets.phase2 import integrity_checks, paths  # noqa: E402
from src.phase3.engine import LOSS_OF, Engine  # noqa: E402
from src.phase3.models import EXPECTED_PARAMS, FreMLP  # noqa: E402
from src.utils import load_config, select_device, set_seed  # noqa: E402
from src.utils.tiling import tiled_forward  # noqa: E402

BASE_CFG = "configs/generalized_lol_v2.yaml"
CONFIGS = {m: f"configs/phase3/tierA_{m}.yaml" for m in ("unet_expaug", "retinexformer", "hvi_cidnet", "darkir_m")}
PHASE2_T4_SECONDS = 757.8          # measured: sum of 50 epoch times (5,600 steps + 50 validations), Phase-2 U-Net, T4
# Recorded Phase-2 per-epoch wall times (each = 112 training steps + 1 validation pass), imported in Stage 0
PHASE2_HISTORY = "experiments/phase3_model_selection/stage0/source_phase2/history.json"
PHASE2_HISTORY_MD5 = "10debcea1c0fe82d41f4d8426dedfa4d"
CALIBRATION_EPOCHS = 3             # epoch 1 reported separately (warm-up), epochs 2..3 = steady state
VAL_TIMED_REPEATS = 3
TIER_A_STEPS, TIER_A_VALS = 5600, 50
GATES = {"calibration_tolerance": 0.30, "max_train_mem_gb": 12.0, "amp_max_skipped_frac": 0.01,
         "amp_max_loss_rel_diff": 0.02, "amp_min_speedup": 1.3, "max_tierA_hours": 3.0,
         "max_dev_export_hours": 1.5, "resume_loss_rel_tol": 1e-4}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def boot_id() -> str:
    """Identifies the machine session (Linux boot id; hostname elsewhere). A new Colab VM has a new boot id."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return platform.node()


def atomic_json(obj, path: Path) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


FULL_SCOPE = "structural checks + per-file MD5 of every prepared file against manifest.json"
STRUCTURAL_SCOPE = ("structural checks only (manifest files present, pair counts, split/scene-disjointness, training-pool "
                    "isolation, image readability/RGB, sizes); per-file MD5 NOT recomputed in this run: see "
                    "integrity_md5_reference for the run that verified it")


def integrity_record(base, data_root, md5_reference=None, checks_fn=integrity_checks) -> dict:
    """P0 integrity. Without a reference: full check (incl. MD5). With a reference: structural checks only, accepted
    only if the referenced P0 ran the full MD5 check, passed, used the same data root, and ran in this machine session."""
    root = str(Path(data_root).resolve())
    ref = None
    if md5_reference is not None:
        r = json.load(open(md5_reference))
        problems = []
        if r.get("integrity_scope") != "full":
            problems.append("referenced P0 did not run the full MD5 check")
        if not r.get("integrity_all_passed"):
            problems.append("referenced P0 did not pass")
        if r.get("data_root") != root:
            problems.append(f"different data root ({r.get('data_root')} vs {root})")
        if r.get("boot_id") != boot_id():
            problems.append("referenced P0 ran in a different machine session (e.g. after a Colab disconnect)")
        if problems:
            raise RuntimeError("--p0-md5-reference rejected: " + "; ".join(problems) + ". Run the full P0 instead.")
        ref = {"path": str(md5_reference), "time": r.get("time"), "boot_id": r.get("boot_id"),
               "md5_files_verified": r.get("integrity_md5_files")}
    full = ref is None
    checks = checks_fn(base, data_root, verify_md5=full)
    rec = {"integrity_scope": "full" if full else "structural",
           "integrity_scope_description": FULL_SCOPE if full else STRUCTURAL_SCOPE,
           "integrity_md5_verified_in_this_run": full,
           "integrity_md5_verified": full,  # legacy key from earlier reports; same meaning as *_in_this_run
           "integrity_md5_reference": ref,
           "integrity_checks_run": [c for c, _, _ in checks],
           "integrity_all_passed": all(ok for _, ok, _ in checks),
           "integrity_failed": [c for c, ok, _ in checks if not ok],
           "boot_id": boot_id(), "data_root": root}
    if full:
        rec["integrity_md5_files"] = len(json.load(open(paths(base, data_root)["manifest"]))["files"])
    return rec


def phase2_epoch_times() -> list:
    p = PROJECT_ROOT / PHASE2_HISTORY
    data = p.read_bytes()
    if hashlib.md5(data).hexdigest() != PHASE2_HISTORY_MD5:
        raise RuntimeError(f"{p}: MD5 differs from the imported Phase-2 history")
    return [e["time_s"] for e in json.loads(data)]


def time_epochs(eng, device, n_epochs, steps_per_epoch, nw) -> list:
    """Phase-2-comparable timing (scripts/train_generalized.py): per epoch, t0 after DataLoader creation, then
    ``steps_per_epoch`` training steps without extra synchronisation, then one validation pass; time_s ends after
    validation (history/checkpoint writes are not included, as in Phase 2). Train and validation parts are split."""
    loaders = eng.val_loaders(nw)  # created once, as in Phase 2
    out, g = [], 0
    for epoch in range(1, n_epochs + 1):
        loader = eng.epoch_loader(epoch, steps_per_epoch, nw)
        t0 = time.time()
        n = 0
        for batch in loader:
            eng.train_step(batch, g)  # returns float(loss): one host sync per step, like Phase 2's loss.item()
            g += 1
            n += 1
        sync(device)
        t1 = time.time()
        eng.validate(loaders)
        sync(device)
        t2 = time.time()
        out.append({"epoch": epoch, "steps": n, "train_s": round(t1 - t0, 3), "val_s": round(t2 - t1, 3),
                    "total_s": round(t2 - t0, 3)})
    return out


def calibration_summary(epochs, phase2_times, tolerance, instrumented=None) -> dict:
    """Compare like with like: a projected 50-epoch total built from the measured epochs (epoch 1 + 49 x median of the
    steady-state epochs) against the recorded Phase-2 total (sum of its 50 epochs). Validation is counted exactly once
    per epoch on both sides. The tolerance is the predefined gate and is not adjusted here."""
    n_p2 = len(phase2_times)
    steady = epochs[1:] if len(epochs) > 1 else epochs
    med_total = statistics.median(e["total_s"] for e in steady)
    med_train = statistics.median(e["train_s"] for e in steady)
    med_val = statistics.median(e["val_s"] for e in steady)
    projected = epochs[0]["total_s"] + (n_p2 - 1) * med_total
    recorded = sum(phase2_times)
    ratio = projected / recorded
    res = {"method": "phase2_comparable_full_epochs", "epochs_measured": epochs,
           "steady_median_epoch_s": med_total, "steady_median_train_s": med_train, "steady_median_val_s": med_val,
           "steady_val_share": med_val / med_total, "steady_s_per_step": med_train / steady[0]["steps"],
           "projected_phase2_equivalent_s": round(projected, 1), "measured_phase2_s": round(recorded, 1),
           "phase2_median_epoch_s": statistics.median(phase2_times), "ratio": round(ratio, 3),
           "median_epoch_ratio": round(med_total / statistics.median(phase2_times), 3),
           "tolerance": tolerance, "pass": abs(ratio - 1) <= tolerance,
           "note": "Phase-2 logs record only per-epoch totals; the train/validation split exists for this session only"}
    if instrumented is not None:
        res["diagnostic_instrumented"] = instrumented
    return res


def timed_validation(eng, device, nw, repeats=VAL_TIMED_REPEATS) -> dict:
    """1 untimed-for-the-estimate warm-up pass (first full-resolution shapes) + median of ``repeats`` timed passes."""
    loaders = eng.val_loaders(nw)
    times = []
    for _ in range(repeats + 1):
        sync(device)
        t0 = time.perf_counter()
        val = eng.validate(loaders)
        sync(device)
        times.append(time.perf_counter() - t0)
    return {"val_pass_first_s": round(times[0], 2), "val_pass_s": round(statistics.median(times[1:]), 2),
            "val_pass_timed_s": [round(t, 2) for t in times[1:]], "val": val}


# ---------------------------------------------------------------- interruption handling
# The notebook stops a pilot by sending SIGTERM to its whole process group, so DataLoader workers can die at the same
# moment. PyTorch then raises "DataLoader worker ... is killed by signal" (RuntimeError) from its own SIGCHLD handler,
# possibly BEFORE our SIGTERM handler has run. A failure is therefore classified as "interrupted" if an interrupt was
# handled OR a SIGTERM is still pending; SIGTERM is blocked while the status is classified and written, so the handler
# cannot fire half-way through the write. Genuine errors (no interrupt in flight) stay "error".
INTERRUPT = {"flag": False, "signal": None, "time": None}


def _on_sigterm(signum, frame):
    if INTERRUPT["flag"]:
        return  # repeated SIGTERM while already stopping: ignore
    INTERRUPT.update(flag=True, signal="SIGTERM", time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    raise KeyboardInterrupt("SIGTERM")


def install_interrupt_handler():
    """Install the SIGTERM handler; returns the previous handler."""
    INTERRUPT.update(flag=False, signal=None, time=None)
    return signal.signal(signal.SIGTERM, _on_sigterm)


def interrupt_in_flight() -> bool:
    return bool(INTERRUPT["flag"]) or signal.SIGTERM in signal.sigpending()


def classify_failure(exc: BaseException) -> str:
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted"
    if interrupt_in_flight():  # e.g. worker killed by the same SIGTERM before our handler ran
        return "interrupted"
    return "failed" if isinstance(exc, SystemExit) else "error"


@contextmanager
def sigterm_blocked(restore_mask):
    """Block SIGTERM for the duration; restore ``restore_mask`` afterwards (a pending SIGTERM is then delivered)."""
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, restore_mask)


def _record_failure(rec: dict, exc: BaseException) -> str:
    status = classify_failure(exc)
    rec["status"] = status
    rec["error"] = f"{type(exc).__name__}: {exc}"
    if status == "interrupted":
        rec["interrupt"] = {"handled": INTERRUPT["flag"], "signal": INTERRUPT["signal"] or
                            ("SIGTERM (pending)" if signal.SIGTERM in signal.sigpending() else type(exc).__name__),
                            "time": INTERRUPT["time"] or time.strftime("%Y-%m-%dT%H:%M:%S")}
    elif status == "error":
        rec["traceback"] = traceback.format_exc()[-4000:]
    return status


def run_guarded(rep: dict, report_path: Path, body) -> str:
    """Run one model's pilot sections. Returns "complete" or "error" (pilot continues with the next model); for an
    interruption the report is saved with status "interrupted" and KeyboardInterrupt is raised (pilot stops)."""
    entry_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])  # query only
    try:
        body()
    except BaseException as e:  # noqa: BLE001
        with sigterm_blocked(entry_mask):
            status = _record_failure(rep, e)
            atomic_json(rep, report_path)
            if status == "error":
                print(rep.get("traceback", ""), flush=True)
        # the mask is restored here: a pending SIGTERM is handled now (raises KeyboardInterrupt unless already flagged)
        if status == "interrupted":
            print(f"INTERRUPTED during {rep.get('section_in_progress')}; partial report saved", flush=True)
            if isinstance(e, KeyboardInterrupt):
                raise
            raise KeyboardInterrupt(f"interrupted ({type(e).__name__}: {e})") from e
        return status
    rep["status"] = "complete"
    rep.pop("section_in_progress", None)
    atomic_json(rep, report_path)
    return "complete"


class RunStatus:
    """Run-level status of one pilot invocation (``pilot_run_status.json`` in its out-dir): which stage (P0, P1, each
    model) is running / complete / interrupted / failed / error, so an interrupted setup or an interruption between
    models is never ambiguous."""

    def __init__(self, out_root: Path, argv):
        self.path = out_root / "pilot_run_status.json"
        self.rec = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(), "boot_id": boot_id(),
                    "argv": list(argv), "stages": {}, "current": None, "outcome": "running"}
        self.write()

    def write(self):
        atomic_json(self.rec, self.path)

    def set(self, stage: str, status: str):
        self.rec["stages"][stage] = status
        self.rec["current"] = stage if status == "running" else None
        if status != "running":
            self.rec["last_finished_stage"] = stage
        self.write()

    def finish(self, outcome: str, exc: BaseException = None):
        if self.rec["outcome"] != "running":
            return  # first outcome wins (e.g. "failed" recorded before sys.exit)
        if exc is not None:
            sub = {}
            outcome = _record_failure(sub, exc)
            self.rec["error"] = sub.get("error")
            self.rec["interrupt"] = sub.get("interrupt")
            if self.rec["current"]:
                self.rec["stages"][self.rec["current"]] = outcome
                self.rec["stopped_in_stage"] = self.rec["current"]
            else:  # between stages (e.g. between two models)
                self.rec["stopped_in_stage"] = f"after {self.rec.get('last_finished_stage', 'start')}"
        self.rec.update(outcome=outcome, finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.write()


def run_steps(cfg, name, device, amp, n_steps, warmup, data_root, nw, record_batches=False):
    """Fresh engine from the fixed seed; steps over the Tier-A batch stream; returns timings and losses."""
    set_seed(cfg["seed"], deterministic=cfg["train"]["deterministic"])
    eng = Engine(cfg, name, device, amp=amp, total_steps=TIER_A_STEPS, data_root=data_root)
    steps_per_epoch = cfg["sampler"]["steps_per_epoch"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    losses, skipped, step_t, wait_t, names = [], 0, [], [], []
    g, epoch, nonfinite = 0, 1, 0
    while g < n_steps:
        t_wait = time.perf_counter()
        for batch in eng.epoch_loader(epoch, steps_per_epoch, nw):
            sync(device)
            t0 = time.perf_counter()
            wait_t.append(t0 - t_wait)
            r = eng.train_step(batch, g)
            sync(device)
            t1 = time.perf_counter()
            step_t.append(t1 - t0)
            losses.append(r["loss"])
            nonfinite += not np.isfinite(r["loss"])
            skipped += r["skipped"]
            if record_batches:
                names.append(list(batch["name"]))
            g += 1
            t_wait = time.perf_counter()
            if g >= n_steps:
                break
        epoch += 1
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    st, wt = np.array(step_t[warmup:]), np.array(wait_t[warmup:])
    return eng, {"steps": n_steps, "s_per_step_compute": float(st.mean()), "s_per_step_wall": float((st + wt).mean()),
                 "data_wait_share": float(wt.sum() / (st.sum() + wt.sum())), "peak_mem_gb": peak,
                 "losses": losses, "skipped": int(skipped), "nonfinite": int(nonfinite), "names": names}


def max_batch(cfg, name, device, amp, sizes=(4, 8, 12, 16)):
    res = {}
    for b in sizes:
        set_seed(cfg["seed"])
        eng = Engine(cfg, name, device, amp=amp, total_steps=TIER_A_STEPS, build_data=False)
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            x, y = torch.rand(b, 3, 256, 256, device=device), torch.rand(b, 3, 256, 256, device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=eng.amp):
                pred, side = eng.model.forward_train(x)
                loss, _ = eng.objective(pred, y, side=side, model=eng.model)
            eng.scaler.scale(loss).backward()
            sync(device)
            res[b] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
        except torch.cuda.OutOfMemoryError:
            res[b] = "OOM"
            del eng
            torch.cuda.empty_cache()
            break
        del eng, x, y, pred, side, loss
        torch.cuda.empty_cache()
    return res


def inference_timing(eng, device, nw, shape=(3999, 5997)):
    eng.model.eval()
    tv = timed_validation(eng, device, nw)
    val = tv.pop("val")
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    x = torch.rand(1, 3, *shape)
    sync(device)
    t0 = time.perf_counter()
    y = tiled_forward(eng.model, x, tile=512, overlap=64, device=device)
    sync(device)
    t_sice = time.perf_counter() - t0
    ok_range = bool(0.0 <= float(y.min()) and float(y.max()) <= 1.0)
    return {**tv, "val_psnr_untrained_NOT_USED": round(val["val_psnr_mean"], 3),
            "sice_20mp_tiled_s": round(t_sice, 2), "sice_shape": list(shape), "sice_20mp_output_in_range": ok_range,
            "sice_20mp_peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if device.type == "cuda" else None}


def resume_test(model, data_root, out_dir, amp, device):
    def run(run_dir, *extra):
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts/train_phase3.py"), "--config", CONFIGS[model],
               "--run-dir", str(run_dir), "--data-root", str(data_root), "--device", device, "--amp", amp,
               "--epochs", "2", "--steps-per-epoch", "10", "--no-val", "--skip-integrity-md5", *extra]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        if p.returncode != 0:
            raise RuntimeError(f"resume test subprocess failed:\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}")
        return p.stdout
    a, b = out_dir / "resume_test" / "A_uninterrupted", out_dir / "resume_test" / "B_interrupted"
    t0 = time.perf_counter()
    log_a = run(a)
    run(b, "--stop-after-epoch", "1")
    interrupted_epochs = len(json.load(open(b / "history.json")))
    run(b, "--resume", str(b / "checkpoints" / "last.pth"))
    ha, hb = json.load(open(a / "history.json")), json.load(open(b / "history.json"))
    sa, sb = json.load(open(a / "logs/train_summary.json")), json.load(open(b / "logs/train_summary.json"))
    rel = abs(ha[1]["train_loss"] - hb[1]["train_loss"]) / max(abs(ha[1]["train_loss"]), 1e-12)
    ckpt_mb = (a / "checkpoints/last.pth").stat().st_size / 2**20
    return {"interrupted_after_epochs": interrupted_epochs,
            "batch_order_identical": [x["batch_names_md5"] for x in ha] == [x["batch_names_md5"] for x in hb],
            "epoch2_loss_uninterrupted": ha[1]["train_loss"], "epoch2_loss_resumed": hb[1]["train_loss"],
            "epoch2_loss_rel_diff": rel, "final_weights_bitwise_identical": sa["final_state_md5"] == sb["final_state_md5"],
            "checkpoint_mb": round(ckpt_mb, 1), "wall_s": round(time.perf_counter() - t0, 1),
            "checkpoint_write_s": [float(x) for x in re.findall(r"ckpt ([0-9.]+)s", log_a)],
            "note": "throwaway weights (2 x 10 steps); never evaluated"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", nargs="*", default=list(CONFIGS), choices=list(CONFIGS),
                    help="empty list = P0 + P1 only (setup cell)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--skip-calibration", action="store_true")
    ap.add_argument("--p0-md5-reference",
                    help="path to a P0_environment.json of a FULL (MD5) check from this machine session; this run then "
                         "repeats only the structural checks. Rejected if the reference is not full/passed/same session.")
    ap.add_argument("--calibration-epochs", type=int, default=CALIBRATION_EPOCHS)
    ap.add_argument("--skip-resume-test", action="store_true")
    ap.add_argument("--sice-shape", type=int, nargs=2, default=[3999, 5997], help="smaller only for CPU dry runs")
    args = ap.parse_args()

    install_interrupt_handler()
    device = select_device(args.device)
    out_root = Path(args.out_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    rs = RunStatus(out_root, sys.argv)
    entry_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    try:
        _pilot(args, device, out_root, rs)
        rs.finish("complete")
    except BaseException as e:  # noqa: BLE001  (P0/P1, between models, or a re-raised model interruption)
        with sigterm_blocked(entry_mask):
            rs.finish(None, e)
        raise


def _pilot(args, device, out_root, rs):
    nw = args.num_workers
    base = load_config(BASE_CFG)

    # ---------------- P0
    rs.set("P0", "running")
    env = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": platform.python_version(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
           "gpu_total_gb": round(torch.cuda.get_device_properties(device).total_memory / 2**30, 2) if device.type == "cuda" else None,
           "cpu_count": os.cpu_count(), "gates": GATES}
    env.update(integrity_record(base, args.data_root, args.p0_md5_reference))
    p = paths(base, args.data_root)
    sample = sorted((Path(p["lol_v2"]) / "Real_captured/Train/Low").iterdir())[0]
    arr = np.asarray(Image.open(sample).convert("RGB"))
    from src.datasets.paired_list import load_rgb
    from src.datasets.paired_transforms import PairedToTensor
    t, _ = PairedToTensor()(load_rgb(str(sample)), load_rgb(str(sample)))
    env["rgb_pipeline_check"] = bool(np.array_equal(np.round(t.numpy() * 255).astype(np.uint8).transpose(1, 2, 0), arr))
    atomic_json(env, out_root / "P0_environment.json")
    print(json.dumps(env, indent=1), flush=True)
    if not (env["integrity_all_passed"] and env["rgb_pipeline_check"]):
        rs.set("P0", "failed")
        rs.finish("failed")
        sys.exit("STOP: P0 failed")
    rs.set("P0", "complete")

    # ---------------- P1
    if not args.skip_calibration:
        rs.set("P1", "running")
        p2 = phase2_epoch_times()
        set_seed(base["seed"], deterministic=base["train"]["deterministic"])
        eng = Engine(base, "unet", device, amp=False, total_steps=TIER_A_STEPS, data_root=args.data_root)
        epochs = time_epochs(eng, device, args.calibration_epochs, base["sampler"]["steps_per_epoch"], nw)
        del eng
        # diagnostic only (not gated): the per-step instrumented rate used by P3, to quantify instrumentation overhead
        e2, r = run_steps(base, "unet", device, False, args.steps, args.warmup, args.data_root, nw)
        del e2
        cal = calibration_summary(epochs, p2, GATES["calibration_tolerance"],
                                  instrumented={"s_per_step_wall_synced": r["s_per_step_wall"],
                                                "s_per_step_compute_synced": r["s_per_step_compute"],
                                                "data_wait_share": r["data_wait_share"]})
        atomic_json(cal, out_root / "P1_calibration.json")
        print("P1", json.dumps({k: v for k, v in cal.items() if k != "epochs_measured"}), flush=True)
        rs.set("P1", "complete")

    # ---------------- per model
    for name in args.models:
        rs.set(name, "running")
        cfg = load_config(CONFIGS[name])
        mdir = out_root / name
        mdir.mkdir(parents=True, exist_ok=True)
        rep = {"model": name, "loss": LOSS_OF[name], "gates": GATES, "status": "running"}
        print(f"\n===== {name}", flush=True)
        atomic_json(rep, mdir / "pilot_report.json")
        def body():
            rep["section_in_progress"] = "P2"
            # P2
            set_seed(cfg["seed"])
            eng = Engine(cfg, name, device, amp=False, total_steps=TIER_A_STEPS, data_root=args.data_root)
            n = eng.n_params()
            batch = next(iter(eng.epoch_loader(1, cfg["sampler"]["steps_per_epoch"], 0)))
            r = eng.train_step(batch, 0)
            eng.model.eval()
            with torch.no_grad():
                y = eng.model(eng.val_real[0]["low"][None].to(device))
            p2 = {"params": n, "expected": EXPECTED_PARAMS[name]["count"], "params_ok": n == EXPECTED_PARAMS[name]["count"],
                  "paper": EXPECTED_PARAMS[name]["paper"], "first_loss": r["loss"], "loss_finite": bool(np.isfinite(r["loss"])),
                  "loss_parts": r["parts"], "eval_output_shape": list(y.shape),
                  "eval_output_in_range": bool(0 <= float(y.min()) and float(y.max()) <= 1)}
            if name == "darkir_m":
                fm = [m for m in eng.model.modules() if isinstance(m, FreMLP)]
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    o = fm[0](torch.rand(1, fm[0].process1[0].in_channels, 48, 40, device=device))
                p2["fremlp_count"] = len(fm)
                p2["fremlp_runs_fp32_under_autocast"] = o.dtype == torch.float32
            if name == "unet_expaug":
                plain = Engine(load_config(BASE_CFG), "unet", device, amp=False, total_steps=TIER_A_STEPS,
                               data_root=args.data_root)
                aug_n, ref_equal, in_range, total = 0, True, True, 0
                # Materialise each stream completely before building the other: with num_workers=0 both loaders draw
                # crops from the same global RNG, so interleaving them would shift every crop.
                la = list(eng.epoch_loader(1, 25, 0))
                lp = list(plain.epoch_loader(1, 25, 0))
                for ba, bp in zip(la, lp):
                    ref_equal &= torch.equal(ba["high"], bp["high"])
                    in_range &= bool(ba["low"].min() >= 0 and ba["low"].max() <= 1)
                    aug_n += int((ba["low"] - bp["low"]).abs().flatten(1).amax(1).gt(1e-6).sum())
                    total += len(ba["low"])
                p2["expaug_reference_unchanged"] = bool(ref_equal)
                p2["expaug_inputs_in_range"] = bool(in_range)
                p2["expaug_fraction_augmented"] = aug_n / total
                p2["expaug_ok"] = bool(ref_equal and in_range and 0.3 <= aug_n / total <= 0.7)
                del plain
            rep["P2_correctness"] = p2
            atomic_json(rep, mdir / "pilot_report.json")  # partial report after each section
            del eng
            torch.cuda.empty_cache() if device.type == "cuda" else None

            rep["section_in_progress"] = "P3_P4"
            # P3/P4
            e32, r32 = run_steps(cfg, name, device, False, args.steps, args.warmup, args.data_root, nw)
            del e32
            torch.cuda.empty_cache() if device.type == "cuda" else None
            e16, r16 = run_steps(cfg, name, device, True, args.steps, args.warmup, args.data_root, nw)
            l32, l16 = np.array(r32["losses"]), np.array(r16["losses"])
            k = min(100, len(l32))
            rel = float(abs(l16[-k:].mean() - l32[-k:].mean()) / abs(l32[-k:].mean()))
            speed = r32["s_per_step_wall"] / r16["s_per_step_wall"]
            amp_ok = bool(r16["nonfinite"] == 0 and r16["skipped"] / args.steps < GATES["amp_max_skipped_frac"]
                          and rel <= GATES["amp_max_loss_rel_diff"] and speed >= GATES["amp_min_speedup"])
            rep["P3_P4"] = {
                "fp32": {k2: v for k2, v in r32.items() if k2 not in ("losses", "names")},
                "amp": {k2: v for k2, v in r16.items() if k2 not in ("losses", "names")},
                "loss_mean_last100_fp32": float(l32[-k:].mean()), "loss_mean_last100_amp": float(l16[-k:].mean()),
                "amp_loss_rel_diff": rel, "amp_speedup": speed, "amp_gates_pass": amp_ok,
                "loss_decreasing_fp32": bool(l32[-50:].mean() < l32[:50].mean())}
            np.savez(mdir / "P4_loss_curves.npz", fp32=l32, amp=l16)
            atomic_json(rep, mdir / "pilot_report.json")  # partial report after each section
            del e16
            torch.cuda.empty_cache() if device.type == "cuda" else None
            chosen_amp = amp_ok
            rep["chosen_precision"] = "amp_fp16" if chosen_amp else "fp32"
            chosen = r16 if chosen_amp else r32
            rep["mem_ok"] = chosen["peak_mem_gb"] is None or chosen["peak_mem_gb"] <= GATES["max_train_mem_gb"]
            rep["section_in_progress"] = "max_batch"
            rep["max_batch_peak_gb"] = max_batch(cfg, name, device, chosen_amp) if device.type == "cuda" else None

            rep["section_in_progress"] = "P5"
            # P5 (always FP32)
            set_seed(cfg["seed"])
            eng = Engine(cfg, name, device, amp=False, total_steps=TIER_A_STEPS, data_root=args.data_root)
            rep["P5_inference_fp32"] = inference_timing(eng, device, nw, tuple(args.sice_shape))
            atomic_json(rep, mdir / "pilot_report.json")  # partial report after each section
            del eng
            torch.cuda.empty_cache() if device.type == "cuda" else None

            rep["section_in_progress"] = "P6"
            # P6
            if not args.skip_resume_test:
                rep["P6_resume"] = resume_test(name, args.data_root, mdir, "on" if chosen_amp else "off", args.device)
                p6 = rep["P6_resume"]
                p6["pass"] = bool(p6["interrupted_after_epochs"] == 1 and p6["batch_order_identical"]
                                  and p6["epoch2_loss_rel_diff"] <= GATES["resume_loss_rel_tol"])

            atomic_json(rep, mdir / "pilot_report.json")
            rep["section_in_progress"] = "P7"
            # P7 (val_pass_s = median of timed passes after a warm-up pass)
            val_s = rep["P5_inference_fp32"]["val_pass_s"]
            ckpt_list = rep.get("P6_resume", {}).get("checkpoint_write_s") or []
            ckpt_s = max(ckpt_list) if ckpt_list else 5.0  # measured Drive write time (max of runs); 5 s if unmeasured
            rep["ckpt_write_s_used"] = ckpt_s
            proj = TIER_A_STEPS * chosen["s_per_step_wall"] + TIER_A_VALS * (val_s + ckpt_s)
            # dev export estimate: SICE (58 x the 20 MP timing) + LoLI (2,997 x 512^2 approx. val-image rate) + val
            per_img = val_s / 156
            export = 58 * rep["P5_inference_fp32"]["sice_20mp_tiled_s"] + 2997 * per_img * (512 * 512) / (600 * 400) + val_s
            rep["P7_projection"] = {"tierA_hours": round(proj / 3600, 2), "tierA_ok": proj / 3600 <= GATES["max_tierA_hours"],
                                    "dev_export_inference_hours_approx": round(export / 3600, 2),
                                    "dev_export_ok": export / 3600 <= GATES["max_dev_export_hours"],
                                    "note": "projection excludes PNG encoding/LPIPS of the export (CPU-bound)"}

        outcome = run_guarded(rep, mdir / "pilot_report.json", body)
        rs.set(name, outcome)
        if outcome == "error" and device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({k: v for k, v in rep.items() if k != "traceback"}, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
