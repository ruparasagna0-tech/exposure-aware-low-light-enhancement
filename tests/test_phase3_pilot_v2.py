"""Tests for the pilot v2 fixes (notebook process cleanup, integrity scope, Phase-2-comparable calibration).
No model is trained or evaluated: timing logic is tested with stub engines, the notebook helper with dummy processes.

    .venv/bin/python -m unittest tests.test_phase3_pilot_v2 -v
"""

import contextlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

os.environ["PHASE3_VGG_RANDOM_INIT"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import scripts.phase3_pilot as pilot  # noqa: E402
from scripts.make_phase3_bundle import FILES, notebook  # noqa: E402

DATA = "data_prepared/phase2"
HAVE_DATA = (PROJECT_ROOT / DATA / "manifest.json").is_file()
NO_DATA = "prepared data not present (data_prepared/phase2); run scripts/prepare_phase2_data.py (docs/DATA.md)"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a reaped-but-not-yet-collected zombie counts as dead
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        return bool(out) and not out.startswith("Z")
    except Exception:  # noqa: BLE001
        return True


def notebook_sh():
    src = notebook("bundle.zip", "0" * 32)["cells"][4]["source"]
    block = src[src.index("import signal\ndef sh"):src.index("PILOT = ")]
    ns = {"os": os, "subprocess": subprocess}
    exec(block, ns)  # noqa: S102  (our own generated code)
    return ns["sh"]


CHILD = r"""
import subprocess, sys, time, signal
if len(sys.argv) > 1 and sys.argv[1] == 'ignore_term':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
print('PIDS', __import__('os').getpid(), g.pid, flush=True)
time.sleep(120)
"""


class TestNotebookProcessCleanup(unittest.TestCase):
    def _interrupt_after_pids(self, extra_args, grace):
        sh = notebook_sh()
        buf = io.StringIO()
        timer = threading.Timer(3.0, lambda: os.kill(os.getpid(), signal.SIGINT))
        timer.start()
        t0 = time.time()
        with contextlib.redirect_stdout(buf), self.assertRaises(KeyboardInterrupt):
            sh([sys.executable, "-c", CHILD, *extra_args], cwd=str(PROJECT_ROOT), grace_s=grace)
        timer.cancel()
        m = re.search(r"PIDS (\d+) (\d+)", buf.getvalue())
        self.assertIsNotNone(m, buf.getvalue())
        child, grandchild = int(m.group(1)), int(m.group(2))
        deadline = time.time() + 15
        while time.time() < deadline and (alive(child) or alive(grandchild)):
            time.sleep(0.2)
        self.assertFalse(alive(child), "child still running after interrupt")
        self.assertFalse(alive(grandchild), "grandchild (e.g. DataLoader worker / resume run) still running")
        self.assertIn("INTERRUPTED: process group", buf.getvalue())
        return time.time() - t0

    def test_interrupt_terminates_whole_process_group(self):
        self.assertLess(self._interrupt_after_pids([], grace=30), 15)

    def test_sigkill_fallback_when_sigterm_ignored(self):
        took = self._interrupt_after_pids(["ignore_term"], grace=1)
        self.assertLess(took, 15)

    def test_nonzero_exit_still_raises(self):
        sh = notebook_sh()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            sh([sys.executable, "-c", "raise SystemExit(3)"], cwd=str(PROJECT_ROOT))


class TestIntegrityScope(unittest.TestCase):
    def stub(self, calls):
        def checks_fn(cfg, root, verify_md5=True):
            calls.append(verify_md5)
            return [("all manifest files present", True, ""), ("prepared files match manifest MD5 (= source bytes)", True, "")]
        return checks_fn

    @unittest.skipUnless(HAVE_DATA, NO_DATA)   # counts the files listed in the prepared-data manifest
    def test_full_scope(self):
        calls = []
        rec = pilot.integrity_record(pilot.load_config(pilot.BASE_CFG), DATA, None, self.stub(calls))
        self.assertEqual(calls, [True])
        self.assertEqual(rec["integrity_scope"], "full")
        self.assertTrue(rec["integrity_md5_verified_in_this_run"] and rec["integrity_md5_verified"])
        self.assertIsNone(rec["integrity_md5_reference"])
        self.assertGreater(rec["integrity_md5_files"], 10000)
        self.assertIn("per-file MD5", rec["integrity_scope_description"])

    def _ref(self, d, **over):
        base = {"integrity_scope": "full", "integrity_all_passed": True, "data_root": str(Path(DATA).resolve()),
                "boot_id": pilot.boot_id(), "time": "t", "integrity_md5_files": 10272}
        base.update(over)
        p = Path(d) / "P0_environment.json"
        p.write_text(json.dumps(base))
        return str(p)

    def test_structural_scope_with_valid_reference(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            rec = pilot.integrity_record(pilot.load_config(pilot.BASE_CFG), DATA, self._ref(d), self.stub(calls))
        self.assertEqual(calls, [False])
        self.assertEqual(rec["integrity_scope"], "structural")
        self.assertFalse(rec["integrity_md5_verified_in_this_run"])
        self.assertEqual(rec["integrity_md5_reference"]["md5_files_verified"], 10272)
        self.assertIn("NOT recomputed", rec["integrity_scope_description"])

    def test_invalid_references_are_rejected_before_any_check(self):
        bad = [{"integrity_scope": "structural"}, {"integrity_all_passed": False}, {"boot_id": "another-vm"},
               {"data_root": "/elsewhere"}]
        for over in bad:
            calls = []
            with tempfile.TemporaryDirectory() as d, self.assertRaises(RuntimeError, msg=over):
                pilot.integrity_record(pilot.load_config(pilot.BASE_CFG), DATA, self._ref(d, **over), self.stub(calls))
            self.assertEqual(calls, [], over)

    def test_old_flag_removed(self):
        p = subprocess.run([sys.executable, "scripts/phase3_pilot.py", "--data-root", DATA, "--out-dir", "/tmp/x",
                            "--p0-no-md5"], capture_output=True, text=True, cwd=PROJECT_ROOT)
        self.assertEqual(p.returncode, 2)  # argparse error: the unconditional shortcut no longer exists


class StubEngine:
    def __init__(self, step_s=0.002, val_s=0.05, first_val_s=None):
        self.step_s, self.val_s, self.first_val_s = step_s, val_s, first_val_s
        self.loader_calls = self.val_loader_calls = self.val_calls = 0

    def val_loaders(self, nw):
        self.val_loader_calls += 1
        return {}

    def epoch_loader(self, epoch, steps, nw):
        self.loader_calls += 1
        return [{"i": i} for i in range(steps)]

    def train_step(self, batch, g):
        time.sleep(self.step_s)

    def validate(self, loaders):
        self.val_calls += 1
        time.sleep(self.first_val_s if (self.first_val_s and self.val_calls == 1) else self.val_s)
        return {"val_psnr_mean": 0.0}


class TestCalibration(unittest.TestCase):
    def test_gate_unchanged(self):
        self.assertEqual(pilot.GATES["calibration_tolerance"], 0.30)

    def test_phase2_history_is_the_recorded_run(self):
        t = pilot.phase2_epoch_times()
        self.assertEqual(len(t), 50)
        self.assertAlmostEqual(sum(t), 757.8, places=1)

    def test_time_epochs_splits_train_and_validation_like_phase2(self):
        eng = StubEngine()
        import torch
        ep = pilot.time_epochs(eng, torch.device("cpu"), 3, 5, 0)
        self.assertEqual(eng.val_loader_calls, 1)        # val loaders built once (Phase 2)
        self.assertEqual(eng.loader_calls, 3)            # one training DataLoader per epoch (Phase 2)
        self.assertEqual(eng.val_calls, 3)               # exactly one validation pass per epoch
        for e in ep:
            self.assertEqual(e["steps"], 5)
            self.assertGreater(e["val_s"], e["train_s"])
            self.assertAlmostEqual(e["total_s"], e["train_s"] + e["val_s"], delta=0.005)

    def test_summary_counts_validation_once_per_epoch(self):
        p2 = pilot.phase2_epoch_times()
        ep = [{"epoch": 1, "steps": 112, "train_s": 12.0, "val_s": 4.0, "total_s": 16.0},
              {"epoch": 2, "steps": 112, "train_s": 11.5, "val_s": 4.0, "total_s": 15.5},
              {"epoch": 3, "steps": 112, "train_s": 11.7, "val_s": 4.0, "total_s": 15.7}]
        s = pilot.calibration_summary(ep, p2, 0.30)
        self.assertAlmostEqual(s["projected_phase2_equivalent_s"], 16.0 + 49 * 15.6, places=1)
        self.assertAlmostEqual(s["ratio"], (16.0 + 49 * 15.6) / sum(p2), places=3)
        self.assertTrue(s["pass"])
        self.assertAlmostEqual(s["steady_median_val_s"], 4.0)
        self.assertAlmostEqual(s["steady_s_per_step"], 11.6 / 112)
        slow = [dict(e, total_s=e["total_s"] * 1.5) for e in ep]
        self.assertFalse(pilot.calibration_summary(slow, p2, 0.30)["pass"])

    def test_timed_validation_excludes_warmup_pass(self):
        import torch
        tv = pilot.timed_validation(StubEngine(val_s=0.02, first_val_s=0.3), torch.device("cpu"), 0, repeats=3)
        self.assertGreater(tv["val_pass_first_s"], 0.25)
        self.assertLess(tv["val_pass_s"], 0.1)
        self.assertEqual(len(tv["val_pass_timed_s"]), 3)


class TestNotebookV2(unittest.TestCase):
    def test_cells_parse_and_use_new_mechanisms(self):
        import ast
        nb = notebook("bundle.zip", "0" * 32)
        text = json.dumps(nb)
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                ast.parse("\n".join(ln for ln in c["source"].split("\n") if not ln.lstrip().startswith("!")))
        for needle in ("start_new_session=True", "os.killpg", "--p0-md5-reference", "boot_id", "leftover pilot"):
            self.assertIn(needle, text)
        self.assertNotIn("--p0-no-md5", text)

    def test_bundle_includes_phase2_history(self):
        self.assertIn("experiments/phase3_model_selection/stage0/source_phase2/history.json", FILES)


if __name__ == "__main__":
    unittest.main()
