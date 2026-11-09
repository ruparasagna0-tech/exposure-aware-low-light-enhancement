"""Tests for the generated Tier-A notebook (no training: the trainer call is replaced by a recorder).

    .venv/bin/python -m unittest tests.test_tierA_notebook -v
"""

import ast
import builtins
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_tierA_notebook import BUNDLE, BUNDLE_MD5, VGG_MD5, notebook  # noqa: E402


def helpers(drive):
    """Exec cell 4 (helpers) with Colab-only pieces stubbed: /proc boot id, torch.cuda name, sh()."""
    import subprocess
    import torch
    src = notebook()["cells"][4]["source"]
    real_open = builtins.open

    def fake_open(p, *a, **k):
        if str(p) == "/proc/sys/kernel/random/boot_id":
            return io.StringIO("test-boot\n")
        return real_open(p, *a, **k)

    class FakeCuda:
        @staticmethod
        def get_device_name(i=0):
            return "Tesla T4"

    class Sub:  # nvidia-smi does not exist on the Mac: stub it; everything else (pgrep) is real
        PIPE, STDOUT, Popen, TimeoutExpired = subprocess.PIPE, subprocess.STDOUT, subprocess.Popen, subprocess.TimeoutExpired

        @staticmethod
        def run(cmd, *a, **k):
            if cmd and cmd[0] == "nvidia-smi":
                return subprocess.CompletedProcess(cmd, 0, stdout="580.82.07, 1590 MHz, 70.00 W", stderr="")
            return subprocess.run(cmd, *a, **k)
    ns = {"DRIVE": drive, "BUNDLE": BUNDLE, "json": json, "os": os, "subprocess": Sub, "open": fake_open,
          "torch": type("T", (), {"__version__": torch.__version__, "version": torch.version, "cuda": FakeCuda})}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(src, ns)  # noqa: S102  (our own generated code)
    calls = []
    ns["sh"] = lambda args, **k: calls.append(list(args))
    return ns, calls


def write_summary(d, epochs=50, steps=5600, test=False, amp=False):
    os.makedirs(f"{d}/logs", exist_ok=True)
    os.makedirs(f"{d}/checkpoints", exist_ok=True)
    open(f"{d}/checkpoints/best.pth", "w").write("x")
    open(f"{d}/checkpoints/last.pth", "w").write("x")
    json.dump({"epochs_completed": epochs, "optimizer_steps_total": steps, "pipeline_test_mode": test, "amp": amp,
               "best_epoch": 40, "best_val_psnr_mean": 18.0, "best_val_real_psnr": 17.0,
               "best_val_synthetic_psnr": 19.0}, open(f"{d}/logs/train_summary.json", "w"))


class TestTierANotebook(unittest.TestCase):
    def test_cells_parse_and_protocol_strings(self):
        nb = notebook()
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                ast.parse("\n".join(ln for ln in c["source"].split("\n") if not ln.lstrip().startswith("!")))
        text = json.dumps(nb)
        for needle in (BUNDLE, BUNDLE_MD5, VGG_MD5, "'--amp', 'off'", "'T4' in gpu", "SICE Part 2 present",
                       "re-implemented", "NOT bit-reproducible", "start_new_session=True", "os.killpg"):
            self.assertIn(needle, text)
        for forbidden in ("--skip-integrity-md5", "--epochs", "--steps-per-epoch", "--no-val", "'--amp', 'on'",
                          "--stop-after-epoch", "evaluate_generalized", "export_outputs"):
            self.assertNotIn(forbidden, text)

    def test_order_is_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            ns, calls = helpers(d)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(AssertionError):
                ns["run_tierA"]("hvi_cidnet")          # retinexformer not complete yet
            self.assertEqual(calls, [])

    def test_fresh_run_arguments_fp32_full_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            ns, calls = helpers(d)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileNotFoundError):
                ns["run_tierA"]("retinexformer")       # sh is a recorder, so report() finds no summary afterwards
            args = calls[0]
            self.assertEqual(args[:2], ["python", "scripts/train_phase3.py"])
            self.assertIn("configs/phase3/tierA_retinexformer.yaml", args)
            self.assertEqual(args[args.index("--amp") + 1], "off")
            self.assertNotIn("--resume", args)
            rd = args[args.index("--run-dir") + 1]
            self.assertTrue(rd.endswith("/experiments/phase3_model_selection/stage1/runs/tierA_retinexformer_s42"))
            label = json.load(open(f"{rd}/RUN_LABEL.json"))
            self.assertEqual((label["precision"], label["seed"]), ("fp32", 42))
            self.assertEqual(len([f for f in os.listdir(f"{rd}/logs") if f.startswith("session_")]), 1)

    def test_partial_run_resumes_and_complete_run_is_not_retrained(self):
        with tempfile.TemporaryDirectory() as d:
            ns, calls = helpers(d)
            runs = f"{d}/experiments/phase3_model_selection/stage1/runs"
            write_summary(f"{runs}/tierA_retinexformer_s42")
            with contextlib.redirect_stdout(io.StringIO()):
                ns["run_tierA"]("retinexformer")       # complete -> report only
            self.assertEqual(calls, [])
            part = f"{runs}/tierA_hvi_cidnet_s42"
            os.makedirs(f"{part}/checkpoints")
            open(f"{part}/checkpoints/last.pth", "w").write("x")
            json.dump([{"epoch": 1}, {"epoch": 2}], open(f"{part}/history.json", "w"))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileNotFoundError):
                ns["run_tierA"]("hvi_cidnet")
            self.assertEqual(calls[0][calls[0].index("--resume") + 1], f"{part}/checkpoints/last.pth")
            self.assertEqual(calls[0][calls[0].index("--amp") + 1], "off")

    def test_pipeline_test_or_amp_summaries_are_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            ns, _ = helpers(d)
            runs = f"{d}/experiments/phase3_model_selection/stage1/runs"
            write_summary(f"{runs}/tierA_retinexformer_s42", epochs=2, steps=20, test=True)
            self.assertEqual(ns["state"]("retinexformer"), "partial")
            buf = io.StringIO()
            write_summary(f"{runs}/tierA_unet_expaug_s42", amp=True)
            with contextlib.redirect_stdout(buf):
                ns["report"]("unet_expaug")
            self.assertIn("CHECK FAILED", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
