"""Phase 5 diagnostic: loaders, sealed-data guards and a CPU dry run on a tiny fake pack (no downloads).

    .venv/bin/python -m unittest tests.test_phase5_diag -v
"""

import contextlib
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.phase5_diagnostic as D  # noqa: E402
from src.phase5 import zoo  # noqa: E402

WEIGHTS = ROOT / "models/phase5_diag_weights"


def fake_pack(d, rng):
    def img(p, mean):
        a = np.clip(rng.normal(mean, 20, (40, 56, 3)), 0, 255).astype("uint8")
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(a).save(p, quality=95)
    for s in ("1", "2"):
        for k, m in enumerate((15, 40, 90, 140, 200), start=1):
            img(d / "SICE_Part1_900" / s / f"{k}.jpg", m)
        img(d / "SICE_Part1_900" / s / "label.jpg", 120)
    for sid in ("30001", "30002"):
        for lvl, m in (("dense", 15), ("moderate", 35), ("light", 70)):
            img(d / "LoLI-Street/Val/low" / f"{lvl}_{sid}.jpg", m)
            img(d / "LoLI-Street/Val/high" / f"{lvl}_{sid}.jpg", 120)
    for n in (1, 2):
        img(d / "LoLI-Street/Test" / f"Test_{n}.jpg", 25)


class TestPhase5Diag(unittest.TestCase):
    def test_luma_and_gain(self):
        x = torch.full((1, 3, 4, 4), 0.1)
        self.assertAlmostEqual(float(zoo.luma(x).mean()), 0.1, places=6)
        y = zoo.build("gain048", None, "cpu")(x)
        self.assertAlmostEqual(float(zoo.luma(y).mean()), 0.48, places=5)

    @unittest.skipUnless(WEIGHTS.exists(), "weights folder not present")
    def test_all_weights_load_and_match_md5(self):
        self.assertEqual(zoo.check_weights(str(WEIGHTS)), {})
        x = torch.rand(1, 3, 40, 56) * 0.3
        for c in zoo.CONDITIONS:
            with torch.no_grad():
                y = zoo.build(c, str(WEIGHTS), "cpu")(x)
            self.assertEqual(tuple(y.shape), (1, 3, 40, 56))
            self.assertTrue(0 <= float(y.min()) and float(y.max()) <= 1, c)

    def test_notebook_cells_parse(self):
        import ast
        from scripts.make_phase5_diag_bundle import notebook
        nb = notebook("b.zip", "0" * 32, "w.zip", "1" * 32, "2" * 32)
        code = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
        for src in code:
            ast.parse(src)
        allcode = "\n".join(code)
        for needle in ("'T4' in gpu", "MD5 mismatch", "Part2", "constraints.txt", "a pinned core package changed"):
            self.assertIn(needle, allcode)
        for bad in (".backward(", "optimizer", "Dataset_Part2.rar"):
            self.assertNotIn(bad, allcode)

    def test_prepare_refuses_part2(self):
        r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/prepare_phase5_diag_data.py"), "--tmp", "/tmp/x",
                            "--sice-rar", "data/SICE/Dataset_Part2.rar", "--out", "/tmp/never_written_phase5"],
                           capture_output=True, text=True, cwd=ROOT)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("sealed", r.stdout + r.stderr)

    def test_no_part2_or_training_in_phase5_code(self):
        for f in ("scripts/phase5_diagnostic.py", "src/phase5/zoo.py"):
            s = (ROOT / f).read_text().lower()
            for bad in ("dataset_part2", ".backward(", "optimizer", "train()"):
                self.assertNotIn(bad, s, f)

    @unittest.skipUnless(WEIGHTS.exists(), "weights folder not present")
    def test_dry_run_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fake_pack(d / "pack", np.random.default_rng(0))
            old = dict(D.SUBSET), dict(D.PANEL_N)
            D.SUBSET.update({"sice": 4, "loli_val": 1, "loli_test": 1})
            D.PANEL_N.update({"sice": 1, "loli_val": 1, "loli_test": 1})
            try:
                a = Namespace(data=str(d / "pack"), weights=str(WEIGHTS), out=str(d / "out"), device="cpu",
                              datasets=D.DATASETS, conditions=["identity", "gain048", "hvi_off_real", "iat_lol"])
                with contextlib.redirect_stdout(io.StringIO()):
                    D.stage_run(a)
                    D.stage_run(a)                    # rerun: skips finished parts
                    D.stage_report(a)
            finally:
                D.SUBSET.clear(); D.SUBSET.update(old[0]); D.PANEL_N.clear(); D.PANEL_N.update(old[1])
            rows = list(csv.DictReader(open(d / "out/per_image/sice__identity.csv")))
            self.assertEqual(len(rows), 10)
            ident = {r["id"]: float(r["psnr"]) for r in rows}
            gain = {r["id"]: float(r["psnr"]) for r in csv.DictReader(open(d / "out/per_image/sice__gain048.csv"))}
            self.assertEqual(set(ident), set(gain))
            s = json.load(open(d / "out/diag_summary.json"))
            self.assertIn("loli_val|light|hvi_off_real", s["tables"])
            self.assertIn("sice|all|iat_lol", s["tables"])
            self.assertEqual(s["tables"]["loli_val|all|identity"]["harm"], 0.0)   # identity never harms itself
            self.assertTrue((d / "out/DIAG_REPORT.md").exists() and (d / "out/plot_sice.png").exists())
            self.assertTrue(any((d / "out/panels/loli_val").glob("*.jpg")))
            lt = list(csv.DictReader(open(d / "out/per_image/loli_test__hvi_off_real.csv")))
            self.assertEqual(len(lt), 2)
            self.assertNotIn("psnr", lt[0])                                         # no reference -> no PSNR


if __name__ == "__main__":
    unittest.main()
