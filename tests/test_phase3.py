"""Phase 3 / Stage 1 (pilot preparation) tests. CPU only, no downloads (VGG-19 built with random weights).

Run from the project root:
    .venv/bin/python -m unittest tests.test_phase3 -v
"""

import difflib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["PHASE3_VGG_RANDOM_INIT"] = "1"   # tests never download ImageNet weights
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch  # noqa: E402
import yaml  # noqa: E402

from src.phase3.engine import LOSS_OF, Engine, state_hash  # noqa: E402
from src.phase3.losses import build_phase3_loss  # noqa: E402
from src.phase3.models import EXPECTED_PARAMS, FreMLP, build_phase3_model, count_params  # noqa: E402
from src.phase3.schedules import RECIPES, lr_at  # noqa: E402
from src.utils import load_checkpoint, load_config, save_checkpoint  # noqa: E402

MODELS = ["unet_expaug", "retinexformer", "hvi_cidnet", "darkir_m"]
SRC = PROJECT_ROOT / "third_party_src"
PAIRS = {  # vendored file -> pinned upstream file
    "retinexformer/RetinexFormer_arch.py": "Retinexformer/basicsr/models/archs/RetinexFormer_arch.py",
    **{f"hvi_cidnet/{f}": f"HVI-CIDNet/{f}" for f in ("net/CIDNet.py", "net/HVI_transform.py", "net/LCA.py",
                                                       "net/transformer_utils.py", "loss/losses.py",
                                                       "loss/loss_utils.py", "loss/vgg_arch.py")},
    **{f"darkir/{f}": f"DarkIR/{f}" for f in ("archs/DarkIR.py", "archs/arch_model.py", "archs/arch_util.py",
                                               "losses/loss.py", "losses/loss_utils.py")},
}
# The upstream clones (with .git) are local-only; the public repository ships only third_party_src/IAT and the
# download manifest. Vendoring tests therefore run only where the pinned clones are present.
HAVE_CLONES = all((SRC / r / ".git").exists() for r in ("Retinexformer", "HVI-CIDNet", "DarkIR"))
NO_CLONES = ("upstream clones not present (third_party_src/{Retinexformer,HVI-CIDNet,DarkIR} with .git); "
             "fetch them at the pinned commits listed in third_party_src/DOWNLOAD_MANIFEST.md to run this test")
HAVE_DATA = (PROJECT_ROOT / "data_prepared/phase2/manifest.json").is_file()
NO_DATA = "prepared data not present (data_prepared/phase2); run scripts/prepare_phase2_data.py (docs/DATA.md)"
PINNED = {"Retinexformer": "1e9a0efce4b306b6701b824768370ff26066c32a",
          "HVI-CIDNet": "eb43d7d91e9a336c66856824ff9e4603ae41f408",
          "DarkIR": "0910291157ad04f210481a002cf6bb480f6a7f18"}


def fake_batch(h=64, w=64, k=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    low = torch.rand(2 * k, 3, h, w, generator=g) * 0.3
    high = (low * 2.5).clamp(0, 1)
    return {"low": low, "high": high, "domain": ["lol_v2_real"] * k + ["lol_v2_synthetic"] * k,
            "name": [f"x{i}" for i in range(2 * k)]}


class TestVendoring(unittest.TestCase):
    @unittest.skipUnless(HAVE_CLONES, NO_CLONES)
    def test_pinned_commits(self):
        for repo, sha in PINNED.items():
            head = subprocess.run(["git", "-C", str(SRC / repo), "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip()
            self.assertEqual(head, sha, repo)

    @unittest.skipUnless(HAVE_CLONES, NO_CLONES)
    def test_only_recorded_patches(self):
        """Current vendored files differ from upstream exactly by the saved PATCHES.diff (no unrecorded edits)."""
        for pkg in ("retinexformer", "hvi_cidnet", "darkir"):
            files = [(v, u) for v, u in PAIRS.items() if v.startswith(pkg + "/")]
            cur = []
            for v, u in files:
                a = (SRC / u).read_text().splitlines(keepends=True)
                b = (PROJECT_ROOT / "third_party" / v).read_text().splitlines(keepends=True)
                cur += [ln for ln in difflib.unified_diff(a, b, n=0) if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
            saved = [ln + "\n" for ln in (PROJECT_ROOT / "third_party" / pkg / "PATCHES.diff").read_text().splitlines()
                     if ln[:1] in "+-" and ln[:3] not in ("+++", "---")]
            self.assertEqual([c.rstrip("\n") for c in cur], [s.rstrip("\n") for s in saved], pkg)
            for ln in cur:  # every patch is either an import/device/API change carrying the marker, or inside a marked block
                if ln.startswith("+") and "[phase3 patch]" not in ln and not ln.startswith("+from .") \
                        and not ln.lstrip("+").lstrip().startswith("#"):
                    self.assertTrue(any(t in ln for t in ("os.environ", "vgg_net = ", "elif vgg_type", "else:",
                                                          "raise ValueError", "_w = None", "weights=_w")), ln)

    @unittest.skipUnless(HAVE_CLONES, NO_CLONES)
    def test_retinexformer_is_byte_identical(self):
        if SRC.is_dir():
            self.assertEqual((SRC / PAIRS["retinexformer/RetinexFormer_arch.py"]).read_bytes(),
                             (PROJECT_ROOT / "third_party/retinexformer/RetinexFormer_arch.py").read_bytes())

    def test_licences_present(self):
        for f in ("retinexformer/LICENSE.txt", "hvi_cidnet/LICENSE", "darkir/LICENSE"):
            self.assertIn("MIT License", (PROJECT_ROOT / "third_party" / f).read_text())
        self.assertIn("Apache License", (PROJECT_ROOT / "third_party/iat/LICENSE").read_text())


class TestModels(unittest.TestCase):
    def test_param_counts_and_paper_convention(self):
        for m in MODELS + ["unet"]:
            n = count_params(build_phase3_model(m))
            e = EXPECTED_PARAMS[m]
            self.assertEqual(n, e["count"], m)
            self.assertLess(abs(n / e["unit"] - e["paper_value"]) / e["paper_value"], 0.01, m)

    def test_eval_output_range_and_odd_sizes(self):
        for m in MODELS:
            torch.manual_seed(0)
            net = build_phase3_model(m).eval()
            for h, w in ((64, 64), (37, 53), (400, 600)):
                with torch.no_grad():
                    y = net(torch.rand(1, 3, h, w))
                self.assertEqual(tuple(y.shape), (1, 3, h, w), (m, h, w))
                self.assertTrue(torch.isfinite(y).all() and y.min() >= 0 and y.max() <= 1, (m, h, w))

    def test_eval_deterministic(self):
        for m in MODELS:
            torch.manual_seed(0)
            net = build_phase3_model(m).eval()
            x = torch.rand(1, 3, 64, 96)
            with torch.no_grad():
                self.assertTrue(torch.equal(net(x), net(x)), m)

    def test_fp32_islands_under_autocast(self):
        cid = build_phase3_model("hvi_cidnet")
        dk = build_phase3_model("darkir_m")
        fm = next(m for m in dk.modules() if isinstance(m, FreMLP))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertEqual(cid.hvi(torch.rand(1, 3, 16, 16)).dtype, torch.float32)
            o = fm(torch.rand(1, fm.process1[0].in_channels, 8, 8).to(torch.bfloat16))
            self.assertEqual(o.dtype, torch.float32)

    def test_rgb_channel_semantics_hvi(self):
        """Pure red/green/blue map to the expected HSV hues inside the HVI transform (channel order = RGB)."""
        cid = build_phase3_model("hvi_cidnet")
        x = torch.zeros(1, 3, 1, 3)
        x[0, 0, 0, 0] = x[0, 1, 0, 1] = x[0, 2, 0, 2] = 1.0
        hvi = cid.hvi(x)[0, :, 0]                     # H = cos(2*pi*hue)*..., V = sin(2*pi*hue)*...
        hue = torch.atan2(hvi[1], hvi[0]) / (2 * torch.pi) % 1
        torch.testing.assert_close(hue, torch.tensor([0.0, 1 / 3, 2 / 3]), atol=1e-4, rtol=0)


class TestLossesAndTraining(unittest.TestCase):
    def test_forward_backward_all_models(self):
        cfg = load_config("configs/generalized_lol_v2.yaml")
        for m in MODELS:
            torch.manual_seed(0)
            eng = Engine(cfg, m, torch.device("cpu"), amp=False, total_steps=5600, build_data=False)
            r = eng.train_step(fake_batch(), 0)
            self.assertTrue(torch.isfinite(torch.tensor(r["loss"])), m)
            # step 2: DarkIR (NAFNet-style) initialises its residual scales beta/gamma to 0, so at step 1 only those
            # scales receive gradients; from step 2 on every block must receive gradients
            r = eng.train_step(fake_batch(seed=1), 1)
            dead = {n.split(".")[1] for n, p in eng.model.named_parameters()
                    if p.grad is None or not bool(p.grad.abs().sum() > 0)}
            # Upstream HVI-CIDNet computes I_LCA5 and then overwrites its output (i_dec2 = ID_block2(...)), so that
            # block never receives gradients in the official model. Kept as published; every other block must train.
            self.assertEqual(dead, {"I_LCA5"} if m == "hvi_cidnet" else set(), m)
            if m == "darkir_m":
                self.assertEqual(set(r["parts"]), {"l1", "perc", "edge", "enhance"})
            if m == "hvi_cidnet":
                self.assertEqual(len(r["parts"]), 8)

    def test_loss_zero_when_prediction_equals_target(self):
        for name in ("l1", "cidnet_official"):
            obj = build_phase3_loss(name)
            x = torch.rand(2, 3, 32, 32)
            model = build_phase3_model("hvi_cidnet")
            loss, _ = obj(x, x.clone(), model=model)
            self.assertLess(float(loss), 1e-5, name)

    def test_batch_composition_enforced(self):
        cfg = load_config("configs/generalized_lol_v2.yaml")
        eng = Engine(cfg, "retinexformer", torch.device("cpu"), amp=False, total_steps=10, build_data=False)
        b = fake_batch()
        b["domain"] = ["lol_v2_real"] * 4
        with self.assertRaises(RuntimeError):
            eng.train_step(b, 0)

    def test_schedules(self):
        T = 5600
        self.assertEqual(lr_at("unet", 0, T), 2e-4)
        self.assertEqual(lr_at("unet", T - 1, T), 2e-4)
        self.assertAlmostEqual(lr_at("darkir_m", 0, T), 5e-4)
        self.assertAlmostEqual(lr_at("darkir_m", T, T), 1e-6)
        self.assertEqual(lr_at("hvi_cidnet", 0, T), 0.0)                       # warm-up from 0
        w = round(3 / 1000 * T)
        self.assertAlmostEqual(lr_at("hvi_cidnet", w, T), 1e-4)
        self.assertAlmostEqual(lr_at("hvi_cidnet", T, T), 1e-7)
        p1 = round(46000 / 150000 * T)
        self.assertAlmostEqual(lr_at("retinexformer", 0, T), 2e-4)
        self.assertAlmostEqual(lr_at("retinexformer", p1, T), 3e-4)            # upstream eta_min of cycle 1
        self.assertAlmostEqual(lr_at("retinexformer", T, T), 1e-6, places=9)
        self.assertEqual(RECIPES["retinexformer"]["grad_clip"], 0.01)

    def test_checkpoint_round_trip(self):
        cfg = load_config("configs/generalized_lol_v2.yaml")
        for m in MODELS:
            torch.manual_seed(0)
            a = Engine(cfg, m, torch.device("cpu"), amp=False, total_steps=10, build_data=False)
            a.train_step(fake_batch(), 0)
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "c.pth"
                save_checkpoint(p, a.model, cfg, **a.state(), epoch=1)
                torch.manual_seed(123)
                b = Engine(cfg, m, torch.device("cpu"), amp=False, total_steps=10, build_data=False)
                # loss networks (frozen VGG-19) are not part of the checkpoint: ImageNet weights in real runs;
                # in tests they are random, so copy them to compare like with like
                b.objective.load_state_dict(a.objective.state_dict())
                b.load_state(load_checkpoint(p))
            self.assertEqual(state_hash(a.model), state_hash(b.model), m)
            x = torch.rand(1, 3, 48, 64)
            a.model.eval(), b.model.eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(a.model(x), b.model(x)), m)
            ra, rb = a.train_step(fake_batch(seed=1), 1), b.train_step(fake_batch(seed=1), 1)
            self.assertEqual(ra["loss"], rb["loss"], m)                       # optimizer state restored too


class TestProtocolAndTrainer(unittest.TestCase):
    def test_configs_keep_phase2_protocol(self):
        base = yaml.safe_load(open("configs/generalized_lol_v2.yaml"))
        for m in MODELS:
            c = yaml.safe_load(open(f"configs/phase3/tierA_{m}.yaml"))
            for k in ("seed", "splits", "sampler", "eval"):
                self.assertEqual(c[k], base[k], (m, k))
            diff = {k for k in set(c["data"]) | set(base["data"]) if c["data"].get(k) != base["data"].get(k)}
            self.assertEqual(diff, {"input_exposure_aug"} if m == "unet_expaug" else set(), m)
            for k in ("epochs", "batch_size", "deterministic", "num_workers"):
                self.assertEqual(c["train"][k], base["train"][k], (m, k))
            self.assertEqual(c["model"]["name"], m)
            self.assertEqual(c["loss"]["name"], LOSS_OF[m])
            self.assertEqual(c["train"]["amp"], "pilot_gated")
            self.assertTrue(c["output"]["dir"].startswith("experiments/phase3_model_selection/stage1/runs/"))

    def _train(self, run_dir, *extra):
        cmd = [sys.executable, "scripts/train_phase3.py", "--config", "configs/phase3/tierA_unet_expaug.yaml",
               "--run-dir", str(run_dir), "--device", "cpu", "--amp", "off", "--epochs", "2",
               "--steps-per-epoch", "3", "--no-val", "--skip-integrity-md5", "--num-workers", "0", *extra]
        return subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

    @unittest.skipUnless(HAVE_DATA, NO_DATA)   # the trainer checks the prepared-data manifest before any step
    def test_resume_is_exact_and_guards_hold(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "A", Path(d) / "B"
            pa = self._train(a)
            self.assertEqual(pa.returncode, 0, pa.stderr[-3000:])
            pb = self._train(b, "--stop-after-epoch", "1")
            self.assertEqual(pb.returncode, 0, pb.stderr[-3000:])
            self.assertEqual(len(json.load(open(b / "history.json"))), 1)
            again = self._train(b)                                              # no --resume: must refuse
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("refusing to overwrite", again.stdout + again.stderr)
            pr = self._train(b, "--resume", str(b / "checkpoints/last.pth"))
            self.assertEqual(pr.returncode, 0, pr.stderr[-3000:])
            ha, hb = json.load(open(a / "history.json")), json.load(open(b / "history.json"))
            self.assertEqual([h["batch_names_md5"] for h in ha], [h["batch_names_md5"] for h in hb])
            self.assertEqual([h["train_loss"] for h in ha], [h["train_loss"] for h in hb])
            sa = json.load(open(a / "logs/train_summary.json"))
            sb = json.load(open(b / "logs/train_summary.json"))
            self.assertEqual(sa["final_state_md5"], sb["final_state_md5"])      # bitwise on CPU
            self.assertTrue(sa["pipeline_test_mode"])

    def test_trainer_refuses_protected_dirs_and_unset_amp(self):
        p = subprocess.run([sys.executable, "scripts/train_phase3.py", "--config", "configs/phase3/tierA_retinexformer.yaml",
                            "--run-dir", "experiments/generalized_lol_v2/x", "--device", "cpu", "--amp", "off"],
                           capture_output=True, text=True, cwd=PROJECT_ROOT)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("refusing to write inside", p.stdout + p.stderr)
        self.assertFalse((PROJECT_ROOT / "experiments/generalized_lol_v2/x").exists())
        p = subprocess.run([sys.executable, "scripts/train_phase3.py", "--config", "configs/phase3/tierA_retinexformer.yaml",
                            "--run-dir", "/tmp/never", "--device", "cpu"], capture_output=True, text=True, cwd=PROJECT_ROOT)
        self.assertNotEqual(p.returncode, 0)                                  # --amp is mandatory


if __name__ == "__main__":
    unittest.main()
