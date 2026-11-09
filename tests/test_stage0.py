"""Phase 3 / Stage 0 tests (standard-library unittest; pytest is not installed).

Run from the project root:
    .venv/bin/python -m unittest tests.test_stage0 -v

Groups
  colour metrics      CIEDE2000 vs published test pairs; Lab anchors; identity; channel-swap sensitivity
  image I/O           RGB channel order through save -> project loader; value range; quantization guard
  checkpoints         save -> load round trip (synthetic); the imported Phase-2 checkpoint (skipped if absent)
  determinism         repeated full-image and tiled inference are bit-identical
  exported outputs    file <-> dataset item correspondence, MD5s, sizes, ranges, re-inference = saved PNG
                      (skipped until exported)
  integrity           references/inputs match the prepared-data MD5 manifest; existing assets unchanged
"""

import csv
import hashlib
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from scripts.export_outputs import output_stem, save_png, tensor_to_uint8  # noqa: E402
from src.datasets.paired_list import load_rgb  # noqa: E402
from src.datasets.paired_transforms import PairedToTensor  # noqa: E402
from src.metrics.colour_metrics import (SHARMA_TEST_PAIRS, ciede2000, pointwise_metrics, srgb_to_lab,  # noqa: E402
                                        structure_metrics)
from src.metrics.evaluation import quantize_uint8  # noqa: E402
from src.models import BaselineUNet  # noqa: E402
from src.utils import load_config, load_model_from_checkpoint, save_checkpoint  # noqa: E402
from src.utils.tiling import tiled_forward  # noqa: E402

STAGE0 = PROJECT_ROOT / "experiments/phase3_model_selection/stage0"
CKPT = STAGE0 / "source_phase2/checkpoints/best.pth"
EXPORT = STAGE0 / "outputs/phase2_unet_best"
DATA = PROJECT_ROOT / "data_prepared/phase2"
HAVE_DATA = (DATA / "manifest.json").is_file()
NO_DATA = "prepared data not present (data_prepared/phase2); run scripts/prepare_phase2_data.py (docs/DATA.md)"
SNAPSHOT = PROJECT_ROOT / "experiments/generalized_lol_v2_expaug/logs/pre_expaug_file_md5.json"
NO_SNAPSHOT = ("local integrity snapshot not present (experiments/generalized_lol_v2_expaug/logs/pre_expaug_file_md5.json); "
               "it records MD5s of the authors' working copy and is not part of the public repository")


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rgb_patches(h=24, w=36) -> np.ndarray:
    """Known image: red | green | blue thirds, plus a white and a black pixel."""
    x = np.zeros((h, w, 3), np.uint8)
    x[:, : w // 3, 0] = 255
    x[:, w // 3: 2 * w // 3, 1] = 255
    x[:, 2 * w // 3:, 2] = 255
    x[0, 0] = (255, 255, 255)
    x[-1, -1] = (0, 0, 0)
    return x


class TestColourMetrics(unittest.TestCase):
    def test_ciede2000_published_pairs(self):
        for a, b, expected in SHARMA_TEST_PAIRS:
            self.assertAlmostEqual(float(ciede2000(np.array(a), np.array(b))), expected, places=4)
            self.assertAlmostEqual(float(ciede2000(np.array(b), np.array(a))), expected, places=4)  # symmetric

    def test_lab_anchors(self):
        lab = srgb_to_lab(np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]], np.uint8))
        np.testing.assert_allclose(lab[0], [100, 0, 0], atol=0.01)
        np.testing.assert_allclose(lab[1], [0, 0, 0], atol=0.01)
        np.testing.assert_allclose(lab[2], [53.24, 80.09, 67.20], atol=0.05)  # standard sRGB red

    @unittest.skipUnless(HAVE_DATA, NO_DATA)   # uses real LoLI-Street images
    def test_identical_images(self):
        x = np.asarray(Image.open(DATA / "LoLI-Street/Val/high/dense_30001.jpg").convert("RGB"))
        m = pointwise_metrics(x, x, strip=100)
        self.assertEqual(m["de00_mean"], 0.0)
        self.assertAlmostEqual(m["chroma_ratio"], 1.0)
        self.assertEqual(m["dL_mean"], 0.0)
        s = structure_metrics(x, x)
        self.assertAlmostEqual(s["edge_corr"], 1.0)
        self.assertAlmostEqual(s["flat_noise_ratio"], 1.0)

    @unittest.skipUnless(HAVE_DATA, NO_DATA)
    def test_strips_equal_whole(self):
        x = np.asarray(Image.open(DATA / "LoLI-Street/Val/low/light_30001.jpg").convert("RGB"))
        y = np.asarray(Image.open(DATA / "LoLI-Street/Val/high/light_30001.jpg").convert("RGB"))
        a, b = pointwise_metrics(x, y, strip=37), pointwise_metrics(x, y, strip=10_000)
        for k in ("de00_mean", "de00_leq_mean", "dL_mean", "chroma_ratio", "lstd_out"):
            self.assertAlmostEqual(a[k], b[k], places=9)

    @unittest.skipUnless(HAVE_DATA, NO_DATA)
    def test_channel_swap_is_detected(self):
        y = np.asarray(Image.open(DATA / "LOL-v2/Synthetic/Test/Normal/r082626aft.png").convert("RGB"))
        m = pointwise_metrics(y[..., ::-1].copy(), y)  # RGB <-> BGR
        self.assertGreater(m["de00_leq_mean"], 5.0)


class TestImageIO(unittest.TestCase):
    def test_rgb_order_through_save_and_project_loader(self):
        x = rgb_patches()
        t = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0
        arr = tensor_to_uint8(quantize_uint8(t))
        np.testing.assert_array_equal(arr, x)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.png"
            save_png(arr, p)
            back = np.asarray(load_rgb(str(p)))
            np.testing.assert_array_equal(back, x)
            self.assertEqual(tuple(back[5, 0]), (255, 0, 0))     # red stays in channel 0
            self.assertEqual(tuple(back[5, -1]), (0, 0, 255))    # blue stays in channel 2
            tt, _ = PairedToTensor()(load_rgb(str(p)), load_rgb(str(p)))
            self.assertTrue(torch.equal(tt, t))                   # loader -> identical [0,1] tensor
            with self.assertRaises(FileExistsError):              # never overwrites
                save_png(arr, p)

    def test_range_and_quantization_guards(self):
        with self.assertRaises(ValueError):
            tensor_to_uint8(torch.full((3, 4, 4), 1.5))
        with self.assertRaises(ValueError):
            tensor_to_uint8(torch.full((3, 4, 4), 0.123456))     # not 8-bit quantized
        a = tensor_to_uint8(quantize_uint8(torch.rand(3, 9, 7)))
        self.assertEqual(a.dtype, np.uint8)
        self.assertEqual(a.shape, (9, 7, 3))

    def test_output_names(self):
        self.assertEqual(output_stem("dense_30001.jpg"), "dense_30001.png")
        self.assertEqual(output_stem("seq46"), "seq46.png")
        self.assertEqual(output_stem("low00730.png"), "low00730.png")


class TestCheckpointsAndDeterminism(unittest.TestCase):
    def test_synthetic_checkpoint_round_trip(self):
        torch.manual_seed(0)
        m = BaselineUNet().eval()
        x = torch.rand(1, 3, 64, 96)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.pth"
            save_checkpoint(p, m, {"model": {"name": "baseline_unet", "in_channels": 3, "out_channels": 3,
                                             "base_channels": 32, "depth": 3}}, epoch=1)
            m2, ck = load_model_from_checkpoint(p)
        with torch.no_grad():
            self.assertTrue(torch.equal(m(x), m2(x)))
        self.assertEqual(ck["epoch"], 1)

    def test_deterministic_inference(self):
        torch.manual_seed(0)
        m = BaselineUNet().eval()
        x = torch.rand(1, 3, 400, 600)
        with torch.no_grad():
            self.assertTrue(torch.equal(m(x), m(x)))
        big = torch.rand(1, 3, 700, 1100)
        a = tiled_forward(m, big, tile=512, overlap=64)
        b = tiled_forward(m, big, tile=512, overlap=64)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(0.0 <= float(a.min()) and float(a.max()) <= 1.0)

    @unittest.skipUnless(CKPT.is_file(), "Phase-2 best.pth not imported yet (scripts/stage0_import_phase2.py)")
    def test_imported_phase2_checkpoint(self):
        prov = json.load(open(CKPT.parents[1] / "provenance.json"))
        self.assertEqual(md5(CKPT), prov["files"]["checkpoints/best.pth"]["md5"])
        self.assertTrue(prov["all_checks_passed"], prov["checks"])
        m, ck = load_model_from_checkpoint(CKPT)
        self.assertEqual(sum(p.numel() for p in m.parameters()), 1_925_667)
        self.assertEqual(ck["epoch"], 45)
        self.assertEqual(ck["config"]["model"], load_config("configs/generalized_lol_v2.yaml")["model"])
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            y1, y2 = m(x), m(x)
        self.assertTrue(torch.equal(y1, y2))
        self.assertTrue(torch.isfinite(y1).all() and 0 <= float(y1.min()) and float(y1.max()) <= 1)


@unittest.skipUnless((EXPORT / "export_manifest.json").is_file(), "outputs not exported yet (scripts/export_outputs.py)")
class TestExportedOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.datasets.phase2 import build_test
        cls.man = json.load(open(EXPORT / "export_manifest.json"))
        cls.ds = build_test(load_config("configs/generalized_lol_v2.yaml"))

    def test_correspondence_md5_and_counts(self):
        for d, info in self.man["domains"].items():
            files = info["files"]
            on_disk = sorted(os.listdir(EXPORT / d))
            self.assertEqual(on_disk, sorted(files), d)                    # no extra / missing files
            items = [v["item"] for v in files.values()]
            self.assertEqual(len(items), len(set(items)), d)               # one file per item
            names = self.ds[d].names
            if self.man.get("limit") is None:
                self.assertEqual(sorted(items), sorted(names), d)          # every dataset item exported
            for fn, v in files.items():
                self.assertEqual(fn, output_stem(v["item"]))
            rows = list(csv.DictReader(open(EXPORT / "per_image" / f"{d}.csv")))
            self.assertEqual(sorted(r["name"] for r in rows), sorted(items), d)
            for fn in list(files)[:: max(1, len(files) // 25)]:           # MD5 spot check (~25 per domain)
                self.assertEqual(md5(EXPORT / d / fn), files[fn]["md5"], fn)

    def test_sizes_and_ranges_match_references(self):
        for d, info in self.man["domains"].items():
            ds, names = self.ds[d], self.ds[d].names
            idx = {n: i for i, n in enumerate(names)}
            for fn, v in list(info["files"].items())[:3]:
                out = np.asarray(Image.open(EXPORT / d / fn))
                self.assertEqual(out.dtype, np.uint8)
                self.assertEqual(out.ndim, 3)
                self.assertEqual(out.shape[2], 3)
                ref = ds[idx[v["item"]]]["high"]
                self.assertEqual(out.shape[:2], tuple(ref.shape[1:]), fn)

    @unittest.skipUnless(CKPT.is_file(), "checkpoint not imported")
    def test_reinference_reproduces_saved_png(self):
        if self.man.get("device") != "cpu":
            self.skipTest("export ran on a non-CPU device; bit-exact CPU comparison not applicable")
        m, _ = load_model_from_checkpoint(CKPT)
        for d in ("lol_v2_real_test", "lol_v2_synthetic_test", "loli_street_val"):
            if d not in self.man["domains"]:
                continue
            ds, files = self.ds[d], self.man["domains"][d]["files"]
            for i in (0, len(ds) - 1):
                item = ds[i]
                fn = output_stem(item["name"])
                if fn not in files:
                    continue
                with torch.no_grad():
                    pred = quantize_uint8(m(item["low"][None]))[0]
                saved = np.asarray(Image.open(EXPORT / d / fn))
                np.testing.assert_array_equal(tensor_to_uint8(pred), saved, err_msg=f"{d}/{fn}")


class TestIntegrity(unittest.TestCase):
    @unittest.skipUnless(HAVE_DATA, NO_DATA)
    def test_test_set_files_match_prepared_manifest(self):
        man = json.load(open(DATA / "manifest.json"))
        expected = {r["dest"]: r["md5"] for r in man["files"]}
        rels = [k for k in expected if k.startswith(("LOL-v2/Real_captured/Test/", "LOL-v2/Synthetic/Test/",
                                                      "LoLI-Street/Val/", "SICE/Dataset_Part1/Label/"))]
        self.assertGreater(len(rels), 6000)
        bad = [k for k in rels if md5(DATA / k) != expected[k]]
        self.assertEqual(bad, [])

    @unittest.skipUnless(SNAPSHOT.is_file(), NO_SNAPSHOT)
    def test_existing_assets_unchanged(self):
        snap = json.load(open("experiments/generalized_lol_v2_expaug/logs/pre_expaug_file_md5.json"))["files"]
        allowed = {"src/datasets/phase2.py"}  # documented off-by-default switch (exposure-aug experiment)
        changed = [k for k, v in snap.items() if k not in allowed and
                   (v.get("md5") if isinstance(v, dict) else v) != (md5(k) if os.path.exists(k) else None)]
        self.assertEqual(changed, [])
        z = Path("~/Downloads/phase2_evaluation_results.zip").expanduser()
        if z.is_file():
            self.assertEqual(md5(z), "e7302a3b660629702bc19bf6af2f5e0c")
            self.assertIsNone(zipfile.ZipFile(z).testzip())


if __name__ == "__main__":
    unittest.main()
