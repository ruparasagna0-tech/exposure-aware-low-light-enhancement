"""SICE official test sequences as an UNSEEN-domain evaluation set (never used for training/validation).

Prepared layout::

    <root>/testing index.txt                    defines the 58 test sequences of Dataset_Part1
    <root>/Dataset_Part1/<id>/<k>.JPG           multi-exposure sequence (k = 1..N)
    <root>/Dataset_Part1/Label/<id>.<ext>       dataset-provided reference ("Label")

Input selection (approved Phase-2 decision):
    the DARKEST exposure whose mean brightness >= MIN_BRIGHTNESS (5 on 0-255).
    Brightness = mean of the full-resolution image converted to 8-bit grayscale
    (PIL "L", ITU-R 601 luma). Exposure order is NOT taken from filenames (the SICE
    file index is not a reliable exposure order; see experiments/dataset_audit).
    For reference only, the "-1 EV" image named by testing index.txt
    (7 images -> #3, 9 images -> #4; undefined otherwise) is also recorded.

Reference: the dataset-provided Label image. It is NOT independently verified ground truth.
Labels whose size is the transpose of the sequence size are rotated (90 or 270 degrees,
chosen by structural correlation with the selected exposure); any other size mismatch is an error.
"""

import os
import re
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .paired_transforms import PairedToTensor

MIN_BRIGHTNESS = 5.0
IMG_EXT = (".jpg", ".jpeg", ".png")


def parse_test_index(text: str) -> List[int]:
    """Sequence ids listed as 'a' or 'a-b' lines in SICE 'testing index.txt'."""
    ids = []
    for line in text.splitlines():
        m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*", line)
        if m:
            a = int(m.group(1))
            ids.extend(range(a, int(m.group(2) or a) + 1))
    return ids


def read_test_ids(root: str) -> List[int]:
    with open(os.path.join(root, "testing index.txt"), encoding="utf-8", errors="replace") as f:
        return parse_test_index(f.read())


def minus_1ev_index(n_exposures: int) -> Optional[int]:
    """Index named by testing index.txt for the '-1 EV' input (reference information only)."""
    return {7: 3, 9: 4}.get(n_exposures)


def brightness(img: Image.Image) -> float:
    return float(np.asarray(img.convert("L"), dtype=np.float64).mean())


def _thumb(img: Image.Image) -> np.ndarray:
    t = np.asarray(img.convert("L").resize((96, 64), Image.Resampling.BOX), dtype=np.float64).ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) or 1.0)


def _open(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def sequence_files(root: str, seq_id: int) -> List[str]:
    d = os.path.join(root, "Dataset_Part1", str(seq_id))
    files = [f for f in os.listdir(d) if f.lower().endswith(IMG_EXT) and not f.startswith(".")]
    return sorted(files, key=lambda f: int(os.path.splitext(f)[0]))


def label_path(root: str, seq_id: int) -> str:
    d = os.path.join(root, "Dataset_Part1", "Label")
    hits = [f for f in os.listdir(d) if os.path.splitext(f)[0] == str(seq_id) and f.lower().endswith(IMG_EXT)]
    if len(hits) != 1:
        raise ValueError(f"SICE sequence {seq_id}: expected one Label file, found {hits}")
    return os.path.join(d, hits[0])


def select_sequence(root: str, seq_id: int, min_brightness: float = MIN_BRIGHTNESS) -> Dict:
    """Measure every exposure, select the input, and resolve Label orientation."""
    files = sequence_files(root, seq_id)
    if not files:
        raise ValueError(f"SICE sequence {seq_id} has no images")
    rel_dir = os.path.join("Dataset_Part1", str(seq_id))
    lum, sizes = [], set()
    for f in files:
        im = _open(os.path.join(root, rel_dir, f))
        lum.append(round(brightness(im), 3))
        sizes.add(im.size)
    if len(sizes) != 1:
        raise ValueError(f"SICE sequence {seq_id}: mixed exposure sizes {sizes}")
    size = sizes.pop()
    valid = [i for i, b in enumerate(lum) if b >= min_brightness]
    if not valid:
        raise ValueError(f"SICE sequence {seq_id}: no exposure with brightness >= {min_brightness}")
    sel = min(valid, key=lambda i: lum[i])
    sel_img = _open(os.path.join(root, rel_dir, files[sel]))

    lp = label_path(root, seq_id)
    lab = _open(lp)
    if lab.size == size:
        rotation = 0
    elif lab.size == (size[1], size[0]):
        rotation = max((90, 270), key=lambda a: float(_thumb(lab.rotate(a, expand=True)) @ _thumb(sel_img)))
    else:
        raise ValueError(f"SICE sequence {seq_id}: Label size {lab.size} incompatible with {size}")
    lab_r = lab.rotate(rotation, expand=True) if rotation else lab
    rule = minus_1ev_index(len(files))
    return {
        "seq_id": seq_id,
        "n_exposures": len(files),
        "files": files,
        "brightness": lum,
        "min_brightness": min_brightness,
        "excluded_too_dark": [files[i] for i in range(len(files)) if lum[i] < min_brightness],
        "selected_file": files[sel],
        "selected_index": int(os.path.splitext(files[sel])[0]),
        "selected_brightness": lum[sel],
        "minus_1ev_index_per_testing_index_txt": rule,
        "minus_1ev_brightness": lum[rule - 1] if rule else None,
        "size": list(size),
        "label_file": os.path.relpath(lp, root),
        "label_rotation_deg": rotation,
        "label_brightness": round(brightness(lab_r), 3),
        "label_corr_with_selected": round(float(_thumb(lab_r) @ _thumb(sel_img)), 4),
    }


class SICETestDataset(Dataset):
    """Items: {"low": selected exposure, "high": Label (orientation-corrected), "name", "seq_id", ...}.

    ``selection``: list of dicts produced by ``select_sequence`` (stored in the split
    directory as sice_test_selection.json so the choice is fixed and auditable).
    """

    def __init__(self, root: str, selection: List[Dict], transform: Optional[Callable] = None):
        self.root = root
        self.selection = list(selection)
        self.transform = transform if transform is not None else PairedToTensor()

    @property
    def names(self) -> List[str]:
        return [f"seq{s['seq_id']}" for s in self.selection]

    def __len__(self) -> int:
        return len(self.selection)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        s = self.selection[idx]
        low = _open(os.path.join(self.root, "Dataset_Part1", str(s["seq_id"]), s["selected_file"]))
        high = _open(os.path.join(self.root, s["label_file"]))
        if s["label_rotation_deg"]:
            high = high.rotate(s["label_rotation_deg"], expand=True)
        if low.size != high.size:
            raise ValueError(f"SICE {s['seq_id']}: input {low.size} vs label {high.size}")
        low_t, high_t = self.transform(low, high)
        return {"low": low_t, "high": high_t, "name": f"seq{s['seq_id']}", "seq_id": s["seq_id"],
                "exposure_file": s["selected_file"], "domain": "sice"}
