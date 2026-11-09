"""LOL-v2 (Real_captured / Synthetic) paired datasets.

Layout (prepared copy, see scripts/prepare_phase2_data.py)::

    <root>/Real_captured/{Train,Test}/Low/lowNNNNN.png  <->  .../Normal/normalNNNNN.png
    <root>/Synthetic/{Train,Test}/Low/<name>.png         <->  .../Normal/<name>.png

Official Train/Test are used as-is. A scene-disjoint validation subset is carved
from Train using a split file (scripts/make_phase2_splits.py); pass ``names`` to
restrict a dataset to the train or val part of that split.
"""

import os
import re
from typing import Callable, Dict, List, Optional, Sequence

from .lol_v1 import list_images
from .paired_list import PairedListDataset

SUBSETS = {"real": "Real_captured", "synthetic": "Synthetic"}


def _normal_name(subset: str, low_name: str) -> str:
    if subset == "real":
        m = re.fullmatch(r"low(\d+)\.png", low_name)
        if not m:
            raise ValueError(f"Unexpected LOL-v2-real low filename: {low_name}")
        return f"normal{m.group(1)}.png"
    return low_name


def lol_v2_pairs(root: str, subset: str, split: str) -> List[Dict]:
    """All pairs of an official split; raises on any orphan/mismatch."""
    if subset not in SUBSETS or split not in ("Train", "Test"):
        raise ValueError(f"subset={subset!r}, split={split!r}")
    base = os.path.join(root, SUBSETS[subset], split)
    lows = list_images(os.path.join(base, "Low"))
    highs = set(list_images(os.path.join(base, "Normal")))
    pairs, expected = [], set()
    for n in lows:
        h = _normal_name(subset, n)
        expected.add(h)
        if h not in highs:
            raise ValueError(f"Missing reference for {subset}/{split}/Low/{n}")
        pairs.append({"low": os.path.join(base, "Low", n), "high": os.path.join(base, "Normal", h),
                      "name": n, "domain": f"lol_v2_{subset}"})
    orphans = highs - expected
    if orphans:
        raise ValueError(f"Orphan references in {subset}/{split}/Normal: {sorted(orphans)[:10]}")
    return pairs


class LOLv2Dataset(PairedListDataset):
    """LOL-v2 pairs of one subset/split, optionally restricted to ``names`` (low filenames)."""

    def __init__(self, root: str, subset: str, split: str, names: Optional[Sequence[str]] = None,
                 transform: Optional[Callable] = None):
        pairs = lol_v2_pairs(root, subset, split)
        if names is not None:
            by_name = {p["name"]: p for p in pairs}
            missing = [n for n in names if n not in by_name]
            if missing:
                raise ValueError(f"Split names not found in {subset}/{split}: {missing[:10]}")
            pairs = [by_name[n] for n in names]
        super().__init__(pairs, transform)
        self.subset, self.split = subset, split
