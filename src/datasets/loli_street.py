"""LoLI-Street official Val set as an UNSEEN-domain paired evaluation set.

Layout (prepared copy)::

    <root>/Val/low/{dense,moderate,light}_<id>.jpg  <->  <root>/Val/high/<same name>

- ids 30001..31000; id 30727 is excluded (its low and high images are entirely black).
- The three darkness levels of an id share one reference image; items carry
  "level" and "scene_id" so results can be reported per level and grouped by id.
- The unpaired Test set and the YOLO annotations are not used in Phase 2.
"""

import os
import re
from typing import Callable, Dict, List, Optional, Sequence

from .lol_v1 import list_images
from .paired_list import PairedListDataset

LEVELS = ("dense", "moderate", "light")
INVALID_VAL_IDS = frozenset({30727})


def loli_street_val_pairs(root: str, levels: Sequence[str] = LEVELS) -> List[Dict]:
    low_dir, high_dir = os.path.join(root, "Val", "low"), os.path.join(root, "Val", "high")
    lows, highs = list_images(low_dir), set(list_images(high_dir))
    if set(lows) != highs:
        raise ValueError(f"LoLI-Street Val low/high mismatch: {sorted(set(lows) ^ highs)[:10]}")
    pairs = []
    for n in lows:
        m = re.fullmatch(r"(dense|moderate|light)_(\d+)\.jpg", n)
        if not m:
            raise ValueError(f"Unexpected LoLI-Street filename: {n}")
        lvl, sid = m.group(1), int(m.group(2))
        if sid in INVALID_VAL_IDS:
            raise ValueError(f"Invalid LoLI-Street id {sid} present in prepared data: {n}")
        if lvl in levels:
            pairs.append({"low": os.path.join(low_dir, n), "high": os.path.join(high_dir, n),
                          "name": n, "level": lvl, "scene_id": sid, "domain": "loli_street"})
    pairs.sort(key=lambda p: (p["scene_id"], LEVELS.index(p["level"])))
    return pairs


class LoLIStreetValDataset(PairedListDataset):
    def __init__(self, root: str, levels: Sequence[str] = LEVELS, transform: Optional[Callable] = None):
        super().__init__(loli_street_val_pairs(root, levels), transform)
