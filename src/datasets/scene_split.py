"""Scene grouping and scene-disjoint train/validation splitting (Phase 2).

Scenes are connected components of the relation "same scene", where two pairs are
linked if
  (a) their low images OR their reference images are pixel-identical (decoded RGB MD5), or
  (b) the Pearson correlation of brightness-normalised grayscale thumbnails is
      >= corr_threshold for low-vs-low OR reference-vs-reference.
Approved settings: LOL-v2-real threshold 0.85 (96x64 thumbnails),
LOL-v2-synthetic threshold 0.95 (64x64 thumbnails).  Filename ids are NOT assumed
to be scene ids.

Split: scenes are ordered by their first member (natural filename order) and
``n_val_scenes`` are drawn with random.Random(seed).sample -> deterministic.
"""

import hashlib
import random
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


def natural_key(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _signature(path: str, thumb_size: Tuple[int, int]):
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    md5 = hashlib.md5(np.asarray(rgb).tobytes()).hexdigest()
    t = np.asarray(rgb.convert("L").resize(tuple(thumb_size), Image.Resampling.BOX), dtype=np.float64).ravel()
    t -= t.mean()
    return md5, t / (np.linalg.norm(t) or 1.0)


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def group_scenes(pairs: Sequence[Dict], corr_threshold: float, thumb_size: Tuple[int, int]) -> Dict:
    """Returns {"scene_of": {name: scene_idx}, "scenes": [[names...]], "pixel_md5": {name: (low, high)}}."""
    names = [p["name"] for p in pairs]
    sig = [(_signature(p["low"], thumb_size), _signature(p["high"], thumb_size)) for p in pairs]
    n = len(pairs)
    uf = _UnionFind(n)
    for side in (0, 1):
        first = {}
        for i, s in enumerate(sig):
            md5 = s[side][0]
            if md5 in first:
                uf.union(i, first[md5])
            else:
                first[md5] = i
    L = np.stack([s[0][1] for s in sig])
    H = np.stack([s[1][1] for s in sig])
    C = np.maximum(L @ L.T, H @ H.T)
    np.fill_diagonal(C, -1.0)
    for a, b in zip(*np.where(np.triu(C >= corr_threshold, 1))):
        uf.union(int(a), int(b))
    comps: Dict[int, List[str]] = {}
    for i in range(n):
        comps.setdefault(uf.find(i), []).append(names[i])
    scenes = sorted((sorted(v, key=natural_key) for v in comps.values()), key=lambda v: natural_key(v[0]))
    scene_of = {nm: k for k, members in enumerate(scenes) for nm in members}
    return {"scene_of": scene_of, "scenes": scenes,
            "pixel_md5": {names[i]: (sig[i][0][0], sig[i][1][0]) for i in range(n)},
            "thumbs": {names[i]: (sig[i][0][1], sig[i][1][1]) for i in range(n)}}


def split_scenes(scenes: List[List[str]], n_val_scenes: int, seed: int):
    """Deterministic scene-level split -> (train_names, val_names, val_scene_indices)."""
    if not 0 < n_val_scenes < len(scenes):
        raise ValueError(f"n_val_scenes={n_val_scenes} invalid for {len(scenes)} scenes")
    val_idx = sorted(random.Random(seed).sample(range(len(scenes)), n_val_scenes))
    vs = set(val_idx)
    val = sorted((nm for k in val_idx for nm in scenes[k]), key=natural_key)
    train = sorted((nm for k, s in enumerate(scenes) if k not in vs for nm in s), key=natural_key)
    return train, val, val_idx
