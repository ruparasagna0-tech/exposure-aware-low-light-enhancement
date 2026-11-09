"""LOL-v1 paired low-light dataset.

Expected layout (read-only, never modified)::

    <root>/
    ├── our485/{low,high}/   485 pairs -> train (450) + val (35)
    └── eval15/{low,high}/    15 pairs -> test

The train/val split is logical: a seeded shuffle of the sorted our485
filenames, so the same seed always yields the same 450/35 partition and no
files are copied or moved.
"""

import os
import random
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

from .paired_transforms import PairedToTensor

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

SPLIT_DIRS = {"train": "our485", "val": "our485", "test": "eval15"}

DEFAULT_VAL_SIZE = 35
DEFAULT_SEED = 42


def _natural_key(name: str):
    """Sort '2.png' before '10.png'."""
    stem = os.path.splitext(name)[0]
    return (0, int(stem), name) if stem.isdigit() else (1, 0, name)


def list_images(folder: str) -> List[str]:
    """Image filenames in ``folder`` (ignores .DS_Store, hidden and non-image files)."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    names = [
        f
        for f in os.listdir(folder)
        if not f.startswith(".")
        and f.lower().endswith(IMG_EXTENSIONS)
        and os.path.isfile(os.path.join(folder, f))
    ]
    return sorted(names, key=_natural_key)


def find_pairs(split_dir: str) -> List[str]:
    """Filenames present in both ``low/`` and ``high/``; raises on any mismatch."""
    low = list_images(os.path.join(split_dir, "low"))
    high = list_images(os.path.join(split_dir, "high"))
    only_low = sorted(set(low) - set(high))
    only_high = sorted(set(high) - set(low))
    if only_low or only_high:
        raise ValueError(
            f"Unpaired images in {split_dir}: "
            f"only in low={only_low[:10]}, only in high={only_high[:10]}"
        )
    if not low:
        raise ValueError(f"No images found in {split_dir}")
    return low


def train_val_indices(
    num_items: int, val_size: int = DEFAULT_VAL_SIZE, seed: int = DEFAULT_SEED
) -> Tuple[List[int], List[int]]:
    """Deterministic train/val index split.

    Uses a local ``random.Random(seed)`` so the result does not depend on (or
    disturb) any global RNG state.
    """
    if not 0 < val_size < num_items:
        raise ValueError(f"val_size={val_size} invalid for {num_items} items")
    indices = list(range(num_items))
    random.Random(seed).shuffle(indices)
    return sorted(indices[val_size:]), sorted(indices[:val_size])


class LOLv1Dataset(Dataset):
    """LOL-v1 low/high pairs.

    Args:
        root: path to ``LOLdataset/`` (contains ``our485`` and ``eval15``).
        split: ``"train"``, ``"val"`` or ``"test"``.
        transform: paired transform ``(low, high) -> (low, high)`` on PIL
            images; defaults to ``PairedToTensor``.
        val_size: number of our485 pairs held out for validation.
        seed: seed for the train/val split.

    Each item is ``{"low": Tensor[3,H,W], "high": Tensor[3,H,W], "name": str}``.
    Images are loaded lazily in ``__getitem__``.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        val_size: int = DEFAULT_VAL_SIZE,
        seed: int = DEFAULT_SEED,
    ):
        if split not in SPLIT_DIRS:
            raise ValueError(f"split must be one of {list(SPLIT_DIRS)}, got '{split}'")
        self.root = os.path.expanduser(root)
        self.split = split
        self.split_dir = os.path.join(self.root, SPLIT_DIRS[split])
        self.low_dir = os.path.join(self.split_dir, "low")
        self.high_dir = os.path.join(self.split_dir, "high")
        self.transform = transform if transform is not None else PairedToTensor()

        names = find_pairs(self.split_dir)
        if split in ("train", "val"):
            train_idx, val_idx = train_val_indices(len(names), val_size, seed)
            self.indices = train_idx if split == "train" else val_idx
            names = [names[i] for i in self.indices]
        else:
            self.indices = list(range(len(names)))
        self.names = names

    def __len__(self) -> int:
        return len(self.names)

    @staticmethod
    def _load_rgb(path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def get_paths(self, idx: int) -> Tuple[str, str]:
        name = self.names[idx]
        return os.path.join(self.low_dir, name), os.path.join(self.high_dir, name)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        low_path, high_path = self.get_paths(idx)
        low = self._load_rgb(low_path)
        high = self._load_rgb(high_path)
        low, high = self.transform(low, high)
        return {"low": low, "high": high, "name": self.names[idx]}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(split={self.split!r}, n={len(self)}, "
            f"dir={self.split_dir!r}, transform={self.transform!r})"
        )
