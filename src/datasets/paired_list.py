"""Generic lazy paired dataset over an explicit list of (low_path, reference_path) pairs.

Shared by the Phase-2 dataset modules (LOL-v2, LoLI-Street). Items follow the
same convention as the LOL-v1 loader: {"low", "high", "name"} with tensors in
[0, 1], plus any extra per-pair metadata (e.g. "domain", "level").
"""

from typing import Callable, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from .paired_transforms import PairedToTensor


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


class PairedListDataset(Dataset):
    """``pairs``: list of dicts with keys "low", "high", "name" (+ optional metadata)."""

    def __init__(self, pairs: List[Dict], transform: Optional[Callable] = None):
        self.pairs = list(pairs)
        self.transform = transform if transform is not None else PairedToTensor()

    @property
    def names(self) -> List[str]:
        return [p["name"] for p in self.pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        p = self.pairs[idx]
        low, high = self.transform(load_rgb(p["low"]), load_rgb(p["high"]))
        item = {"low": low, "high": high, "name": p["name"]}
        item.update({k: v for k, v in p.items() if k not in ("low", "high", "name")})
        return item
