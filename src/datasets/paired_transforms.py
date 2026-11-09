"""Paired transforms for low/high image pairs.

Every transform takes and returns a ``(low, high)`` tuple. Random parameters
(crop position, flip decision) are drawn ONCE per call and applied to both
images, so the pair stays pixel-aligned.

Randomness uses torch's global RNG, so it follows ``torch.manual_seed`` and is
re-seeded per DataLoader worker automatically.
"""

from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torchvision.transforms.functional as TF
from PIL import Image

Pair = Tuple[Union[Image.Image, torch.Tensor], Union[Image.Image, torch.Tensor]]


def _image_size(img: Union[Image.Image, torch.Tensor]) -> Tuple[int, int]:
    """Return (height, width) for a PIL image or a CxHxW tensor."""
    width, height = TF.get_image_size(img)
    return height, width


class PairedCompose:
    """Apply a sequence of paired transforms in order."""

    def __init__(self, transforms: Sequence[Callable[[Pair], Pair]]):
        self.transforms = list(transforms)

    def __call__(self, low, high) -> Pair:
        for t in self.transforms:
            low, high = t(low, high)
        return low, high

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"{self.__class__.__name__}([{inner}])"


class PairedRandomCrop:
    """Crop the same random ``size x size`` window from both images."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, low, high) -> Pair:
        h, w = _image_size(low)
        if (h, w) != _image_size(high):
            raise ValueError(
                f"Low/high size mismatch: {(h, w)} vs {_image_size(high)}"
            )
        ch, cw = self.size
        if ch > h or cw > w:
            raise ValueError(f"Crop size {self.size} larger than image {(h, w)}")
        top = int(torch.randint(0, h - ch + 1, (1,)).item())
        left = int(torch.randint(0, w - cw + 1, (1,)).item())
        return TF.crop(low, top, left, ch, cw), TF.crop(high, top, left, ch, cw)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class PairedRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, low, high) -> Pair:
        if torch.rand(1).item() < self.p:
            return TF.hflip(low), TF.hflip(high)
        return low, high

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class PairedRandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, low, high) -> Pair:
        if torch.rand(1).item() < self.p:
            return TF.vflip(low), TF.vflip(high)
        return low, high

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class PairedToTensor:
    """PIL -> float32 tensor in [0, 1], shape (3, H, W)."""

    def __call__(self, low, high) -> Pair:
        return TF.to_tensor(low), TF.to_tensor(high)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def build_paired_transforms(
    split: str,
    crop_size: Optional[int] = None,
    hflip: bool = True,
    vflip: bool = False,
) -> PairedCompose:
    """Default transforms per split.

    - ``train``: optional random crop + random horizontal flip + ToTensor.
      Vertical flip is available (``vflip=True``) but off by default, since
      LOL scenes have a natural up/down orientation.
    - ``val`` / ``test``: ToTensor only (deterministic, full resolution).
    """
    ops: List[Callable] = []
    if split == "train":
        if crop_size is not None:
            ops.append(PairedRandomCrop(crop_size))
        if hflip:
            ops.append(PairedRandomHorizontalFlip())
        if vflip:
            ops.append(PairedRandomVerticalFlip())
    elif split not in ("val", "test"):
        raise ValueError(f"Unknown split '{split}'")
    ops.append(PairedToTensor())
    return PairedCompose(ops)
