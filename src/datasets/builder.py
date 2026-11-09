"""Build LOL-v1 datasets from a config dict (keeps scripts free of path/transform wiring)."""

from typing import Optional

from ..utils.config import resolve_path
from .lol_v1 import LOLv1Dataset
from .paired_transforms import build_paired_transforms


def build_lol_v1_dataset(cfg: dict, split: str, root: Optional[str] = None) -> LOLv1Dataset:
    """``split`` is the loader split name: "train", "val" or "test".

    Training gets random crop + hflip (per config); val/test are deterministic
    full-resolution ToTensor.
    """
    d = cfg["data"]
    if split == "train":
        tf = build_paired_transforms("train", crop_size=d["crop_size"], hflip=d["hflip"], vflip=d["vflip"])
    else:
        tf = build_paired_transforms(split)
    return LOLv1Dataset(
        str(resolve_path(root or d["root"])),
        split=split,
        transform=tf,
        val_size=d["val_size"],
        seed=d["split_seed"],
    )
