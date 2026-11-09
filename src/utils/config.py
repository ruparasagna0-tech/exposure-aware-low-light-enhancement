"""Config loading and project path helpers."""

import copy
from pathlib import Path
from typing import Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def load_config(path: Union[str, Path]) -> dict:
    with open(resolve_path(path), "r") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: Union[str, Path]) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(copy.deepcopy(cfg), f, sort_keys=False)


def resolve_path(path: Union[str, Path]) -> Path:
    """Absolute paths are kept; relative paths are taken from the project root.

    This makes the same config work locally and on Colab (as long as the
    project folder layout is the same), independent of the current directory.
    """
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def ensure_outside_data(path: Union[str, Path]) -> Path:
    """Refuse output locations inside the project's data/ folder."""
    p = Path(path).expanduser().resolve()
    if p == DATA_DIR or DATA_DIR in p.parents:
        raise ValueError(f"Refusing to write inside the dataset folder: {p}")
    return p
