"""Checkpoint save/load."""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

from ..models import build_model


def save_checkpoint(path: Union[str, Path], model, config: dict, **extra) -> None:
    """Save model weights + the config needed to rebuild it.

    ``extra`` may include optimizer_state, epoch, metrics, history, best_val_psnr.
    Written to a temp file first, then renamed, so an interrupted save (e.g. a
    Colab disconnect) never leaves a corrupt checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model_state": model.state_dict(), "config": config, **extra}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Union[str, Path], map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=True)


def load_model_from_checkpoint(
    path: Union[str, Path], device: Optional[torch.device] = None
) -> Tuple[torch.nn.Module, dict]:
    """Rebuild the model from the config stored in the checkpoint; returns (model, checkpoint)."""
    ckpt = load_checkpoint(path)
    model = build_model(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model_state"])
    if device is not None:
        model.to(device)
    model.eval()
    return model, ckpt
