"""Device selection, seeding and small runtime helpers."""

import random

import numpy as np
import torch


def select_device(preference: str = "auto") -> torch.device:
    """``auto`` picks CUDA, then Apple MPS, then CPU. Explicit choices are validated."""
    pref = (preference or "auto").lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    if pref == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device=mps requested but MPS is not available")
    return torch.device(pref)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch (CPU/CUDA/MPS).

    ``deterministic`` makes cuDNN pick deterministic kernels. Bit-exact
    reproducibility across GPUs/driver versions is still not guaranteed by
    PyTorch, but runs on the same setup should match closely.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive Python/NumPy seeds from torch's per-worker seed."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
