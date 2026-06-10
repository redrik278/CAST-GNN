"""Randomness control for deterministic EEG experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        Seed value.
    deterministic:
        If ``True``, configures PyTorch CUDA backends for deterministic
        behaviour.  This may reduce speed but improves reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def seed_worker(worker_id: int) -> None:
    """Seed a PyTorch DataLoader worker deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
