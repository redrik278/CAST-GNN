"""Input/output helpers used across training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file."""
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    """Write a DataFrame to CSV, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path) -> None:
    """Write an object to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_numpy(array: np.ndarray, path: str | Path) -> None:
    """Save a NumPy array as ``.npy``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_numpy(path: str | Path) -> np.ndarray:
    """Load a NumPy array from ``.npy``."""
    return np.load(path, allow_pickle=False)


def save_checkpoint(obj: Any, path: str | Path) -> None:
    """Save a PyTorch checkpoint object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """Load a PyTorch checkpoint object."""
    return torch.load(path, map_location=map_location)
