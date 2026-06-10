"""Device utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def get_device(device: str | None = "auto", prefer_gpu: bool = True) -> torch.device:
    """Select a computation device.

    Parameters
    ----------
    device:
        ``"auto"``, ``"cpu"``, ``"cuda"``, or a concrete device string such as
        ``"cuda:0"``.
    prefer_gpu:
        Used only when ``device`` is ``"auto"`` or ``None``.
    """
    if device is None or device == "auto":
        if prefer_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return requested


def move_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors in nested containers to a device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    return obj
