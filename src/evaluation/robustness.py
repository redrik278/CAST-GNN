"""Robustness perturbations and evaluation utilities."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_classification_metrics
from src.utils.device import move_to_device


def add_gaussian_noise(x: torch.Tensor, snr_db: float = 20.0) -> torch.Tensor:
    """Add Gaussian noise at a specified signal-to-noise ratio."""
    power = x.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12)
    noise_power = power / (10 ** (snr_db / 10.0))
    noise = torch.randn_like(x) * torch.sqrt(noise_power)
    return x + noise


def apply_channel_dropout(x: torch.Tensor, drop_prob: float = 0.10) -> torch.Tensor:
    """Randomly zero full EEG channels."""
    if x.ndim < 3:
        return x
    # Works for [N, B, C, T] or [B, C, T]
    channel_dim = -2
    mask_shape = list(x.shape)
    mask_shape[-1] = 1
    mask = torch.rand(mask_shape, device=x.device) > drop_prob
    return x * mask


def apply_temporal_crop(x: torch.Tensor, crop_ratio: float = 0.90) -> torch.Tensor:
    """Crop a central temporal segment and pad back to original length."""
    t = x.shape[-1]
    new_t = max(1, int(t * crop_ratio))
    start = (t - new_t) // 2
    cropped = x[..., start:start + new_t]
    pad_left = start
    pad_right = t - new_t - pad_left
    return torch.nn.functional.pad(cropped, (pad_left, pad_right))


def apply_amplitude_scaling(x: torch.Tensor, scale_range: tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
    """Scale sample amplitudes by random factors."""
    low, high = scale_range
    shape = list(x.shape)
    shape[-1] = 1
    scales = low + (high - low) * torch.rand(shape, device=x.device)
    return x * scales


def apply_band_noise(x: torch.Tensor, band_index: int, noise_std: float = 0.05) -> torch.Tensor:
    """Add noise to one precomputed frequency-band channel.

    Assumes input shape ``[N, bands, channels, time]`` or ``[bands, channels, time]``.
    """
    x = x.clone()
    if x.ndim == 4:
        x[:, band_index] = x[:, band_index] + noise_std * torch.randn_like(x[:, band_index])
    elif x.ndim == 3:
        x[band_index] = x[band_index] + noise_std * torch.randn_like(x[band_index])
    return x


PERTURBATIONS: dict[str, Callable[..., torch.Tensor]] = {
    "gaussian_noise": add_gaussian_noise,
    "channel_dropout": apply_channel_dropout,
    "temporal_crop": apply_temporal_crop,
    "amplitude_scaling": apply_amplitude_scaling,
}


@torch.no_grad()
def evaluate_robustness(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    perturbation_name: str,
    perturbation_kwargs: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Evaluate model performance under a selected perturbation."""
    if perturbation_name not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation: {perturbation_name}")

    perturbation_kwargs = perturbation_kwargs or {}
    perturb = PERTURBATIONS[perturbation_name]
    model.eval()

    all_probs, all_preds, all_targets = [], [], []
    for batch in dataloader:
        batch = move_to_device(batch, device)
        batch = deepcopy(batch)
        batch["x"] = perturb(batch["x"], **perturbation_kwargs)
        try:
            output = model(batch)
        except TypeError:
            output = model(batch["x"])
        logits = output["logits"] if isinstance(output, dict) else output
        if isinstance(logits, dict):
            logits = next(iter(logits.values()))
        probs = torch.softmax(logits, dim=1)
        targets = batch.get("y", batch.get("label", batch.get("targets")))
        all_probs.append(probs.cpu())
        all_preds.append(probs.argmax(dim=1).cpu())
        all_targets.append(targets.cpu())

    probs = torch.cat(all_probs).numpy()
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    return compute_classification_metrics(targets, preds, probs)


def compute_relative_degradation(clean_metric: float, perturbed_metric: float) -> float:
    """Return relative degradation from clean to perturbed performance."""
    if clean_metric == 0 or not np.isfinite(clean_metric):
        return float("nan")
    return float((clean_metric - perturbed_metric) / clean_metric)
