"""Perturbation-validated interpretability utilities.

These methods quantify how trained models use channels, edges, and spectral
bands.  They describe model behaviour only and should not be interpreted as
causal neurophysiological evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils.device import move_to_device
from src.utils.io import write_csv


@torch.no_grad()
def _confidence_for_true_class(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    valid = targets != -100
    conf = torch.zeros_like(targets, dtype=probs.dtype, device=probs.device)
    conf[valid] = probs[valid, targets[valid].long()]
    return conf


@torch.no_grad()
def channel_perturbation_importance(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> pd.DataFrame:
    """Estimate channel importance by masking each channel and measuring confidence drop."""
    model.eval()
    drops = []
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="channel importance", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_to_device(batch, device)
        x = batch["x"]
        targets = batch.get("y", batch.get("label", batch.get("targets")))

        base_out = model(batch)
        base_logits = base_out["logits"] if isinstance(base_out, dict) else base_out
        if isinstance(base_logits, dict):
            base_logits = next(iter(base_logits.values()))
        base_conf = _confidence_for_true_class(base_logits, targets)

        n_channels = x.shape[-2]
        for c in range(n_channels):
            perturbed = dict(batch)
            x_masked = x.clone()
            x_masked[..., c, :] = 0.0
            perturbed["x"] = x_masked
            out = model(perturbed)
            logits = out["logits"] if isinstance(out, dict) else out
            if isinstance(logits, dict):
                logits = next(iter(logits.values()))
            conf = _confidence_for_true_class(logits, targets)
            drop = (base_conf - conf).detach().cpu().numpy()
            drops.extend([{"channel_index": c, "confidence_drop": float(v)} for v in drop])

    return pd.DataFrame(drops).groupby("channel_index", as_index=False)["confidence_drop"].mean()


@torch.no_grad()
def band_masking_importance(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    band_names: list[str] | None = None,
    max_batches: int | None = None,
) -> pd.DataFrame:
    """Estimate frequency-band importance by masking one band at a time."""
    model.eval()
    records = []
    band_names = band_names or ["5-8Hz", "8-13Hz", "13-30Hz", "30-40Hz"]

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="band importance", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_to_device(batch, device)
        x = batch["x"]
        targets = batch.get("y", batch.get("label", batch.get("targets")))

        base_out = model(batch)
        base_logits = base_out["logits"] if isinstance(base_out, dict) else base_out
        if isinstance(base_logits, dict):
            base_logits = next(iter(base_logits.values()))
        base_conf = _confidence_for_true_class(base_logits, targets)

        n_bands = x.shape[-3] if x.ndim >= 4 else x.shape[0]
        for b in range(n_bands):
            perturbed = dict(batch)
            x_masked = x.clone()
            if x.ndim == 4:
                x_masked[:, b] = 0.0
            elif x.ndim == 3:
                x_masked[b] = 0.0
            perturbed["x"] = x_masked
            out = model(perturbed)
            logits = out["logits"] if isinstance(out, dict) else out
            if isinstance(logits, dict):
                logits = next(iter(logits.values()))
            conf = _confidence_for_true_class(logits, targets)
            drops = (base_conf - conf).detach().cpu().numpy()
            name = band_names[b] if b < len(band_names) else f"band_{b}"
            records.extend({"band": name, "confidence_drop": float(v)} for v in drops)

    return pd.DataFrame(records).groupby("band", as_index=False)["confidence_drop"].mean()


@torch.no_grad()
def aggregate_attention_relevance(attention_weights: torch.Tensor | list[torch.Tensor]) -> np.ndarray:
    """Aggregate attention weights across layers, heads, and batches."""
    if isinstance(attention_weights, list):
        tensors = [a.detach().cpu() for a in attention_weights if torch.is_tensor(a)]
        if not tensors:
            return np.array([])
        attn = torch.stack(tensors).mean(dim=0)
    else:
        attn = attention_weights.detach().cpu()

    # Accept [layers, batch, heads, nodes, nodes], [batch, heads, nodes, nodes], or [heads, nodes, nodes].
    while attn.ndim > 2:
        attn = attn.mean(dim=0)
    return attn.numpy()


def edge_occlusion_importance(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    edge_list: list[tuple[int, int]],
    max_edges: int | None = None,
) -> pd.DataFrame:
    """Placeholder-safe edge occlusion interface.

    This function requires the model to support an ``edge_mask`` field in the
    input batch.  If the current model does not support this, it raises a clear
    error instead of silently producing invalid explanations.
    """
    if not hasattr(model, "supports_edge_mask"):
        raise NotImplementedError(
            "edge_occlusion_importance requires the model to implement edge-mask support."
        )
    # The detailed edge-mask implementation is model-specific.
    raise NotImplementedError("Implement edge masking according to the final graph module interface.")


def save_interpretability_outputs(results: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """Save interpretability result tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        write_csv(df, output_dir / f"{name}.csv")
