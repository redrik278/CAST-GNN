"""Graph construction modules for CAST-GNN."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch import nn

try:
    import mne
except Exception:  # pragma: no cover
    mne = None


def normalize_adjacency(adj: torch.Tensor, add_self_loops: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """Row-normalize an adjacency matrix."""
    a = adj.float()
    if add_self_loops:
        eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
        a = torch.maximum(a, eye)
    denom = a.sum(dim=-1, keepdim=True).clamp_min(eps)
    return a / denom


def topk_adjacency(adj: torch.Tensor, k: int, exclude_self: bool = True) -> torch.Tensor:
    """Keep top-k positive connections per node and symmetrize."""
    a = adj.clone().float()
    n = a.shape[-1]
    if exclude_self:
        a.fill_diagonal_(0.0)
    k = max(1, min(k, n - 1))
    values, idx = torch.topk(a, k=k, dim=-1)
    mask = torch.zeros_like(a)
    mask.scatter_(dim=-1, index=idx, src=torch.ones_like(values))
    out = a * mask
    out = torch.maximum(out, out.T)
    out = torch.clamp(out, min=0.0)
    return out


def build_anatomical_adjacency(
    channel_names: Sequence[str],
    montage_name: str = "standard_1020",
    sigma: float = 0.08,
    fallback_k: int = 4,
) -> torch.Tensor:
    """Build anatomical adjacency from electrode coordinates when available.

    If MNE or montage positions are unavailable, a conservative ring-like
    fallback topology is used.
    """
    c = len(channel_names)
    if c == 0:
        raise ValueError("channel_names must not be empty")
    if mne is not None:
        try:
            montage = mne.channels.make_standard_montage(montage_name)
            pos = montage.get_positions()["ch_pos"]
            coords = []
            missing = False
            for ch in channel_names:
                key = ch.replace(".", "").upper()
                # Try several forms because EEGMMIDB names sometimes include dots.
                candidates = [ch, ch.upper(), key, key.capitalize()]
                found = None
                for cand in candidates:
                    if cand in pos:
                        found = pos[cand]
                        break
                if found is None:
                    missing = True
                    break
                coords.append(found)
            if not missing:
                xyz = torch.tensor(np.asarray(coords), dtype=torch.float32)
                d = torch.cdist(xyz, xyz, p=2)
                adj = torch.exp(-(d ** 2) / (2 * sigma**2))
                adj.fill_diagonal_(1.0)
                return normalize_adjacency(adj)
        except Exception:
            pass

    # Fallback: connect each node to nearby index neighbours.
    adj = torch.zeros(c, c, dtype=torch.float32)
    for i in range(c):
        for offset in range(1, fallback_k + 1):
            adj[i, (i + offset) % c] = 1.0
            adj[i, (i - offset) % c] = 1.0
    adj.fill_diagonal_(1.0)
    return normalize_adjacency(adj)


def estimate_functional_adjacency(
    train_signals: torch.Tensor | np.ndarray,
    top_k: int = 8,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate functional adjacency from training data only.

    Parameters
    ----------
    train_signals:
        Array with shape [N, C, T] or [N, B, C, T].
    top_k:
        Number of positive connections retained per node.
    """
    x = torch.as_tensor(train_signals, dtype=torch.float32)
    if x.ndim == 4:
        x = x.mean(dim=1)  # [N,C,T]
    if x.ndim != 3:
        raise ValueError("train_signals must have shape [N,C,T] or [N,B,C,T]")
    n, c, t = x.shape
    flat = x.permute(1, 0, 2).reshape(c, n * t)
    flat = flat - flat.mean(dim=1, keepdim=True)
    flat = flat / flat.std(dim=1, keepdim=True).clamp_min(eps)
    corr = flat @ flat.T / max(flat.shape[1] - 1, 1)
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = torch.clamp(corr, min=0.0, max=1.0)
    corr.fill_diagonal_(1.0)
    return normalize_adjacency(topk_adjacency(corr, k=top_k))


class HybridGraphBuilder(nn.Module):
    """Hybrid anatomical-functional-learnable adjacency builder.

    Implements: A = alpha * A_anat + beta * A_func + gamma * A_learn, where the
    mixture coefficients are softmax-normalized.
    """

    def __init__(
        self,
        num_nodes: int,
        anat_adj: Optional[torch.Tensor] = None,
        func_adj: Optional[torch.Tensor] = None,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        if anat_adj is None:
            anat_adj = torch.eye(num_nodes, dtype=torch.float32)
        if func_adj is None:
            func_adj = torch.eye(num_nodes, dtype=torch.float32)
        self.register_buffer("anat_adj", normalize_adjacency(anat_adj.float()))
        self.register_buffer("func_adj", normalize_adjacency(func_adj.float()))
        learn = torch.eye(num_nodes, dtype=torch.float32) + init_scale * torch.randn(num_nodes, num_nodes)
        self.learn_adj_logits = nn.Parameter(learn)
        self.mix_logits = nn.Parameter(torch.zeros(3))

    def learned_adjacency(self) -> torch.Tensor:
        sym = 0.5 * (self.learn_adj_logits + self.learn_adj_logits.T)
        adj = torch.sigmoid(sym)
        adj.fill_diagonal_(1.0)
        return normalize_adjacency(adj)

    def forward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        weights = torch.softmax(self.mix_logits, dim=0)
        a_learn = self.learned_adjacency()
        adj = weights[0] * self.anat_adj + weights[1] * self.func_adj + weights[2] * a_learn
        adj = normalize_adjacency(adj)
        info = {"alpha_beta_gamma": weights, "a_learn": a_learn, "a_anat": self.anat_adj, "a_func": self.func_adj}
        return adj, info
