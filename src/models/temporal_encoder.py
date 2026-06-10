"""Temporal and band-aware encoding modules for CAST-GNN."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class DatasetInputStem(nn.Module):
    """Dataset-specific temporal convolutional input stem.

    Converts one band of EEG from [N, C, T] to [N, C, T', F].  Channels are
    processed independently with shared temporal filters, preserving channel
    identity before graph construction.
    """

    def __init__(self, feature_dim: int = 64, kernel_size: int = 7, stride: int = 4, dropout: float = 0.30) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(1, feature_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(feature_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Tensor with shape [N, C, T].
        """
        if x.ndim != 3:
            raise ValueError("DatasetInputStem expects input with shape [N, C, T]")
        n, c, t = x.shape
        y = x.reshape(n * c, 1, t)
        y = self.conv(y)
        y = self.bn(y)
        y = self.act(y)
        y = self.dropout(y)
        # [N*C, F, T'] -> [N, C, T', F]
        y = y.transpose(1, 2).reshape(n, c, -1, self.feature_dim)
        return y


class MultiScaleTemporalEncoder(nn.Module):
    """Multi-scale temporal convolutional encoder over per-channel features.

    Input shape is [N, C, T', F].  The encoder applies parallel temporal
    convolutions with kernel sizes 3, 7, and 15, then returns both a time-resolved
    representation [N, C, T', F] and a pooled channel embedding [N, C, F].
    """

    def __init__(self, feature_dim: int = 64, kernels: Sequence[int] = (3, 7, 15), dropout: float = 0.30) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.kernels = tuple(kernels)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(feature_dim, feature_dim, kernel_size=k, padding=k // 2, groups=1, bias=False)
                for k in self.kernels
            ]
        )
        self.project = nn.Linear(feature_dim * len(self.kernels), feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError("MultiScaleTemporalEncoder expects [N, C, T, F]")
        n, c, t, f = x.shape
        y = x.reshape(n * c, t, f).transpose(1, 2)  # [N*C, F, T]
        outs = [conv(y).transpose(1, 2) for conv in self.convs]  # each [N*C,T,F]
        y = torch.cat(outs, dim=-1)
        y = self.project(y)
        y = self.act(y)
        y = self.dropout(y)
        residual = x.reshape(n * c, t, f)
        y = self.norm(y + residual)
        y_time = y.reshape(n, c, t, f)
        y_pooled = y_time.mean(dim=2)  # [N,C,F]
        return y_time, y_pooled


class BandAttention(nn.Module):
    """Learnable attention over frequency-band embeddings."""

    def __init__(self, feature_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, band_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse band embeddings.

        Parameters
        ----------
        band_embeddings:
            Tensor with shape [N, B, C, F].

        Returns
        -------
        fused:
            Tensor with shape [N, C, F].
        weights:
            Band weights with shape [N, B].
        """
        if band_embeddings.ndim != 4:
            raise ValueError("BandAttention expects [N, B, C, F]")
        # Score each band using a channel-averaged representation.
        band_summary = band_embeddings.mean(dim=2)  # [N,B,F]
        scores = self.score(band_summary).squeeze(-1)  # [N,B]
        weights = torch.softmax(scores, dim=1)
        fused = (band_embeddings * weights[:, :, None, None]).sum(dim=1)
        return fused, weights
