"""Lightweight temporal/dependency convolution module for CAST-GNN."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    """Residual 1D convolutional block with dilation."""

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.30) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.dropout(y)
        return F.gelu(y + residual)


class LightweightTCN(nn.Module):
    """Compact residual TCN.

    Input and output shape: [N, S, F].  In CAST-GNN, S may represent a latent
    node axis or a retained temporal axis depending on the model configuration.
    """

    def __init__(self, feature_dim: int = 64, dilations: tuple[int, ...] = (1, 2), kernel_size: int = 5, dropout: float = 0.30) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TemporalConvBlock(feature_dim, kernel_size=kernel_size, dilation=d, dropout=dropout) for d in dilations]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("LightweightTCN expects [N, S, F]")
        y = x.transpose(1, 2)  # [N,F,S]
        for block in self.blocks:
            y = block(y)
        return y.transpose(1, 2)


class AttentiveTemporalPooling(nn.Module):
    """Attention pooling over sequence dimension."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.Tanh(), nn.Linear(feature_dim, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("AttentiveTemporalPooling expects [N, S, F]")
        scores = self.score(x).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        pooled = (x * weights[..., None]).sum(dim=1)
        return pooled, weights
