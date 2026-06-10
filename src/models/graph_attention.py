"""Pure PyTorch graph attention layers for CAST-GNN."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Multi-head graph attention with adjacency-constrained neighborhoods."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.30, concat: bool = False) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_heads = int(num_heads)
        self.concat = concat
        self.head_dim = out_dim if not concat else out_dim // num_heads
        if concat and out_dim % num_heads != 0:
            raise ValueError("out_dim must be divisible by num_heads when concat=True")
        self.proj = nn.Linear(in_dim, self.num_heads * self.head_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(self.num_heads, self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply graph attention.

        Parameters
        ----------
        x:
            Node features with shape [N, L, F].
        adjacency:
            Graph adjacency with shape [L, L] or [N, L, L]. Non-positive values
            are treated as absent edges.
        """
        if x.ndim != 3:
            raise ValueError("x must have shape [N, L, F]")
        n, l, _ = x.shape
        h = self.proj(x).view(n, l, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [N,H,L,D]
        src = (h * self.att_src[None, :, None, :]).sum(dim=-1)  # [N,H,L]
        dst = (h * self.att_dst[None, :, None, :]).sum(dim=-1)  # [N,H,L]
        e = self.leaky_relu(src[:, :, :, None] + dst[:, :, None, :])  # [N,H,L,L]

        if adjacency.ndim == 2:
            mask = adjacency[None, None, :, :] > 0
        elif adjacency.ndim == 3:
            mask = adjacency[:, None, :, :] > 0
        else:
            raise ValueError("adjacency must have shape [L,L] or [N,L,L]")
        e = e.masked_fill(~mask, torch.finfo(e.dtype).min)
        attention = torch.softmax(e, dim=-1)
        attention = self.dropout(attention)
        out = torch.matmul(attention, h)  # [N,H,L,D]
        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(n, l, self.num_heads * self.head_dim)
        else:
            out = out.mean(dim=1)
        return out, attention


class GraphAttentionEncoder(nn.Module):
    """Two-layer residual graph attention encoder."""

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = feature_dim
        for _ in range(num_layers):
            self.layers.append(GraphAttentionLayer(in_dim, hidden_dim, num_heads=num_heads, dropout=dropout, concat=False))
            self.norms.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attentions: list[torch.Tensor] = []
        out = x
        for layer, norm in zip(self.layers, self.norms):
            residual = out
            out_layer, att = layer(out, adjacency)
            if residual.shape[-1] == out_layer.shape[-1]:
                out = norm(residual + self.dropout(out_layer))
            else:
                out = norm(self.dropout(out_layer))
            out = F.gelu(out)
            attentions.append(att)
        return out, attentions
