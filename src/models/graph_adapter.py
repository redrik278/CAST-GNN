"""Dataset-specific graph adapters for CAST-GNN."""

from __future__ import annotations

import torch
from torch import nn

from .graph_builder import normalize_adjacency


class GraphAdapter(nn.Module):
    """Map dataset-specific channel nodes into a shared latent graph space.

    The adapter uses a learnable assignment matrix P with shape [L, C], where C
    is the number of dataset-specific channels and L is the number of latent
    graph nodes.  It does not impose one-to-one electrode correspondence between
    datasets.
    """

    def __init__(self, in_nodes: int, latent_nodes: int = 32, feature_dim: int = 64, dropout: float = 0.30) -> None:
        super().__init__()
        self.in_nodes = int(in_nodes)
        self.latent_nodes = int(latent_nodes)
        self.feature_dim = int(feature_dim)
        self.assignment_logits = nn.Parameter(torch.randn(latent_nodes, in_nodes) * 0.02)
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(feature_dim),
        )

    def assignment_matrix(self) -> torch.Tensor:
        """Return row-normalised latent-to-channel assignment matrix."""
        return torch.softmax(self.assignment_logits, dim=-1)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Adapt node features and adjacency to latent graph space.

        Parameters
        ----------
        node_features:
            Tensor with shape [N, C, F].
        adjacency:
            Tensor with shape [C, C].
        """
        if node_features.ndim != 3:
            raise ValueError("node_features must have shape [N, C, F]")
        if node_features.shape[1] != self.in_nodes:
            raise ValueError(f"Expected {self.in_nodes} nodes, got {node_features.shape[1]}")
        p = self.assignment_matrix()  # [L,C]
        latent = torch.einsum("lc,ncf->nlf", p, node_features)
        latent = self.feature_proj(latent)
        latent_adj = p @ adjacency @ p.T
        latent_adj = normalize_adjacency(latent_adj)
        return latent, latent_adj, p
