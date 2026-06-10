"""Main CAST-GNN model implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from .temporal_encoder import DatasetInputStem, MultiScaleTemporalEncoder, BandAttention
from .graph_builder import HybridGraphBuilder, build_anatomical_adjacency
from .graph_adapter import GraphAdapter
from .graph_attention import GraphAttentionEncoder
from .tcn import LightweightTCN, AttentiveTemporalPooling
from .heads import MultiTaskHeads, TASK_NUM_CLASSES


class CASTGNN(nn.Module):
    """Cross-Dataset Adaptive Spectro-Temporal Graph Network.

    The model expects batches with `x` shaped [N, B, C, T], where B is the number
    of spectral bands.  Because EEGMMIDB and MILimbEEG have different channel
    counts, each batch should normally contain samples from only one dataset.
    """

    def __init__(
        self,
        dataset_configs: Dict[str, Dict[str, Any]],
        task_num_classes: Optional[Dict[str, int]] = None,
        latent_nodes: int = 32,
        feature_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.30,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.dataset_configs = dataset_configs
        self.latent_nodes = latent_nodes
        self.feature_dim = feature_dim

        self.stems = nn.ModuleDict()
        self.graph_builders = nn.ModuleDict()
        self.adapters = nn.ModuleDict()

        for name, cfg in dataset_configs.items():
            c = int(cfg["num_channels"])
            ch_names = cfg.get("channel_names") or [f"Ch{i+1}" for i in range(c)]
            self.stems[name] = DatasetInputStem(feature_dim=feature_dim, dropout=dropout)
            anat_adj = cfg.get("anat_adj")
            if anat_adj is None:
                anat_adj = build_anatomical_adjacency(ch_names, fallback_k=min(4, max(c - 1, 1)))
            func_adj = cfg.get("func_adj")
            self.graph_builders[name] = HybridGraphBuilder(num_nodes=c, anat_adj=anat_adj, func_adj=func_adj)
            self.adapters[name] = GraphAdapter(in_nodes=c, latent_nodes=latent_nodes, feature_dim=feature_dim, dropout=dropout)

        self.temporal_encoder = MultiScaleTemporalEncoder(feature_dim=feature_dim, dropout=dropout)
        self.band_attention = BandAttention(feature_dim=feature_dim)
        self.graph_encoder = GraphAttentionEncoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.tcn = LightweightTCN(feature_dim=hidden_dim, dropout=dropout)
        self.pooling = AttentiveTemporalPooling(hidden_dim)
        self.heads = MultiTaskHeads(input_dim=hidden_dim, task_num_classes=task_num_classes or TASK_NUM_CLASSES, dropout=dropout)

    @staticmethod
    def _infer_dataset(batch_dataset: Any) -> str:
        if isinstance(batch_dataset, str):
            return batch_dataset
        if isinstance(batch_dataset, (list, tuple)):
            unique = sorted(set(map(str, batch_dataset)))
            if len(unique) != 1:
                raise ValueError(
                    "A CASTGNN batch must contain one dataset because channel counts differ. "
                    f"Received datasets: {unique}"
                )
            return unique[0]
        raise TypeError("dataset must be a string or a list/tuple of strings")

    def encode(self, x: torch.Tensor, dataset: str) -> Dict[str, Any]:
        """Encode an EEG batch into a shared representation."""
        if dataset not in self.stems:
            raise KeyError(f"Unknown dataset {dataset!r}. Available: {list(self.stems)}")
        if x.ndim != 4:
            raise ValueError("x must have shape [N, B, C, T]")

        n, b, c, t = x.shape
        band_embeddings = []
        band_time = []
        stem = self.stems[dataset]
        for band_idx in range(b):
            h = stem(x[:, band_idx])  # [N,C,T',F]
            h_time, h_pool = self.temporal_encoder(h)
            band_time.append(h_time)
            band_embeddings.append(h_pool)
        band_embeddings_tensor = torch.stack(band_embeddings, dim=1)  # [N,B,C,F]
        node_features, band_weights = self.band_attention(band_embeddings_tensor)  # [N,C,F]

        adjacency, graph_info = self.graph_builders[dataset]()
        latent_features, latent_adj, assignment = self.adapters[dataset](node_features, adjacency)
        graph_features, attentions = self.graph_encoder(latent_features, latent_adj)
        dep_features = self.tcn(graph_features)  # [N,L,F]
        pooled, pool_weights = self.pooling(dep_features)
        return {
            "features": pooled,
            "node_features": node_features,
            "latent_features": latent_features,
            "graph_features": graph_features,
            "adjacency": adjacency,
            "latent_adjacency": latent_adj,
            "assignment": assignment,
            "attention": attentions,
            "band_weights": band_weights,
            "pool_weights": pool_weights,
            "graph_info": graph_info,
        }

    def forward(
        self,
        batch: Dict[str, Any] | torch.Tensor,
        dataset: Optional[str] = None,
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Forward pass.

        Parameters
        ----------
        batch:
            Either a tensor [N,B,C,T] or a dictionary containing `x`, `dataset`,
            and optionally `task_name`.
        dataset:
            Dataset name when `batch` is a tensor.
        task_name:
            Optional task head to evaluate. If omitted, all heads are returned.
        """
        if isinstance(batch, dict):
            x = batch["x"]
            ds = dataset or self._infer_dataset(batch.get("dataset"))
            task = task_name or batch.get("task_name")
        else:
            x = batch
            if dataset is None:
                raise ValueError("dataset must be provided when batch is a tensor")
            ds = dataset
            task = task_name

        enc = self.encode(x, ds)
        logits = self.heads(enc["features"], task_name=task, dataset=ds) if task is not None else self.heads(enc["features"])
        enc["logits"] = logits
        enc["dataset"] = ds
        enc["task_name"] = task
        return enc
