"""Task-specific prediction heads for CAST-GNN."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


TASK_NUM_CLASSES = {
    "A1_non_task_vs_task": 2,
    "A2_rest_vs_task": 2,
    "B_execution_vs_imagery": 2,
    "C_upper_vs_lower": 2,
    "D_left_vs_right": 2,
    "E_multiclass_motor:eegmmidb": 4,
    "E_multiclass_motor:milimbeeg": 6,
    "F_fine_grained:eegmmidb": 8,
    "F_fine_grained:milimbeeg": 12,
}


class TaskHead(nn.Module):
    """Two-layer MLP classification head."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.30) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskHeads(nn.Module):
    """Container for task-specific heads."""

    def __init__(self, input_dim: int, task_num_classes: Optional[Dict[str, int]] = None, hidden_dim: int = 128, dropout: float = 0.30) -> None:
        super().__init__()
        self.task_num_classes = dict(task_num_classes or TASK_NUM_CLASSES)
        self.heads = nn.ModuleDict(
            {name.replace(":", "__"): TaskHead(input_dim, n_cls, hidden_dim=hidden_dim, dropout=dropout) for name, n_cls in self.task_num_classes.items()}
        )

    @staticmethod
    def make_key(task_name: str, dataset: Optional[str] = None) -> str:
        if task_name in {"E_multiclass_motor", "F_fine_grained"}:
            if dataset is None:
                raise ValueError(f"dataset must be provided for task {task_name}")
            return f"{task_name}:{dataset}"
        return task_name

    @staticmethod
    def _module_key(key: str) -> str:
        return key.replace(":", "__")

    def forward(self, features: torch.Tensor, task_name: Optional[str] = None, dataset: Optional[str] = None):
        if task_name is None:
            return {k.replace("__", ":"): head(features) for k, head in self.heads.items()}
        key = self.make_key(task_name, dataset=dataset)
        module_key = self._module_key(key)
        if module_key not in self.heads:
            raise KeyError(f"No prediction head found for {key!r}")
        return self.heads[module_key](features)
