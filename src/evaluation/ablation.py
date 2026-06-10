"""Ablation-study utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


@dataclass
class AblationSpec:
    """Specification for one ablation setting."""

    name: str
    overrides: dict[str, Any]


DEFAULT_ABLATIONS = [
    AblationSpec("full_cast_gnn", {}),
    AblationSpec("without_graph_adapter", {"use_graph_adapters": False}),
    AblationSpec("without_functional_adjacency", {"use_functional_adjacency": False}),
    AblationSpec("without_learnable_adjacency", {"use_learnable_adjacency": False}),
    AblationSpec("without_band_attention", {"use_band_attention": False}),
    AblationSpec("without_coral", {"lambda1": 0.0, "use_coral": False}),
    AblationSpec("without_tcn", {"use_tcn": False}),
    AblationSpec("anatomical_graph_only", {"graph_mode": "anatomical"}),
    AblationSpec("functional_graph_only", {"graph_mode": "functional"}),
]


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow-copied config with overrides applied."""
    cfg = dict(config)
    cfg.update(overrides)
    return cfg


def run_ablation_suite(
    base_config: dict[str, Any],
    task_name: str,
    seed: int,
    train_fn: Callable[..., dict[str, float]],
    ablations: list[AblationSpec] | None = None,
) -> pd.DataFrame:
    """Run a suite of ablations using a supplied training function.

    The ``train_fn`` should accept ``config``, ``task_name``, and ``seed``.
    """
    records = []
    for spec in ablations or DEFAULT_ABLATIONS:
        cfg = apply_overrides(base_config, spec.overrides)
        metrics = train_fn(config=cfg, task_name=task_name, seed=seed)
        records.append({"ablation": spec.name, "seed": seed, **metrics})
    return pd.DataFrame(records)
