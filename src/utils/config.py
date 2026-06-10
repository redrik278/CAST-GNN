"""Configuration utilities for reproducible CAST-GNN experiments.

The project uses YAML files for dataset, model, and experiment settings.  The
helpers in this module support recursive merging so that a default config can
be safely overridden by dataset-specific and experiment-specific files.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to a YAML file.

    Returns
    -------
    dict
        Parsed configuration dictionary. Empty YAML files return an empty dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return cfg


def update_nested(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively update ``base`` with values from ``override``.

    Nested dictionaries are merged rather than replaced wholesale.  Lists and
    scalar values are replaced by the overriding configuration.
    """
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, Mapping)
        ):
            update_nested(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def merge_configs(*configs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge multiple configuration mappings recursively.

    Later configurations take precedence over earlier configurations.
    """
    merged: dict[str, Any] = {}
    for cfg in configs:
        if cfg is None:
            continue
        update_nested(merged, cfg)
    return merged


def load_many_configs(*paths: str | Path | None) -> dict[str, Any]:
    """Load and recursively merge multiple YAML files."""
    configs = [load_config(path) for path in paths if path is not None]
    return merge_configs(*configs)


def save_config(cfg: Mapping[str, Any], path: str | Path) -> None:
    """Save a configuration dictionary as YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False, allow_unicode=True)


def get_by_path(cfg: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Return a nested config value using dot notation.

    Examples
    --------
    ``get_by_path(cfg, "training.lr", 1e-3)``
    """
    current: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current
