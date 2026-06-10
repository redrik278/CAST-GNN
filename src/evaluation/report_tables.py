"""Manuscript-ready table formatting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_confidence_interval


def format_mean_sd(values, decimals: int = 3) -> str:
    """Format mean ± standard deviation."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return "NA"
    mean = values.mean()
    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    return f"{mean:.{decimals}f}±{sd:.{decimals}f}"


def format_mean_ci(values, decimals: int = 3) -> str:
    """Format mean with 95% confidence interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return "NA"
    mean = values.mean()
    low, high = compute_confidence_interval(values)
    return f"{mean:.{decimals}f} ({low:.{decimals}f}–{high:.{decimals}f})"


def _pivot_metric_table(results_df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in results_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in metrics:
            if metric in group.columns:
                row[metric] = format_mean_sd(group[metric].values)
        rows.append(row)
    return pd.DataFrame(rows)


def create_within_dataset_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create a within-dataset performance table."""
    metrics = ["accuracy", "macro_f1", "balanced_accuracy", "kappa", "mcc", "ece"]
    group_cols = [c for c in ["dataset", "task", "model"] if c in results_df.columns]
    return _pivot_metric_table(results_df, group_cols, metrics)


def create_transfer_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create a cross-dataset transfer table."""
    metrics = ["accuracy", "macro_f1", "balanced_accuracy", "kappa", "mcc"]
    group_cols = [c for c in ["source", "target", "task", "model"] if c in results_df.columns]
    return _pivot_metric_table(results_df, group_cols, metrics)


def create_ablation_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create an ablation table."""
    metrics = ["macro_f1", "balanced_accuracy", "mcc"]
    group_cols = [c for c in ["ablation", "task"] if c in results_df.columns]
    return _pivot_metric_table(results_df, group_cols, metrics)


def create_robustness_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create a robustness table."""
    metrics = ["clean_macro_f1", "perturbed_macro_f1", "relative_degradation"]
    group_cols = [c for c in ["perturbation", "level", "task"] if c in results_df.columns]
    return _pivot_metric_table(results_df, group_cols, metrics)


def create_calibration_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create a calibration table."""
    metrics = ["ece_before", "ece_after", "brier_before", "brier_after"]
    group_cols = [c for c in ["dataset", "task", "model"] if c in results_df.columns]
    return _pivot_metric_table(results_df, group_cols, metrics)
