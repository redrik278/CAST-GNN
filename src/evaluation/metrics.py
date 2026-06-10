"""Classification metrics and summary statistics."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray | Iterable[int],
    y_pred: np.ndarray | Iterable[int],
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute standard classification metrics.

    Invalid labels marked as ``-100`` are removed before evaluation.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid = y_true != -100
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if y_prob is not None:
        y_prob = np.asarray(y_prob)[valid]

    if len(y_true) == 0:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "weighted_f1": float("nan"),
            "balanced_accuracy": float("nan"),
            "kappa": float("nan"),
            "mcc": float("nan"),
            "roc_auc": float("nan"),
        }

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float("nan"),
    }

    if y_prob is not None:
        try:
            classes = np.unique(y_true)
            if y_prob.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            elif len(classes) > 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            metrics["roc_auc"] = float("nan")

    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None) -> np.ndarray:
    """Return a confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=labels)


def compute_confidence_interval(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float]:
    """Compute a normal-approximation confidence interval."""
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean
    z = 1.96 if abs(confidence - 0.95) < 1e-9 else 1.96
    se = values.std(ddof=1) / math.sqrt(len(values))
    return mean - z * se, mean + z * se


def summarize_seed_metrics(metrics_list: list[dict[str, float]]) -> pd.DataFrame:
    """Summarise metrics across seeds or folds."""
    df = pd.DataFrame(metrics_list)
    rows = []
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        ci_low, ci_high = compute_confidence_interval(vals)
        rows.append({
            "metric": col,
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "n": int(len(vals)),
        })
    return pd.DataFrame(rows)
