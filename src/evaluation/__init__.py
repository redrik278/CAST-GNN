"""Evaluation package for CAST-GNN."""

from .metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_confidence_interval,
    summarize_seed_metrics,
)
from .calibration import (
    expected_calibration_error,
    brier_score,
    TemperatureScaler,
    plot_reliability_diagram,
)

__all__ = [
    "compute_classification_metrics",
    "compute_confusion_matrix",
    "compute_confidence_interval",
    "summarize_seed_metrics",
    "expected_calibration_error",
    "brier_score",
    "TemperatureScaler",
    "plot_reliability_diagram",
]
