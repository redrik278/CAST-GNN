"""Validation-based early stopping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Track validation performance and decide when to stop training.

    Parameters
    ----------
    patience:
        Number of epochs without improvement before stopping.
    mode:
        ``"max"`` for metrics such as macro-F1, ``"min"`` for losses.
    min_delta:
        Minimum absolute improvement needed to reset patience.
    """

    patience: int = 20
    mode: str = "max"
    min_delta: float = 1e-4

    def __post_init__(self) -> None:
        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be either 'max' or 'min'")
        self.best: float | None = None
        self.num_bad_epochs: int = 0
        self.best_epoch: int = -1
        self.epoch: int = -1

    def is_improvement(self, metric: float) -> bool:
        """Return True if a metric improves over the current best."""
        if self.best is None:
            return True
        if self.mode == "max":
            return metric > self.best + self.min_delta
        return metric < self.best - self.min_delta

    def step(self, metric: float) -> bool:
        """Update state and return True if the metric improved."""
        self.epoch += 1
        if self.is_improvement(metric):
            self.best = metric
            self.best_epoch = self.epoch
            self.num_bad_epochs = 0
            return True
        self.num_bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        """Whether training should stop."""
        return self.num_bad_epochs >= self.patience
