"""Calibration analysis and temperature scaling."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute expected calibration error."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    valid = labels != -100
    probs = probs[valid]
    labels = labels[valid]
    if len(labels) == 0:
        return float("nan")

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == labels

    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidences > low) & (confidences <= high)
        if mask.any():
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += mask.mean() * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute multiclass Brier score."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    valid = labels != -100
    probs = probs[valid]
    labels = labels[valid]
    if len(labels) == 0:
        return float("nan")

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels.astype(int)] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


class TemperatureScaler(nn.Module):
    """Validation-only temperature scaling.

    Fit this module on validation logits and labels, then apply it to test
    logits.  Do not fit the temperature on the test set.
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(init_temperature))))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp(min=1e-3, max=100.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 200,
        lr: float = 0.01,
    ) -> "TemperatureScaler":
        """Fit temperature using validation logits and labels."""
        self.train()
        valid = labels != -100
        logits = logits[valid]
        labels = labels[valid].long()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    n_bins: int = 15,
    title: str = "Reliability diagram",
) -> None:
    """Plot a reliability diagram."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    valid = labels != -100
    probs = probs[valid]
    labels = labels[valid]

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == labels
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    bin_centers, bin_accs, bin_confs = [], [], []
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidences > low) & (confidences <= high)
        if mask.any():
            bin_centers.append((low + high) / 2)
            bin_accs.append(accuracies[mask].mean())
            bin_confs.append(confidences[mask].mean())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.bar(bin_centers, bin_accs, width=1.0 / n_bins, alpha=0.6, edgecolor="black")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
