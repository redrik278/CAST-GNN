"""Loss functions for CAST-GNN.

The model combines supervised classification, optional CORAL feature alignment,
and graph regularisation.  The implementation is intentionally defensive so it
can work with both single-task outputs and dictionaries of task-specific logits.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def _covariance(features: torch.Tensor) -> torch.Tensor:
    """Compute an unbiased covariance matrix for a batch of features."""
    if features.ndim > 2:
        features = features.flatten(start_dim=1)
    n = features.shape[0]
    if n <= 1:
        return torch.zeros(
            features.shape[1], features.shape[1],
            device=features.device,
            dtype=features.dtype,
        )
    centered = features - features.mean(dim=0, keepdim=True)
    return centered.T @ centered / (n - 1)


def classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute cross-entropy classification loss.

    Parameters
    ----------
    logits:
        Tensor with shape ``[N, num_classes]``.
    targets:
        Integer class labels with shape ``[N]``.  Values equal to
        ``ignore_index`` are ignored.
    ignore_index:
        Label value ignored during loss computation.
    class_weights:
        Optional per-class weights.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits [N, C], got shape {tuple(logits.shape)}")
    targets = targets.long()
    valid = targets != ignore_index
    if valid.sum().item() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits[valid],
        targets[valid],
        weight=class_weights,
        ignore_index=ignore_index,
    )


def multitask_classification_loss(
    logits_by_task: Mapping[str, torch.Tensor] | torch.Tensor,
    targets_by_task: Mapping[str, torch.Tensor] | torch.Tensor,
    ignore_index: int = -100,
    class_weights: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute classification loss for one or more task heads.

    If tensors are provided, this reduces to ordinary cross entropy.  If
    dictionaries are provided, losses are averaged across tasks that have at
    least one valid target.
    """
    if torch.is_tensor(logits_by_task):
        if not torch.is_tensor(targets_by_task):
            raise TypeError("Tensor logits require tensor targets.")
        return classification_loss(logits_by_task, targets_by_task, ignore_index)

    if not isinstance(targets_by_task, Mapping):
        raise TypeError("Dictionary logits require dictionary targets.")

    losses: list[torch.Tensor] = []
    for task_name, logits in logits_by_task.items():
        if task_name not in targets_by_task:
            continue
        weights = None
        if class_weights is not None and task_name in class_weights:
            weights = class_weights[task_name].to(logits.device)
        loss = classification_loss(
            logits=logits,
            targets=targets_by_task[task_name].to(logits.device),
            ignore_index=ignore_index,
            class_weights=weights,
        )
        if torch.isfinite(loss):
            losses.append(loss)

    if not losses:
        # keep graph connectivity to logits
        first = next(iter(logits_by_task.values()))
        return first.sum() * 0.0

    return torch.stack(losses).mean()


def coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    """Correlation alignment loss between source and target feature batches.

    The loss aligns second-order statistics and is suitable for stable
    cross-dataset EEG representation alignment.
    """
    if source_features.numel() == 0 or target_features.numel() == 0:
        return source_features.sum() * 0.0 + target_features.sum() * 0.0

    if source_features.ndim > 2:
        source_features = source_features.flatten(start_dim=1)
    if target_features.ndim > 2:
        target_features = target_features.flatten(start_dim=1)

    d = source_features.shape[1]
    if target_features.shape[1] != d:
        raise ValueError("Source and target features must have the same feature dimension.")

    c_s = _covariance(source_features)
    c_t = _covariance(target_features)
    return torch.mean((c_s - c_t) ** 2) / (4.0 * d * d)


def graph_regularization_loss(
    a_learn: torch.Tensor | None,
    a_anat: torch.Tensor | None = None,
    delta: float = 0.10,
) -> torch.Tensor:
    """Regularise the learnable graph.

    The first term encourages sparsity; the second discourages excessive drift
    from anatomical priors when available.
    """
    if a_learn is None:
        return torch.tensor(0.0)

    sparsity = a_learn.abs().mean()
    if a_anat is None:
        return sparsity

    a_anat = a_anat.to(device=a_learn.device, dtype=a_learn.dtype)
    drift = torch.mean((a_learn - a_anat) ** 2)
    return sparsity + delta * drift


def total_loss(
    cls_loss: torch.Tensor,
    coral: torch.Tensor | None = None,
    graph_reg: torch.Tensor | None = None,
    lambda1: float = 0.05,
    lambda2: float = 1e-4,
) -> torch.Tensor:
    """Combine classification, CORAL, and graph regularisation losses."""
    loss = cls_loss
    if coral is not None:
        loss = loss + lambda1 * coral
    if graph_reg is not None:
        loss = loss + lambda2 * graph_reg
    return loss
