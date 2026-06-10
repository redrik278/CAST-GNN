"""Training package for CAST-GNN."""

from .losses import (
    classification_loss,
    multitask_classification_loss,
    coral_loss,
    graph_regularization_loss,
    total_loss,
)
from .early_stopping import EarlyStopping
from .trainer import Trainer, TrainerConfig

__all__ = [
    "classification_loss",
    "multitask_classification_loss",
    "coral_loss",
    "graph_regularization_loss",
    "total_loss",
    "EarlyStopping",
    "Trainer",
    "TrainerConfig",
]
