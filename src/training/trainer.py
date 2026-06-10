"""Generic training engine for CAST-GNN models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.evaluation.metrics import compute_classification_metrics
from src.training.early_stopping import EarlyStopping
from src.training.losses import (
    coral_loss,
    graph_regularization_loss,
    multitask_classification_loss,
    total_loss,
)
from src.utils.device import move_to_device
from src.utils.io import save_checkpoint, write_csv


@dataclass
class TrainerConfig:
    """Configuration for model training."""

    num_epochs: int = 100
    lambda_coral: float = 0.05
    lambda_graph: float = 1e-4
    graph_delta: float = 0.10
    ignore_index: int = -100
    early_stopping_patience: int = 20
    monitor: str = "macro_f1"
    monitor_mode: str = "max"
    use_amp: bool = False
    grad_clip_norm: float | None = 5.0
    log_interval: int = 10


class Trainer:
    """Reusable trainer for within-dataset, transfer, and joint experiments.

    The trainer expects batches to be dictionaries containing at least ``x`` and
    a target label.  The target can be supplied as ``y``, ``label``, or as
    ``targets``/``targets_by_task`` for multi-task training.  The model output
    can be either a tensor of logits or a dictionary with a ``logits`` entry.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        output_dir: str | Path,
        config: TrainerConfig | None = None,
        scheduler: Any | None = None,
        logger: Any | None = None,
        metric_fn: Callable[..., dict[str, float]] = compute_classification_metrics,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or TrainerConfig()
        self.logger = logger
        self.metric_fn = metric_fn
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp and device.type == "cuda")
        self.history: list[dict[str, Any]] = []

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    def _extract_logits(self, output: Any) -> torch.Tensor | Mapping[str, torch.Tensor]:
        if torch.is_tensor(output):
            return output
        if isinstance(output, Mapping):
            if "logits" in output:
                return output["logits"]
            if "outputs" in output:
                return output["outputs"]
        raise KeyError("Model output must be a tensor or contain key 'logits'.")

    def _extract_features(self, output: Any) -> torch.Tensor | None:
        if isinstance(output, Mapping):
            for key in ("features", "shared_features", "embedding", "z"):
                if key in output and torch.is_tensor(output[key]):
                    return output[key]
        return None

    def _extract_graphs(self, output: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not isinstance(output, Mapping):
            return None, None
        a_learn = output.get("A_learn", output.get("a_learn", output.get("learned_adjacency")))
        a_anat = output.get("A_anat", output.get("a_anat", output.get("anatomical_adjacency")))
        return a_learn, a_anat

    def _extract_targets(self, batch: Mapping[str, Any]) -> torch.Tensor | Mapping[str, torch.Tensor]:
        if "targets_by_task" in batch:
            return batch["targets_by_task"]
        if "targets" in batch:
            return batch["targets"]
        if "task_labels" in batch:
            return batch["task_labels"]
        if "y" in batch:
            return batch["y"]
        if "label" in batch:
            return batch["label"]
        raise KeyError("Batch must contain targets as y, label, targets, or targets_by_task.")

    def _compute_loss(
        self,
        output: Any,
        batch: Mapping[str, Any],
        source_features: torch.Tensor | None = None,
        target_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = self._extract_logits(output)
        targets = self._extract_targets(batch)
        cls = multitask_classification_loss(
            logits,
            targets,
            ignore_index=self.config.ignore_index,
        )

        c_loss = None
        if source_features is not None and target_features is not None:
            c_loss = coral_loss(source_features, target_features)

        a_learn, a_anat = self._extract_graphs(output)
        g_loss = graph_regularization_loss(a_learn, a_anat, delta=self.config.graph_delta)

        loss = total_loss(
            cls_loss=cls,
            coral=c_loss,
            graph_reg=g_loss,
            lambda1=self.config.lambda_coral,
            lambda2=self.config.lambda_graph,
        )

        parts = {
            "loss": float(loss.detach().cpu()),
            "loss_cls": float(cls.detach().cpu()),
            "loss_graph": float(g_loss.detach().cpu()) if g_loss is not None else 0.0,
            "loss_coral": float(c_loss.detach().cpu()) if c_loss is not None else 0.0,
        }
        return loss, parts

    def _forward(self, batch: Mapping[str, Any]) -> Any:
        """Forward pass with the full batch.

        CAST-GNN implementations often expect metadata such as ``dataset`` or
        ``task_name`` in addition to the EEG tensor.  Passing the full batch is
        therefore safer than passing only ``x``.
        """
        try:
            return self.model(batch)
        except TypeError:
            return self.model(batch["x"])

    def train_epoch(self, train_loader: DataLoader, epoch: int = 0) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        totals: dict[str, float] = {"loss": 0.0, "loss_cls": 0.0, "loss_graph": 0.0, "loss_coral": 0.0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            batch = move_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                output = self._forward(batch)
                loss, parts = self._compute_loss(output, batch)

            self.scaler.scale(loss).backward()
            if self.config.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for key, value in parts.items():
                totals[key] += value
            n_batches += 1

            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix(loss=parts["loss"])

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> dict[str, Any]:
        """Collect predictions, probabilities, targets, and features."""
        self.model.eval()
        logits_all: list[torch.Tensor] = []
        targets_all: list[torch.Tensor] = []
        features_all: list[torch.Tensor] = []
        losses: list[float] = []

        for batch in tqdm(loader, desc="predict", leave=False):
            batch = move_to_device(batch, self.device)
            output = self._forward(batch)
            loss, parts = self._compute_loss(output, batch)
            losses.append(parts["loss"])

            logits = self._extract_logits(output)
            targets = self._extract_targets(batch)

            if isinstance(logits, Mapping):
                # For generic prediction, use the first task with available targets.
                task = next(iter(logits.keys()))
                logits = logits[task]
                if isinstance(targets, Mapping):
                    targets = targets[task]
            if not torch.is_tensor(targets):
                raise TypeError("Targets must resolve to a tensor for prediction.")

            logits_all.append(logits.detach().cpu())
            targets_all.append(targets.detach().cpu())

            features = self._extract_features(output)
            if features is not None:
                features_all.append(features.detach().cpu())

        logits_cat = torch.cat(logits_all, dim=0) if logits_all else torch.empty(0)
        targets_cat = torch.cat(targets_all, dim=0) if targets_all else torch.empty(0, dtype=torch.long)
        probs = torch.softmax(logits_cat, dim=1) if logits_cat.numel() else torch.empty(0)
        preds = probs.argmax(dim=1) if probs.numel() else torch.empty(0, dtype=torch.long)

        result = {
            "logits": logits_cat,
            "probs": probs,
            "preds": preds,
            "targets": targets_cat,
            "loss": float(sum(losses) / max(len(losses), 1)),
        }
        if features_all:
            result["features"] = torch.cat(features_all, dim=0)
        return result

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate a dataloader using classification metrics."""
        pred = self.predict(loader)
        targets = pred["targets"].numpy()
        preds = pred["preds"].numpy()
        probs = pred["probs"].numpy() if pred["probs"].numel() else None
        metrics = self.metric_fn(targets, preds, probs)
        metrics["loss"] = pred["loss"]
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int | None = None,
        checkpoint_name: str = "best_model.pt",
    ) -> pd.DataFrame:
        """Train with early stopping on a validation metric."""
        num_epochs = num_epochs or self.config.num_epochs
        stopper = EarlyStopping(
            patience=self.config.early_stopping_patience,
            mode=self.config.monitor_mode,
            min_delta=1e-4,
        )
        best_path = self.output_dir / checkpoint_name

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader, epoch=epoch)
            val_metrics = self.evaluate(val_loader)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(self.config.monitor, val_metrics["loss"]))
                else:
                    self.scheduler.step()

            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
            self.history.append(row)
            self._log(str(row))

            monitor_value = val_metrics.get(self.config.monitor)
            if monitor_value is None:
                monitor_value = -val_metrics["loss"] if self.config.monitor_mode == "max" else val_metrics["loss"]
            improved = stopper.step(float(monitor_value))
            if improved:
                self.save_checkpoint(best_path, epoch=epoch, metrics=val_metrics)
            if stopper.should_stop:
                self._log(f"Early stopping at epoch {epoch}. Best epoch: {stopper.best_epoch}.")
                break

        history_df = pd.DataFrame(self.history)
        write_csv(history_df, self.output_dir / "history.csv")
        return history_df

    def save_checkpoint(self, path: str | Path, epoch: int, metrics: Mapping[str, float] | None = None) -> None:
        """Save model, optimizer, scheduler, and trainer configuration."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": dict(metrics or {}),
            "trainer_config": asdict(self.config),
        }
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        save_checkpoint(state, path)

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> dict[str, Any]:
        """Load a checkpoint into the model and return checkpoint metadata."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt
