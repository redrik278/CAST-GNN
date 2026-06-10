"""Within-dataset CAST-GNN training entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import EEGTaskDataset
from src.models.cast_gnn import CASTGNN
from src.models.heads import TASK_NUM_CLASSES
from src.training.trainer import Trainer, TrainerConfig
from src.utils import get_device, set_seed, setup_logging, write_csv


def _filter_split(metadata: pd.DataFrame, split: str) -> pd.DataFrame:
    if "split" not in metadata.columns:
        raise ValueError("Metadata must contain a 'split' column.")
    return metadata[metadata["split"].astype(str).str.lower() == split.lower()].reset_index(drop=True)


def build_within_dataloaders(
    metadata_path: str | Path,
    processed_dir: str | Path,
    task_name: str,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/validation/test dataloaders from metadata with split labels."""
    metadata = pd.read_csv(metadata_path)
    train_ds = EEGTaskDataset(_filter_split(metadata, "train"), processed_dir, task_name)
    val_ds = EEGTaskDataset(_filter_split(metadata, "val"), processed_dir, task_name)
    test_ds = EEGTaskDataset(_filter_split(metadata, "test"), processed_dir, task_name)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def run_within_dataset_training(
    dataset_name: str,
    config: dict[str, Any],
    task_name: str,
    seed: int = 42,
) -> dict[str, float]:
    """Run subject-wise within-dataset training.

    Expected config keys include ``metadata_path``, ``processed_dir``,
    ``output_dir``, ``learning_rate``, ``weight_decay``, and ``batch_size``.
    """
    set_seed(seed)
    device = get_device(config.get("device", "auto"))
    output_dir = Path(config.get("output_dir", "outputs")) / "within" / dataset_name / task_name / f"seed_{seed}"
    logger = setup_logging(output_dir / "logs", "train.log")

    train_loader, val_loader, test_loader = build_within_dataloaders(
        metadata_path=config["metadata_path"],
        processed_dir=config["processed_dir"],
        task_name=task_name,
        batch_size=int(config.get("batch_size", 32)),
        num_workers=int(config.get("num_workers", 0)),
    )

    dataset_configs = {
        dataset_name: {
            "num_channels": int(config.get("num_channels", 64 if dataset_name == "eegmmidb" else 16)),
            "functional_k": int(config.get("functional_k", 8 if dataset_name == "eegmmidb" else 4)),
        }
    }

    task_classes = config.get("task_num_classes", TASK_NUM_CLASSES)
    model = CASTGNN(dataset_configs=dataset_configs, task_num_classes=task_classes)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config.get("num_epochs", 100)),
    )

    trainer_cfg = TrainerConfig(
        num_epochs=int(config.get("num_epochs", 100)),
        early_stopping_patience=int(config.get("early_stopping_patience", 20)),
        use_amp=bool(config.get("use_amp", False)),
    )
    trainer = Trainer(model, optimizer, device, output_dir, trainer_cfg, scheduler=scheduler, logger=logger)
    trainer.fit(train_loader, val_loader)

    test_metrics = trainer.evaluate(test_loader)
    write_csv(pd.DataFrame([test_metrics]), output_dir / "test_metrics.csv")
    return test_metrics
