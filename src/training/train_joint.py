"""Joint training utilities for EEGMMIDB and MILimbEEG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.data.dataset import EEGTaskDataset
from src.models.cast_gnn import CASTGNN
from src.models.heads import TASK_NUM_CLASSES
from src.training.trainer import Trainer, TrainerConfig
from src.utils import get_device, set_seed, setup_logging, write_csv


def _split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"].astype(str).str.lower() == split.lower()].reset_index(drop=True)


def run_joint_training(
    config: dict[str, Any],
    task_name: str,
    seed: int = 42,
    use_graph_adapters: bool = True,
    use_coral: bool = True,
) -> dict[str, float]:
    """Run joint training across EEGMMIDB and MILimbEEG.

    The function concatenates datasets at the dataloader level while preserving
    dataset identifiers in each sample.  The model is expected to use these
    identifiers to route examples through dataset-specific stems/adapters.
    """
    set_seed(seed)
    device = get_device(config.get("device", "auto"))

    variant = f"adapter_{int(use_graph_adapters)}_coral_{int(use_coral)}"
    output_dir = Path(config.get("output_dir", "outputs")) / "joint" / task_name / variant / f"seed_{seed}"
    logger = setup_logging(output_dir / "logs", "train.log")

    eeg_meta = pd.read_csv(config["eegmmidb_metadata_path"])
    mil_meta = pd.read_csv(config["milimbeeg_metadata_path"])

    train_ds = ConcatDataset([
        EEGTaskDataset(_split(eeg_meta, "train"), config["eegmmidb_processed_dir"], task_name),
        EEGTaskDataset(_split(mil_meta, "train"), config["milimbeeg_processed_dir"], task_name),
    ])
    val_ds = ConcatDataset([
        EEGTaskDataset(_split(eeg_meta, "val"), config["eegmmidb_processed_dir"], task_name),
        EEGTaskDataset(_split(mil_meta, "val"), config["milimbeeg_processed_dir"], task_name),
    ])
    test_ds = ConcatDataset([
        EEGTaskDataset(_split(eeg_meta, "test"), config["eegmmidb_processed_dir"], task_name),
        EEGTaskDataset(_split(mil_meta, "test"), config["milimbeeg_processed_dir"], task_name),
    ])

    batch_size = int(config.get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    dataset_configs = {
        "eegmmidb": {"num_channels": int(config.get("eegmmidb_num_channels", 64)), "functional_k": int(config.get("eegmmidb_functional_k", 8))},
        "milimbeeg": {"num_channels": int(config.get("milimbeeg_num_channels", 16)), "functional_k": int(config.get("milimbeeg_functional_k", 4))},
    }
    model = CASTGNN(
        dataset_configs=dataset_configs,
        task_num_classes=config.get("task_num_classes", TASK_NUM_CLASSES),
        use_graph_adapters=use_graph_adapters,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-3)), weight_decay=float(config.get("weight_decay", 1e-4)))
    trainer_cfg = TrainerConfig(
        num_epochs=int(config.get("num_epochs", 100)),
        lambda_coral=float(config.get("lambda1", 0.05 if use_coral else 0.0)),
        lambda_graph=float(config.get("lambda2", 1e-4)),
        early_stopping_patience=int(config.get("early_stopping_patience", 20)),
        use_amp=bool(config.get("use_amp", False)),
    )
    trainer = Trainer(model, optimizer, device, output_dir, trainer_cfg, logger=logger)
    trainer.fit(train_loader, val_loader)

    metrics = trainer.evaluate(test_loader)
    write_csv(pd.DataFrame([metrics]), output_dir / "joint_test_metrics.csv")
    return metrics
