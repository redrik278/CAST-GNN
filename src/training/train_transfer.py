"""Cross-dataset transfer training utilities."""

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


VALID_TRANSFER_TASKS = {
    "B_execution_vs_imagery",
    "C_upper_vs_lower",
    "D_left_vs_right",
    "task_b",
    "task_c",
    "task_d",
}


def _split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"].astype(str).str.lower() == split.lower()].reset_index(drop=True)


def run_transfer_training(
    source_dataset: str,
    target_dataset: str,
    config: dict[str, Any],
    task_name: str,
    seed: int = 42,
) -> dict[str, float]:
    """Train on a source dataset and evaluate on a held-out target dataset.

    This function enforces that only scientifically comparable tasks are used
    for transfer.  In practice, use Tasks B, C, and D.
    """
    if task_name not in VALID_TRANSFER_TASKS and not task_name.lower().startswith(("b_", "c_", "d_")):
        raise ValueError(f"Task {task_name!r} is not recommended for direct cross-dataset transfer.")

    set_seed(seed)
    device = get_device(config.get("device", "auto"))
    output_dir = Path(config.get("output_dir", "outputs")) / "transfer" / f"{source_dataset}_to_{target_dataset}" / task_name / f"seed_{seed}"
    logger = setup_logging(output_dir / "logs", "train.log")

    src_meta = pd.read_csv(config["source_metadata_path"])
    tgt_meta = pd.read_csv(config["target_metadata_path"])

    source_train = EEGTaskDataset(_split(src_meta, "train"), config["source_processed_dir"], task_name)
    source_val = EEGTaskDataset(_split(src_meta, "val"), config["source_processed_dir"], task_name)
    target_test = EEGTaskDataset(_split(tgt_meta, "test"), config["target_processed_dir"], task_name)

    train_loader = DataLoader(source_train, batch_size=int(config.get("batch_size", 32)), shuffle=True)
    val_loader = DataLoader(source_val, batch_size=int(config.get("batch_size", 32)), shuffle=False)
    test_loader = DataLoader(target_test, batch_size=int(config.get("batch_size", 32)), shuffle=False)

    dataset_configs = {
        source_dataset: {"num_channels": int(config.get("source_num_channels", 64)), "functional_k": int(config.get("source_functional_k", 8))},
        target_dataset: {"num_channels": int(config.get("target_num_channels", 16)), "functional_k": int(config.get("target_functional_k", 4))},
    }
    model = CASTGNN(dataset_configs=dataset_configs, task_num_classes=config.get("task_num_classes", TASK_NUM_CLASSES))

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-3)), weight_decay=float(config.get("weight_decay", 1e-4)))
    trainer_cfg = TrainerConfig(
        num_epochs=int(config.get("num_epochs", 100)),
        lambda_coral=float(config.get("lambda1", 0.05)),
        lambda_graph=float(config.get("lambda2", 1e-4)),
        early_stopping_patience=int(config.get("early_stopping_patience", 20)),
        use_amp=bool(config.get("use_amp", False)),
    )
    trainer = Trainer(model, optimizer, device, output_dir, trainer_cfg, logger=logger)
    trainer.fit(train_loader, val_loader)

    metrics = trainer.evaluate(test_loader)
    write_csv(pd.DataFrame([metrics]), output_dir / "target_test_metrics.csv")
    return metrics
