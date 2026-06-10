"""PyTorch dataset wrappers for CAST-GNN.

The dataset expects a metadata CSV/DataFrame where each row points to a processed
sample file.  Processed samples may be `.npy`, `.npz`, or `.pt` files containing
an EEG tensor with shape [bands, channels, samples].
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .label_harmonization import get_task_column


class EEGTaskDataset(Dataset):
    """Dataset for single-task EEG decoding.

    Parameters
    ----------
    metadata_df:
        DataFrame containing at least `sample_path`, `dataset`, `subject_id`, and
        the task-specific label column.
    task_name:
        Human-readable task name, e.g. `B_execution_vs_imagery`.
    processed_dir:
        Optional base directory prepended to relative sample paths.
    transform:
        Optional callable applied to the loaded tensor.
    label_col:
        Optional explicit label column.  If omitted, inferred from `task_name`.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        task_name: str,
        processed_dir: str | Path | None = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        label_col: Optional[str] = None,
        sample_path_col: str = "sample_path",
        drop_missing_labels: bool = True,
    ) -> None:
        self.task_name = task_name
        self.label_col = label_col or get_task_column(task_name)
        self.processed_dir = Path(processed_dir) if processed_dir is not None else None
        self.transform = transform
        self.sample_path_col = sample_path_col

        if self.label_col not in metadata_df.columns:
            raise KeyError(f"Metadata does not contain label column {self.label_col!r}")
        if sample_path_col not in metadata_df.columns:
            raise KeyError(f"Metadata does not contain sample path column {sample_path_col!r}")

        df = metadata_df.copy()
        if drop_missing_labels:
            df = df[df[self.label_col].notna()].reset_index(drop=True)
        self.metadata = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def _resolve_path(self, sample_path: str | Path) -> Path:
        path = Path(sample_path)
        if path.is_absolute() or self.processed_dir is None:
            return path
        return self.processed_dir / path

    @staticmethod
    def _load_tensor(path: Path) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Processed sample file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".npy":
            arr = np.load(path)
            return torch.from_numpy(arr).float()
        if suffix == ".npz":
            data = np.load(path)
            key = "x" if "x" in data.files else data.files[0]
            return torch.from_numpy(data[key]).float()
        if suffix in {".pt", ".pth"}:
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, torch.Tensor):
                return obj.float()
            if isinstance(obj, dict):
                if "x" not in obj:
                    raise KeyError(f"Torch file {path} is a dict but does not contain key 'x'")
                return obj["x"].float()
        raise ValueError(f"Unsupported processed sample format: {path.suffix}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata.iloc[idx]
        path = self._resolve_path(row[self.sample_path_col])
        x = self._load_tensor(path)
        if self.transform is not None:
            x = self.transform(x)

        label = int(row[self.label_col])
        task_mask = torch.tensor(1, dtype=torch.bool)
        sample_id = str(row.get("sample_id", path.stem))
        return {
            "x": x,  # [bands, channels, time]
            "y": torch.tensor(label, dtype=torch.long),
            "task_name": self.task_name,
            "task_mask": task_mask,
            "dataset": str(row.get("dataset", "unknown")),
            "subject_id": str(row.get("subject_id", "unknown")),
            "sample_id": sample_id,
        }


def eeg_collate_fn(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate EEGTaskDataset samples into a batch.

    This function assumes all samples in a batch have the same channel count.  For
    joint training with heterogeneous channel counts, use dataset-specific batch
    sampling or separate dataloaders per dataset.
    """
    x = torch.stack([item["x"] for item in batch], dim=0)
    y = torch.stack([item["y"] for item in batch], dim=0)
    task_mask = torch.stack([item["task_mask"] for item in batch], dim=0)
    return {
        "x": x,
        "y": y,
        "task_name": batch[0]["task_name"],
        "task_mask": task_mask,
        "dataset": [item["dataset"] for item in batch],
        "subject_id": [item["subject_id"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
    }


def make_dataloader(
    dataset: EEGTaskDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a PyTorch DataLoader with the project collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=eeg_collate_fn,
    )
