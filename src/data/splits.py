"""Subject-wise splitting utilities for CAST-GNN.

All split utilities enforce that no subject appears in more than one data split.
This is essential for leakage-safe EEG decoding because subject-specific signal
structure can inflate performance when epochs from the same participant are
shared across train and test partitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split


def create_subjectwise_split(
    metadata_df: pd.DataFrame,
    subject_col: str = "subject_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a leakage-safe train/validation/test split at subject level."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")
    if subject_col not in metadata_df.columns:
        raise KeyError(f"metadata_df must contain {subject_col!r}")

    subjects = np.array(sorted(metadata_df[subject_col].dropna().unique()))
    if len(subjects) < 3:
        raise ValueError("At least three subjects are required for train/val/test splitting")

    train_subjects, temp_subjects = train_test_split(
        subjects,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
    )
    rel_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        train_size=rel_val_ratio,
        random_state=seed,
        shuffle=True,
    )

    split_map = {s: "train" for s in train_subjects}
    split_map.update({s: "val" for s in val_subjects})
    split_map.update({s: "test" for s in test_subjects})

    out = metadata_df.copy()
    out["split"] = out[subject_col].map(split_map)
    if out["split"].isna().any():
        raise RuntimeError("Some samples were not assigned to a split")
    verify_no_subject_leakage(out, subject_col=subject_col, split_col="split", raise_error=True)
    return out


def create_group_kfold_splits(
    metadata_df: pd.DataFrame,
    subject_col: str = "subject_id",
    n_splits: int = 5,
    seed: int = 42,
) -> List[pd.DataFrame]:
    """Create group-wise K-fold split DataFrames.

    Each returned DataFrame contains columns `fold` and `split`, with the held-out
    fold labelled as `test` and all other samples labelled as `train`.  A separate
    validation split can be carved out from training subjects later if required.
    """
    if subject_col not in metadata_df.columns:
        raise KeyError(f"metadata_df must contain {subject_col!r}")
    groups = metadata_df[subject_col].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    splits: List[pd.DataFrame] = []
    dummy_y = np.zeros(len(metadata_df))
    for fold, (train_idx, test_idx) in enumerate(gkf.split(metadata_df, dummy_y, groups), start=1):
        df = metadata_df.copy()
        df["fold"] = fold
        df["split"] = "train"
        df.iloc[test_idx, df.columns.get_loc("split")] = "test"
        verify_no_subject_leakage(df, subject_col=subject_col, split_col="split", raise_error=True)
        splits.append(df)
    return splits


def create_train_val_from_train_subjects(
    split_df: pd.DataFrame,
    subject_col: str = "subject_id",
    split_col: str = "split",
    val_ratio_within_train: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a validation subset from subjects currently labelled as train."""
    df = split_df.copy()
    train_subjects = np.array(sorted(df.loc[df[split_col] == "train", subject_col].dropna().unique()))
    if len(train_subjects) < 2:
        raise ValueError("At least two training subjects are needed to create validation split")
    train_keep, val_subjects = train_test_split(
        train_subjects,
        test_size=val_ratio_within_train,
        random_state=seed,
        shuffle=True,
    )
    df.loc[df[subject_col].isin(val_subjects), split_col] = "val"
    verify_no_subject_leakage(df, subject_col=subject_col, split_col=split_col, raise_error=True)
    return df


def verify_no_subject_leakage(
    split_df: pd.DataFrame,
    subject_col: str = "subject_id",
    split_col: str = "split",
    raise_error: bool = False,
) -> bool:
    """Verify that each subject appears in exactly one split."""
    grouped = split_df.groupby(subject_col)[split_col].nunique(dropna=True)
    leaked = grouped[grouped > 1]
    ok = leaked.empty
    if not ok and raise_error:
        raise ValueError(f"Subject leakage detected for subjects: {leaked.index.tolist()}")
    return ok


def save_split(split_df: pd.DataFrame, output_path: str | Path) -> None:
    """Save split DataFrame as CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(path, index=False)


def load_split(path: str | Path) -> pd.DataFrame:
    """Load a split CSV file."""
    return pd.read_csv(path)
