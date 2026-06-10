"""MILimbEEG CSV loader for CAST-GNN.

The loader recursively scans CSV files, parses subject/task/mode metadata from
file paths, identifies numeric EEG columns, removes non-signal columns where
possible, and returns trial arrays in [channels, samples] format.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .label_harmonization import MILIMBEEG_MOVEMENT_MAP, parse_milimbeeg_label


MOVEMENT_LABELS = {"BEO", "REST", "CLH", "CRH", "DLF", "PLF", "DRF", "PRF"}
EXECUTION_TOKENS = {"execution", "exec", "me", "motor_execution", "real", "movement"}
IMAGERY_TOKENS = {"imagery", "imagination", "imagined", "mi", "motor_imagery"}


class MILimbEEGLoader:
    """Loader for MILimbEEG CSV trials.

    Parameters
    ----------
    root_dir:
        Root directory containing CSV files.
    expected_channels:
        Expected number of EEG channels, usually 16.
    expected_sfreq:
        Expected sampling frequency, usually 125 Hz.
    """

    def __init__(self, root_dir: str | Path, expected_channels: int = 16, expected_sfreq: float = 125.0) -> None:
        self.root_dir = Path(root_dir)
        self.expected_channels = int(expected_channels)
        self.expected_sfreq = float(expected_sfreq)

    def scan_files(self) -> pd.DataFrame:
        """Recursively scan CSV files and return parsed metadata."""
        files = sorted(self.root_dir.rglob("*.csv"))
        rows: List[Dict[str, Any]] = []
        for fp in files:
            rows.append({"file_path": str(fp), **self.parse_file_metadata(fp)})
        return pd.DataFrame(rows)

    def parse_file_metadata(self, file_path: str | Path) -> Dict[str, Any]:
        """Parse subject ID, movement label, mode, and trial ID from a path.

        The public MILimbEEG distributions are not always packaged with the same
        folder names.  This parser therefore uses robust token matching rather
        than assuming one exact naming convention.
        """
        path = Path(file_path)
        text = " ".join(path.parts).replace("-", "_")
        tokens = re.split(r"[^A-Za-z0-9]+", text)
        tokens_clean = [t for t in tokens if t]
        tokens_upper = [t.upper() for t in tokens_clean]
        tokens_lower = [t.lower() for t in tokens_clean]

        # Subject extraction: supports S01, sub01, subject_01, person01, etc.
        subject_id: Optional[str] = None
        subject_patterns = [
            r"(?:^|[^A-Za-z])S(?:UBJECT)?_?(\d{1,3})(?:$|[^A-Za-z0-9])",
            r"SUB(?:JECT)?_?(\d{1,3})",
            r"PERSON_?(\d{1,3})",
        ]
        joined = "_".join(tokens_clean)
        for pat in subject_patterns:
            m = re.search(pat, joined, flags=re.IGNORECASE)
            if m:
                subject_id = f"S{int(m.group(1)):03d}"
                break
        if subject_id is None:
            # Fallback: use nearest token that looks like S01.
            for tok in tokens_clean:
                m = re.match(r"^[sS](\d{1,3})$", tok)
                if m:
                    subject_id = f"S{int(m.group(1)):03d}"
                    break

        movement_label: Optional[str] = None
        for tok in tokens_upper:
            if tok in MOVEMENT_LABELS:
                movement_label = "Rest" if tok == "REST" else tok
                break

        mode: Optional[str] = None
        for tok in tokens_lower:
            if tok in EXECUTION_TOKENS:
                mode = "execution"
                break
            if tok in IMAGERY_TOKENS:
                mode = "imagery"
                break
        if movement_label in {"BEO", "Rest"}:
            mode = "baseline" if movement_label == "BEO" else "rest"

        trial_id: Optional[str] = None
        trial_match = re.search(r"(?:trial|tr|t)_?(\d+)", joined, flags=re.IGNORECASE)
        if trial_match:
            trial_id = f"T{int(trial_match.group(1)):04d}"
        else:
            # Fallback to file stem for uniqueness.
            trial_id = path.stem

        metadata = {
            "dataset": "milimbeeg",
            "subject_id": subject_id,
            "movement_label": movement_label,
            "mode": mode,
            "trial_id": trial_id,
            "filename_valid": bool(subject_id is not None and movement_label is not None),
        }
        parsed_label = parse_milimbeeg_label(metadata)
        if parsed_label is not None:
            metadata.update(parsed_label)
        return metadata

    def detect_signal_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect numeric EEG signal columns.

        Non-signal columns such as time, index, sample counters, marker, or label
        are excluded when their names suggest metadata rather than EEG channels.
        """
        non_signal_patterns = re.compile(r"time|index|sample|counter|marker|label|class|event|stim", re.IGNORECASE)
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        candidate_cols = [c for c in numeric_cols if not non_signal_patterns.search(str(c))]
        if len(candidate_cols) >= self.expected_channels:
            return candidate_cols[: self.expected_channels]
        # Fallback: return the last expected numeric columns, which often hold channels after sample/time columns.
        if len(numeric_cols) >= self.expected_channels:
            return numeric_cols[-self.expected_channels :]
        return numeric_cols

    def load_csv_trial(self, file_path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load one CSV trial as [channels, samples] with parsed metadata."""
        path = Path(file_path)
        metadata = {"file_path": str(path), **self.parse_file_metadata(path)}
        df = pd.read_csv(path)
        signal_cols = self.detect_signal_columns(df)
        if len(signal_cols) == 0:
            raise ValueError(f"No numeric signal columns detected in {path}")
        signal = df[signal_cols].to_numpy(dtype=np.float32).T
        metadata.update(
            {
                "n_channels": int(signal.shape[0]),
                "n_samples": int(signal.shape[1]),
                "sfreq": self.expected_sfreq,
                "channel_names": ",".join(map(str, signal_cols)),
            }
        )
        if metadata.get("sample_id") is None:
            subject = metadata.get("subject_id") or "unknown_subject"
            label = metadata.get("movement_label") or "unknown_label"
            mode = metadata.get("mode") or "unknown_mode"
            trial = metadata.get("trial_id") or path.stem
            metadata["sample_id"] = f"{subject}_{label}_{mode}_{trial}"
        return signal, metadata

    @staticmethod
    def standardize_trial_length(
        signal: np.ndarray,
        target_samples: int = 500,
        mode: str = "trim_or_pad",
    ) -> np.ndarray:
        """Standardize a trial to a fixed number of samples.

        If the signal is longer than the target length, it is center-cropped.  If
        shorter, it is padded with zeros at the end.  This operation should be
        logged by the caller when applied to real data.
        """
        arr = np.asarray(signal, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("signal must have shape [channels, samples]")
        n = arr.shape[1]
        if n == target_samples:
            return arr
        if n > target_samples:
            start = (n - target_samples) // 2
            return arr[:, start : start + target_samples]
        if mode != "trim_or_pad":
            raise ValueError(f"Unsupported mode {mode!r}")
        pad_width = target_samples - n
        return np.pad(arr, ((0, 0), (0, pad_width)), mode="constant")
