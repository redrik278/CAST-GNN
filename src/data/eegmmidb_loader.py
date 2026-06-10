"""EEGMMIDB loader for CAST-GNN.

This module loads PhysioNet EEG Motor Movement/Imagery Dataset EDF files using
MNE, parses subject/run identifiers, reconstructs run-specific event labels, and
extracts fixed-length epochs.  Signals returned by this loader are expressed in
microvolts to make downstream quality checks easier to interpret.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import mne
except Exception:  # pragma: no cover - allows static import without MNE installed
    mne = None

from .label_harmonization import reconstruct_eegmmidb_label


EEGMMIDB_FILENAME_RE = re.compile(r"S(?P<subject>\d{3})R(?P<run>\d{2})", re.IGNORECASE)


class EEGMMIDBLoader:
    """Loader for PhysioNet EEGMMIDB EDF recordings.

    Parameters
    ----------
    root_dir:
        Directory containing EDF files, possibly nested.
    target_sfreq:
        Optional target sampling frequency for downstream cross-dataset work.
        The loader does not resample by default unless `resample=True` is passed
        to `load_raw_edf`.
    preload:
        Passed to MNE when loading EDF files.
    verbose:
        MNE verbosity flag.
    """

    def __init__(
        self,
        root_dir: str | Path,
        target_sfreq: float = 125.0,
        preload: bool = False,
        verbose: bool | str = "ERROR",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.target_sfreq = float(target_sfreq)
        self.preload = preload
        self.verbose = verbose

    def scan_files(self) -> pd.DataFrame:
        """Find EDF files recursively and return parsed file metadata."""
        files = sorted(list(self.root_dir.rglob("*.edf")) + list(self.root_dir.rglob("*.EDF")))
        rows: List[Dict[str, Any]] = []
        for fp in files:
            parsed = self.parse_filename(fp)
            rows.append({"file_path": str(fp), **parsed})
        return pd.DataFrame(rows)

    @staticmethod
    def parse_filename(file_path: str | Path) -> Dict[str, Any]:
        """Parse EEGMMIDB subject and run IDs from a filename."""
        path = Path(file_path)
        match = EEGMMIDB_FILENAME_RE.search(path.stem)
        if not match:
            return {"subject_id": None, "run_id": None, "filename_valid": False}
        subject = match.group("subject")
        run = int(match.group("run"))
        return {"subject_id": f"S{subject}", "run_id": run, "filename_valid": True}

    def _require_mne(self) -> None:
        if mne is None:
            raise ImportError("mne is required to load EEGMMIDB EDF files. Install it with `pip install mne`.")

    def load_raw_edf(self, file_path: str | Path, resample: bool = False):
        """Load an EDF recording with MNE.

        Parameters
        ----------
        file_path:
            Path to EDF file.
        resample:
            If True, resample the raw object to `target_sfreq`.  Resampling is
            performed after loading and should only be used after train/test
            leakage risks have been considered.
        """
        self._require_mne()
        raw = mne.io.read_raw_edf(str(file_path), preload=self.preload or resample, verbose=self.verbose)
        if resample and abs(raw.info["sfreq"] - self.target_sfreq) > 1e-6:
            raw.resample(self.target_sfreq, npad="auto", verbose=self.verbose)
        return raw

    def extract_annotations(self, raw: Any, subject_id: str, run_id: int, file_path: Optional[str] = None) -> pd.DataFrame:
        """Extract raw and reconstructed event annotations from an MNE Raw object."""
        rows: List[Dict[str, Any]] = []
        for ann in raw.annotations:
            raw_label = str(ann["description"])
            mapped = reconstruct_eegmmidb_label(run_id, raw_label)
            if mapped is None:
                mapped = {
                    "dataset": "eegmmidb",
                    "state": "unknown",
                    "mode": "unknown",
                    "body_part": "unknown",
                    "semantic_label": "unknown",
                    "raw_label": raw_label,
                    "run_id": run_id,
                }
            rows.append(
                {
                    "dataset": "eegmmidb",
                    "file_path": file_path,
                    "subject_id": subject_id,
                    "run_id": run_id,
                    "onset_sec": float(ann["onset"]),
                    "duration_sec": float(ann["duration"]),
                    **mapped,
                }
            )
        return pd.DataFrame(rows)

    def load_signal_uv(self, raw: Any, picks: Optional[Iterable[str]] = None) -> Tuple[np.ndarray, List[str], float]:
        """Return EEG data as [channels, samples] in microvolts."""
        if picks is None:
            picks = raw.ch_names
        data = raw.get_data(picks=list(picks)) * 1e6
        channel_names = list(picks)
        sfreq = float(raw.info["sfreq"])
        return data.astype(np.float32), channel_names, sfreq

    def extract_epochs(
        self,
        raw: Any,
        events_df: pd.DataFrame,
        epoch_length: float = 4.0,
        include_states: Optional[Iterable[str]] = None,
        file_path: Optional[str] = None,
    ) -> Tuple[List[np.ndarray], pd.DataFrame]:
        """Extract fixed-length epochs from event onsets.

        Parameters
        ----------
        raw:
            MNE Raw object.
        events_df:
            DataFrame returned by `extract_annotations`.
        epoch_length:
            Duration of each epoch in seconds.
        include_states:
            Optional set of states to include, e.g. {"task", "rest"}.
        file_path:
            Optional file path stored in sample metadata.

        Returns
        -------
        epochs:
            List of arrays with shape [channels, samples] in microvolts.
        metadata:
            DataFrame with one row per epoch.
        """
        data_uv, channel_names, sfreq = self.load_signal_uv(raw)
        n_samples = data_uv.shape[1]
        win = int(round(epoch_length * sfreq))
        include = set(include_states) if include_states is not None else None

        epochs: List[np.ndarray] = []
        rows: List[Dict[str, Any]] = []
        for _, row in events_df.iterrows():
            if include is not None and row.get("state") not in include:
                continue
            onset = int(round(float(row["onset_sec"]) * sfreq))
            end = onset + win
            if onset < 0 or end > n_samples:
                continue
            epoch = data_uv[:, onset:end]
            sample_id = f"{row['subject_id']}_R{int(row['run_id']):02d}_{int(onset)}"
            epochs.append(epoch.astype(np.float32))
            meta = row.to_dict()
            meta.update(
                {
                    "sample_id": sample_id,
                    "file_path": file_path or meta.get("file_path"),
                    "sfreq": sfreq,
                    "n_channels": len(channel_names),
                    "n_samples": win,
                    "channel_names": ",".join(channel_names),
                }
            )
            rows.append(meta)
        return epochs, pd.DataFrame(rows)

    def load_file_metadata(self, file_path: str | Path) -> Dict[str, Any]:
        """Load basic EDF metadata without extracting epochs."""
        parsed = self.parse_filename(file_path)
        raw = self.load_raw_edf(file_path, resample=False)
        return {
            "file_path": str(file_path),
            **parsed,
            "sfreq": float(raw.info["sfreq"]),
            "n_channels": int(len(raw.ch_names)),
            "n_samples": int(raw.n_times),
            "duration_sec": float(raw.n_times / raw.info["sfreq"]),
            "channel_names": ",".join(raw.ch_names),
            "n_annotations": len(raw.annotations),
        }
