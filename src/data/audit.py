"""Dataset audit utilities for EEGMMIDB and MILimbEEG.

The audit layer is deliberately independent from model training.  It records
readability, shape, channel count, sampling rate, missing/non-finite values,
flat channels, amplitude warnings, and duration/sample-count anomalies.  Files
should never be silently dropped; warnings and exclusion candidates are logged in
CSV audit reports.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class AuditConfig:
    """Configuration for dataset audit checks."""

    expected_channels: int
    expected_sfreq: Optional[float] = None
    expected_samples: Optional[int] = None
    amplitude_threshold_uv: float = 500.0
    flat_std_threshold: float = 1e-8
    min_sample_ratio: float = 0.95


@dataclass
class AuditRecord:
    """A single file/trial/epoch audit record."""

    file_path: str
    readable: bool
    dataset: str
    subject_id: Optional[str] = None
    run_id: Optional[int] = None
    n_channels: Optional[int] = None
    n_samples: Optional[int] = None
    sfreq: Optional[float] = None
    duration_sec: Optional[float] = None
    channel_names: Optional[str] = None
    has_missing: Optional[bool] = None
    has_non_finite: Optional[bool] = None
    n_flat_channels: Optional[int] = None
    flat_channels: Optional[str] = None
    amplitude_warning: Optional[bool] = None
    sample_count_warning: Optional[bool] = None
    channel_count_warning: Optional[bool] = None
    sampling_rate_warning: Optional[bool] = None
    exclusion_candidate: Optional[bool] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def check_numeric_validity(signal: np.ndarray) -> Dict[str, bool]:
    """Check missing and non-finite values in a numeric signal array."""
    arr = np.asarray(signal)
    return {
        "has_missing": bool(np.isnan(arr).any()),
        "has_non_finite": bool(~np.isfinite(arr).all()),
    }


def detect_flat_channels(signal: np.ndarray, threshold: float = 1e-8) -> List[int]:
    """Return channel indices with near-zero standard deviation."""
    arr = np.asarray(signal)
    if arr.ndim != 2:
        raise ValueError("signal must have shape [channels, samples]")
    std = np.nanstd(arr, axis=1)
    return np.where(std <= threshold)[0].astype(int).tolist()


def detect_amplitude_outliers(signal: np.ndarray, threshold_uv: float = 500.0) -> bool:
    """Flag signals exceeding a conservative absolute amplitude threshold.

    The function assumes the signal is in microvolts.  If units are unknown,
    users should interpret this as a warning rather than an automatic exclusion.
    """
    arr = np.asarray(signal)
    if arr.size == 0:
        return True
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return True
    return bool(np.nanmax(np.abs(finite)) > threshold_uv)


def robust_outlier_flags(values: Iterable[float], z_threshold: float = 8.0) -> np.ndarray:
    """Return robust median-absolute-deviation outlier flags."""
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return np.array([], dtype=bool)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return np.zeros_like(x, dtype=bool)
    robust_z = 0.6745 * (x - med) / mad
    return np.abs(robust_z) > z_threshold


def audit_signal_array(
    signal: np.ndarray,
    config: AuditConfig,
    file_path: str,
    dataset: str,
    subject_id: Optional[str] = None,
    run_id: Optional[int] = None,
    sfreq: Optional[float] = None,
    channel_names: Optional[Iterable[str]] = None,
) -> AuditRecord:
    """Audit a loaded signal array of shape [channels, samples]."""
    arr = np.asarray(signal)
    if arr.ndim != 2:
        raise ValueError("signal must have shape [channels, samples]")

    n_channels, n_samples = arr.shape
    validity = check_numeric_validity(arr)
    flat = detect_flat_channels(arr, threshold=config.flat_std_threshold)
    amplitude_warning = detect_amplitude_outliers(arr, threshold_uv=config.amplitude_threshold_uv)

    channel_count_warning = bool(n_channels != config.expected_channels)
    sampling_rate_warning = False
    if config.expected_sfreq is not None and sfreq is not None:
        sampling_rate_warning = bool(abs(float(sfreq) - float(config.expected_sfreq)) > 1e-6)

    sample_count_warning = False
    if config.expected_samples is not None:
        lower = int(np.floor(config.expected_samples * config.min_sample_ratio))
        sample_count_warning = bool(n_samples < lower or n_samples != config.expected_samples)

    failed_channel_ratio = len(flat) / max(n_channels, 1)
    exclusion_candidate = bool(
        validity["has_non_finite"]
        or failed_channel_ratio > 0.25
        or channel_count_warning
        or (config.expected_samples is not None and n_samples < int(config.expected_samples * config.min_sample_ratio))
    )

    duration = float(n_samples / sfreq) if sfreq not in (None, 0) else None
    return AuditRecord(
        file_path=str(file_path),
        readable=True,
        dataset=dataset,
        subject_id=subject_id,
        run_id=run_id,
        n_channels=int(n_channels),
        n_samples=int(n_samples),
        sfreq=float(sfreq) if sfreq is not None else None,
        duration_sec=duration,
        channel_names=",".join(channel_names) if channel_names is not None else None,
        has_missing=validity["has_missing"],
        has_non_finite=validity["has_non_finite"],
        n_flat_channels=len(flat),
        flat_channels=",".join(map(str, flat)),
        amplitude_warning=amplitude_warning,
        sample_count_warning=sample_count_warning,
        channel_count_warning=channel_count_warning,
        sampling_rate_warning=sampling_rate_warning,
        exclusion_candidate=exclusion_candidate,
        error_message=None,
    )


def failed_audit_record(file_path: str, dataset: str, error: Exception) -> AuditRecord:
    """Create an audit record for an unreadable or malformed file."""
    return AuditRecord(
        file_path=str(file_path),
        readable=False,
        dataset=dataset,
        exclusion_candidate=True,
        error_message=f"{type(error).__name__}: {error}",
    )


def summarize_audit(records: List[AuditRecord | Dict[str, Any]]) -> pd.DataFrame:
    """Convert audit records to a DataFrame."""
    rows: List[Dict[str, Any]] = []
    for record in records:
        if isinstance(record, AuditRecord):
            rows.append(record.to_dict())
        else:
            rows.append(dict(record))
    return pd.DataFrame(rows)


def save_audit_report(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save an audit report as CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def audit_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary table from a full audit DataFrame."""
    if df.empty:
        return pd.DataFrame()
    summary = {
        "n_files": len(df),
        "n_readable": int(df.get("readable", pd.Series(dtype=bool)).fillna(False).sum()),
        "n_exclusion_candidates": int(df.get("exclusion_candidate", pd.Series(dtype=bool)).fillna(False).sum()),
        "n_channel_warnings": int(df.get("channel_count_warning", pd.Series(dtype=bool)).fillna(False).sum()),
        "n_sampling_rate_warnings": int(df.get("sampling_rate_warning", pd.Series(dtype=bool)).fillna(False).sum()),
        "n_sample_count_warnings": int(df.get("sample_count_warning", pd.Series(dtype=bool)).fillna(False).sum()),
        "n_amplitude_warnings": int(df.get("amplitude_warning", pd.Series(dtype=bool)).fillna(False).sum()),
    }
    if "subject_id" in df.columns:
        summary["n_subjects"] = int(df["subject_id"].dropna().nunique())
    if "run_id" in df.columns:
        summary["n_runs"] = int(df["run_id"].dropna().nunique())
    return pd.DataFrame([summary])
