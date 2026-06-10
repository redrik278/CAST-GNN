"""Signal preprocessing for CAST-GNN.

Includes bandpass filtering, resampling, band decomposition, fixed-length
standardisation, and training-subject-only normalisation.  All functions assume
signals are shaped as [channels, samples] unless otherwise stated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as scipy_signal


DEFAULT_BANDS: List[Tuple[float, float]] = [(5, 8), (8, 13), (13, 30), (30, 40)]


def bandpass_filter(
    x: np.ndarray,
    sfreq: float,
    low_freq: float = 5.0,
    high_freq: float = 40.0,
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filtering."""
    arr = np.asarray(x, dtype=np.float32)
    nyq = sfreq / 2.0
    if not 0 < low_freq < high_freq < nyq:
        raise ValueError(f"Invalid band [{low_freq}, {high_freq}] for sfreq={sfreq}")
    sos = scipy_signal.butter(order, [low_freq / nyq, high_freq / nyq], btype="bandpass", output="sos")
    return scipy_signal.sosfiltfilt(sos, arr, axis=axis).astype(np.float32)


def resample_signal(x: np.ndarray, original_sfreq: float, target_sfreq: float = 125.0, axis: int = -1) -> np.ndarray:
    """Resample a signal using polyphase filtering."""
    arr = np.asarray(x, dtype=np.float32)
    if abs(original_sfreq - target_sfreq) < 1e-9:
        return arr
    # Use rational approximation for stable polyphase resampling.
    from fractions import Fraction

    ratio = Fraction(float(target_sfreq) / float(original_sfreq)).limit_denominator(1000)
    y = scipy_signal.resample_poly(arr, up=ratio.numerator, down=ratio.denominator, axis=axis)
    return y.astype(np.float32)


def standardize_length(x: np.ndarray, target_samples: int = 500, mode: str = "center") -> np.ndarray:
    """Trim or pad a signal to a fixed number of samples."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim < 1:
        raise ValueError("input must have at least one dimension")
    n = arr.shape[-1]
    if n == target_samples:
        return arr
    if n > target_samples:
        start = (n - target_samples) // 2 if mode == "center" else 0
        return arr[..., start : start + target_samples]
    pad_width = [(0, 0)] * arr.ndim
    pad_width[-1] = (0, target_samples - n)
    return np.pad(arr, pad_width, mode="constant").astype(np.float32)


def extract_frequency_bands(
    x: np.ndarray,
    sfreq: float,
    bands: Sequence[Tuple[float, float]] = DEFAULT_BANDS,
    order: int = 4,
) -> np.ndarray:
    """Create band-specific views of an EEG signal.

    Parameters
    ----------
    x:
        Signal array with shape [channels, samples].
    sfreq:
        Sampling frequency in Hz.
    bands:
        Iterable of `(low, high)` frequency tuples.

    Returns
    -------
    np.ndarray
        Array with shape [n_bands, channels, samples].
    """
    arr = np.asarray(x, dtype=np.float32)
    band_arrays = [bandpass_filter(arr, sfreq, low, high, order=order) for low, high in bands]
    return np.stack(band_arrays, axis=0).astype(np.float32)


def preprocess_trial(
    x: np.ndarray,
    sfreq: float,
    target_sfreq: float = 125.0,
    target_samples: int = 500,
    bands: Sequence[Tuple[float, float]] = DEFAULT_BANDS,
    broad_low: float = 5.0,
    broad_high: float = 40.0,
) -> np.ndarray:
    """Resample, broad-band filter, standardize length, and decompose into bands.

    Returns an array with shape [bands, channels, target_samples].
    """
    y = resample_signal(x, original_sfreq=sfreq, target_sfreq=target_sfreq)
    y = bandpass_filter(y, sfreq=target_sfreq, low_freq=broad_low, high_freq=broad_high)
    y = standardize_length(y, target_samples=target_samples)
    return extract_frequency_bands(y, sfreq=target_sfreq, bands=bands)


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray


class SubjectWiseNormalizer:
    """Channel-wise normaliser fitted only on training-subject samples.

    The class expects arrays with shape [n_samples, bands, channels, time] or
    [n_samples, channels, time].  Normalisation is computed across samples and
    time while preserving band and channel dimensions.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self.stats: Optional[NormalizationStats] = None

    def fit(self, x: np.ndarray) -> "SubjectWiseNormalizer":
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 4:  # [N, B, C, T]
            mean = arr.mean(axis=(0, 3), keepdims=True)
            std = arr.std(axis=(0, 3), keepdims=True)
        elif arr.ndim == 3:  # [N, C, T]
            mean = arr.mean(axis=(0, 2), keepdims=True)
            std = arr.std(axis=(0, 2), keepdims=True)
        else:
            raise ValueError("x must have shape [N,B,C,T] or [N,C,T]")
        std = np.maximum(std, self.eps)
        self.stats = NormalizationStats(mean=mean.astype(np.float32), std=std.astype(np.float32))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.stats is None:
            raise RuntimeError("SubjectWiseNormalizer must be fitted before transform")
        arr = np.asarray(x, dtype=np.float32)
        return ((arr - self.stats.mean) / self.stats.std).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def compute_target_samples(window_seconds: float, sfreq: float) -> int:
    """Return the integer number of samples for a given window duration."""
    return int(round(float(window_seconds) * float(sfreq)))
