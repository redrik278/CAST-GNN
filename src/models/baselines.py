"""Baseline models and classical feature pipelines for CAST-GNN experiments."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from mne.decoding import CSP
except Exception:  # pragma: no cover
    CSP = None


class EEGNet(nn.Module):
    """Compact EEGNet-style CNN baseline.

    Input shape: [N, B, C, T]. Bands are treated as input channels after averaging
    or concatenation.  This implementation averages across bands by default for
    compatibility with classical EEGNet layouts.
    """

    def __init__(self, num_channels: int, num_classes: int, samples: int = 500, dropout: float = 0.50) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.depthwise = nn.Conv2d(8, 16, kernel_size=(num_channels, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.sep_depth = nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), groups=16, bias=False)
        self.sep_point = nn.Conv2d(16, 16, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool2d((1, 16))
        self.classifier = nn.Linear(16 * 16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.mean(dim=1)  # [N,C,T]
        x = x.unsqueeze(1)  # [N,1,C,T]
        x = F.elu(self.bn1(self.temporal(x)))
        x = F.elu(self.bn2(self.depthwise(x)))
        x = F.avg_pool2d(x, kernel_size=(1, 4))
        x = self.dropout(x)
        x = self.sep_depth(x)
        x = self.sep_point(x)
        x = F.elu(self.bn3(x))
        x = F.avg_pool2d(x, kernel_size=(1, 8))
        x = self.dropout(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


class ShallowConvNet(nn.Module):
    """ShallowConvNet-style EEG baseline."""

    def __init__(self, num_channels: int, num_classes: int, dropout: float = 0.50) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(1, 40, kernel_size=(1, 25), bias=False)
        self.spatial = nn.Conv2d(40, 40, kernel_size=(num_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(40)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool2d((1, 20))
        self.fc = nn.Linear(40 * 20, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.mean(dim=1)
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = torch.square(x)
        x = F.avg_pool2d(x, kernel_size=(1, 35), stride=(1, 7))
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class DeepConvNet(nn.Module):
    """DeepConvNet-style EEG baseline."""

    def __init__(self, num_channels: int, num_classes: int, dropout: float = 0.50) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 10), bias=False),
            nn.Conv2d(25, 25, kernel_size=(num_channels, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(dropout),
        )
        self.block2 = self._block(25, 50, dropout)
        self.block3 = self._block(50, 100, dropout)
        self.block4 = self._block(100, 200, dropout)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(200, num_classes)

    @staticmethod
    def _block(in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, 10), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.mean(dim=1)
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))


class SimpleTCNBaseline(nn.Module):
    """Simple temporal convolution baseline."""

    def __init__(self, num_channels: int, num_classes: int, hidden: int = 64, dropout: float = 0.30) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_channels, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.mean(dim=1)
        z = self.net(x).squeeze(-1)
        return self.fc(z)


class SimpleGraphTemporalBaseline(nn.Module):
    """Simple graph-temporal baseline using fixed adjacency propagation."""

    def __init__(self, num_channels: int, num_classes: int, adjacency: Optional[torch.Tensor] = None, hidden: int = 64) -> None:
        super().__init__()
        if adjacency is None:
            adjacency = torch.eye(num_channels)
        self.register_buffer("adjacency", adjacency.float())
        self.temporal = nn.Conv1d(num_channels, hidden, kernel_size=7, padding=3)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.mean(dim=1)
        x = torch.einsum("ij,njt->nit", self.adjacency, x)
        z = F.gelu(self.temporal(x))
        z = z.mean(dim=-1)
        return self.fc(z)


def _to_numpy_epochs(x: np.ndarray | torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    if arr.ndim == 4:
        arr = arr.mean(axis=1)
    if arr.ndim != 3:
        raise ValueError("Expected epochs with shape [N,C,T] or [N,B,C,T]")
    return arr.astype(np.float64)


def run_csp_lda(train_data, train_labels, test_data, n_components: int = 6):
    """Fit CSP + LDA and return test predictions and probabilities."""
    if CSP is None:
        raise ImportError("mne is required for CSP. Install with `pip install mne`.")
    x_train = _to_numpy_epochs(train_data)
    x_test = _to_numpy_epochs(test_data)
    clf = Pipeline(
        [
            ("csp", CSP(n_components=n_components, reg=None, log=True, norm_trace=False)),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )
    clf.fit(x_train, np.asarray(train_labels))
    pred = clf.predict(x_test)
    prob = clf.predict_proba(x_test) if hasattr(clf, "predict_proba") else None
    return pred, prob, clf


def run_fbcsp_svm(train_data, train_labels, test_data, bands: Sequence[tuple[float, float]] | None = None, n_components: int = 4):
    """Approximate FBCSP + SVM using already band-decomposed input if available.

    If data has shape [N,B,C,T], CSP features are extracted per band and
    concatenated. If data is [N,C,T], it falls back to standard CSP features.
    """
    if CSP is None:
        raise ImportError("mne is required for CSP. Install with `pip install mne`.")
    x_train = train_data.detach().cpu().numpy() if isinstance(train_data, torch.Tensor) else np.asarray(train_data)
    x_test = test_data.detach().cpu().numpy() if isinstance(test_data, torch.Tensor) else np.asarray(test_data)
    y = np.asarray(train_labels)

    if x_train.ndim == 3:
        x_train = x_train[:, None]
        x_test = x_test[:, None]
    features_train = []
    features_test = []
    csps = []
    for b in range(x_train.shape[1]):
        csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
        ft = csp.fit_transform(x_train[:, b], y)
        fs = csp.transform(x_test[:, b])
        features_train.append(ft)
        features_test.append(fs)
        csps.append(csp)
    f_train = np.concatenate(features_train, axis=1)
    f_test = np.concatenate(features_test, axis=1)
    clf = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced"))])
    clf.fit(f_train, y)
    pred = clf.predict(f_test)
    prob = clf.predict_proba(f_test)
    return pred, prob, {"csps": csps, "clf": clf}


def hjorth_parameters(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Hjorth activity, mobility, and complexity per channel."""
    dx = np.diff(x, axis=-1)
    ddx = np.diff(dx, axis=-1)
    var_x = np.var(x, axis=-1) + 1e-8
    var_dx = np.var(dx, axis=-1) + 1e-8
    var_ddx = np.var(ddx, axis=-1) + 1e-8
    activity = var_x
    mobility = np.sqrt(var_dx / var_x)
    complexity = np.sqrt(var_ddx / var_dx) / (mobility + 1e-8)
    return activity, mobility, complexity


def extract_psd_hjorth_features(signals, sfreq: float = 125.0, bands: Sequence[tuple[float, float]] = ((5, 8), (8, 13), (13, 30), (30, 40))) -> np.ndarray:
    """Extract PSD bandpower and Hjorth features from EEG epochs."""
    x = _to_numpy_epochs(signals)
    n, c, _ = x.shape
    freqs, psd = welch(x, fs=sfreq, axis=-1, nperseg=min(256, x.shape[-1]))
    band_features = []
    for low, high in bands:
        idx = (freqs >= low) & (freqs <= high)
        power = psd[:, :, idx].mean(axis=-1)
        band_features.append(power)
    activity, mobility, complexity = hjorth_parameters(x)
    features = np.concatenate(band_features + [activity, mobility, complexity], axis=1)
    return features.astype(np.float32)


def run_psd_hjorth_random_forest(train_data, train_labels, test_data, sfreq: float = 125.0, n_estimators: int = 300):
    """Fit PSD/Hjorth + RandomForest baseline."""
    x_train = extract_psd_hjorth_features(train_data, sfreq=sfreq)
    x_test = extract_psd_hjorth_features(test_data, sfreq=sfreq)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight="balanced_subsample", n_jobs=-1)
    clf.fit(x_train, np.asarray(train_labels))
    pred = clf.predict(x_test)
    prob = clf.predict_proba(x_test)
    return pred, prob, clf
