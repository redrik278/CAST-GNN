"""Statistical tests for repeated-seed/fold comparisons."""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_t_test(values_a, values_b) -> dict[str, float]:
    """Paired t-test."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    stat, p = stats.ttest_rel(a, b, nan_policy="omit")
    return {"statistic": float(stat), "p_value": float(p)}


def wilcoxon_signed_rank(values_a, values_b) -> dict[str, float]:
    """Wilcoxon signed-rank test."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    stat, p = stats.wilcoxon(a, b)
    return {"statistic": float(stat), "p_value": float(p)}


def bootstrap_confidence_interval(values, n_bootstrap: int = 10000, confidence: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_bootstrap)]
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def effect_size_cohens_d(values_a, values_b) -> float:
    """Cohen's d for paired differences."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    diff = a - b
    sd = diff.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(diff.mean() / sd)
