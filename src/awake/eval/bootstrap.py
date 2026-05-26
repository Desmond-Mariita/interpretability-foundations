"""Paired bootstrap CIs and pairwise difference tests over per-example metrics."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Args:
        values: 1-D array of per-example metric values.
        n_resamples: Number of bootstrap resamples to draw.
        alpha: Two-sided significance level; CI is at ``1 - alpha`` confidence.
        seed: Integer seed for the random number generator (determinism).

    Returns:
        ``(low, mean, high)`` at the ``1 - alpha`` confidence level.
    """
    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = values[rng.integers(0, n, n)].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, float(values.mean()), hi


def paired_diff_test(a: np.ndarray, b: np.ndarray, n_resamples: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap test of ``mean(a) - mean(b)`` over shared examples.

    Args:
        a: Per-example metric values for method A.
        b: Per-example metric values for method B; must match ``a`` in shape.
        n_resamples: Number of bootstrap resamples to draw.
        seed: Integer seed for the random number generator (determinism).

    Returns:
        Dict with ``mean_diff``, ``ci_low``, ``ci_high`` and a two-sided
        bootstrap ``p_value`` (fraction of resampled diffs crossing zero).

    Raises:
        ValueError: If ``a`` and ``b`` do not have the same shape.
    """
    if a.shape != b.shape:
        raise ValueError("paired arrays must have equal shape")
    diff = a - b
    lo, mean_diff, hi = bootstrap_ci(diff, n_resamples=n_resamples, seed=seed)
    rng = np.random.default_rng(seed + 1)
    n = diff.size
    centered = diff - diff.mean()
    resampled = np.array([centered[rng.integers(0, n, n)].mean() for _ in range(n_resamples)])
    p = float((np.abs(resampled) >= abs(diff.mean())).mean())
    return {"mean_diff": mean_diff, "ci_low": lo, "ci_high": hi, "p_value": p}
