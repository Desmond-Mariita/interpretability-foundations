"""Paired bootstrap CIs and pairwise difference tests over per-example metrics."""

from __future__ import annotations

from collections.abc import Callable

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


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Cluster (group-resampled) bootstrap CI for ``metric_fn(y_true, y_pred)``.

    Resamples GROUPS (e.g. sentences) with replacement -- the correct unit of independence when
    rows within a group are correlated. ``metric_fn`` is called ``metric_fn(y_true', y_pred')``.

    Returns:
        ``(lo, mean, hi)`` at the ``1 - alpha`` level (``mean`` = metric on the full sample).
    """
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    grp = np.asarray(groups)
    uniq = np.unique(grp)
    by_group = {gr: np.flatnonzero(grp == gr) for gr in uniq}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        drawn = uniq[rng.integers(0, uniq.size, uniq.size)]
        idx = np.concatenate([by_group[gr] for gr in drawn])
        stats[i] = metric_fn(yt[idx], yp[idx])
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, float(metric_fn(yt, yp)), hi


def paired_cluster_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired cluster bootstrap CI for ``metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b)``.

    Both metrics are recomputed on the SAME resampled groups each replicate (paired), so the CI
    correctly reflects their correlation on a shared test set.

    Returns:
        ``(lo, mean_diff, hi)`` at the ``1 - alpha`` level.
    """
    yt, pa, pb = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    grp = np.asarray(groups)
    uniq = np.unique(grp)
    by_group = {gr: np.flatnonzero(grp == gr) for gr in uniq}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        drawn = uniq[rng.integers(0, uniq.size, uniq.size)]
        idx = np.concatenate([by_group[gr] for gr in drawn])
        diffs[i] = metric_fn(yt[idx], pa[idx]) - metric_fn(yt[idx], pb[idx])
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return lo, float(metric_fn(yt, pa) - metric_fn(yt, pb)), hi
