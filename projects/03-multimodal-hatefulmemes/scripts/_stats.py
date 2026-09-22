"""P3-local statistical helpers: paired metric bootstrap, modality shares, interaction.

These live in project 03 rather than ``src/awake/eval`` because the corrected
estimands are specific to this project's evaluation protocol (paired AUROC on
shared resampled example indices; signed-vs-magnitude modality shares). The
generic shared bootstrap helpers remain untouched.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

EPS = 1e-9


def _draw_indices(rng: np.random.Generator, n: int, n_resamples: int) -> np.ndarray:
    """Draw ``n_resamples`` resamples of example indices, shape ``(n_resamples, n)``."""
    return rng.integers(0, n, size=(n_resamples, n))


def paired_metric_bootstrap(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 2000,
    seed: int = 0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Paired bootstrap CI for ``metric_fn(y, scores_a) - metric_fn(y, scores_b)``.

    The SAME resampled example indices are used for both models on every
    resample, and each metric is recomputed inside the resample, so the CI
    reflects the correlation between the two models on the shared test set.

    Degenerate resamples (e.g. a resampled label vector with a single class,
    for which AUROC raises ``ValueError``) are counted and skipped; they are
    never silently coerced into a number.

    Args:
        y: True binary labels, shape ``(n,)``.
        scores_a: Per-example scores for model A, shape ``(n,)``.
        scores_b: Per-example scores for model B, shape ``(n,)``.
        metric_fn: Metric function ``(y, scores) -> float``; may raise
            ``ValueError`` on degenerate inputs (like ``roc_auc_score``).
        n_resamples: Number of bootstrap resamples to draw.
        seed: Integer seed for the random number generator (determinism).
        rng: Optional pre-built generator (used for pairing tests); overrides
            ``seed`` when given.

    Returns:
        Dict with ``point_diff`` (metric difference on the full sample),
        ``ci_low``/``ci_high`` (percentile 95% CI over valid resamples),
        ``n_resamples``, ``n_valid`` (resamples that produced a metric), and
        ``seed``.

    Raises:
        ValueError: If ``y``, ``scores_a``, ``scores_b`` disagree in length, or
            if every resample is degenerate (the CI is then undefined).
    """
    y = np.asarray(y)
    a = np.asarray(scores_a)
    b = np.asarray(scores_b)
    if not (y.ndim == 1 and y.shape == a.shape == b.shape):
        raise ValueError("y, scores_a, scores_b must be 1-D arrays of equal length")
    if rng is None:
        rng = np.random.default_rng(seed)
    n = y.size
    diffs: list[float] = []
    for idx in _draw_indices(rng, n, n_resamples):
        try:
            diffs.append(float(metric_fn(y[idx], a[idx]) - metric_fn(y[idx], b[idx])))
        except ValueError:
            continue  # degenerate resample: counted via n_valid, never coerced
    n_valid = len(diffs)
    if n_valid == 0:
        raise ValueError(
            f"all {n_resamples} bootstrap resamples were degenerate for the given metric"
        )
    return {
        "point_diff": float(metric_fn(y, a) - metric_fn(y, b)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_resamples": int(n_resamples),
        "n_valid": int(n_valid),
        "seed": int(seed),
    }


def paired_auroc_bootstrap(
    y: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap CI for ``AUROC(y, probs_a) - AUROC(y, probs_b)``.

    This is the corrected estimand for the fused-vs-unimodal comparisons:
    example indices are resampled as paired units and both AUROCs are
    recomputed inside each resample (see :func:`paired_metric_bootstrap`).
    A resample whose label vector contains a single class is degenerate and
    is skipped (some sklearn versions warn and return 0.5 there instead of
    raising; both models would then trivially tie, which would bias the CI).
    """
    from sklearn.metrics import roc_auc_score

    def _safe_auroc(yi: np.ndarray, si: np.ndarray) -> float:
        if np.unique(yi).size < 2:
            raise ValueError("degenerate resample: single-class label vector")
        return float(roc_auc_score(yi, si))

    return paired_metric_bootstrap(y, probs_a, probs_b, _safe_auroc, n_resamples, seed)


def signed_image_share(phi_image: float, phi_text: float, eps: float = EPS) -> float:
    """Direction of the image contribution, in ``[-1, 1]``.

    ``+1`` = the entire non-zero contribution pushes the target score up via
    the image; ``-1`` = it pushes the score down. The sign is a *direction*,
    not a dominance measure: a large negative image contribution dominates the
    text in magnitude while producing a negative share.

    Returns ``0.0`` when both contributions have (near-)zero magnitude, i.e.
    the direction is undefined.
    """
    total = abs(phi_image) + abs(phi_text)
    if total < eps:
        return 0.0
    return float(phi_image / (total + eps))


def magnitude_share_image(phi_image: float, phi_text: float, eps: float = EPS) -> float:
    """Relative magnitude of the image contribution, in ``[0, 1]``.

    ``abs(phi_image) / (abs(phi_image) + abs(phi_text))``: the share of the
    total absolute contribution carried by the image, independent of sign.
    Returns ``0.5`` (balanced) when both contributions have (near-)zero
    magnitude, since no modality can then be said to dominate.
    """
    total = abs(phi_image) + abs(phi_text)
    if total < eps:
        return 0.5
    return float(abs(phi_image) / total)


def interaction(v_empty: float, v_img: float, v_txt: float, v_ab: float) -> float:
    """Non-additivity of the coalition game: ``v(ab) - v(a) - v(b) + v(empty)``.

    Interpreted narrowly: how far the chosen value function departs from
    additivity under the specified interventional game. It is not a measure of
    semantic synergy or causal interaction between modalities.
    """
    return float(v_ab - v_img - v_txt + v_empty)
