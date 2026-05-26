"""ERASER faithfulness metrics, scored on the original predicted class.

comprehensiveness = p_j(x) - p_j(x with top-k rationale erased)
sufficiency       = p_j(x) - p_j(x with only top-k rationale kept)
where j is the predicted class fixed per example, and the top-k budget is a
fixed dataset fraction (k_d), not the per-example gold length.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from awake.eval.erasure import erase, top_k_mask

PredictFn = Callable[[np.ndarray], np.ndarray]


def _prob_j(predict_fn: PredictFn, token_ids: np.ndarray, predicted_class: int) -> float:
    """Return predicted probability for ``predicted_class`` on a single example."""
    probs = predict_fn(token_ids[None, :])
    return float(probs[0, predicted_class])


def comprehensiveness(
    predict_fn: PredictFn,
    token_ids: np.ndarray,
    scores: np.ndarray,
    visible_mask: np.ndarray,
    predicted_class: int,
    mask_token_id: int,
    k_fraction: float,
) -> float:
    """Drop in predicted-class prob when the top-k rationale is erased.

    Args:
        predict_fn: Callable ``(batch: ndarray) -> ndarray`` returning class
            probabilities with shape ``(batch, num_classes)``.
        token_ids: 1-D array of input token ids.
        scores: Per-token importance scores (same length as ``token_ids``).
        visible_mask: Boolean mask; True for tokens eligible for selection.
        predicted_class: Index of the class whose probability is tracked.
        mask_token_id: Token id substituted into erased positions.
        k_fraction: Fraction of visible tokens selected as the rationale.

    Returns:
        ``p_j(x) - p_j(x_erased)``; positive when the rationale is important.
    """
    base = _prob_j(predict_fn, token_ids, predicted_class)
    rationale = top_k_mask(scores, visible_mask, k_fraction)
    reduced = erase(token_ids, keep_mask=~rationale, mask_token_id=mask_token_id)
    return base - _prob_j(predict_fn, reduced, predicted_class)


def sufficiency(
    predict_fn: PredictFn,
    token_ids: np.ndarray,
    scores: np.ndarray,
    visible_mask: np.ndarray,
    predicted_class: int,
    mask_token_id: int,
    k_fraction: float,
) -> float:
    """Drop in predicted-class prob when only the top-k rationale is kept.

    Args:
        predict_fn: Callable ``(batch: ndarray) -> ndarray`` returning class
            probabilities with shape ``(batch, num_classes)``.
        token_ids: 1-D array of input token ids.
        scores: Per-token importance scores (same length as ``token_ids``).
        visible_mask: Boolean mask; True for tokens eligible for selection.
        predicted_class: Index of the class whose probability is tracked.
        mask_token_id: Token id substituted into erased positions.
        k_fraction: Fraction of visible tokens selected as the rationale.

    Returns:
        ``p_j(x) - p_j(x_kept)``; near zero when the rationale alone suffices.
    """
    base = _prob_j(predict_fn, token_ids, predicted_class)
    rationale = top_k_mask(scores, visible_mask, k_fraction)
    # keep rationale + special tokens (non-visible) so structure is preserved
    keep = rationale | (~visible_mask)
    kept = erase(token_ids, keep_mask=keep, mask_token_id=mask_token_id)
    return base - _prob_j(predict_fn, kept, predicted_class)
