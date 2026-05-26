"""Length-preserving token erasure for faithfulness metrics.

We replace erased tokens with the tokenizer's mask id rather than deleting
them, so positional embeddings stay aligned (deletion shifts positions and
produces out-of-distribution inputs). This is a deliberate, documented
deviation from ERASER's literal token-removal (see ADR 002).
"""

from __future__ import annotations

import math

import numpy as np


def erase(token_ids: np.ndarray, keep_mask: np.ndarray, mask_token_id: int) -> np.ndarray:
    """Replace tokens where ``keep_mask`` is False with ``mask_token_id``.

    Args:
        token_ids: 1-D array of input token ids.
        keep_mask: Boolean array; True keeps the original token.
        mask_token_id: Id substituted into erased positions.

    Returns:
        A new array, same shape, with erased positions masked.
    """
    out = token_ids.copy()
    out[~keep_mask] = mask_token_id
    return out


def top_k_mask(scores: np.ndarray, visible_mask: np.ndarray, k_fraction: float) -> np.ndarray:
    """Boolean mask selecting the top ``k_fraction`` of visible tokens by score.

    Args:
        scores: Per-token importance.
        visible_mask: True for tokens eligible for selection.
        k_fraction: Fraction in ``[0, 1]`` of visible tokens to select.

    Returns:
        Boolean mask, True for selected (top-scoring visible) tokens.
    """
    mask = np.zeros_like(visible_mask, dtype=bool)
    visible_idx = np.flatnonzero(visible_mask)
    if visible_idx.size == 0 or k_fraction <= 0:
        return mask
    k = min(visible_idx.size, math.ceil(k_fraction * visible_idx.size))
    order = visible_idx[np.argsort(scores[visible_idx])[::-1]]
    mask[order[:k]] = True
    return mask
