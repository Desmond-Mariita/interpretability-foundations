"""Plausibility metrics vs. human rationales (word level).

Subword attributions are aggregated to whitespace-word level by taking the
max absolute score among a word's subwords. LIME, already whitespace-level,
bypasses this via an identity word_ids mapping. All functions are
tokenizer-agnostic: callers pass ``word_ids`` (e.g. from a fast tokenizer's
``word_ids()``) and a clipped gold mask, never a live tokenizer.
"""

from __future__ import annotations

import numpy as np


def aggregate_subwords_to_words(
    scores: np.ndarray, word_ids: list[int | None], n_words: int
) -> np.ndarray:
    """Aggregate subword scores to word level by max absolute value.

    Args:
        scores: Per-subword scores.
        word_ids: Word index per subword; None for special tokens.
        n_words: Total number of whitespace words in the (clipped) document.

    Returns:
        Array of length ``n_words`` with the max |score| per word (0 if a word
        has no surviving subwords).
    """
    word_scores = np.zeros(n_words, dtype=float)
    for score, wid in zip(scores, word_ids, strict=False):
        if wid is None:
            continue
        word_scores[wid] = max(word_scores[wid], abs(float(score)))
    return word_scores
