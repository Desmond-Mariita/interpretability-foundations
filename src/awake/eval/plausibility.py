"""Plausibility metrics vs. human rationales (word level).

Subword attributions are aggregated to whitespace-word level by taking the
max absolute score among a word's subwords. LIME, already whitespace-level,
bypasses this via an identity word_ids mapping. All functions are
tokenizer-agnostic: callers pass ``word_ids`` (e.g. from a fast tokenizer's
``word_ids()``) and a clipped gold mask, never a live tokenizer.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


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


def clip_gold_mask_to_window(gold_mask: np.ndarray, n_visible_words: int) -> np.ndarray:
    """Clip a full-document word-level gold mask to the visible window."""
    return gold_mask[:n_visible_words]


def token_prf1_at_k(
    word_scores: np.ndarray, gold_mask: np.ndarray, k: int
) -> tuple[float, float, float]:
    """Token precision/recall/F1 when the top-k scored words are predicted.

    Args:
        word_scores: Word-level attribution scores.
        gold_mask: Binary word-level human rationale (clipped to window).
        k: Number of words to mark as predicted rationale (the k_d budget).

    Returns:
        ``(precision, recall, f1)``.
    """
    k = max(0, min(k, word_scores.size))
    pred = np.zeros_like(gold_mask, dtype=bool)
    if k > 0:
        top = np.argsort(word_scores)[::-1][:k]
        pred[top] = True
    gold = gold_mask.astype(bool)
    tp = int(np.sum(pred & gold))
    p = tp / pred.sum() if pred.sum() else 0.0
    r = tp / gold.sum() if gold.sum() else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def token_auprc(word_scores: np.ndarray, gold_mask: np.ndarray) -> float:
    """Area under the precision-recall curve over continuous word scores."""
    gold = gold_mask.astype(int)
    if gold.sum() == 0 or gold.sum() == gold.size:
        return float("nan")
    return float(average_precision_score(gold, word_scores))


def token_iou(pred_mask: np.ndarray, gold_mask: np.ndarray) -> float:
    """Custom token-set IoU (NOT ERASER's span-level IoU). Diagnostic only."""
    pred = pred_mask.astype(bool)
    gold = gold_mask.astype(bool)
    union = int(np.sum(pred | gold))
    return int(np.sum(pred & gold)) / union if union else 0.0
