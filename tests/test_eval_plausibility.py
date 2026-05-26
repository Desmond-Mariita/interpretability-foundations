import math

import numpy as np
import pytest

from awake.eval.plausibility import (
    aggregate_subwords_to_words,
    clip_gold_mask_to_window,
    token_auprc,
    token_iou,
    token_prf1_at_k,
)


@pytest.mark.unit
def test_aggregate_takes_max_abs_over_word_subwords():
    # word_ids maps each subword to a word index; None for special tokens.
    word_ids = [None, 0, 0, 1, None]
    scores = np.array([9.0, 0.2, -0.8, 0.5, 9.0])
    word_scores = aggregate_subwords_to_words(scores, word_ids, n_words=2)
    # word 0: max(|0.2|, |-0.8|) = 0.8 ; word 1: 0.5
    assert word_scores.tolist() == [0.8, 0.5]


@pytest.mark.unit
def test_clip_gold_mask_to_window():
    gold = np.array([0, 1, 1, 0, 1])  # 5 words in full doc
    out = clip_gold_mask_to_window(gold, n_visible_words=3)
    assert out.tolist() == [0, 1, 1]


@pytest.mark.unit
def test_token_prf1_at_k_selects_top_scores():
    word_scores = np.array([0.1, 0.9, 0.8, 0.2])
    gold = np.array([0, 1, 1, 0])
    p, r, f1 = token_prf1_at_k(word_scores, gold, k=2)
    assert (p, r, f1) == pytest.approx((1.0, 1.0, 1.0))


@pytest.mark.unit
def test_token_auprc_perfect_ranking_is_one():
    word_scores = np.array([0.1, 0.9, 0.8, 0.2])
    gold = np.array([0, 1, 1, 0])
    assert token_auprc(word_scores, gold) == pytest.approx(1.0)


@pytest.mark.unit
def test_token_iou_custom_metric():
    pred = np.array([0, 1, 1, 0], dtype=bool)
    gold = np.array([0, 1, 0, 0], dtype=bool)
    # intersection=1, union=2 -> 0.5
    assert token_iou(pred, gold) == pytest.approx(0.5)


@pytest.mark.unit
def test_token_auprc_all_positive_gold_returns_nan():
    word_scores = np.array([0.9, 0.8, 0.7])
    gold = np.array([1, 1, 1])  # all positive -> degenerate -> NaN
    assert math.isnan(token_auprc(word_scores, gold))


@pytest.mark.unit
def test_token_auprc_all_negative_gold_returns_nan():
    word_scores = np.array([0.9, 0.8, 0.7])
    gold = np.array([0, 0, 0])  # all negative -> degenerate -> NaN
    assert math.isnan(token_auprc(word_scores, gold))


@pytest.mark.unit
def test_token_prf1_at_k_zero_k_returns_zeros():
    word_scores = np.array([0.9, 0.8, 0.7])
    gold = np.array([0, 1, 0])
    p, r, f1 = token_prf1_at_k(word_scores, gold, k=0)
    # no predictions -> precision undefined (0), recall 0, f1 0
    assert (p, r, f1) == pytest.approx((0.0, 0.0, 0.0))
