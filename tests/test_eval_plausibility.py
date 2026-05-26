import numpy as np
import pytest
from awake.eval.plausibility import aggregate_subwords_to_words


@pytest.mark.unit
def test_aggregate_takes_max_abs_over_word_subwords():
    # word_ids maps each subword to a word index; None for special tokens.
    word_ids = [None, 0, 0, 1, None]
    scores = np.array([9.0, 0.2, -0.8, 0.5, 9.0])
    word_scores = aggregate_subwords_to_words(scores, word_ids, n_words=2)
    # word 0: max(|0.2|, |-0.8|) = 0.8 ; word 1: 0.5
    assert word_scores.tolist() == [0.8, 0.5]
