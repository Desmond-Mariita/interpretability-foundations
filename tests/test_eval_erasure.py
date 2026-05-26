import numpy as np
import pytest

from awake.eval.erasure import erase, top_k_mask


@pytest.mark.unit
def test_erase_replaces_dropped_positions_and_preserves_length():
    ids = np.array([5, 6, 7, 8])
    keep = np.array([True, False, True, False])
    out = erase(ids, keep_mask=keep, mask_token_id=99)
    assert out.shape == ids.shape
    assert out.tolist() == [5, 99, 7, 99]


@pytest.mark.unit
def test_top_k_mask_selects_highest_scores_among_visible():
    scores = np.array([0.1, 0.9, 0.5, 0.3])
    visible = np.array([True, True, True, False])
    # k_fraction 0.5 of 3 visible -> ceil(1.5)=2 -> indices 1 and 2
    mask = top_k_mask(scores, visible_mask=visible, k_fraction=0.5)
    assert mask.tolist() == [False, True, True, False]
