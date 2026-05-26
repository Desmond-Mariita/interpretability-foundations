import numpy as np
import pytest
from awake.eval.faithfulness import comprehensiveness, sufficiency


def _toy_predict_fn(batch: np.ndarray) -> np.ndarray:
    # p(class 1) = fraction of tokens equal to the signal id 7.
    p1 = (batch == 7).mean(axis=1)
    return np.stack([1 - p1, p1], axis=1)


@pytest.mark.unit
def test_comprehensiveness_positive_when_important_token_removed():
    ids = np.array([7, 7, 0, 0])              # p(class1)=0.5
    scores = np.array([1.0, 1.0, 0.0, 0.0])   # signal tokens rank top
    visible = np.array([True, True, True, True])
    comp = comprehensiveness(_toy_predict_fn, ids, scores, visible,
                             predicted_class=1, mask_token_id=0, k_fraction=0.5)
    assert comp > 0  # removing the two 7s drops p(class1) from 0.5 to 0


@pytest.mark.unit
def test_sufficiency_small_when_rationale_alone_preserves_prediction():
    ids = np.array([7, 7, 0, 0])
    scores = np.array([1.0, 1.0, 0.0, 0.0])
    visible = np.array([True, True, True, True])
    suff = sufficiency(_toy_predict_fn, ids, scores, visible,
                       predicted_class=1, mask_token_id=0, k_fraction=0.5)
    # keeping only the two 7s -> p(class1)=0.5 == original -> suff ~ 0
    assert abs(suff) < 1e-9
