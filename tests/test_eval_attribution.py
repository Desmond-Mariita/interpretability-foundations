import numpy as np
import pytest

from awake.eval.attribution import TokenAttribution


@pytest.mark.unit
def test_token_attribution_validates_lengths():
    attr = TokenAttribution(
        tokens=["a", "b"],
        offsets=[(0, 1), (2, 3)],
        scores=np.array([0.1, 0.9]),
        visible_mask=np.array([True, True]),
        predicted_class=1,
        class_scores=np.array([0.3, 0.7]),
    )
    assert attr.n_tokens == 2
    assert attr.predicted_class == 1


@pytest.mark.unit
def test_token_attribution_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        TokenAttribution(
            tokens=["a", "b"],
            offsets=[(0, 1)],
            scores=np.array([0.1, 0.9]),
            visible_mask=np.array([True, True]),
            predicted_class=0,
            class_scores=np.array([0.5, 0.5]),
        )
