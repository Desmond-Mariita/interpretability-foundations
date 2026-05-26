import numpy as np
import pytest
from awake.eval.explainers.random_baseline import RandomExplainer


@pytest.mark.unit
def test_random_explainer_is_seed_deterministic_and_right_length():
    ex = {"input_ids": [2, 5, 6, 3], "visible_mask": [False, True, True, False],
          "offsets": [(0, 0), (0, 1), (2, 3), (0, 0)], "tokens": ["[CLS]", "a", "b", "[SEP]"],
          "predicted_class": 1, "class_scores": [0.4, 0.6]}
    r1 = RandomExplainer(seed=0).attribute(ex)
    r2 = RandomExplainer(seed=0).attribute(ex)
    assert r1.n_tokens == 4
    assert np.allclose(r1.scores, r2.scores)
