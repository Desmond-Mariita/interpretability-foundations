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


@pytest.mark.smoke
def test_gradient_x_input_runs_on_stub():
    import importlib, sys, pathlib
    sys.path.insert(0, str(pathlib.Path("projects/02-text-eraser/scripts").resolve()))
    stub = importlib.import_module("_stub_model")
    from awake.eval.explainers.gradient_x_input import GradientXInputExplainer
    model, tok = stub.build_stub_model_and_tokenizer()
    gxi = GradientXInputExplainer(model, tok, device="cpu")
    ex = {"text": "w5 w6 w7", "predicted_class": None}
    attr = gxi.attribute(ex)
    assert attr.n_tokens > 0
    assert np.all(np.isfinite(attr.scores))
