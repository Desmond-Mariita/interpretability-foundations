import numpy as np
import pytest

from awake.eval.explainers.random_baseline import RandomExplainer


@pytest.mark.unit
def test_random_explainer_is_seed_deterministic_and_right_length():
    ex = {
        "input_ids": [2, 5, 6, 3],
        "visible_mask": [False, True, True, False],
        "offsets": [(0, 0), (0, 1), (2, 3), (0, 0)],
        "tokens": ["[CLS]", "a", "b", "[SEP]"],
        "predicted_class": 1,
        "class_scores": [0.4, 0.6],
    }
    r1 = RandomExplainer(seed=0).attribute(ex)
    r2 = RandomExplainer(seed=0).attribute(ex)
    assert r1.n_tokens == 4
    assert np.allclose(r1.scores, r2.scores)


@pytest.mark.smoke
def test_random_explainer_text_path_tokenizes():
    """RandomExplainer accepts the {'text': ...} example the driver passes."""
    import importlib
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path("projects/02-text-eraser/scripts").resolve()))
    stub = importlib.import_module("_stub_model")
    _, tok = stub.build_stub_model_and_tokenizer()
    attr = RandomExplainer(tokenizer=tok, seed=0).attribute({"text": "w5 w6 w7"})
    assert attr.n_tokens > 0
    assert np.all(np.isfinite(attr.scores))


@pytest.mark.smoke
def test_gradient_x_input_runs_on_stub():
    import importlib
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path("projects/02-text-eraser/scripts").resolve()))
    stub = importlib.import_module("_stub_model")
    from awake.eval.explainers.gradient_x_input import GradientXInputExplainer

    model, tok = stub.build_stub_model_and_tokenizer()
    gxi = GradientXInputExplainer(model, tok, device="cpu")
    ex = {"text": "w5 w6 w7", "predicted_class": None}
    attr = gxi.attribute(ex)
    assert attr.n_tokens > 0
    assert np.all(np.isfinite(attr.scores))


# append to tests/test_eval_explainers.py
@pytest.mark.smoke
def test_integrated_gradients_runs_on_stub():
    import importlib
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path("projects/02-text-eraser/scripts").resolve()))
    stub = importlib.import_module("_stub_model")
    from awake.eval.explainers.integrated_gradients import IntegratedGradientsExplainer

    model, tok = stub.build_stub_model_and_tokenizer()
    ig = IntegratedGradientsExplainer(model, tok, device="cpu", n_steps=4)
    attr = ig.attribute({"text": "w5 w6 w7", "predicted_class": None})
    assert attr.n_tokens > 0
    assert np.all(np.isfinite(attr.scores))


# append to tests/test_eval_explainers.py
@pytest.mark.smoke
def test_lime_runs_on_stub():
    import importlib
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path("projects/02-text-eraser/scripts").resolve()))
    stub = importlib.import_module("_stub_model")
    from awake.eval.explainers.lime_text import LimeExplainer

    model, tok = stub.build_stub_model_and_tokenizer()
    lime = LimeExplainer(model, tok, device="cpu", num_samples=20)
    attr = lime.attribute({"text": "w5 w6 w7", "predicted_class": None})
    assert attr.n_tokens == 3  # whitespace-level: one score per word
    assert attr.word_level is True


# append to tests/test_eval_explainers.py
@pytest.mark.slow
def test_shap_partition_runs_on_stub():
    pytest.importorskip("shap")
    import importlib
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path("projects/02-text-eraser/scripts").resolve()))
    stub = importlib.import_module("_stub_model")
    from awake.eval.explainers.shap_partition import ShapPartitionExplainer

    model, tok = stub.build_stub_model_and_tokenizer()
    sx = ShapPartitionExplainer(model, tok, device="cpu", max_evals=50)
    attr = sx.attribute({"text": "w5 w6 w7", "predicted_class": None})
    assert attr.n_tokens > 0
