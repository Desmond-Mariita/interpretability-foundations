from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from awake.eval.explainers.random_baseline import RandomExplainer

# ---------------------------------------------------------------------------
# Unit tests for GradientXInputExplainer (mocked model + tokenizer)
# ---------------------------------------------------------------------------


def _make_mock_tokenizer_and_encoding(n_tokens: int = 5):
    """Return a mock tokenizer that produces a fixed encoding of *n_tokens* tokens."""
    enc = MagicMock()
    enc.__contains__ = lambda self, key: key in {"input_ids", "attention_mask"}
    # offset_mapping pop
    enc.pop.return_value = torch.zeros(1, n_tokens, 2, dtype=torch.long)
    enc.word_ids.return_value = [None] + [0] * (n_tokens - 2) + [None]
    enc.__getitem__ = lambda self, key: torch.ones(1, n_tokens, dtype=torch.long)
    # Support .to(device) - return self
    enc.to = lambda device: enc
    tokenizer = MagicMock()
    tokenizer.return_value = enc
    tokenizer.convert_ids_to_tokens.return_value = [f"t{i}" for i in range(n_tokens)]
    return tokenizer, enc


def _make_mock_model(n_tokens: int = 5, n_classes: int = 2):
    """Return a mock HF model that produces zero logits and a zero embedding layer."""
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model

    # Embedding layer returns a leaf tensor so .grad works
    emb_layer = MagicMock()

    def fake_call(ids):
        t = torch.zeros(1, n_tokens, 8, requires_grad=False)
        return t.clone().detach()

    emb_layer.side_effect = fake_call
    model.get_input_embeddings.return_value = emb_layer

    # model(**enc) or model(inputs_embeds=..., attention_mask=...) return logits
    logits_tensor = torch.zeros(1, n_classes)
    output = MagicMock()
    output.logits = logits_tensor
    model.return_value = output
    return model


def _make_gxi_model_with_grad(n_tokens: int = 5, n_classes: int = 2):
    """Return a mock model where forward produces a logits tensor with a grad_fn.

    GradientXInputExplainer calls ``logits[0, pred].backward()``, so the output
    logits must have ``requires_grad=True`` (or a grad_fn).  We achieve this by
    having the mock model call an nn.Linear on the incoming embeds so that the
    autograd graph is real.
    """
    import torch.nn as nn

    linear = nn.Linear(8, n_classes, bias=False)

    class _FakeOutput:
        def __init__(self, logits):
            self.logits = logits

    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model

    emb_layer = MagicMock()
    emb_layer.return_value = None  # overridden per test

    model.get_input_embeddings.return_value = emb_layer

    def fake_forward(inputs_embeds, attention_mask):
        # inputs_embeds: (1, n_tokens, 8)
        out = linear(inputs_embeds).mean(dim=1)  # (1, n_classes)
        return _FakeOutput(out)

    model.side_effect = None
    model.__call__ = fake_forward
    # MagicMock's __call__ goes through __call__ attribute; use return_value trick
    # instead: patch via a wrapping function stored on the mock.
    model.configure_mock(**{"return_value": None})
    # Simplest: make model callable via side_effect
    model.side_effect = fake_forward

    return model, emb_layer, linear


@pytest.mark.unit
def test_gradient_x_input_unit_predicted_class_none():
    """GradientXInputExplainer: predicted_class=None branch uses argmax."""
    from awake.eval.explainers.gradient_x_input import GradientXInputExplainer

    n_tokens, n_classes = 5, 2
    tokenizer, _enc = _make_mock_tokenizer_and_encoding(n_tokens)
    model, emb_layer, _ = _make_gxi_model_with_grad(n_tokens, n_classes)

    # The embedding layer must return a leaf tensor (detached, no grad_fn) that
    # GradientXInputExplainer then clones + requires_grad_(True).
    emb_layer.return_value = torch.zeros(1, n_tokens, 8)
    emb_layer.side_effect = None

    gxi = GradientXInputExplainer(model, tokenizer, device="cpu")
    attr = gxi.attribute({"text": "hello world", "predicted_class": None})

    assert attr.n_tokens == n_tokens
    assert np.all(np.isfinite(attr.scores))


@pytest.mark.unit
def test_gradient_x_input_unit_predicted_class_provided():
    """GradientXInputExplainer: predicted_class provided skips argmax branch."""
    from awake.eval.explainers.gradient_x_input import GradientXInputExplainer

    n_tokens, n_classes = 5, 2
    tokenizer, _enc = _make_mock_tokenizer_and_encoding(n_tokens)
    model, emb_layer, _ = _make_gxi_model_with_grad(n_tokens, n_classes)

    emb_layer.return_value = torch.zeros(1, n_tokens, 8)
    emb_layer.side_effect = None

    gxi = GradientXInputExplainer(model, tokenizer, device="cpu")
    attr = gxi.attribute({"text": "hello world", "predicted_class": 1})

    assert attr.n_tokens == n_tokens
    assert attr.predicted_class == 1


# ---------------------------------------------------------------------------
# Unit tests for IntegratedGradientsExplainer (mocked model + tokenizer)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_integrated_gradients_unit_predicted_class_none():
    """IntegratedGradientsExplainer: predicted_class=None branch uses argmax."""
    from awake.eval.explainers.integrated_gradients import IntegratedGradientsExplainer

    n_tokens, n_classes = 4, 2
    tokenizer, _enc = _make_mock_tokenizer_and_encoding(n_tokens)
    model = _make_mock_model(n_tokens, n_classes)
    # pad_token_id needed for baseline construction
    tokenizer.pad_token_id = 0

    # _forward calls model(input_ids=..., attention_mask=...).logits
    real_logits = torch.zeros(1, n_classes)
    inner_output = MagicMock()
    inner_output.logits = real_logits
    model.return_value = inner_output

    with patch("awake.eval.explainers.integrated_gradients.LayerIntegratedGradients") as MockLIG:
        mock_lig_inst = MagicMock()
        # attribute returns zero attributions with same shape as input_ids (1, n_tokens, 8)
        mock_lig_inst.attribute.return_value = torch.zeros(1, n_tokens, 8)
        MockLIG.return_value = mock_lig_inst

        ig = IntegratedGradientsExplainer(model, tokenizer, device="cpu", n_steps=2)
        attr = ig.attribute({"text": "a b c d", "predicted_class": None})

    assert attr.n_tokens == n_tokens
    assert np.all(np.isfinite(attr.scores))
    assert attr.predicted_class == 0


@pytest.mark.unit
def test_integrated_gradients_unit_predicted_class_provided():
    """IntegratedGradientsExplainer: predicted_class provided skips argmax."""
    from awake.eval.explainers.integrated_gradients import IntegratedGradientsExplainer

    n_tokens, n_classes = 4, 2
    tokenizer, _enc = _make_mock_tokenizer_and_encoding(n_tokens)
    model = _make_mock_model(n_tokens, n_classes)
    tokenizer.pad_token_id = 0

    real_logits = torch.zeros(1, n_classes)
    inner_output = MagicMock()
    inner_output.logits = real_logits
    model.return_value = inner_output

    with patch("awake.eval.explainers.integrated_gradients.LayerIntegratedGradients") as MockLIG:
        mock_lig_inst = MagicMock()
        mock_lig_inst.attribute.return_value = torch.zeros(1, n_tokens, 8)
        MockLIG.return_value = mock_lig_inst

        ig = IntegratedGradientsExplainer(model, tokenizer, device="cpu", n_steps=2)
        attr = ig.attribute({"text": "a b c d", "predicted_class": 1})

    assert attr.n_tokens == n_tokens
    assert attr.predicted_class == 1


# ---------------------------------------------------------------------------
# Unit tests for LimeExplainer (mocked model + tokenizer)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lime_explainer_unit_predicted_class_none():
    """LimeExplainer: predicted_class=None branch uses argmax of probs."""
    from awake.eval.explainers.lime_text import LimeExplainer

    model = _make_mock_model()
    tokenizer = MagicMock()
    tokenizer.return_value = MagicMock(
        **{"to.return_value": MagicMock(**{"__getitem__": lambda self, key: None})}
    )
    # Patch _predict_proba to return canned probs to avoid the real tokenizer path
    with patch.object(LimeExplainer, "_predict_proba") as mock_pp:
        # Return shape (1, 2) for the base call and (n_samples, 2) for LIME's calls
        mock_pp.side_effect = lambda texts: np.tile([0.3, 0.7], (len(texts), 1))

        with patch("awake.eval.explainers.lime_text.LimeTextExplainer") as MockLTE:
            mock_exp_inst = MagicMock()
            # as_map returns {pred: [(word_idx, weight), ...]}
            mock_exp_inst.explain_instance.return_value = MagicMock(
                **{"as_map.return_value": {1: [(0, 0.5), (1, -0.3)]}}
            )
            MockLTE.return_value = mock_exp_inst

            le = LimeExplainer(model, tokenizer, device="cpu", num_samples=10)
            attr = le.attribute({"text": "foo bar baz", "predicted_class": None})

    assert attr.n_tokens == 3  # 3 whitespace words
    assert attr.word_level is True
    assert attr.predicted_class == 1  # argmax of [0.3, 0.7]


@pytest.mark.unit
def test_lime_explainer_unit_predicted_class_provided():
    """LimeExplainer: predicted_class provided skips argmax."""
    from awake.eval.explainers.lime_text import LimeExplainer

    model = _make_mock_model()
    tokenizer = MagicMock()

    with patch.object(LimeExplainer, "_predict_proba") as mock_pp:
        mock_pp.side_effect = lambda texts: np.tile([0.6, 0.4], (len(texts), 1))

        with patch("awake.eval.explainers.lime_text.LimeTextExplainer") as MockLTE:
            mock_exp_inst = MagicMock()
            mock_exp_inst.explain_instance.return_value = MagicMock(
                **{"as_map.return_value": {0: [(0, 0.9)]}}
            )
            MockLTE.return_value = mock_exp_inst

            le = LimeExplainer(model, tokenizer, device="cpu", num_samples=10)
            attr = le.attribute({"text": "hello world", "predicted_class": 0})

    assert attr.n_tokens == 2
    assert attr.predicted_class == 0


@pytest.mark.unit
def test_lime_explainer_unit_predict_proba_batching():
    """LimeExplainer._predict_proba processes texts in batches without error."""
    from awake.eval.explainers.lime_text import LimeExplainer

    n_classes = 2
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model

    # side_effect: return logits shaped to the actual chunk size
    def fake_forward(**enc_kwargs):
        # The tokenizer mock returns an enc_mock; model(**enc_mock) is called.
        # We don't have the actual batch size here, so return batch_size=2 logits
        # and let the test verify the concat shape.
        out = MagicMock()
        out.logits = torch.zeros(2, n_classes)
        return out

    model.side_effect = fake_forward

    tokenizer = MagicMock()
    enc_mock = MagicMock()
    enc_mock.to.return_value = enc_mock
    tokenizer.return_value = enc_mock

    with patch("awake.eval.explainers.lime_text.LimeTextExplainer"):
        le = LimeExplainer(model, tokenizer, device="cpu", num_samples=5, batch_size=2)

    # Verify _predict_proba returns an array; shape depends on mock output (2 per call)
    result = le._predict_proba(["a", "b", "c"])
    # 3 texts split into 2 batches: [2, 1]. Mock returns 2 for each batch = 4 rows.
    # Shape check: it's a numpy array with 2 columns
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[1] == n_classes


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
