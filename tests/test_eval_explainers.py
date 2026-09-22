"""Real differentiable synthetic tests for Project 2's fixed-input boundary."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from awake.eval.explainers.gradient_x_input import GradientXInputExplainer
from awake.eval.explainers.integrated_gradients import IntegratedGradientsExplainer
from awake.eval.explainers.lime_text import LimeExplainer
from awake.eval.explainers.random_baseline import RandomExplainer
from awake.eval.visible_words import canonical_visible

pytestmark = pytest.mark.unit


class CharacterTokenizer:
    """One character per subword; exact offsets, no vocabulary/network needed."""

    mask_token_id = 4
    pad_token_id = 0

    def __call__(self, text, max_length=512, **kwargs):
        chars = [(i, c) for i, c in enumerate(text) if not c.isspace()][: max_length - 2]
        return {
            "input_ids": [1] + [5 + ord(c) % 50 for _, c in chars] + [2],
            "attention_mask": [1] * (len(chars) + 2),
            "offset_mapping": [(0, 0)] + [(i, i + 1) for i, _ in chars] + [(0, 0)],
            "special_tokens_mask": [1] + [0] * len(chars) + [1],
        }


class LinearClassifier(torch.nn.Module):
    """Known logit function; records actual forwarded IDs for tail-leak tests."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 2)
        with torch.no_grad():
            self.embedding.weight[:, 0] = torch.arange(64) / 100
            self.embedding.weight[:, 1] = 0
        self.seen = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask):
        self.seen.extend(input_ids.detach().cpu().tolist())
        value = (self.embedding(input_ids) * attention_mask[..., None]).sum((1, 2))
        return SimpleNamespace(logits=torch.stack([-value, value + 1], dim=1))


def example(model, visible):
    with torch.no_grad():
        logits = model(
            torch.tensor([visible.input_ids]), torch.tensor([visible.attention_mask])
        ).logits
    probs = logits.softmax(-1)[0].numpy()
    return {"visible": visible, "predicted_class": int(probs.argmax()), "class_scores": probs}


@pytest.fixture
def fixture():
    tok = CharacterTokenizer()
    visible = canonical_visible("wow! wow! unhappy TAIL", tok, max_length=17)
    model = LinearClassifier()
    return model, tok, visible, example(model, visible)


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["random", "gxi", "ig", "lime"])
def test_core_methods_share_original_target_and_exact_word_ids(fixture, kind):
    model, tok, visible, ex = fixture
    methods = {
        "random": RandomExplainer(seed=7),
        "gxi": GradientXInputExplainer(model, tok),
        "ig": IntegratedGradientsExplainer(model, tok, n_steps=8),
        "lime": LimeExplainer(model, tok, num_samples=40, batch_size=3, seed=7),
    }
    attr = methods[kind].attribute(ex)
    assert attr.tokens == visible.words == ["wow!", "wow!", "unhappy"]
    assert attr.offsets == visible.spans
    assert attr.predicted_class == ex["predicted_class"] == 1
    assert attr.word_level and np.isfinite(attr.scores).all()
    assert np.allclose(attr.class_scores, ex["class_scores"])
    for ids in model.seen:
        # Only the frozen original token or MASK can occur at each position.
        assert len(ids) == len(visible.input_ids)
        for j, value in enumerate(ids):
            if visible.token_word[j] < 0:
                assert value == visible.input_ids[j]
            else:
                assert value in (visible.input_ids[j], tok.mask_token_id)


@pytest.mark.unit
def test_gradients_are_logits_and_ig_completeness_is_unaggregated(fixture):
    model, tok, visible, ex = fixture
    ig = IntegratedGradientsExplainer(model, tok, n_steps=8).attribute(ex)
    gx = GradientXInputExplainer(model, tok).attribute(ex)
    # Analytical logit gradient = 1; probability gradients would be near zero.
    expected = visible.aggregate(np.array(visible.input_ids) / 100)
    assert np.allclose(gx.scores, expected, atol=1e-6)
    expected_ig = visible.aggregate((np.array(visible.input_ids) - tok.mask_token_id) / 100)
    assert np.allclose(ig.scores, expected_ig, atol=1e-6)
    assert abs(ig.diagnostics["completeness_residual"]) < 1e-5
    assert abs(ig.diagnostics["captum_delta"]) < 1e-5
    assert not np.isclose(sum(ig.scores), ig.diagnostics["attribution_sum"])


@pytest.mark.unit
def test_partial_context_and_specials_preserved_in_ig_and_lime():
    model, tok = LinearClassifier(), CharacterTokenizer()
    visible = canonical_visible("ok incomplete SECRET", tok, max_length=7)
    ex = example(model, visible)
    assert visible.words == ["ok"] and visible.text == "ok inc"
    for method in [
        IntegratedGradientsExplainer(model, tok, n_steps=4),
        LimeExplainer(model, tok, num_samples=10),
    ]:
        method.attribute(ex)
    for ids in model.seen:
        for j, wid in enumerate(visible.token_word):
            if wid < 0:
                assert ids[j] == visible.input_ids[j]
    assert any(ids[1:3] == [4, 4] for ids in model.seen)


@pytest.mark.unit
def test_lime_feature_mapping_has_known_word_effects_and_repeat_positions():
    model, tok = LinearClassifier(), CharacterTokenizer()
    visible = canonical_visible("a bb a", tok)
    attr = LimeExplainer(model, tok, num_samples=1000, seed=1).attribute(example(model, visible))
    assert attr.tokens == ["a", "bb", "a"]
    assert attr.scores[1] > attr.scores[0]
    assert attr.scores[0] == pytest.approx(attr.scores[2], rel=0.02)
    assert attr.diagnostics["weighted_r2"] > 0.99


@pytest.mark.unit
def test_random_seed_is_stable_and_word_based(fixture):
    _, _, visible, ex = fixture
    a = RandomExplainer(seed=42).attribute(ex)
    b = RandomExplainer(seed=42).attribute(ex)
    assert np.array_equal(a.scores, b.scores)
    assert len(a.scores) == len(visible.words)
    assert not np.array_equal(a.scores, RandomExplainer(seed=43).attribute(ex).scores)


@pytest.mark.unit
@pytest.mark.parametrize("change", ["target", "input", "scores"])
def test_bad_target_or_input_rejected(fixture, change):
    model, tok, _, ex = fixture
    if change == "target":
        ex["predicted_class"] = 0
    elif change == "input":
        ex["visible"] = "original full review"
    else:
        ex["class_scores"] = [np.nan, 1]
    with pytest.raises(ValueError):
        GradientXInputExplainer(model, tok).attribute(ex)


@pytest.mark.unit
def test_changed_model_prediction_rejected(fixture):
    model, tok, _, ex = fixture
    ex["predicted_class"] = 0
    ex["class_scores"] = [1, 0]
    for method in [
        GradientXInputExplainer(model, tok),
        IntegratedGradientsExplainer(model, tok),
        LimeExplainer(model, tok, num_samples=8),
    ]:
        with pytest.raises(ValueError, match="target changed"):
            method.attribute(ex)


@pytest.mark.smoke
def test_tiny_roberta_without_download_preserves_token_forward():
    from transformers import RobertaConfig, RobertaForSequenceClassification

    torch.manual_seed(1)
    model = RobertaForSequenceClassification(
        RobertaConfig(
            vocab_size=64,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=24,
            max_position_embeddings=64,
            pad_token_id=0,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
    ).eval()
    tok = CharacterTokenizer()
    visible = canonical_visible("a bb", tok)
    ex = example(model, visible)
    for method in [
        GradientXInputExplainer(model, tok),
        IntegratedGradientsExplainer(model, tok, n_steps=8),
        LimeExplainer(model, tok, num_samples=10),
    ]:
        attr = method.attribute(ex)
        assert attr.predicted_class == ex["predicted_class"]
        assert np.isfinite(attr.scores).all()
