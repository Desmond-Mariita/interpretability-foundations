import pytest

from awake.eval.text_occlusion import occlusion_importance


def _score_fn(tokens):
    """Score = number of 'bad' tokens (additive ground truth)."""
    return float(sum(t == "bad" for t in tokens))


@pytest.mark.unit
def test_occlusion_ranks_influential_token_first():
    tokens = ["the", "movie", "is", "bad"]
    out = occlusion_importance(tokens, _score_fn, top_k=2)
    assert out[0][0] == "bad"
    assert out[0][1] == pytest.approx(1.0)  # removing "bad" drops score by 1
    assert len(out) == 2


@pytest.mark.unit
def test_occlusion_top_k_caps_results():
    tokens = ["bad", "bad", "good", "ok"]
    out = occlusion_importance(tokens, _score_fn, top_k=1)
    assert len(out) == 1 and out[0][1] == pytest.approx(1.0)
