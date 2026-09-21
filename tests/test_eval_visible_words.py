"""Adversarial offset/word alignment fixtures, without models or downloads."""

import json
from dataclasses import replace

import numpy as np
import pytest
from tests.test_eval_explainers import CharacterTokenizer

from awake.eval.visible_words import VisibleWords, canonical_visible, top_words

pytestmark = pytest.mark.unit


@pytest.mark.unit
@pytest.mark.parametrize(
    "text,expected",
    [
        ("good, bad!", ["good,", "bad!"]),
        ("very very good", ["very", "very", "good"]),
        ("unhappy", ["unhappy"]),
        ("  a\tbb\nccc ", ["a", "bb", "ccc"]),
        ("café 好", ["café", "好"]),
    ],
)
def test_span_identity_and_repeated_position_ids(text, expected):
    visible = canonical_visible(text, CharacterTokenizer())
    assert visible.words == expected
    assert visible.original_word_ids == list(range(len(expected)))
    for i, (a, b) in enumerate(visible.spans):
        assert text[a:b] == expected[i]
        assert i in visible.token_word
    assert visible.token_word[0] == visible.token_word[-1] == -1
    assert VisibleWords.from_json(visible.to_json()).fingerprint == visible.fingerprint


@pytest.mark.unit
@pytest.mark.parametrize(
    "length,words,partial,prefix",
    [
        (4, ["ok"], [], "ok"),
        (7, ["ok"], [1], "ok inc"),
        (14, ["ok", "incomplete"], [], "ok incomplete"),
    ],
)
def test_truncation_boundary_policy(length, words, partial, prefix):
    v = canonical_visible("ok incomplete SECRET", CharacterTokenizer(), length)
    assert v.words == words and v.partial_word_ids == partial and v.text == prefix
    assert v.clip_gold([1, 1, 1]).tolist() == [1] * len(words)
    assert "SECRET" not in v.text


@pytest.mark.unit
def test_max_abs_aggregation_not_padding_or_repeating():
    v = canonical_visible("a bcd", CharacterTokenizer())
    assert v.aggregate(np.array([999, -2, 3, -7, 1, 999])).tolist() == [2, 7]
    with pytest.raises(ValueError, match="length"):
        v.aggregate(np.array([2, 7]))
    with pytest.raises(ValueError):
        v.aggregate(np.full(len(v.input_ids), np.nan))


@pytest.mark.unit
def test_mask_word_maps_all_subwords_and_preserves_specials():
    v = canonical_visible("a bcd", CharacterTokenizer())
    masked = v.perturb(np.array([[1, 0], [0, 1]]), 4)
    assert masked[0].tolist() == [v.input_ids[0], v.input_ids[1], 4, 4, 4, v.input_ids[-1]]
    assert masked[1, 1] == 4
    assert masked[1, 2:-1].tolist() == v.input_ids[2:-1]


@pytest.mark.unit
@pytest.mark.parametrize("mask", [[[1]], [[1, 2]], [1, 0]])
def test_wrong_perturbation_shapes_raise(mask):
    v = canonical_visible("a b", CharacterTokenizer())
    with pytest.raises(ValueError):
        v.perturb(np.array(mask), 4)


@pytest.mark.unit
def test_gold_clipping_uses_original_whitespace_ids():
    v = canonical_visible("yes, yes, incomplete", CharacterTokenizer(), 12)
    assert v.words == ["yes,", "yes,"]
    assert v.clip_gold([0, 1, 1]).tolist() == [0, 1]
    with pytest.raises(ValueError):
        v.clip_gold([1, 0])
    with pytest.raises(ValueError):
        v.clip_gold([0, 2, 1])


@pytest.mark.unit
@pytest.mark.parametrize(
    "field,value",
    [
        ("input_ids", []),
        ("words", []),
        ("words", ["wrong", "bb"]),
        ("token_word", [-1, 1, 1, 1, -1]),
        ("original_word_ids", [1, 0]),
        ("original_word_ids", [0, 4]),
        ("offsets", [(0, 0), (0, 1), (2, 3), (3, 99), (0, 0)]),
        ("spans", [(0, 1), (1, 4)]),
    ],
)
def test_mismatched_identity_fails_loudly(field, value):
    v = canonical_visible("a bb", CharacterTokenizer())
    with pytest.raises(ValueError):
        replace(v, **{field: value})


@pytest.mark.unit
def test_old_contract_rejected():
    data = json.loads(canonical_visible("a", CharacterTokenizer()).to_json())
    data["contract"] = "v1"
    with pytest.raises(ValueError, match="stale"):
        VisibleWords.from_json(json.dumps(data))


@pytest.mark.unit
def test_no_complete_word_is_explicit_failure():
    with pytest.raises(ValueError, match="empty"):
        canonical_visible("incomplete", CharacterTokenizer(), 3)


@pytest.mark.unit
def test_budget_is_complete_words_ceil_and_stable_ties():
    assert top_words(np.array([2, 2, 0]), 0.2).tolist() == [True, False, False]
    assert not top_words(np.array([2, 1]), 0).any()
    assert top_words(np.array([2, 1]), 1).all()
    for scores, frac in [(np.array([np.nan]), 0.2), (np.array([[1]]), 0.2), (np.array([1]), 2)]:
        with pytest.raises(ValueError):
            top_words(scores, frac)
