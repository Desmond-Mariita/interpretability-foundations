"""Unit tests for the pure VQA consistency core."""

import pytest

from awake.eval.vqa_consistency import (
    accuracy,
    consistency_rate,
    explanation_leaks_answer,
    extract_choice,
    normalize_text,
    parse_rate,
    pipeline_divergence,
    rationale_leaks_answer,
)


@pytest.mark.unit
def test_normalize_text_lowercases_strips_collapses_and_drops_punctuation():
    assert normalize_text("  A Red, Car!! ") == "a red car"
    assert normalize_text("It's GREEN.") == "it s green"
    assert normalize_text("") == ""
    assert normalize_text("multiple   spaces\tand\nnewlines") == "multiple spaces and newlines"


@pytest.mark.unit
def test_extract_choice_strict_first_line_letter():
    choices = ["red car", "blue truck", "green bus", "yellow van"]
    assert extract_choice("Answer: B\nBecause it is blue.", choices) == (1, "strict")
    assert extract_choice("answer : C", choices) == (2, "strict")
    # strict wins even if other choice text also appears
    assert extract_choice("Answer: A\nthe blue truck is wrong", choices) == (0, "strict")


@pytest.mark.unit
def test_extract_choice_text_fallback_unique_substring():
    choices = ["red car", "blue truck", "green bus", "yellow van"]
    assert extract_choice("I think it is a blue truck.", choices) == (1, "text")


@pytest.mark.unit
def test_extract_choice_ambiguous_or_missing_returns_none():
    choices = ["cat", "black cat", "dog", "bird"]
    # output contains "black cat" -> both "cat" and "black cat" match -> ambiguous
    assert extract_choice("clearly a black cat", choices) == (None, "none")
    assert extract_choice("no opinion here", choices) == (None, "none")


@pytest.mark.unit
def test_extract_choice_strict_letter_out_of_range_falls_through():
    choices = ["red car", "blue truck"]  # only A, B valid
    # 'Answer: D' is out of range -> not strict; no text match -> none
    assert extract_choice("Answer: D", choices) == (None, "none")


@pytest.mark.unit
def test_explanation_leaks_answer_substring_match_normalized():
    assert explanation_leaks_answer("Because it is a RED car.", "red car") is True
    assert explanation_leaks_answer("Because it is blue.", "red car") is False
    assert explanation_leaks_answer("anything", "") is False  # empty choice never leaks


@pytest.mark.unit
def test_rationale_leaks_answer_any_rationale_matches():
    assert rationale_leaks_answer(["it flies", "it is a red car"], "red car") is True
    assert rationale_leaks_answer(["it flies", "it is blue"], "red car") is False
    assert rationale_leaks_answer([], "red car") is False


@pytest.mark.unit
def test_leakage_does_not_match_bare_letter():
    # a lone letter token must NOT cause a false leak (the choice TEXT is what matters)
    assert explanation_leaks_answer("the answer is A", "red car") is False


@pytest.mark.unit
def test_parse_rate_counts_non_none_fraction():
    assert parse_rate([0, 1, None, 2]) == 0.75
    assert parse_rate([None, None]) == 0.0
    assert parse_rate([]) == 0.0


@pytest.mark.unit
def test_accuracy_none_counts_as_wrong():
    assert accuracy([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    assert accuracy([0, None, 2, 1], [0, 1, 2, 3]) == 0.5  # None and 1!=3 both wrong
    assert accuracy([], []) == 0.0


@pytest.mark.unit
def test_accuracy_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy([0, 1], [0])


@pytest.mark.unit
def test_consistency_rate_primary_none_either_side_is_inconsistent():
    orig = [0, 1, 2, 3]
    abl = [0, 1, None, 0]  # item0,1 consistent; item2 None; item3 differs
    assert consistency_rate(orig, abl) == 0.5  # 2 of 4


@pytest.mark.unit
def test_consistency_rate_paired_only_drops_unparseable_pairs():
    orig = [0, 1, None, 3]
    abl = [0, 1, 2, None]  # paired-parsed pairs: (0,0)(1,1) only -> both consistent
    assert consistency_rate(orig, abl, paired_only=True) == 1.0
    # primary: 2 consistent of 4
    assert consistency_rate(orig, abl) == 0.5


@pytest.mark.unit
def test_consistency_rate_empty_and_mismatch():
    assert consistency_rate([], []) == 0.0
    assert consistency_rate([0], [0], paired_only=True) == 1.0
    with pytest.raises(ValueError):
        consistency_rate([0, 1], [0])


@pytest.mark.unit
def test_pipeline_divergence_overall_and_contingency():
    #            gold:  0    1    2    3
    a = [0, 1, 2, 0]  # correct, correct, correct, wrong
    b = [0, 2, 2, 1]  # correct, wrong,   correct, wrong
    gold = [0, 1, 2, 3]
    out = pipeline_divergence(a, b, gold)
    # disagreements: item1 (1 vs 2), item3 (0 vs 1) -> 2/4
    assert out["overall"] == 0.5
    c = out["contingency"]
    # items 0 and 2: both correct + agree -> both_correct accumulates agree: 2
    assert c["both_correct"] == {"agree": 2, "disagree": 0}
    # item1: a correct, b wrong, disagree
    assert c["a_correct_b_wrong"] == {"agree": 0, "disagree": 1}
    # item3: both wrong, disagree
    assert c["both_wrong"] == {"agree": 0, "disagree": 1}
    assert c["a_wrong_b_correct"] == {"agree": 0, "disagree": 0}


@pytest.mark.unit
def test_pipeline_divergence_none_counts_as_disagree():
    out = pipeline_divergence([None, 0], [0, 0], [0, 0])
    assert out["overall"] == 0.5  # item0 None vs 0 disagrees; item1 agrees


@pytest.mark.unit
def test_public_api_reexports_vqa_consistency():
    import awake.eval as e

    for name in (
        "normalize_text",
        "extract_choice",
        "explanation_leaks_answer",
        "rationale_leaks_answer",
        "parse_rate",
        "accuracy",
        "consistency_rate",
        "pipeline_divergence",
    ):
        assert hasattr(e, name), name
