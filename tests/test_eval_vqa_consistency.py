"""Unit tests for the pure VQA consistency core."""

import pytest

from awake.eval.vqa_consistency import (
    accuracy,
    explanation_leaks_answer,
    extract_choice,
    normalize_text,
    parse_rate,
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
