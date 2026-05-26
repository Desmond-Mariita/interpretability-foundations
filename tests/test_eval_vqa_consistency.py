"""Unit tests for the pure VQA consistency core."""

import pytest

from awake.eval.vqa_consistency import extract_choice, normalize_text


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
