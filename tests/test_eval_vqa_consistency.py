"""Unit tests for the pure VQA consistency core."""

import pytest

from awake.eval.vqa_consistency import normalize_text


@pytest.mark.unit
def test_normalize_text_lowercases_strips_collapses_and_drops_punctuation():
    assert normalize_text("  A Red, Car!! ") == "a red car"
    assert normalize_text("It's GREEN.") == "it s green"
    assert normalize_text("") == ""
    assert normalize_text("multiple   spaces\tand\nnewlines") == "multiple spaces and newlines"
