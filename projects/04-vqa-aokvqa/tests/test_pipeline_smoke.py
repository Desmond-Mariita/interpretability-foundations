"""Smoke tests for prompt formatting and the stub generator (no models, no dataset)."""

import pytest

from _prompts import format_choices, render
from _stub import stub_generate


@pytest.mark.smoke
def test_format_choices_letters_options():
    assert format_choices(["red", "blue"]) == "A. red\nB. blue"


@pytest.mark.smoke
def test_render_fills_placeholders():
    tmpl = "Q: {question}\n{choices_block}"
    out = render(tmpl, question="why?", choices_block="A. x")
    assert out == "Q: why?\nA. x"


@pytest.mark.smoke
def test_stub_generate_returns_answer_line_and_is_deterministic():
    item = {"id": "q1", "choices": ["red", "blue", "green", "black"], "correct_choice_idx": 2}
    out = stub_generate("any prompt", item)
    assert out.splitlines()[0].startswith("Answer: ")
    assert out == stub_generate("any prompt", item)  # deterministic
