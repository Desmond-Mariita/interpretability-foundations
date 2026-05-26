"""Smoke tests for prompt formatting and the stub generator (no models, no dataset)."""

import importlib

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


def _items():
    return [
        {"id": "q1", "question": "color?", "choices": ["red", "blue", "green", "black"],
         "correct_choice_idx": 0},
        {"id": "q2", "question": "animal?", "choices": ["cat", "dog", "fox", "owl"],
         "correct_choice_idx": 1},
    ]


@pytest.mark.smoke
def test_run_pipeline_b_rows_have_expected_fields():
    mod = importlib.import_module("10_run_pipelines")
    prompts = {"answer": "Q: {question}\n{choices_block}"}
    rows = mod.run_pipeline_b(_items(), vlm_generate=lambda p, img: stub_generate(p, _items()[0]),
                              prompts=prompts)
    r = rows[0]
    assert set(r) >= {"id", "answer_idx", "explanation", "raw_output", "parsed_by", "expl_leaks"}
    assert r["parsed_by"] == "strict"
    assert r["answer_idx"] == 0


@pytest.mark.smoke
def test_run_pipeline_a_includes_caption():
    mod = importlib.import_module("10_run_pipelines")
    items = _items()
    prompts = {"answer_with_caption": "{caption}\nQ: {question}\n{choices_block}"}
    rows = mod.run_pipeline_a(
        items,
        caption_fn=lambda path: "a description",
        llm_generate=lambda p: stub_generate(p, items[0]),
        prompts=prompts,
    )
    assert rows[0]["caption"] == "a description"
    assert format_choices(items[0]["choices"]).startswith("A. red")


@pytest.mark.smoke
def test_ablate_arms_differ_only_by_explanation():
    mod = importlib.import_module("20_probe")
    items = _items()
    gen_rows = [{"id": "q1", "answer_idx": 0, "explanation": "because red"},
                {"id": "q2", "answer_idx": 1, "explanation": "because dog"}]
    prompts = {
        "ablate_with_expl": "prev: {explanation}\nQ: {question}\n{choices_block}",
        "ablate_no_expl": "Q: {question}\n{choices_block}",
    }
    seen = {}

    def vlm_generate(prompt, image_path):
        seen["with_expl"] = "prev:" in prompt
        return stub_generate(prompt, items[0])

    rows_with = mod.ablate_pipeline_b(items, gen_rows, vlm_generate, prompts, with_expl=True)
    assert seen["with_expl"] is True
    rows_without = mod.ablate_pipeline_b(items, gen_rows, vlm_generate, prompts, with_expl=False)
    assert seen["with_expl"] is False
    assert {"id", "ablated_idx", "parsed_by"} <= set(rows_with[0])
