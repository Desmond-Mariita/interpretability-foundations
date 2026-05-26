"""Smoke test for data preparation logic (no HF download, no real images)."""

import importlib

import pytest


@pytest.mark.smoke
def test_prepare_rows_adds_leakage_flag_and_keeps_fields():
    """Prepared rows gain a leakage flag and image_path while keeping core fields."""
    mod = importlib.import_module("00_data")
    raw = [
        {
            "id": "q1",
            "question": "color?",
            "choices": ["red", "blue", "green", "black"],
            "correct_choice_idx": 0,
            "rationales": ["it is clearly red", "warm tone"],
        },
        {
            "id": "q2",
            "question": "animal?",
            "choices": ["cat", "dog", "fox", "owl"],
            "correct_choice_idx": 1,
            "rationales": ["it barks", "a pet"],
        },
    ]
    rows = mod.prepare_rows(raw)
    assert rows[0]["leakage_flag"] is True  # "it is clearly red" leaks gold "red"
    assert rows[1]["leakage_flag"] is False  # no rationale contains "dog"
    assert rows[0]["image_path"].endswith("q1.jpg")
    assert set(rows[0]) >= {
        "id",
        "question",
        "choices",
        "correct_choice_idx",
        "rationales",
        "leakage_flag",
        "image_path",
    }
