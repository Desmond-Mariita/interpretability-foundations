"""Smoke test for the eval aggregator (pure; no plotting, no files)."""

import importlib

import pytest


@pytest.mark.smoke
def test_compute_subset_metrics_shapes_and_delta():
    mod = importlib.import_module("30_eval")
    gold = [0, 1, 2, 3]
    gen = {
        "A": {"answer_idx": [0, 1, 2, 0], "expl_leaks": [True, False, False, False],
              "parsed_by": ["strict"] * 4},
        "B": {"answer_idx": [0, 1, 2, 3], "expl_leaks": [False] * 4,
              "parsed_by": ["strict"] * 4},
        "B7": {"answer_idx": [0, 1, 2, 3], "expl_leaks": [False] * 4,
               "parsed_by": ["strict"] * 4},
    }
    abl = {  # ablated answers per pipeline per arm
        "A": {"expl": [0, 1, 2, 0], "noexpl": [0, 0, 0, 0]},
        "B": {"expl": [0, 1, 2, 3], "noexpl": [0, 1, 2, 3]},
        "B7": {"expl": [0, 1, 2, 3], "noexpl": [0, 1, 2, 3]},
    }
    m = mod.compute_subset_metrics(gen, abl, gold, n_resamples=200, seed=0)
    assert set(m["pipelines"]) == {"A", "B", "B7"}
    a = m["pipelines"]["A"]
    assert 0.0 <= a["accuracy"] <= 1.0
    assert set(a["consistency"]) >= {"with_expl", "no_expl", "delta", "delta_ci"}
    assert set(a["parse_rate"]) == {"answer", "abl_expl", "abl_noexpl"}
    assert set(m["divergence"]) == {"A_vs_B", "A_vs_B7", "B_vs_B7"}
    assert "contingency" in m["divergence"]["A_vs_B"]
