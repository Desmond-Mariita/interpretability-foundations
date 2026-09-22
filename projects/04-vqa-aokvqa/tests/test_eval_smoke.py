"""Smoke test for the eval aggregator (pure; no plotting, no files)."""

import importlib

import pytest


@pytest.mark.smoke
def test_compute_subset_metrics_shapes_and_delta():
    """Subset metrics return the expected schema and the right delta sign."""
    mod = importlib.import_module("30_eval")
    gold = [0, 1, 2, 3]
    gen = {
        "A": {
            "answer_idx": [0, 1, 2, 0],
            "expl_leaks": [True, False, False, False],
            "parsed_by": ["strict"] * 4,
        },
        "B": {"answer_idx": [0, 1, 2, 3], "expl_leaks": [False] * 4, "parsed_by": ["strict"] * 4},
        "B7": {"answer_idx": [0, 1, 2, 3], "expl_leaks": [False] * 4, "parsed_by": ["strict"] * 4},
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
    # Pipeline A: with_expl consistency 1.0, no_expl 0.5 -> delta 0.5 (guards sign flip).
    assert a["consistency"]["with_expl"] == pytest.approx(1.0)
    assert a["consistency"]["no_expl"] == pytest.approx(0.5)
    assert a["consistency"]["delta"] == pytest.approx(0.5)
    assert set(a["parse_rate"]) == {"answer", "abl_expl", "abl_noexpl"}
    assert set(m["divergence"]) == {"A_vs_B", "A_vs_B7", "B_vs_B7"}
    assert "contingency" in m["divergence"]["A_vs_B"]


@pytest.mark.smoke
def test_render_hero_figure_writes_four_panel_png(tmp_path):
    """The hero renderer writes a PNG from aggregate metrics (no models, no data)."""
    mod = importlib.import_module("30_eval")
    metrics = {
        "pipelines": {
            p: {
                "accuracy": 0.5,
                "parse_rate_answer": 0.9,
                "expl_leak_rate": 0.3,
                "consistency": {"delta": 0.1, "delta_ci": [0.05, 0.15]},
            }
            for p in ("A", "B", "B7")
        }
    }
    out = tmp_path / "hero.png"
    mod.render_hero_figure(metrics, out)
    assert out.exists() and out.stat().st_size > 1000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
