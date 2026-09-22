"""Unit tests for the repaired P3 statistical/semantic contracts (scripts/_stats.py).

Covers: paired AUROC bootstrap (known sign, exact point delta, shared resample
indices, degenerate resamples, determinism), signed-vs-magnitude modality
shares (including the phi_image=-10 / phi_text=+1 counterexample and the
zero/zero case), Shapley efficiency, and interaction non-additivity.
"""

import importlib
import math
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

stats = importlib.import_module("_stats")


def _paired_scores(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic balanced labels + two score arrays with a known AUROC ordering."""
    rng = np.random.default_rng(seed)
    n = 200
    y = np.tile([0, 1], n // 2)
    a = y + rng.normal(scale=0.8, size=n)  # strongly correlated -> high AUROC
    b = np.full(n, 0.5)  # uninformative -> AUROC 0.5
    return y, a, b


class _FixedRng:
    """Minimal rng stub returning a precomputed index matrix once."""

    def __init__(self, idx_matrix: np.ndarray):
        self.idx_matrix = idx_matrix
        self.calls = 0

    def integers(self, low, high, size):
        assert self.calls == 0
        self.calls += 1
        return self.idx_matrix


@pytest.mark.unit
def test_paired_auroc_bootstrap_known_sign_and_exact_point_diff():
    """Point diff equals AUROC_A - AUROC_B on the full sample and has the known sign."""
    y, a, b = _paired_scores()
    out = stats.paired_auroc_bootstrap(y, a, b, n_resamples=500, seed=0)
    assert out["point_diff"] == pytest.approx(roc_auc_score(y, a) - roc_auc_score(y, b))
    assert out["point_diff"] > 0.3  # a strongly separates, b is chance
    assert out["ci_low"] < out["ci_high"]
    assert math.isfinite(out["ci_low"]) and math.isfinite(out["ci_high"])
    assert out["n_valid"] == 500
    assert out["n_resamples"] == 500
    assert out["seed"] == 0


@pytest.mark.unit
def test_paired_metric_bootstrap_uses_same_indices_for_both_models():
    """Both metric calls inside a resample receive the SAME resampled indices."""
    rng_data = np.random.default_rng(1)
    n = 50
    y = rng_data.integers(0, 2, n).astype(float)
    a = rng_data.normal(size=n)
    b = rng_data.normal(size=n)
    n_resamples = 40
    idx_matrix = np.random.default_rng(2).integers(0, n, size=(n_resamples, n))
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def recording_metric(yi, si):
        calls.append((yi.copy(), si.copy()))
        return float(si.mean())

    out = stats.paired_metric_bootstrap(
        y, a, b, recording_metric, n_resamples=n_resamples, seed=0, rng=_FixedRng(idx_matrix)
    )
    assert out["n_valid"] == n_resamples
    # The first 2 * n_resamples calls are the paired resample evaluations; the
    # final two calls compute point_diff on the full sample.
    assert len(calls) == 2 * n_resamples + 2
    for k, idx in enumerate(idx_matrix):
        y_a, s_a = calls[2 * k]
        y_b, s_b = calls[2 * k + 1]
        assert np.array_equal(y_a, y[idx]) and np.array_equal(s_a, a[idx])
        assert np.array_equal(y_b, y[idx]) and np.array_equal(s_b, b[idx])
        assert np.array_equal(s_a, a[idx])  # same idx applied to model A and B
    # CI matches a manual recomputation over the same fixed indices.
    manual = [a[idx].mean() - b[idx].mean() for idx in idx_matrix]
    assert out["ci_low"] == pytest.approx(float(np.percentile(manual, 2.5)))
    assert out["ci_high"] == pytest.approx(float(np.percentile(manual, 97.5)))


@pytest.mark.unit
def test_paired_auroc_bootstrap_degenerate_resamples_counted_not_coerced():
    """Single-class resamples raise inside AUROC, are skipped, and n_valid reflects it."""
    rng = np.random.default_rng(3)
    n = 4
    y = np.array([0.0, 1.0, 0.0, 1.0])
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    out = stats.paired_auroc_bootstrap(y, a, b, n_resamples=2000, seed=0)
    assert 0 < out["n_valid"] < 2000  # P(single-class resample) = 2 * 0.5^4 > 0
    assert math.isfinite(out["ci_low"]) and math.isfinite(out["ci_high"])
    assert out["point_diff"] == pytest.approx(roc_auc_score(y, a) - roc_auc_score(y, b))


@pytest.mark.unit
def test_paired_metric_bootstrap_all_degenerate_raises():
    """If every resample is degenerate the CI is undefined: raise, never return nonsense."""

    def always_degenerate(yi, si):
        raise ValueError("single-class labels")

    y = np.array([0.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="degenerate"):
        stats.paired_metric_bootstrap(y, y, y, always_degenerate, n_resamples=10, seed=0)


@pytest.mark.unit
def test_paired_metric_bootstrap_shape_mismatch_raises():
    """Length mismatch between labels and scores is rejected up front."""
    with pytest.raises(ValueError, match="equal length"):
        stats.paired_metric_bootstrap(np.zeros(4), np.zeros(5), np.zeros(5), lambda y, s: 0.0)


@pytest.mark.unit
def test_paired_auroc_bootstrap_deterministic_under_fixed_seed():
    """Same seed -> identical output dict."""
    y, a, b = _paired_scores()
    out1 = stats.paired_auroc_bootstrap(y, a, b, n_resamples=200, seed=7)
    out2 = stats.paired_auroc_bootstrap(y, a, b, n_resamples=200, seed=7)
    assert out1 == out2


@pytest.mark.unit
def test_shapley_2player_efficiency_holds():
    """phi_image + phi_text == v(ab) - v(empty) for the exact 2-player game."""
    from awake.eval.modality_shapley import shapley_2player

    for v_empty, v_img, v_txt, v_ab in [(0.0, 1.0, 2.0, 3.5), (-1.0, -0.5, 0.0, 2.0)]:
        phi_img, phi_txt = shapley_2player(v_empty, v_img, v_txt, v_ab)
        assert math.isclose(phi_img + phi_txt, v_ab - v_empty, rel_tol=1e-12)


@pytest.mark.unit
def test_sign_and_magnitude_are_distinguished_counterexample():
    """phi_image=-10, phi_text=+1: image dominates in MAGNITUDE despite a negative sign."""
    signed = stats.signed_image_share(-10.0, 1.0)
    magnitude = stats.magnitude_share_image(-10.0, 1.0)
    assert signed < 0  # direction: the image pushes the margin down
    assert magnitude == pytest.approx(10.0 / 11.0)
    assert magnitude > 0.5  # magnitude: the image carries most of the contribution
    assert stats.magnitude_share_image(1.0, -10.0) == pytest.approx(1.0 / 11.0)  # complements


@pytest.mark.unit
def test_shares_zero_zero_defined():
    """Zero/zero contribution is defined: magnitude 0.5 (balanced), signed 0 (no direction)."""
    assert stats.magnitude_share_image(0.0, 0.0) == 0.5
    assert stats.signed_image_share(0.0, 0.0) == 0.0


@pytest.mark.unit
def test_shares_complementary():
    """Image and text magnitude shares sum to 1 for any non-zero pair."""
    rng = np.random.default_rng(4)
    for _ in range(20):
        p_img, p_txt = rng.normal(size=2)
        if abs(p_img) + abs(p_txt) < 1e-9:
            continue
        m_img = stats.magnitude_share_image(p_img, p_txt)
        m_txt = stats.magnitude_share_image(p_txt, p_img)
        assert m_img + m_txt == pytest.approx(1.0)


@pytest.mark.unit
def test_interaction_additive_game_is_zero():
    """An additive value function has zero interaction."""
    v_empty, v_img, v_txt = 1.0, 2.0, 3.0
    v_ab = v_img + v_txt - v_empty  # perfectly additive
    assert stats.interaction(v_empty, v_img, v_txt, v_ab) == pytest.approx(0.0)


@pytest.mark.unit
def test_interaction_nonadditive_game_has_expected_sign():
    """Super-/sub-additive games give the expected interaction sign and magnitude."""
    v_empty, v_img, v_txt = 1.0, 2.0, 3.0
    assert stats.interaction(v_empty, v_img, v_txt, 5.0) == pytest.approx(1.0)  # super-additive
    assert stats.interaction(v_empty, v_img, v_txt, 3.5) == pytest.approx(-0.5)  # sub-additive


@pytest.mark.unit
def test_repaired_metrics_json_schema_contract():
    """The committed metrics.json carries the corrected schema.

    Pre-rerun, per-example-derived quantities are null; after the rerun they are
    floats/dicts. Both states satisfy the contract.
    """
    import json

    metrics_path = Path(__file__).resolve().parents[1] / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    diffs = metrics["auroc_diffs"]
    for pair in ("fused_vs_image", "fused_vs_text"):
        assert isinstance(diffs[pair]["point_diff"], float)
        assert diffs[pair]["ci_low"] is None or isinstance(diffs[pair]["ci_low"], float)
        assert diffs[pair]["ci_high"] is None or isinstance(diffs[pair]["ci_high"], float)
        assert diffs[pair]["seed"] == 0
    attr = metrics["attribution"]
    assert set(attr) >= {
        "mean_abs_phi",
        "mean_signed_phi",
        "signed_share_mean",
        "magnitude_share_mean",
        "magnitude_share_of_mean_abs",
        "interaction",
    }
    mshare = attr["magnitude_share_of_mean_abs"]
    assert isinstance(mshare["image"], float) and isinstance(mshare["text"], float)
    assert mshare["image"] + mshare["text"] == pytest.approx(1.0)
    assert metrics["background"]["seed"] == 1337
    # The pre-rerun repair provenance records the removed mislabeled diffs; after a
    # rerun the file is regenerated by the pipeline scripts and that block is absent.
    removed = metrics.get("provenance", {}).get("removed_prob_array_diffs")
    if removed is not None:
        assert removed["fused_vs_image"] == pytest.approx(-0.0304, abs=1e-4)
