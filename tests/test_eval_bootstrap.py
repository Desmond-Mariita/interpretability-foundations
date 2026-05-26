import numpy as np
import pytest

from awake.eval.bootstrap import bootstrap_ci, paired_diff_test


@pytest.mark.unit
def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    rng_vals = np.linspace(0.0, 1.0, 200)
    lo, mean, hi = bootstrap_ci(rng_vals, n_resamples=500, alpha=0.05, seed=0)
    assert lo < mean < hi
    lo2, mean2, hi2 = bootstrap_ci(rng_vals, n_resamples=500, alpha=0.05, seed=0)
    assert (lo, mean, hi) == (lo2, mean2, hi2)  # seeded determinism


@pytest.mark.unit
def test_paired_diff_test_detects_clear_difference():
    a = np.full(100, 0.8)
    b = np.full(100, 0.2)
    res = paired_diff_test(a, b, n_resamples=500, seed=0)
    assert res["mean_diff"] == pytest.approx(0.6)
    assert res["p_value"] < 0.05


@pytest.mark.unit
def test_paired_diff_test_raises_on_shape_mismatch():
    a = np.array([0.5, 0.6, 0.7])
    b = np.array([0.5, 0.6])
    with pytest.raises(ValueError, match="equal shape"):
        paired_diff_test(a, b)
