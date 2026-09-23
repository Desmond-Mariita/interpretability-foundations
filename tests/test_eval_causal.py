"""Unit tests for the P5 v1.1 causal number-intervention core (awake.eval.causal).

These lock the methodology defined in docs/decisions/006: probe-direction recovery in
residual space, the direction-only patch, sign conventions, and the lemma-cluster bootstrap.
"""

import numpy as np
import pytest

from awake.eval.causal import (
    agreement_margin,
    align_direction_to_class,
    cluster_mean_bootstrap,
    direction_patch,
    directional_rate,
    donor_directed_shift,
    orthogonal_unit_vector,
    patch_changes_only_projection,
    residual_space_score,
    standardized_to_residual_direction,
    unit_direction,
)

pytestmark = pytest.mark.unit  # pure unit tests (no external resources)

D = 32


def _random_vec(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


# ---------------------------------------------------------------------------
# 1. standardized coefficient -> residual-space direction equivalence
# ---------------------------------------------------------------------------


def test_residual_space_score_matches_standardized_score():
    """The standardized classifier score and the residual-space score must agree."""
    w_z = _random_vec(D, 0)
    mean = _random_vec(D, 1)
    scale = np.abs(_random_vec(D, 2)) + 0.1
    b = 0.37
    w_x = standardized_to_residual_direction(w_z, scale)
    rng = np.random.default_rng(3)
    for _ in range(10):
        h = rng.standard_normal(D)
        z = (h - mean) / scale
        assert float(w_z @ z + b) == pytest.approx(
            residual_space_score(h, w_x, b, mean, scale), abs=1e-12
        )


def test_standardized_to_residual_direction_shape_mismatch_raises():
    with pytest.raises(ValueError):
        standardized_to_residual_direction(_random_vec(D, 0), _random_vec(D + 1, 1))


def test_unit_direction_has_unit_norm():
    w_x = _random_vec(D, 4)
    u = unit_direction(w_x)
    assert np.linalg.norm(u) == pytest.approx(1.0, abs=1e-12)
    # same direction as w_x
    assert float(w_x @ u) > 0


def test_unit_direction_zero_raises():
    with pytest.raises(ValueError):
        unit_direction(np.zeros(D))


# ---------------------------------------------------------------------------
# 2. direction sign / class mapping
# ---------------------------------------------------------------------------


def test_align_direction_sign_maps_positive_projection_to_plural():
    """classes_=[0,1] (0=Sing, 1=Plur): positive projection must be the plural direction."""
    u = _random_vec(D, 5)
    out = align_direction_to_class(u, classes=[0, 1], positive_class=1)
    assert not out["flipped"]
    assert out["mapping"] == {0: 0, 1: 1}
    assert np.allclose(out["u"], u)


def test_align_direction_sign_flips_when_class_order_reversed():
    """classes_=[1,0]: the direction must be sign-flipped so positive projection = Plur."""
    u = _random_vec(D, 6)
    out = align_direction_to_class(u, classes=[1, 0], positive_class=1)
    assert out["flipped"]
    # after alignment the mapping is in LABEL space: negative -> Sing (0), positive -> Plur (1)
    assert out["mapping"] == {0: 0, 1: 1}
    assert np.allclose(out["u"], -u)


def test_align_direction_rejects_bad_classes():
    with pytest.raises(ValueError):
        align_direction_to_class(_random_vec(D, 7), classes=[0], positive_class=1)
    with pytest.raises(ValueError):
        align_direction_to_class(_random_vec(D, 7), classes=[0, 2], positive_class=1)


def test_align_direction_output_is_json_serializable():
    """The class mapping must be plain Python ints (sklearn returns numpy ints)."""
    import json

    out = align_direction_to_class(
        _random_vec(D, 7), classes=[np.int64(1), np.int64(0)], positive_class=1
    )
    assert type(out["mapping"][0]) is int
    json.dumps({"mapping": out["mapping"], "flipped": out["flipped"]})  # must not raise


# ---------------------------------------------------------------------------
# 3/4. direction-only patch changes only the u projection
# ---------------------------------------------------------------------------


def test_direction_patch_replaces_only_the_projection():
    """Patched residual differs from the original only along u; orthogonal part unchanged."""
    h_r = _random_vec(D, 8)
    h_d = _random_vec(D, 9)
    u = unit_direction(_random_vec(D, 10))
    out = direction_patch(h_r, h_d, u)
    # new projection equals donor projection
    assert float(u @ out["h_prime"]) == pytest.approx(out["a_d"], abs=1e-12)
    assert out["a_r"] == pytest.approx(float(u @ h_r), abs=1e-12)
    assert out["a_d"] == pytest.approx(float(u @ h_d), abs=1e-12)
    # orthogonal component unchanged (independent asserts per the required test list)
    orth_r = h_r - (u @ h_r) * u
    orth_p = out["h_prime"] - (u @ out["h_prime"]) * u
    assert np.allclose(orth_r, orth_p, atol=1e-12)
    assert patch_changes_only_projection(h_r, out["h_prime"], u)
    # delta lies along u with the right magnitude
    assert np.allclose(out["delta"], (out["a_d"] - out["a_r"]) * u, atol=1e-12)
    assert out["norm_delta"] == pytest.approx(abs(out["a_d"] - out["a_r"]), abs=1e-12)


def test_direction_patch_shape_mismatch_raises():
    with pytest.raises(ValueError):
        direction_patch(_random_vec(D, 0), _random_vec(D + 1, 1), _random_vec(D, 2))


def test_patch_changes_only_projection_detects_orthogonal_leak():
    """A patch with an orthogonal component must fail the projection-only check."""
    h_r = _random_vec(D, 11)
    u = unit_direction(_random_vec(D, 12))
    v = orthogonal_unit_vector(seed=0, d=D, u=u)
    h_prime = h_r + 0.5 * v  # pure orthogonal perturbation, no u component
    assert not patch_changes_only_projection(h_r, h_prime, u)


# ---------------------------------------------------------------------------
# 5. no-op reproduces baseline (donor == recipient)
# ---------------------------------------------------------------------------


def test_direction_patch_noop_when_donor_is_recipient():
    """A donor identical to the recipient produces an exact no-op (zero delta)."""
    h = _random_vec(D, 13)
    u = unit_direction(_random_vec(D, 14))
    out = direction_patch(h, h, u)
    assert out["norm_delta"] == 0.0
    assert np.allclose(out["h_prime"], h, atol=0.0)


# ---------------------------------------------------------------------------
# 6/7. random direction: orthogonality and norm matching
# ---------------------------------------------------------------------------


def test_orthogonal_unit_vector_is_unit_and_orthogonal():
    u = unit_direction(_random_vec(D, 15))
    for seed in range(3):
        v = orthogonal_unit_vector(seed=seed, d=D, u=u)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-12)
        assert float(u @ v) == pytest.approx(0.0, abs=1e-12)


def test_orthogonal_unit_vector_is_deterministic():
    u = unit_direction(_random_vec(D, 16))
    v1 = orthogonal_unit_vector(seed=7, d=D, u=u)
    v2 = orthogonal_unit_vector(seed=7, d=D, u=u)
    assert np.array_equal(v1, v2)


def test_orthogonal_unit_vector_shape_mismatch_raises():
    with pytest.raises(ValueError):
        orthogonal_unit_vector(seed=0, d=D, u=np.ones(D + 1))


def test_random_perturbation_matches_primary_norm():
    """C2: ||delta_random|| must equal the primary direction-only patch norm."""
    h_r = _random_vec(D, 17)
    h_d = _random_vec(D, 18)
    u = unit_direction(_random_vec(D, 19))
    primary = direction_patch(h_r, h_d, u)
    v = orthogonal_unit_vector(seed=1, d=D, u=u)
    delta_random = primary["norm_delta"] * v
    assert np.linalg.norm(delta_random) == pytest.approx(primary["norm_delta"], abs=1e-12)


# ---------------------------------------------------------------------------
# 8/9. sign conventions
# ---------------------------------------------------------------------------


def test_agreement_margin_sign_convention():
    """A = y * M: plural subject passes M through; singular subject negates it."""
    m = 2.5
    assert agreement_margin(m, +1) == pytest.approx(2.5)
    assert agreement_margin(m, -1) == pytest.approx(-2.5)
    assert agreement_margin(-2.5, -1) == pytest.approx(2.5)


def test_donor_directed_shift_sign_convention():
    """E = y_d * (M_patched - M_baseline); positive = shift toward the donor number."""
    # plural donor, patch raises M -> positive E
    assert donor_directed_shift(m_patched=3.0, m_baseline=1.0, y_donor=+1) == pytest.approx(2.0)
    # singular donor, patch raises M (toward plural!) -> negative E
    assert donor_directed_shift(m_patched=3.0, m_baseline=1.0, y_donor=-1) == pytest.approx(-2.0)
    # singular donor, patch lowers M -> positive E (toward singular)
    assert donor_directed_shift(m_patched=-1.0, m_baseline=1.0, y_donor=-1) == pytest.approx(2.0)


def test_directional_rate():
    assert directional_rate(np.array([1.0, -1.0, 0.0, 2.0])) == pytest.approx(0.5)
    assert directional_rate(np.array([])) == 0.0
    assert directional_rate(np.array([1e-9, -1e-9])) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 13. lemma-cluster bootstrap
# ---------------------------------------------------------------------------


def test_cluster_mean_bootstrap_basic():
    """CI covers the mean; clusters are the resampling unit; deterministic under seed."""
    rng = np.random.default_rng(0)
    values, groups = [], []
    for g in range(20):  # 20 clusters, 5 rows each
        base = rng.standard_normal()
        for _ in range(5):
            values.append(base + 0.1 * rng.standard_normal())
            groups.append(f"lemma_{g}")
    values = np.array(values)
    groups = np.array(groups)
    lo, mean, hi = cluster_mean_bootstrap(values, groups, n_resamples=2000, seed=0)
    assert lo <= mean <= hi
    assert mean == pytest.approx(float(values.mean()))
    lo2, _, hi2 = cluster_mean_bootstrap(values, groups, n_resamples=2000, seed=0)
    assert (lo2, hi2) == (lo, hi)


def test_cluster_mean_bootstrap_resamples_groups_not_rows():
    """Clustered data: bootstrap over 2 big clusters must NOT behave like row bootstrap."""
    values = np.array([10.0] * 100 + [-10.0] * 100)
    groups = np.array(["a"] * 100 + ["b"] * 100)
    lo, mean, hi = cluster_mean_bootstrap(values, groups, n_resamples=2000, seed=0)
    assert mean == pytest.approx(0.0)
    # resampling whole clusters yields only {+10, 0, -10} resample means: extreme CIs
    assert (lo, hi) == (-10.0, 10.0)


def test_cluster_mean_bootstrap_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cluster_mean_bootstrap(np.ones(3), np.array(["a", "b"]), n_resamples=10)
