"""Unit tests for the pure probing core."""

import pytest

from awake.eval.probing import (
    assign_control_labels,
    balanced_accuracy,
    base_rate,
    control_vector,
    emergence_point,
    majority_class,
    selectivity,
    type_overlap,
)


@pytest.mark.unit
def test_balanced_accuracy_is_mean_per_class_recall():
    # class 0: 3 gold, 2 correct -> recall 2/3; class 1: 1 gold, 1 correct -> recall 1
    y_true = [0, 0, 0, 1]
    y_pred = [0, 0, 1, 1]
    assert balanced_accuracy(y_true, y_pred) == pytest.approx((2 / 3 + 1.0) / 2)


@pytest.mark.unit
def test_balanced_accuracy_chance_is_half_under_imbalance():
    y_true = [0] * 9 + [1]
    y_pred = [0] * 10  # predict all-negative
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.5)  # recall0=1, recall1=0


@pytest.mark.unit
def test_balanced_accuracy_empty_and_mismatch():
    assert balanced_accuracy([], []) == 0.0
    with pytest.raises(ValueError):
        balanced_accuracy([0, 1], [0])


@pytest.mark.unit
def test_base_rate_majority_class_selectivity():
    assert base_rate([0, 0, 1, 1]) == 0.5
    assert base_rate([]) == 0.0
    assert majority_class([0, 0, 0, 1]) == 0
    assert majority_class([1, 1, 0]) == 1
    assert selectivity(0.9, 0.6) == pytest.approx(0.3)


@pytest.mark.unit
def test_assign_control_labels_token_rate_matched_and_deterministic():
    # Realistic mild-Zipf spread (~40 types, no type exceeds ~10% of tokens), so the
    # token-rate match is tight rather than dominated by a single frequent type.
    counts = {
        f"t{i:02d}": c
        for i, c in enumerate(
            [
                10,
                9,
                9,
                8,
                8,
                8,
                7,
                7,
                7,
                7,
                6,
                6,
                6,
                6,
                6,
                5,
                5,
                5,
                5,
                5,
                4,
                4,
                4,
                4,
                4,
                3,
                3,
                3,
                3,
                3,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
            ]
        )
    }
    all_types = set(counts)
    m1 = assign_control_labels(all_types, counts, target_base_rate=0.3, seed=0)
    m2 = assign_control_labels(all_types, counts, target_base_rate=0.3, seed=0)
    assert m1 == m2  # deterministic
    assert set(m1) == all_types and set(m1.values()) <= {0, 1}
    # realised TOKEN-level positive share within tolerance of 0.3
    tot = sum(counts.values())
    pos = sum(counts[t] for t, lbl in m1.items() if lbl == 1)
    assert abs(pos / tot - 0.3) < 0.12
    assert assign_control_labels(all_types, counts, 0.3, seed=1) != m1  # seed changes map


@pytest.mark.unit
def test_control_vector_maps_and_raises_on_unseen():
    m = {"a": 0, "b": 1}
    assert control_vector(["a", "b", "a"], m) == [0, 1, 0]
    with pytest.raises(KeyError):
        control_vector(["a", "z"], m)


@pytest.mark.unit
def test_type_overlap_seen_and_oov_token_rates():
    train = ["a", "a", "b"]
    test = ["a", "a", "c", "c"]  # "a" seen (2 tokens), "c" oov (2 tokens)
    out = type_overlap(train, test)
    assert out["seen_type_token_rate"] == pytest.approx(0.5)
    assert out["oov_type_token_rate"] == pytest.approx(0.5)


@pytest.mark.unit
def test_type_overlap_empty_test_is_zero():
    out = type_overlap(["a", "b"], [])
    assert out == {"seen_type_token_rate": 0.0, "oov_type_token_rate": 0.0}


@pytest.mark.unit
def test_emergence_point_peak_and_earliest_within_ci():
    sel = {"embedding": 0.0, "block_0": 0.1, "block_1": 0.45, "block_2": 0.5, "ln_f": 0.9}
    ci = {
        "embedding": (-0.05, 0.05),
        "block_0": (0.0, 0.2),
        "block_1": (0.38, 0.52),  # overlaps peak's [0.42, 0.58]
        "block_2": (0.42, 0.58),  # peak
        "ln_f": (0.8, 1.0),  # excluded from emergence (not a depth point)
    }
    out = emergence_point(sel, ci)
    assert out["peak"] == "block_2"  # ln_f ignored despite higher selectivity
    assert out["earliest_within_peak_ci"] == "block_1"


@pytest.mark.unit
def test_public_api_reexports_probing_without_shadowing_accuracy():
    import awake.eval as e

    for name in (
        "assign_control_labels",
        "control_vector",
        "balanced_accuracy",
        "base_rate",
        "majority_class",
        "selectivity",
        "type_overlap",
        "emergence_point",
        "cluster_bootstrap_ci",
        "paired_cluster_bootstrap",
    ):
        assert hasattr(e, name), name
    # existing P4 accuracy must remain (probing must NOT shadow it with a different contract)
    assert e.accuracy([0, None, 2], [0, 1, 2]) == pytest.approx(2 / 3)
