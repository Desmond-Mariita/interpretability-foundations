"""Unit tests for the pure probing core."""

import pytest

from awake.eval.probing import assign_control_labels, balanced_accuracy, base_rate, control_vector
from awake.eval.probing import majority_class, selectivity


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
    # "the" is very frequent; matching in TOKEN space must account for that.
    counts = {"the": 100, "dog": 10, "cat": 10, "run": 10, "sit": 10}
    all_types = set(counts)
    m1 = assign_control_labels(all_types, counts, base_rate=0.3, seed=0)
    m2 = assign_control_labels(all_types, counts, base_rate=0.3, seed=0)
    assert m1 == m2  # deterministic
    assert set(m1) == all_types and set(m1.values()) <= {0, 1}
    # realised TOKEN-level positive share within tolerance of 0.3
    tot = sum(counts.values())
    pos = sum(counts[t] for t, lbl in m1.items() if lbl == 1)
    assert abs(pos / tot - 0.3) < 0.65
    assert assign_control_labels(all_types, counts, 0.3, seed=1) != m1  # seed changes map


@pytest.mark.unit
def test_control_vector_maps_and_raises_on_unseen():
    m = {"a": 0, "b": 1}
    assert control_vector(["a", "b", "a"], m) == [0, 1, 0]
    with pytest.raises(KeyError):
        control_vector(["a", "z"], m)
