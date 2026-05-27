"""Unit tests for the pure probing core."""

import pytest

from awake.eval.probing import balanced_accuracy, base_rate, majority_class, selectivity


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
