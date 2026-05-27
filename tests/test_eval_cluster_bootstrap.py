"""Unit tests for cluster (sentence-grouped) bootstrap helpers."""

import numpy as np
import pytest

from awake.eval.bootstrap import cluster_bootstrap_ci, paired_cluster_bootstrap


def _acc(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


@pytest.mark.unit
def test_cluster_bootstrap_ci_is_deterministic_and_brackets_mean():
    y_true = [1, 1, 0, 0, 1, 1]
    y_pred = [1, 1, 0, 0, 1, 0]      # 5/6 correct
    groups = [0, 0, 1, 1, 2, 2]
    lo, mean, hi = cluster_bootstrap_ci(y_true, y_pred, groups, _acc, n_resamples=200, seed=0)
    assert lo <= mean <= hi
    assert (lo, mean, hi) == cluster_bootstrap_ci(y_true, y_pred, groups, _acc, n_resamples=200, seed=0)


@pytest.mark.unit
def test_paired_cluster_bootstrap_detects_positive_gap():
    # model a perfect, model b always-wrong -> paired diff strongly positive, CI excludes 0
    y_true = [1, 0, 1, 0, 1, 0, 1, 0]
    pred_a = list(y_true)
    pred_b = [1 - y for y in y_true]
    groups = [0, 0, 1, 1, 2, 2, 3, 3]
    lo, mean, _hi = paired_cluster_bootstrap(y_true, pred_a, pred_b, groups, _acc,
                                             n_resamples=300, seed=0)
    assert mean == pytest.approx(1.0)
    assert lo > 0.0
