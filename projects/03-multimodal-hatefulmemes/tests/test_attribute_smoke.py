import importlib
import numpy as np
import pytest

eval_mod = importlib.import_module("11_eval")


@pytest.mark.unit
def test_metric_block_keys_and_ranges():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 50)
    scores = y + rng.normal(scale=0.5, size=y.size)   # correlated with y -> auroc > 0.5
    block = eval_mod.metric_block(y, scores, n_boot=200, seed=0)
    for k in ("auroc", "auprc", "acc"):
        assert {"mean", "lo", "hi"} <= set(block[k])
        assert 0.0 <= block[k]["mean"] <= 1.0
    assert block["auroc"]["lo"] <= block["auroc"]["mean"] <= block["auroc"]["hi"]
