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


# append to test_attribute_smoke.py
stub = importlib.import_module("_stub")
train_mod = importlib.import_module("10_train")
attr_mod = importlib.import_module("20_attribute")


@pytest.mark.smoke
def test_attribute_rows_have_image_text_shares():
    img, txt, y = stub.tiny_embeddings(n=60, d=8)
    heads = train_mod.fit_heads(img, txt, y, {"n_estimators": 20}, seed=0)
    ib, tb = img[:15], txt[:15]
    rows = attr_mod.attribute_split(img[:10], txt[:10], heads["fused"], ib, tb)
    assert len(rows) == 10
    r0 = rows[0]
    assert {"phi_image", "phi_text", "total", "share"} <= set(r0)
    assert -1.0 <= r0["share"] <= 1.0
