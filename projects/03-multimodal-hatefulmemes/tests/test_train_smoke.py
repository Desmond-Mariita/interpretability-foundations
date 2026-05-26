import importlib

import numpy as np
import pytest

stub = importlib.import_module("_stub")
train_mod = importlib.import_module("10_train")


@pytest.mark.smoke
def test_fit_heads_returns_three_boosters():
    img, txt, y = stub.tiny_embeddings()
    heads = train_mod.fit_heads(img, txt, y, lgbm_params={"n_estimators": 20}, seed=0)
    assert set(heads) == {"fused", "image", "text"}
    feats = np.concatenate([img, txt], axis=1)
    assert heads["fused"].predict(feats, raw_score=True).shape == (img.shape[0],)


bg_mod = importlib.import_module("15_background")


@pytest.mark.smoke
def test_sample_background_shapes():
    img, txt, _ = stub.tiny_embeddings(n=40, d=8)
    ib, tb = bg_mod.sample_background(img, txt, n=10, seed=0)
    assert ib.shape == (10, 8) and tb.shape == (10, 8)
    ib2, _tb2 = bg_mod.sample_background(img, txt, n=10, seed=0)
    assert (ib == ib2).all()   # deterministic under fixed seed
