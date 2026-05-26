import importlib

import numpy as np
import pytest

stub = importlib.import_module("_stub_model")
adapter_mod = importlib.import_module("_model_adapter")


@pytest.mark.smoke
def test_model_adapter_predict_proba_shape():
    model, tok = stub.build_stub_model_and_tokenizer()
    adapter = adapter_mod.HFModelAdapter(model, tok, device="cpu")
    ids = np.array([[tok.cls_token_id, 5, 6, tok.sep_token_id]])
    probs = adapter.predict_proba(ids)
    assert probs.shape == (1, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
