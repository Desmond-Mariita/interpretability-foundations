import importlib

import numpy as np
import pytest

stub = importlib.import_module("_stub_model")
adapter_mod = importlib.import_module("_model_adapter")
train_mod = importlib.import_module("10_train")


@pytest.mark.smoke
def test_model_adapter_predict_proba_shape():
    model, tok = stub.build_stub_model_and_tokenizer()
    adapter = adapter_mod.HFModelAdapter(model, tok, device="cpu")
    ids = np.array([[tok.cls_token_id, 5, 6, tok.sep_token_id]])
    probs = adapter.predict_proba(ids)
    assert probs.shape == (1, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.smoke
def test_train_one_step_on_stub(tmp_path):
    model, tok = stub.build_stub_model_and_tokenizer()
    texts = ["w5 w6 w7", "w8 w9 w10"]
    labels = [0, 1]
    out = train_mod.train_loop(model, tok, texts, labels, out_dir=tmp_path,
                               epochs=1, batch_size=2, lr=1e-3, max_len=16, fp16=False)
    assert (tmp_path / "model_sha256.txt").exists()
    assert out["sha256"]
