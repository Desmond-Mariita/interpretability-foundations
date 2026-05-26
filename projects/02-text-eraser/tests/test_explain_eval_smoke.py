import importlib
import sys
import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(pathlib.Path("projects/02-text-eraser/scripts").resolve()))
stub = importlib.import_module("_stub_model")
explain_mod = importlib.import_module("20_explain")


@pytest.mark.smoke
def test_explain_writes_cache_with_model_hash(tmp_path):
    model, tok = stub.build_stub_model_and_tokenizer()
    df = pd.DataFrame({"text": ["w5 w6 w7", "w8 w9 w10"], "label": [0, 1]})
    path = explain_mod.run_one_explainer(
        "grad_x_input", model, tok, df, out_dir=tmp_path,
        model_sha="abc123", device="cpu",
    )
    meta = pq.read_table(path).schema.metadata
    assert meta[b"model_sha256"] == b"abc123"
    assert meta[b"explainer_name"] == b"grad_x_input"
