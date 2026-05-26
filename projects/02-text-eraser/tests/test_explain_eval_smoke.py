"""Smoke and unit tests for the P2 explain (20_explain) and eval (30_eval) scripts."""

import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(pathlib.Path("projects/02-text-eraser/scripts").resolve()))
stub = importlib.import_module("_stub_model")
explain_mod = importlib.import_module("20_explain")


@pytest.mark.smoke
def test_explain_writes_cache_with_model_hash(tmp_path):
    """Verify that run_one_explainer writes a Parquet file with the expected model hash and explainer name in its metadata."""
    model, tok = stub.build_stub_model_and_tokenizer()
    df = pd.DataFrame({"text": ["w5 w6 w7", "w8 w9 w10"], "label": [0, 1]})
    path = explain_mod.run_one_explainer(
        "grad_x_input",
        model,
        tok,
        df,
        out_dir=tmp_path,
        model_sha="abc123",
        device="cpu",
    )
    meta = pq.read_table(path).schema.metadata
    assert meta[b"model_sha256"] == b"abc123"
    assert meta[b"explainer_name"] == b"grad_x_input"


eval_mod = importlib.import_module("30_eval")


@pytest.mark.unit
def test_expected_calibration_error_zero_for_perfect():
    """ECE is near-zero when predicted confidences exactly match per-bin accuracy."""
    # confidences equal accuracy in each bin -> ECE small
    probs = np.array([0.9, 0.9, 0.1, 0.1])
    preds = np.array([1, 1, 0, 0])
    labels = np.array([1, 1, 0, 0])
    ece = eval_mod.expected_calibration_error(probs, preds, labels, n_bins=5)
    assert ece == pytest.approx(0.1, abs=0.05)
