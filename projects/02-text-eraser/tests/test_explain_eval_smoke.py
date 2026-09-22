"""Offline end-to-end canonical explanation, cache, and evaluation checks."""

import importlib
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tests.test_eval_explainers import CharacterTokenizer, LinearClassifier, example

from awake.eval.visible_words import canonical_visible

sys.path.insert(0, str(pathlib.Path("projects/02-text-eraser/scripts").resolve()))
explain_mod = importlib.import_module("20_explain")
eval_mod = importlib.import_module("30_eval")
contract = importlib.import_module("_contract")


@pytest.mark.smoke
def test_all_methods_cache_to_metrics(tmp_path):
    """Run every core method through persisted mappings and common interventions."""
    tok, model = CharacterTokenizer(), LinearClassifier()
    visible = canonical_visible("a bb a incomplete SECRET", tok, 9)
    intact = example(model, visible)
    sub = pd.DataFrame(
        [
            {
                "visible_json": visible.to_json(),
                "predicted_class": intact["predicted_class"],
                "class_scores": intact["class_scores"],
                "gold_visible": [1, 0, 1],
            }
        ]
    )
    cfg = explain_mod.load_config("explainers")
    manifest = {"fixture": "offline"}

    def predict(ids, attention_mask):
        with torch.no_grad():
            return (
                model(input_ids=torch.tensor(ids), attention_mask=torch.tensor(attention_mask))
                .logits.softmax(-1)
                .numpy()
            )

    adapter = SimpleNamespace(predict_proba=predict, tokenizer=tok)
    for name in explain_mod.EXPLAINERS:
        path = explain_mod.run_one_explainer(name, model, tok, sub, tmp_path, manifest, "cpu", cfg)
        table = pq.read_table(path)
        df = contract.validate_cache(table, manifest, name, sub)
        results = eval_mod.evaluate(adapter, sub, df, cfg)
        assert set(results) == {"comprehensiveness", "sufficiency", "aopc", "token_f1", "auprc"}
        assert all(np.isfinite(v).all() for v in results.values())
        with pytest.raises(ValueError, match="stale"):
            contract.validate_cache(table.replace_schema_metadata({}), manifest, name, sub)
        with pytest.raises(ValueError, match="stale"):
            contract.validate_cache(table, {"fixture": "different"}, name, sub)
        for changed in [
            df.iloc[:-1],
            pd.concat([df, df.iloc[:1]]),
            df.assign(word="WRONG"),
            df.assign(predicted_class=0),
        ]:
            corrupt = pa.Table.from_pandas(changed).replace_schema_metadata(table.schema.metadata)
            with pytest.raises(ValueError, match="identity"):
                contract.validate_cache(corrupt, manifest, name, sub)
        with pytest.raises(ValueError, match="identity"):
            eval_mod._scores_for_row(df.iloc[:-1], 0, visible)
        with pytest.raises(ValueError, match="nonfinite"):
            eval_mod._scores_for_row(df.assign(score=np.nan), 0, visible)


@pytest.mark.unit
def test_expected_calibration_error():
    """Check positive-class reliability including zero confidence."""
    assert eval_mod.expected_calibration_error([0, 1], None, [0, 1]) == 0
    assert eval_mod.expected_calibration_error(
        [0.9, 0.9, 0.1, 0.1], None, [1, 1, 0, 0]
    ) == pytest.approx(0.1)
