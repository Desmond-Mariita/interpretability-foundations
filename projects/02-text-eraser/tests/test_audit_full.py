"""Focused gates protecting the full-run publication verifier (26_audit_full).

Each test fabricates a minimal but fully consistent run directory, mutates one
publication-critical aspect, and asserts the verifier's verdict. The verifier must
return a failed verdict on every mutation and pass the intact run.
"""

import importlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.test_eval_explainers import CharacterTokenizer

from awake.eval.visible_words import CONTRACT, TARGET_SCALAR, canonical_visible

sys.path.insert(0, str(pathlib.Path("projects/02-text-eraser/scripts").resolve()))
audit_mod = importlib.import_module("26_audit_full")
contract = importlib.import_module("_contract")

pytestmark = pytest.mark.unit

TEXTS = ("a bb a incomplete SECRET", "second visible TAIL")
FAKE_IDENTITY = {
    "schema": "p2-fresh-training-v2",
    "training_sha256": "fake-training",
    "weights_sha256": "fake-weights",
    "tokenizer_fingerprint": "fake-tokenizer",
    "files": {},
}


def _write_parquet(run_dir, name, visibles, preds, rid):
    """One word-score row per visible word with run-bound metadata."""
    rows = []
    for i, v in enumerate(visibles):
        for j, word in enumerate(v.words):
            rows.append(
                {
                    "example_id": i,
                    "word_idx": j,
                    "word": word,
                    "score": float(j + 1) / (len(v.words) + 1),
                    "input_fingerprint": v.fingerprint,
                    "predicted_class": int(preds[i]),
                }
            )
    table = pa.Table.from_pandas(pd.DataFrame(rows)).replace_schema_metadata(
        {
            "contract": CONTRACT,
            "target_scalar": TARGET_SCALAR,
            "run_id": rid,
            "explainer_name": name,
        }
    )
    pq.write_table(table, run_dir / f"{name}.parquet")


def _write_diagnostics(run_dir, name, n):
    diags = [{"example_id": i} for i in range(n)]
    if name == "integrated_gradients":
        diags = [
            {
                "example_id": i,
                "n_steps": 800,
                "endpoint_logit_difference": 0.5,
                "completeness_residual": 0.001,
                "captum_delta": 0.001,
                "attribution_sum": 0.5,
            }
            for i in range(n)
        ]
    if name == "lime":
        diags = [
            {
                "example_id": i,
                "weighted_r2": 0.7,
                "local_prediction": 1.0,
                "intact_logit": 1.0,
            }
            for i in range(n)
        ]
    (run_dir / f"{name}_diagnostics.json").write_text(json.dumps(diags, indent=2))


def _completion_files():
    return [
        "manifest.json",
        "subsample.parquet",
        *[f"{m}.parquet" for m in audit_mod.EXPLAINERS],
        *[f"{m}_diagnostics.json" for m in audit_mod.EXPLAINERS],
    ]


def _rewrite_complete(run_dir, rid):
    complete = {
        "run_id": rid,
        "files": {
            name: contract.file_hash(run_dir / name)
            for name in _completion_files()
            if (run_dir / name).exists()
        },
    }
    (run_dir / "COMPLETE.json").write_text(json.dumps(complete, indent=2))


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Fabricate a consistent two-example run; return paths and helpers."""
    tok = CharacterTokenizer()
    visibles = [canonical_visible(t, tok, 512) for t in TEXTS]
    probs = [np.array([0.7, 0.3]), np.array([0.3, 0.7])]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    run_dir = tmp_path / "run"
    prepared_dir = tmp_path / "prepared"
    project_dir = tmp_path / "project"
    assets_dir = tmp_path / "assets"
    for d in (run_dir, prepared_dir, project_dir, assets_dir):
        d.mkdir()
    monkeypatch.setattr(audit_mod._paths, "PREPARED", prepared_dir)
    monkeypatch.setattr(audit_mod._paths, "PROJECT_ROOT", project_dir)
    monkeypatch.setattr(audit_mod._paths, "ASSETS", assets_dir)
    monkeypatch.setattr(audit_mod, "checkpoint_identity", lambda d: dict(FAKE_IDENTITY))
    monkeypatch.setattr(audit_mod, "_load_tokenizer", lambda: tok)

    sub_rows = []
    for i, v in enumerate(visibles):
        gold_mask = np.ones(v.original_word_count, dtype=int)
        sub_rows.append(
            {
                "text": TEXTS[i],
                "visible_json": v.to_json(),
                "gold_mask": gold_mask.tolist(),
                "gold_visible": v.clip_gold(gold_mask).tolist(),
                "class_scores": probs[i].tolist(),
                "predicted_class": int(np.argmax(probs[i])),
            }
        )
    sub = pd.DataFrame(sub_rows)
    sub.to_parquet(run_dir / "subsample.parquet")
    prepared = pd.DataFrame({"visible_json": [v.to_json() for v in visibles], "text": list(TEXTS)})
    prepared.to_parquet(prepared_dir / "test.parquet")

    manifest = {
        "contract": CONTRACT,
        "target_scalar": TARGET_SCALAR,
        "checkpoint": dict(FAKE_IDENTITY),
        "explainer_config": audit_mod.load_config("explainers"),
        "prepared_test_sha256": contract.file_hash(prepared_dir / "test.parquet"),
        "full_test": True,
        "n": 2,
        "full_test_n": 2,
        "experiment_version": "fresh-v2",
        "retrained": True,
        "input_fingerprints": [v.fingerprint for v in visibles],
        "git_commit": head,
        "seed": 0,
        "device": "cpu",
    }
    rid = contract.run_identity(manifest)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    preds = [int(np.argmax(p)) for p in probs]
    for name in audit_mod.EXPLAINERS:
        _write_parquet(run_dir, name, visibles, preds, rid)
        _write_diagnostics(run_dir, name, 2)
    _rewrite_complete(run_dir, rid)

    metrics = {
        "schema": "p2-explainer-metrics-fresh-v2",
        "status": "corrected_full_run",
        "run_id": rid,
        "evaluation_code": {"git_commit": head},
        "diagnostics": {
            "failed_examples": 0,
            "method_coverage": {name: 2 for name in audit_mod.EXPLAINERS},
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    published_metrics = project_dir / "metrics.json"
    published_metrics.write_text(json.dumps(metrics, indent=2))
    figure = assets_dir / "faithfulness_plausibility.png"
    figure.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 64)
    mtime_ns = os.stat(published_metrics).st_mtime_ns
    os.utime(figure, ns=(mtime_ns, mtime_ns))
    return {"run_dir": run_dir, "manifest": manifest, "rid": rid, "visibles": visibles}


def test_intact_run_passes(world):
    """A correct, complete, run-bound artifact set must pass every gate."""
    result = audit_mod.audit(world["run_dir"])
    assert result["passed"], result["failures"]
    assert result["n"] == 2


def test_wrong_run_id_fails(world):
    """A mismatched COMPLETE run ID must fail publication."""
    run_dir = world["run_dir"]
    complete = json.loads((run_dir / "COMPLETE.json").read_text())
    complete["run_id"] = "0" * 64
    (run_dir / "COMPLETE.json").write_text(json.dumps(complete))
    assert not audit_mod.audit(run_dir)["passed"]


def test_missing_file_fails(world):
    """A missing explainer artifact must fail publication."""
    (world["run_dir"] / "lime.parquet").unlink()
    assert not audit_mod.audit(world["run_dir"])["passed"]


def test_bad_file_hash_fails(world):
    """Any completion-listed file whose bytes change must fail publication."""
    run_dir = world["run_dir"]
    path = run_dir / "random_diagnostics.json"
    data = json.loads(path.read_text())
    data.append({"example_id": 99})
    path.write_text(json.dumps(data))
    assert not audit_mod.audit(run_dir)["passed"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_broken_example_ids_fail(world, mutation):
    """Missing or duplicated example rows in a cache must fail publication."""
    run_dir = world["run_dir"]
    df = pd.read_parquet(run_dir / "random.parquet")
    if mutation == "missing":
        df = df[df.example_id != 1]
    else:
        df = pd.concat([df, df[df.example_id == 0]])
    table = pa.Table.from_pandas(df).replace_schema_metadata(
        pq.read_table(run_dir / "random.parquet").schema.metadata
    )
    pq.write_table(table, run_dir / "random.parquet")
    result = audit_mod.audit(run_dir)
    assert not result["passed"]
    assert any("random" in f for f in result["failures"])


def test_wrong_ig_step_count_fails(world):
    """A manifest/config with unconverged IG steps must fail publication."""
    run_dir = world["run_dir"]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["explainer_config"]["ig"]["n_steps"] = 50
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    rid = contract.run_identity(manifest)
    _rewrite_complete(run_dir, rid)
    result = audit_mod.audit(run_dir)
    assert not result["passed"]
    assert any("n_steps" in f for f in result["failures"])


def test_excessive_completeness_residual_fails(world):
    """An IG completeness residual above tolerance must fail publication."""
    run_dir = world["run_dir"]
    path = run_dir / "integrated_gradients_diagnostics.json"
    diags = json.loads(path.read_text())
    diags[1]["completeness_residual"] = 100.0
    path.write_text(json.dumps(diags))
    _rewrite_complete(run_dir, world["rid"])
    result = audit_mod.audit(run_dir)
    assert not result["passed"]
    assert any("completeness residual" in f for f in result["failures"])


def test_stale_contract_fails(world):
    """A run declaring a stale contract/version must fail publication."""
    run_dir = world["run_dir"]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["contract"] = "eraser-visible-words-v1"
    manifest["experiment_version"] = "fresh-v1"
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    rid = contract.run_identity(manifest)
    _rewrite_complete(run_dir, rid)
    assert not audit_mod.audit(run_dir)["passed"]


def test_incomplete_method_set_fails(world):
    """Missing method coverage in artifacts and metrics must fail publication."""
    run_dir = world["run_dir"]
    for suffix in ("lime.parquet", "lime_diagnostics.json"):
        (run_dir / suffix).unlink()
    _rewrite_complete(run_dir, world["rid"])
    assert not audit_mod.audit(run_dir)["passed"]


def test_main_exit_codes(world, monkeypatch):
    """The CLI must exit zero on pass and non-zero on publication failure."""
    monkeypatch.setattr(sys, "argv", ["26_audit_full.py", "--run-dir", str(world["run_dir"])])
    audit_mod.main()  # intact world: no SystemExit
    complete = json.loads((world["run_dir"] / "COMPLETE.json").read_text())
    complete["run_id"] = "1" * 64
    (world["run_dir"] / "COMPLETE.json").write_text(json.dumps(complete))
    with pytest.raises(SystemExit) as exc:
        audit_mod.main()
    assert exc.value.code != 0
