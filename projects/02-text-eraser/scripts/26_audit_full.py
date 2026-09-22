"""Audit a completed full-run attribution pipeline before publication.

Sibling of ``25_audit_pilot.py``: the pilot audit asks whether it is safe to spend
the compute; this full-run audit asks whether the completed run is scientifically
safe to publish. It verifies provenance, identity, cache integrity and the
numerical convergence gate. It never recomputes science and never writes or
modifies any audited artifact; it only writes ``full_audit.json`` with the verdict.

A failed publication-critical gate exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import _paths
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from _contract import checkpoint_identity, file_hash, run_identity, validate_cache
from _paths import load_config
from transformers import AutoTokenizer

from awake.eval.visible_words import (
    CONTRACT,
    TARGET_SCALAR,
    VisibleWords,
    canonical_visible,
    top_words,
)

EXPLAINERS = ["random", "grad_x_input", "integrated_gradients", "lime"]
METRICS_SCHEMA = "p2-explainer-metrics-fresh-v2"
PUBLISHED_STATUS = "corrected_full_run"
REQUIRED_IG_STEPS = 800


def _failures(condition: bool, message: str, failures: list[str]) -> None:
    """Record a gate failure without aborting, so the audit reports everything."""
    if not condition:
        failures.append(message)


def _load_tokenizer():
    """Tokenizer from the audited checkpoint; tokenizer identity is checked in use."""
    return AutoTokenizer.from_pretrained(_paths.MODEL_DIR, local_files_only=True)


def audit(run_dir: Path) -> dict:
    """Run every publication-critical gate; return the verdict dict."""
    failures: list[str] = []
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())

    # Gate 1: manifest structure and contract/version identity (stale v1 fails here).
    _failures(
        manifest.get("contract") == CONTRACT, "manifest contract is not the v2 contract", failures
    )
    _failures(
        manifest.get("target_scalar") == TARGET_SCALAR,
        "manifest target scalar is not the corrected fixed logit",
        failures,
    )
    _failures(
        manifest.get("experiment_version") == "fresh-v2" and manifest.get("retrained") is True,
        "manifest is not the fresh-v2 experiment",
        failures,
    )
    _failures(
        manifest.get("full_test") is True, "manifest does not declare the full test split", failures
    )
    n = int(manifest.get("n", -1))
    _failures(n > 0 and n == manifest.get("full_test_n"), "manifest n != full_test_n", failures)

    # Gate 2: checkpoint identity, including schema (historical/smoke checkpoints fail).
    try:
        identity = checkpoint_identity(_paths.MODEL_DIR)
    except ValueError as exc:
        identity = None
        failures.append(f"checkpoint identity unverifiable: {exc}")
    if identity is not None:
        _failures(
            identity.get("schema") == "p2-fresh-training-v2", "checkpoint is not fresh-v2", failures
        )
        _failures(
            manifest.get("checkpoint") == identity,
            "manifest checkpoint identity mismatch",
            failures,
        )

    # Gate 3: explainer config identity, including the converged IG step count.
    cfg = load_config("explainers")
    _failures(
        manifest.get("explainer_config") == cfg,
        "manifest explainer config != current config",
        failures,
    )
    _failures(
        cfg.get("ig", {}).get("n_steps") == REQUIRED_IG_STEPS
        and manifest.get("explainer_config", {}).get("ig", {}).get("n_steps") == REQUIRED_IG_STEPS,
        f"IG n_steps is not {REQUIRED_IG_STEPS}",
        failures,
    )

    # Gate 4: subsample is bound to the authoritative prepared test split.
    prepared = pd.read_parquet(_paths.PREPARED / "test.parquet")
    _failures(
        len(prepared) == n == manifest["full_test_n"],
        "run does not cover the full prepared test split",
        failures,
    )
    _failures(
        file_hash(_paths.PREPARED / "test.parquet") == manifest.get("prepared_test_sha256"),
        "prepared test split hash mismatch",
        failures,
    )
    # Gate 5: run identity and every completion-listed file hash.
    complete = json.loads((run_dir / "COMPLETE.json").read_text())
    _failures(
        complete.get("run_id") == run_identity(manifest),
        "COMPLETE run_id != run_identity(manifest)",
        failures,
    )
    for name, sha in complete.get("files", {}).items():
        _failures(
            (run_dir / name).exists() and file_hash(run_dir / name) == sha,
            f"file hash mismatch or missing: {name}",
            failures,
        )
    # Gate 6: the implementing commit still exists in the repository.
    repo_root = Path(__file__).resolve().parents[2]
    commit = manifest.get("git_commit")
    _failures(
        commit
        and subprocess.run(
            ["git", "cat-file", "-e", commit + "^{commit}"], cwd=repo_root, capture_output=True
        ).returncode
        == 0,
        "manifest git commit is not present in the repository",
        failures,
    )

    # Gate 7: subsample example coverage — no duplicates, no missing examples.
    sub = pd.read_parquet(run_dir / "subsample.parquet")
    _failures(len(sub) == n, "subsample row count != manifest n", failures)
    fingerprints = [VisibleWords.from_json(s).fingerprint for s in sub.visible_json]
    _failures(
        fingerprints == manifest.get("input_fingerprints"),
        "subsample fingerprints != manifest input fingerprints",
        failures,
    )
    _failures(len(set(fingerprints)) == n, "duplicate examples in subsample", failures)

    # Gate 8: canonical mapping identity per example (tokenizer identity in use).
    tok = _load_tokenizer()
    for i, row in sub.reset_index(drop=True).iterrows():
        v = VisibleWords.from_json(row.visible_json)
        rebuilt = canonical_visible(row.text, tok, 512)
        _failures(
            rebuilt.fingerprint == v.fingerprint,
            f"example {i}: canonical fingerprint mismatch",
            failures,
        )
        _failures(
            np.array_equal(v.clip_gold(row.gold_mask), row.gold_visible),
            f"example {i}: gold clipping mismatch",
            failures,
        )
        _failures(
            np.argmax(row.class_scores) == row.predicted_class,
            f"example {i}: predicted class != argmax(class scores)",
            failures,
        )
        if not np.isfinite(row.class_scores).all():
            failures.append(f"example {i}: non-finite class scores")

    # Gate 9: method coverage, cache identity, finiteness, and per-method diagnostics.
    ig_ok, lime_ok = True, True
    validated = {}
    for name in EXPLAINERS:
        for suffix in (f"{name}.parquet", f"{name}_diagnostics.json"):
            _failures((run_dir / suffix).exists(), f"missing artifact: {suffix}", failures)
        if not (run_dir / f"{name}.parquet").exists():
            continue
        try:
            df = validate_cache(pq.read_table(run_dir / f"{name}.parquet"), manifest, name, sub)
        except ValueError as exc:
            failures.append(f"{name}: cache validation failed: {exc}")
            continue
        validated[name] = df
        ids = df.groupby("example_id").size()
        _failures(sorted(ids.index) == list(range(n)), f"{name}: missing example rows", failures)
        words_per_example = [len(VisibleWords.from_json(s).words) for s in sub.visible_json]
        _failures(
            ids.tolist() == words_per_example,
            f"{name}: duplicate or missing word rows per example",
            failures,
        )
        _failures(np.isfinite(df.score).all(), f"{name}: non-finite scores", failures)
        diags = json.loads((run_dir / f"{name}_diagnostics.json").read_text())
        _failures(len(diags) == n, f"{name}: diagnostics count != n", failures)
        for j, d in enumerate(diags):
            _failures(
                int(d["example_id"]) == j,
                f"{name}: diagnostics example_id sequence broken at {j}",
                failures,
            )
            if name == "integrated_gradients":
                _failures(
                    d.get("n_steps") == REQUIRED_IG_STEPS,
                    f"IG example {j}: n_steps != {REQUIRED_IG_STEPS}",
                    failures,
                )
                tol = 0.05 + 0.01 * abs(d["endpoint_logit_difference"])
                if not (
                    np.isfinite(d["completeness_residual"])
                    and abs(d["completeness_residual"]) <= tol
                ):
                    ig_ok = False
                    failures.append(
                        f"IG example {j}: completeness residual {d['completeness_residual']:.4f} "
                        f"exceeds tolerance {tol:.4f}"
                    )
            if name == "lime" and not all(
                np.isfinite(d[k]) for k in ("weighted_r2", "local_prediction", "intact_logit")
            ):
                lime_ok = False
                failures.append(f"LIME example {j}: non-finite surrogate diagnostics")

    # Gate 10: perturbation contract holds for the exact top-k intervention evaluation uses.
    k_d = manifest["explainer_config"]["k_d"]
    for name, df in validated.items():
        for i, row in sub.reset_index(drop=True).iterrows():
            v = VisibleWords.from_json(row.visible_json)
            scores = df[df.example_id == i].sort_values("word_idx").score.to_numpy()
            keep = top_words(scores, k_d)
            perturbed = v.perturb(keep[None, :], tok.mask_token_id)[0]
            word_map = np.array(v.token_word)
            expected = np.array(v.input_ids)
            for w in range(len(v.words)):
                expected[word_map == w] = (
                    np.array(v.input_ids)[word_map == w] if keep[w] else tok.mask_token_id
                )
            _failures(
                np.array_equal(perturbed, expected),
                f"{name} example {i}: perturbation contract violated",
                failures,
            )

    # Gate 11: published run metrics exist and describe this exact run.
    run_metrics_path = run_dir / "metrics.json"
    _failures(run_metrics_path.exists(), "run metrics.json missing", failures)
    published_metrics_path = _paths.PROJECT_ROOT / "metrics.json"
    _failures(published_metrics_path.exists(), "published metrics.json missing", failures)
    run_metrics, published_metrics = None, None
    if run_metrics_path.exists() and published_metrics_path.exists():
        run_metrics = json.loads(run_metrics_path.read_text())
        published_metrics = json.loads(published_metrics_path.read_text())
        _failures(
            run_metrics.get("schema") == METRICS_SCHEMA, "metrics schema is not fresh-v2", failures
        )
        _failures(
            run_metrics.get("status") == PUBLISHED_STATUS,
            f"publication status != {PUBLISHED_STATUS}",
            failures,
        )
        _failures(
            run_metrics.get("run_id") == run_identity(manifest),
            "metrics run_id != run_identity(manifest)",
            failures,
        )
        _failures(
            run_metrics.get("evaluation_code", {}).get("git_commit") == manifest.get("git_commit"),
            "evaluation commit != manifest commit",
            failures,
        )
        _failures(
            published_metrics == run_metrics, "published metrics.json != run metrics.json", failures
        )
        # Gate 12: the published figure exists and was produced with the published metrics.
        figure = _paths.ASSETS / "faithfulness_plausibility.png"
        _failures(
            figure.exists() and figure.stat().st_size > 0,
            "published figure missing or empty",
            failures,
        )
        if figure.exists() and published_metrics_path.exists():
            delta = abs(figure.stat().st_mtime - published_metrics_path.stat().st_mtime)
            _failures(
                delta <= 300,
                "published figure timestamp does not match published metrics",
                failures,
            )

    # Gate 13: eval diagnostics report no failed examples and full method coverage.
    if run_metrics is not None:
        diag = run_metrics.get("diagnostics", {})
        _failures(diag.get("failed_examples") == 0, "evaluation reports failed examples", failures)
        _failures(
            set(diag.get("method_coverage", [])) == set(EXPLAINERS),
            "evaluation method coverage incomplete",
            failures,
        )

    result = {
        "schema": "p2-full-audit-v1",
        "passed": not failures,
        "run_id": run_identity(manifest),
        "n": n,
        "failures": failures,
        "gates": {
            "contract": manifest["contract"],
            "experiment_version": manifest.get("experiment_version"),
            "checkpoint_schema": identity.get("schema") if identity else None,
            "ig_n_steps": cfg["ig"]["n_steps"],
            "ig_completeness_pass": ig_ok,
            "lime_finite_pass": lime_ok,
            "publication_status": (run_metrics or {}).get("status"),
        },
    }
    (run_dir / "full_audit.json").write_text(json.dumps(result, indent=2, default=int) + "\n")
    return result


def main() -> None:
    """CLI: audit the given run dir (default: the versioned v2 cache)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=_paths.CACHE_DIR)
    args = ap.parse_args()
    result = audit(args.run_dir)
    print(json.dumps(result, indent=2, default=int))
    if not result["passed"]:
        raise SystemExit("Full-run audit FAILED; see failures above")


if __name__ == "__main__":
    main()
