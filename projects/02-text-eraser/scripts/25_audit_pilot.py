"""Audit real-data v2 pilot identities and perturbations before full attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from _contract import checkpoint_identity, file_hash, run_identity, validate_cache
from _paths import MODEL_DIR
from transformers import AutoTokenizer

from awake.eval.visible_words import VisibleWords, canonical_visible, top_words


def main():
    """Check concrete mappings, fixed context, clipping and numerical diagnostics."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    assert not manifest["full_test"] and 2 <= manifest["n"] <= 5
    assert manifest["checkpoint"] == checkpoint_identity(MODEL_DIR)
    complete = json.loads((args.run_dir / "COMPLETE.json").read_text())
    assert complete["run_id"] == run_identity(manifest)
    for name, sha in complete["files"].items():
        assert file_hash(args.run_dir / name) == sha
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    sub = pd.read_parquet(args.run_dir / "subsample.parquet")
    names = ["random", "grad_x_input", "integrated_gradients", "lime"]
    caches = {
        name: validate_cache(pq.read_table(args.run_dir / f"{name}.parquet"), manifest, name, sub)
        for name in names
    }
    ig = json.loads((args.run_dir / "integrated_gradients_diagnostics.json").read_text())
    lime = json.loads((args.run_dir / "lime_diagnostics.json").read_text())
    rows = []
    for i, row in sub.reset_index(drop=True).iterrows():
        v = VisibleWords.from_json(row.visible_json)
        assert canonical_visible(row.text, tok, 512).fingerprint == v.fingerprint
        assert np.array_equal(v.clip_gold(row.gold_mask), row.gold_visible)
        assert np.argmax(row.class_scores) == row.predicted_class
        word_map = np.array(v.token_word)
        for df in caches.values():
            scores = df[df.example_id == i].sort_values("word_idx").score.to_numpy()
            assert len(scores) == len(v.words) and np.isfinite(scores).all()
            top = top_words(scores, manifest["explainer_config"]["k_d"])
            for keep in (top, ~top, np.zeros(len(top), dtype=bool)):
                perturbed = v.perturb(keep[None, :], tok.mask_token_id)[0]
                assert len(perturbed) == len(v.input_ids)
                assert np.array_equal(perturbed[word_map < 0], np.array(v.input_ids)[word_map < 0])
                for w in range(len(v.words)):
                    expected = (
                        np.array(v.input_ids)[word_map == w] if keep[w] else tok.mask_token_id
                    )
                    assert np.all(perturbed[word_map == w] == expected)
        tolerance = 0.05 + 0.01 * abs(ig[i]["endpoint_logit_difference"])
        residual_ok = abs(ig[i]["completeness_residual"]) <= tolerance
        fit_finite = all(
            np.isfinite(lime[i][k]) for k in ("weighted_r2", "local_prediction", "intact_logit")
        )
        rows.append(
            {
                "example_id": i,
                "input_fingerprint": v.fingerprint,
                "original_words": v.original_word_count,
                "visible_words": len(v.words),
                "subwords": len(v.input_ids),
                "partial_words": len(v.partial_word_ids),
                "multi_subword_words": sum(np.sum(word_map == w) > 1 for w in range(len(v.words))),
                "target_class": int(row.predicted_class),
                "gold_visible": int(sum(row.gold_visible)),
                "rationale_coverage": float(row.truncation_coverage),
                "mapping_mask_special_tail_clipping_checks": "passed",
                "ig": ig[i],
                "ig_tolerance": tolerance,
                "ig_pass": bool(residual_ok),
                "lime": lime[i],
                "lime_finite": bool(fit_finite),
            }
        )
    passed = all(r["ig_pass"] and r["lime_finite"] for r in rows)
    result = {
        "schema": "p2-pilot-audit-v2",
        "passed": passed,
        "run_id": run_identity(manifest),
        "numerical_rule": "IG abs residual <= 0.05 + 0.01 * abs endpoint difference; finite LIME diagnostics; inspect R2 and intact error, not method ranking",
        "examples": rows,
    }
    # Convert numpy counters to Python integers without changing numeric measurements.
    (args.run_dir / "pilot_audit.json").write_text(json.dumps(result, indent=2, default=int) + "\n")
    print(json.dumps(result, indent=2, default=int))
    if not passed:
        raise SystemExit("Pilot numerical diagnostic requires investigation")


if __name__ == "__main__":
    main()
