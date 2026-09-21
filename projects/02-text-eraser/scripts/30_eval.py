"""Evaluate only complete, identity-verified v2 canonical word caches."""

from __future__ import annotations

import argparse
import json
from itertools import combinations, pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from _contract import checkpoint_identity, code_identity, file_hash, run_identity, validate_cache
from _model_adapter import HFModelAdapter
from _paths import ASSETS, CACHE_DIR, MODEL_DIR, PROJECT_ROOT, load_config
from sklearn.metrics import f1_score

from awake.eval.bootstrap import bootstrap_ci, paired_diff_test
from awake.eval.plausibility import token_auprc
from awake.eval.visible_words import CONTRACT, VisibleWords, top_words

EXPLAINERS = ["random", "grad_x_input", "integrated_gradients", "lime"]


def expected_calibration_error(conf, _preds, labels, n_bins=10) -> float:
    """Positive-class reliability error on intact inputs, not perturbed inputs."""
    conf, labels = np.asarray(conf), np.asarray(labels)
    error = 0.0
    for i, (lo, hi) in enumerate(pairwise(np.linspace(0, 1, n_bins + 1))):
        mask = ((conf >= lo) if i == 0 else (conf > lo)) & (conf <= hi)
        if mask.any():
            error += mask.mean() * abs(labels[mask].mean() - conf[mask].mean())
    return float(error)


def _scores_for_row(attr_df, example_id, visible):
    """Require exact canonical word IDs, strings, input identity and finite scores."""
    rows = attr_df[attr_df.example_id == example_id].sort_values("word_idx")
    if (
        rows.word_idx.tolist() != list(range(len(visible.words)))
        or rows.word.tolist() != visible.words
        or rows.input_fingerprint.tolist() != [visible.fingerprint] * len(visible.words)
    ):
        raise ValueError("canonical attribution identity/length mismatch")
    scores = rows.score.to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite attribution scores")
    return scores


def score_example(predict_proba, visible, scores, gold, pred, mask_id, cfg, expected_probs=None):
    """Word budgets drive identical frozen-token interventions across methods."""
    gold = np.asarray(gold, dtype=int)
    if gold.shape != (len(visible.words),) or not np.isin(gold, [0, 1]).all():
        raise ValueError("visible gold length/value mismatch")
    rationale = top_words(scores, cfg["k_d"])
    keep = np.vstack(
        [
            np.ones(len(scores), dtype=bool),
            ~rationale,
            rationale,
            *[~top_words(scores, f) for f in cfg["aopc_bins"]],
        ]
    )
    probs = predict_proba(
        visible.perturb(keep, mask_id), np.tile(visible.attention_mask, (len(keep), 1))
    )
    if int(probs[0].argmax()) != pred:
        raise ValueError("original prediction changed; cache/checkpoint mismatch")
    if expected_probs is not None and not np.allclose(
        probs[0], expected_probs, atol=1e-5, rtol=1e-5
    ):
        raise ValueError("intact probabilities changed since attribution")
    values = probs[:, pred]
    tp = int((rationale & gold.astype(bool)).sum())
    precision = tp / rationale.sum() if rationale.any() else 0.0
    recall = tp / gold.sum() if gold.any() else 0.0
    return {
        "comprehensiveness": float(values[0] - values[1]),
        "sufficiency": float(values[0] - values[2]),
        "aopc": float((values[0] - values[3:]).mean()),
        "token_f1": float(2 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0,
        "auprc": token_auprc(scores, gold),
    }


def evaluate(adapter, sub, attr_df, cfg):
    """Return per-example scores; never resample/mutate token identities."""
    values = []
    for i, row in sub.reset_index(drop=True).iterrows():
        visible = VisibleWords.from_json(row["visible_json"])
        values.append(
            score_example(
                adapter.predict_proba,
                visible,
                _scores_for_row(attr_df, i, visible),
                row["gold_visible"],
                int(row["predicted_class"]),
                adapter.tokenizer.mask_token_id,
                cfg,
                expected_probs=row["class_scores"],
            )
        )
    return {k: np.array([v[k] for v in values]) for k in values[0]}


def main() -> None:
    """Verify provenance, compute corrected metrics, optionally publish full-run aggregates."""
    import matplotlib
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    completion = json.loads((args.run_dir / "COMPLETE.json").read_text())
    if completion["run_id"] != run_identity(manifest) or manifest["contract"] != CONTRACT:
        raise ValueError("incomplete/stale attribution run")
    for name, digest in completion["files"].items():
        if file_hash(args.run_dir / name) != digest:
            raise ValueError("run file changed: " + name)
    if checkpoint_identity(MODEL_DIR) != manifest["checkpoint"]:
        raise ValueError("checkpoint changed since attribution")
    cfg = load_config("explainers")
    if cfg != manifest["explainer_config"]:
        raise ValueError("configuration changed since attribution")
    sub = pd.read_parquet(args.run_dir / "subsample.parquet")
    if len(sub) != manifest["n"]:
        raise ValueError("sample count mismatch")
    if args.publish and (not manifest["full_test"] or len(sub) != manifest["full_test_n"]):
        raise ValueError("pilot results cannot replace the current headline")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    adapter = HFModelAdapter(model, tok, args.device)
    per_method, results = {}, {}
    boot = cfg["bootstrap"]
    for name in EXPLAINERS:
        df = validate_cache(pq.read_table(args.run_dir / f"{name}.parquet"), manifest, name, sub)
        per_method[name] = evaluate(adapter, sub, df, cfg)
        results[name] = {}
        for metric, values in per_method[name].items():
            valid = values[np.isfinite(values)]
            lo, mean, hi = (
                bootstrap_ci(valid, boot["n_resamples"], boot["alpha"], boot["seed"])
                if len(valid)
                else (None, None, None)
            )
            results[name][metric] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(valid)}
    pairs = list(combinations(EXPLAINERS, 2))
    comparisons = {}
    for a, b in pairs:
        test = paired_diff_test(
            per_method[a]["comprehensiveness"],
            per_method[b]["comprehensiveness"],
            boot["n_resamples"],
            boot["seed"],
        )
        # Finite Monte Carlo resolution, not a claim that p is literally zero.
        test["p_value"] = (test["p_value"] * boot["n_resamples"] + 1) / (boot["n_resamples"] + 1)
        test["significant"] = test["p_value"] < boot["alpha"] / len(pairs)
        comparisons[f"{a}_vs_{b}"] = test
    probs = np.array(sub.class_scores.tolist())
    labels = sub.label.to_numpy()
    diagnostics = {
        "n": len(sub),
        "accuracy": float((probs.argmax(1) == labels).mean()),
        "macro_f1": float(f1_score(labels, probs.argmax(1), average="macro")),
        "ece": expected_calibration_error(probs[:, 1], None, labels),
        "coverage_mean": float(sub.truncation_coverage.mean()),
        "partial_word_examples": sum(
            bool(VisibleWords.from_json(v).partial_word_ids) for v in sub.visible_json
        ),
    }
    ig = json.loads((args.run_dir / "integrated_gradients_diagnostics.json").read_text())
    residuals = np.abs([d["completeness_residual"] for d in ig])
    diagnostics["ig_absolute_completeness_residual"] = {
        "median": float(np.median(residuals)),
        "max": float(np.max(residuals)),
        "p95": float(np.quantile(residuals, 0.95)),
    }
    out = {
        "status": "corrected_full_run" if manifest["full_test"] else "corrected_pilot",
        "contract": CONTRACT,
        "run_id": run_identity(manifest),
        "provenance": manifest,
        "evaluation_code": code_identity(PROJECT_ROOT.parents[1]),
        "diagnostics": diagnostics,
        "metrics": results,
        "pairwise_comprehensiveness": comparisons,
        "bonferroni_alpha": boot["alpha"] / len(pairs),
        "uncertainty": "example bootstrap conditional on checkpoint and explainer seed",
    }
    (args.run_dir / "metrics.json").write_text(json.dumps(out, indent=2, allow_nan=False))
    if args.publish:
        (PROJECT_ROOT / "metrics.json").write_text(json.dumps(out, indent=2, allow_nan=False))
        fig, ax = plt.subplots(figsize=(6, 5))
        for name in EXPLAINERS:
            x, y = results[name]["aopc"]["mean"], results[name]["auprc"]["mean"]
            ax.scatter(x, y, label=name)
            ax.annotate(name, (x, y))
        ax.set(
            xlabel="Perturbation sensitivity (word-mask AOPC)",
            ylabel="Plausibility (visible-word AUPRC)",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(ASSETS / "faithfulness_plausibility.png", dpi=150)
    print(
        json.dumps(
            {"status": out["status"], "diagnostics": diagnostics, "metrics": results}, indent=2
        )
    )


if __name__ == "__main__":
    main()
