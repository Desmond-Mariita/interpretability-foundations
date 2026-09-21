"""Regenerate all core explanations under a versioned, fingerprinted contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from _contract import checkpoint_identity, code_identity, file_hash, run_identity
from _explainers import build_explainer
from _model_adapter import HFModelAdapter
from _paths import CACHE_DIR, DATA_PATH, MODEL_DIR, PREPARED, PROJECT_ROOT, ensure_dirs, load_config

from awake.eval.visible_words import CONTRACT, TARGET_SCALAR, VisibleWords

EXPLAINERS = ["random", "grad_x_input", "integrated_gradients", "lime"]


def run_one_explainer(name, model, tok, df, out_dir, manifest, device, cfg):
    """Write canonical word scores and per-example diagnostics, bound to a run."""
    explainer = build_explainer(name, model, tok, cfg, device)
    rows, diagnostics = [], []
    for i, row in df.reset_index(drop=True).iterrows():
        visible = VisibleWords.from_json(row["visible_json"])
        attr = explainer.attribute(
            {
                "visible": visible,
                "predicted_class": int(row["predicted_class"]),
                "class_scores": row["class_scores"],
            }
        )
        if (
            attr.tokens != visible.words
            or list(map(tuple, attr.offsets)) != list(map(tuple, visible.spans))
            or attr.predicted_class != int(row["predicted_class"])
            or not attr.word_level
        ):
            raise ValueError("explainer violated canonical identity")
        for j, (word, score) in enumerate(zip(attr.tokens, attr.scores, strict=True)):
            rows.append(
                {
                    "example_id": int(i),
                    "word_idx": j,
                    "word": word,
                    "score": float(score),
                    "input_fingerprint": visible.fingerprint,
                    "predicted_class": attr.predicted_class,
                }
            )
        diagnostics.append({"example_id": int(i), **attr.diagnostics})
        print(f"{name}: {i + 1}/{len(df)}", flush=True)
    table = pa.Table.from_pandas(pd.DataFrame(rows)).replace_schema_metadata(
        {
            "contract": CONTRACT,
            "target_scalar": TARGET_SCALAR,
            "run_id": run_identity(manifest),
            "explainer_name": name,
        }
    )
    out = Path(out_dir) / f"{name}.parquet"
    pq.write_table(table, out)
    (Path(out_dir) / f"{name}_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    return out


def main() -> None:
    """Reuse a verified classifier; full test split by default, separate pilot dirs."""
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--run-dir", type=Path, default=CACHE_DIR)
    args = ap.parse_args()
    if args.limit is not None and (
        args.limit <= 0 or args.run_dir.resolve() == CACHE_DIR.resolve()
    ):
        raise ValueError("positive pilot limit requires a separate --run-dir")
    ensure_dirs()
    cfg = load_config("explainers")
    data_cfg = load_config("data")
    identity = checkpoint_identity(MODEL_DIR)
    code = code_identity(PROJECT_ROOT.parents[1])
    # Verify the versioned source archive; extracted-data identity is also captured below.
    archive = DATA_PATH / "movies.tar.gz"
    if file_hash(archive) != data_cfg["sha256"]:
        raise ValueError("dataset archive hash mismatch")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    adapter = HFModelAdapter(model, tok, device=args.device)
    full = pd.read_parquet(PREPARED / "test.parquet")
    sub = full if args.limit is None else full.head(args.limit).copy()
    if sub.empty:
        raise ValueError("empty evaluation sample")
    probs = []
    from awake.eval.visible_words import canonical_visible

    for _, row in sub.iterrows():
        visible = VisibleWords.from_json(row["visible_json"])
        rebuilt = canonical_visible(row["text"], tok, data_cfg["max_seq_len"])
        if (
            rebuilt.fingerprint != visible.fingerprint
            or list(row["input_ids"]) != visible.input_ids
        ):
            raise ValueError("prepared input no longer matches checkpoint tokenizer")
        if visible.clip_gold(row["gold_mask"]).tolist() != list(row["gold_visible"]):
            raise ValueError("prepared gold clipping mismatch")
        probs.append(
            adapter.predict_proba(
                np.array([visible.input_ids]), np.array([visible.attention_mask])
            )[0]
        )
    sub = sub.copy()
    sub["class_scores"] = [p.tolist() for p in probs]
    sub["predicted_class"] = [int(p.argmax()) for p in probs]
    manifest = {
        "contract": CONTRACT,
        "target_scalar": TARGET_SCALAR,
        **code,
        "checkpoint": identity,
        "dataset": data_cfg,
        "prepared_test_sha256": file_hash(PREPARED / "test.parquet"),
        "explainer_config": cfg,
        "seed": cfg["bootstrap"]["seed"],
        "n": len(sub),
        "full_test_n": len(full),
        "full_test": args.limit is None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
        "retrained": False,
        "aggregation": "max_absolute_subword; absolute_LIME_coefficient; uniform_word_random",
        "perturbation": "mask_all_subwords_of_complete_words; fixed_partial_context",
        "input_fingerprints": [VisibleWords.from_json(s).fingerprint for s in sub.visible_json],
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    completion = args.run_dir / "COMPLETE.json"
    completion.unlink(missing_ok=True)
    sub.to_parquet(args.run_dir / "subsample.parquet")
    (args.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for name in EXPLAINERS:
        run_one_explainer(name, model, tok, sub, args.run_dir, manifest, args.device, cfg)
    completion.write_text(
        json.dumps(
            {
                "run_id": run_identity(manifest),
                "files": {
                    name: file_hash(args.run_dir / name)
                    for name in [
                        "manifest.json",
                        "subsample.parquet",
                        *[f"{name}.parquet" for name in EXPLAINERS],
                        *[f"{name}_diagnostics.json" for name in EXPLAINERS],
                    ]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
