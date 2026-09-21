"""Project 2 provenance and cache identity checks; never infer missing metadata."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from awake.eval.visible_words import CONTRACT, TARGET_SCALAR, VisibleWords


def file_hash(path: Path) -> str:
    """SHA-256 a file without loading model weights into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_identity(model_dir: Path) -> dict:
    """Verify the historical training hash and fingerprint all current files."""
    files = sorted(p for p in model_dir.rglob("*") if p.is_file())
    if not (model_dir / "model_sha256.txt").exists():
        raise ValueError("original trained checkpoint/hash missing; do not retrain silently")
    historical = hashlib.sha256()
    for p in files:
        if p.name not in {"model_sha256.txt", "train_meta.json"}:
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    historical.update(chunk)
    expected = (model_dir / "model_sha256.txt").read_text().strip()
    if historical.hexdigest() != expected:
        raise ValueError("trained checkpoint no longer matches its original training hash")
    return {
        "training_sha256": expected,
        "files": {str(p.relative_to(model_dir)): file_hash(p) for p in files},
        "model_revision": "local-finetuned-sha256:" + expected,
        "tokenizer_revision": "checkpoint-local files in files manifest",
        "upstream_hub_revision": "not recorded by original training pipeline",
    }


def run_identity(manifest: dict) -> str:
    """Stable identity binds outputs to input, model, config and implementation."""
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def code_identity(repo: Path) -> dict:
    """Capture commit plus tracked dirty state; prohibit uncommitted run code."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    paths = ["src/awake/eval", "projects/02-text-eraser/scripts", "projects/02-text-eraser/configs"]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=repo, text=True
    )
    if dirty.strip():
        raise ValueError("commit the repair code/config before generating scientific outputs")
    return {"git_commit": commit}


def validate_cache(table, manifest, name, sub):
    """Reject stale metadata, missing/duplicate rows and different word identities."""
    meta = table.schema.metadata or {}
    required = {
        b"contract": CONTRACT,
        b"target_scalar": TARGET_SCALAR,
        b"run_id": run_identity(manifest),
        b"explainer_name": name,
    }
    if any(meta.get(k, b"").decode() != v for k, v in required.items()):
        raise ValueError("stale or mismatched attribution cache; regenerate")
    df = table.to_pandas()
    expected_rows = 0
    for i, row in sub.reset_index(drop=True).iterrows():
        visible = VisibleWords.from_json(row["visible_json"])
        rows = df[df.example_id == i].sort_values("word_idx")
        expected_rows += len(visible.words)
        if (
            rows.word_idx.tolist() != list(range(len(visible.words)))
            or rows.word.tolist() != visible.words
            or rows.input_fingerprint.tolist() != [visible.fingerprint] * len(visible.words)
            or rows.predicted_class.tolist() != [int(row["predicted_class"])] * len(visible.words)
        ):
            raise ValueError("attribution word/input/target identity mismatch")
    if len(df) != expected_rows:
        raise ValueError("unexpected attribution rows")
    return df
