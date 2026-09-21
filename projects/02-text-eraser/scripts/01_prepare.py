"""Parse ERASER Movies into prepared parquet with frozen visible sequences."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from _paths import DATA_PATH, MODEL_DIR, PREPARED, ensure_dirs, load_config

from awake.eval.visible_words import canonical_visible

LABELS = {"NEG": 0, "POS": 1}


def is_comparison(example: dict, docid: str) -> bool:
    """True if any evidence span references a document other than ``docid``."""
    for group in example["evidences"]:
        for ev in group:
            if ev["docid"] != docid:
                return True
    return False


def build_record(example: dict, doc_text: str) -> dict:
    """Build a word-level record (text, label, words, gold_mask, n_words)."""
    words = doc_text.split()
    gold = np.zeros(len(words), dtype=int)
    for group in example["evidences"]:
        for ev in group:
            gold[ev["start_token"] : ev["end_token"]] = 1
    return {
        "annotation_id": example["annotation_id"],
        "label": LABELS[example["classification"]],
        "text": doc_text,
        "words": words,
        "gold_mask": gold.tolist(),
        "n_words": len(words),
    }


def freeze_visible(text: str, tokenizer, max_len: int) -> dict:
    """Freeze original token IDs and derive explicit complete visible-word spans."""
    visible = canonical_visible(text, tokenizer, max_len)
    return {"input_ids": visible.input_ids, "visible_json": visible.to_json()}


def main() -> None:
    """Parse each split, freeze visible sequences, write prepared parquet."""
    from transformers import AutoTokenizer

    ensure_dirs()
    cfg_d = load_config("data")
    # Reuse the exact saved tokenizer; never silently fetch a different revision.
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    stats = {"dropped_comparison": 0}

    for split in ("train", "val", "test"):
        rows = []
        with open(DATA_PATH / "movies" / f"{split}.jsonl") as f:
            for line in f:
                ex = json.loads(line)
                docid = ex["annotation_id"]
                if is_comparison(ex, docid):
                    stats["dropped_comparison"] += 1
                    continue
                doc_text = (DATA_PATH / "movies" / "docs" / docid).read_text()
                rec = build_record(ex, doc_text)
                vis = freeze_visible(rec["text"], tok, cfg_d["max_seq_len"])
                rec.update(vis)
                from awake.eval.visible_words import VisibleWords

                visible = VisibleWords.from_json(rec["visible_json"])
                clipped = visible.clip_gold(rec["gold_mask"])
                rec["gold_visible"] = clipped.tolist()
                rec["truncation_coverage"] = (
                    float(clipped.sum() / sum(rec["gold_mask"])) if sum(rec["gold_mask"]) else 1.0
                )
                rows.append(rec)
        pd.DataFrame(rows).to_parquet(PREPARED / f"{split}.parquet")
        print(f"{split}: {len(rows)} examples")

    (PREPARED.parent / "prepare_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"stats: {stats}")


if __name__ == "__main__":
    main()
