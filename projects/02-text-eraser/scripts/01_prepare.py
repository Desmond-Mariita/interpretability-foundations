"""Parse ERASER Movies into prepared parquet with frozen visible sequences."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from _paths import DATA_PATH, PREPARED, ensure_dirs, load_config

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
    """Tokenize once, truncate to max_len, capture offsets + word_ids."""
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
        return_tensors=None,
    )
    word_ids = enc.word_ids()
    return {
        "input_ids": enc["input_ids"],
        "offsets": enc["offset_mapping"],
        "word_ids": word_ids,
    }


def truncation_coverage(word_ids: list[int | None], gold_mask: list[int]) -> float:
    """Fraction of gold-rationale words that survive truncation."""
    gold_words = {i for i, g in enumerate(gold_mask) if g}
    if not gold_words:
        return 1.0
    visible_words = {w for w in word_ids if w is not None}
    return len(gold_words & visible_words) / len(gold_words)


def main() -> None:
    """Parse each split, freeze visible sequences, write prepared parquet."""
    from transformers import AutoTokenizer

    ensure_dirs()
    cfg_d = load_config("data")
    cfg_m = load_config("model")
    tok = AutoTokenizer.from_pretrained(cfg_m["model_name"])
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
                rec["truncation_coverage"] = truncation_coverage(vis["word_ids"], rec["gold_mask"])
                rows.append(rec)
        pd.DataFrame(rows).to_parquet(PREPARED / f"{split}.parquet")
        print(f"{split}: {len(rows)} examples")

    (PREPARED.parent / "prepare_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"stats: {stats}")


if __name__ == "__main__":
    main()
