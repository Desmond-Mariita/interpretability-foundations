"""Prepare A-OKVQA validation: decode images, build leakage flag, write parquet.

Pure ``prepare_rows`` is smoke-tested; ``main`` (download + image decode) is slow.
"""

from __future__ import annotations

from awake.eval.vqa_consistency import rationale_leaks_answer

from _paths import IMAGES, PREPARED, ensure_dirs, load_config


def prepare_rows(raw: list[dict]) -> list[dict]:
    """Build prepared rows with a leakage flag and image_path (no I/O).

    Args:
        raw: Items with ``id, question, choices, correct_choice_idx, rationales``.

    Returns:
        Rows augmented with ``leakage_flag`` (any rationale leaks gold choice text)
        and ``image_path`` (``<IMAGES>/<id>.jpg``).
    """
    rows = []
    for r in raw:
        gold_text = r["choices"][int(r["correct_choice_idx"])]
        rows.append({
            "id": r["id"],
            "question": r["question"],
            "choices": list(r["choices"]),
            "correct_choice_idx": int(r["correct_choice_idx"]),
            "rationales": list(r.get("rationales", [])),
            "leakage_flag": rationale_leaks_answer(list(r.get("rationales", [])), gold_text),
            "image_path": str(IMAGES / f"{r['id']}.jpg"),
        })
    return rows


def main() -> None:  # pragma: no cover - slow path, exercised only in the real run
    """Download A-OKVQA, decode images idempotently, write prepared parquet."""
    import pandas as pd
    from datasets import load_dataset

    cfg = load_config("data")
    ensure_dirs(IMAGES, PREPARED)
    ds = load_dataset(cfg["dataset"], split=cfg["split"])
    if cfg.get("subset_n"):
        ds = ds.shuffle(seed=cfg["seed"]).select(range(cfg["subset_n"]))

    raw = []
    for ex in ds:
        ex_id = str(ex.get("question_id", ex.get("id")))
        img_path = IMAGES / f"{ex_id}.jpg"
        if not img_path.exists():  # idempotent: skip re-decode
            ex["image"].convert("RGB").save(img_path, format="JPEG")
        raw.append({
            "id": ex_id,
            "question": ex["question"],
            "choices": ex["choices"],
            "correct_choice_idx": ex["correct_choice_idx"],
            "rationales": ex.get("rationales", []),
        })

    rows = prepare_rows(raw)
    pd.DataFrame(rows).to_parquet(PREPARED / "val.parquet", index=False)
    print(f"prepared {len(rows)} items; leak={sum(r['leakage_flag'] for r in rows)}")


if __name__ == "__main__":  # pragma: no cover
    main()
