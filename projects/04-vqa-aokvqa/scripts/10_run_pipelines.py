"""Run both VQA pipelines and write per-item answer + explanation parquet.

The ``run_pipeline_*`` functions take injectable generate callables (real models in
``main``; stubs in smoke tests). The first line of output is treated as the answer
line and any following lines as the explanation.
"""

from __future__ import annotations

from collections.abc import Callable

from _paths import GEN, PREPARED, ensure_dirs, load_config
from _prompts import format_choices, render

from awake.eval.vqa_consistency import explanation_leaks_answer, extract_choice


def _split_answer_explanation(raw: str) -> tuple[str, str]:
    """Return (answer_line, explanation) from a raw generation."""
    lines = raw.splitlines()
    answer_line = lines[0] if lines else ""
    explanation = "\n".join(lines[1:]).strip()
    return answer_line, explanation


def _row(item: dict, raw: str, caption: str | None = None) -> dict:
    idx, parsed_by = extract_choice(raw, item["choices"])
    _, explanation = _split_answer_explanation(raw)
    chosen_text = item["choices"][idx] if idx is not None else ""
    row = {
        "id": item["id"],
        "answer_idx": idx,
        "explanation": explanation,
        "raw_output": raw,
        "parsed_by": parsed_by,
        "expl_leaks": explanation_leaks_answer(explanation, chosen_text),
    }
    if caption is not None:
        row["caption"] = caption
    return row


def run_pipeline_a(items: list[dict], caption_fn: Callable[[str], str],
                   llm_generate: Callable[[str], str], prompts: dict) -> list[dict]:
    """Caption-then-LLM: caption each image, then answer from (question, caption, choices)."""
    rows = []
    for item in items:
        caption = caption_fn(item.get("image_path", ""))
        prompt = render(prompts["answer_with_caption"], question=item["question"],
                        caption=caption, choices_block=format_choices(item["choices"]))
        rows.append(_row(item, llm_generate(prompt), caption=caption))
    return rows


def run_pipeline_b(items: list[dict], vlm_generate: Callable[[str, str | None], str],
                   prompts: dict) -> list[dict]:
    """Direct VLM: answer from (question, image, choices)."""
    rows = []
    for item in items:
        prompt = render(prompts["answer"], question=item["question"],
                        choices_block=format_choices(item["choices"]))
        rows.append(_row(item, vlm_generate(prompt, item.get("image_path"))))
    return rows


def main() -> None:  # pragma: no cover - slow path
    """Load each model in turn and write outputs/gen/{A,B,B7}.parquet."""
    import gc

    import pandas as pd
    import torch
    from _models import (
        load_blip2_captioner,
        load_qwen_lm,
        load_qwen_vl,
        model_revisions,
    )

    cfg = load_config("pipelines")
    gen_cfg, prompts = cfg["generation"], cfg["prompts"]["main"]
    ensure_dirs(GEN)
    items = pd.read_parquet(PREPARED / "val.parquet").to_dict("records")

    def _free():
        gc.collect()
        torch.cuda.empty_cache()

    # Pipeline A: BLIP-2 caption -> Qwen LM
    caption_fn = load_blip2_captioner(cfg["models"]["blip2"], gen_cfg)
    captions = {it["id"]: caption_fn(it["image_path"]) for it in items}
    del caption_fn
    _free()
    llm = load_qwen_lm(cfg["models"]["qwen_lm"], gen_cfg)
    rows_a = []  # captions already computed above; answer with cached captions
    for it in items:
        prompt = render(prompts["answer_with_caption"], question=it["question"],
                        caption=captions[it["id"]], choices_block=format_choices(it["choices"]))
        rows_a.append(_row(it, llm(prompt), caption=captions[it["id"]]))
    pd.DataFrame(rows_a).to_parquet(GEN / "A.parquet", index=False)
    del llm
    _free()

    # Pipeline B (3B) and B7 (7B)
    for key, mid in (("B", cfg["models"]["qwen_vl_3b"]), ("B7", cfg["models"]["qwen_vl_7b"])):
        vlm = load_qwen_vl(mid, gen_cfg)
        rows = run_pipeline_b(items, vlm_generate=vlm, prompts=prompts)
        pd.DataFrame(rows).to_parquet(GEN / f"{key}.parquet", index=False)
        del vlm
        _free()

    revs = model_revisions(cfg["models"])
    (GEN / "model_revisions.json").write_text(__import__("json").dumps(revs, indent=2))
    print("wrote A/B/B7 parquet")


if __name__ == "__main__":  # pragma: no cover
    main()
