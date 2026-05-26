"""Vision-ablation probe: re-answer with the image/caption removed, two arms.

with_expl arm includes the model's own prior explanation; the no_expl baseline arm
is identical except it omits that explanation. Pipeline A ablates by replacing the
caption with the null-caption string; Pipeline B ablates by passing image_path=None
(the loader substitutes a black tile).
"""

from __future__ import annotations

from collections.abc import Callable

from _paths import GEN, PREPARED, ensure_dirs, load_config
from _prompts import format_choices, render

from awake.eval.vqa_consistency import extract_choice


def _expl_by_id(gen_rows: list[dict]) -> dict[str, str]:
    return {r["id"]: r.get("explanation", "") for r in gen_rows}


def _ablated_row(item: dict, raw: str) -> dict:
    idx, parsed_by = extract_choice(raw, item["choices"])
    return {"id": item["id"], "ablated_idx": idx, "parsed_by": parsed_by, "raw_output": raw}


def _prompt_for(item: dict, expl: str, prompts: dict, with_expl: bool, caption: str | None) -> str:
    key = "ablate_with_expl" if with_expl else "ablate_no_expl"
    kwargs = {"question": item["question"], "choices_block": format_choices(item["choices"])}
    if with_expl:
        kwargs["explanation"] = expl
    if caption is not None:
        kwargs["caption"] = caption
    return render(prompts[key], **kwargs)


def ablate_pipeline_b(items: list[dict], gen_rows: list[dict],
                      vlm_generate: Callable[[str, str | None], str], prompts: dict,
                      with_expl: bool) -> list[dict]:
    """Re-answer Pipeline B with a black tile (image_path=None)."""
    expl = _expl_by_id(gen_rows)
    rows = []
    for item in items:
        prompt = _prompt_for(item, expl.get(item["id"], ""), prompts, with_expl, caption=None)
        rows.append(_ablated_row(item, vlm_generate(prompt, None)))
    return rows


def ablate_pipeline_a(items: list[dict], gen_rows: list[dict],
                      llm_generate: Callable[[str], str], prompts: dict, with_expl: bool,
                      null_caption: str) -> list[dict]:
    """Re-answer Pipeline A with the caption replaced by the null-caption string."""
    expl = _expl_by_id(gen_rows)
    rows = []
    for item in items:
        prompt = _prompt_for(item, expl.get(item["id"], ""), prompts, with_expl,
                             caption=null_caption)
        rows.append(_ablated_row(item, llm_generate(prompt)))
    return rows


def main() -> None:  # pragma: no cover - slow path
    """Run both ablation arms for A/B/B7 and write *_ablated_{expl,noexpl}.parquet."""
    import gc

    import pandas as pd
    import torch
    from _models import load_qwen_lm, load_qwen_vl

    cfg = load_config("pipelines")
    gen_cfg, prompts = cfg["generation"], cfg["prompts"]["main"]
    null_caption = cfg["null_caption"]
    ensure_dirs(GEN)
    items = pd.read_parquet(PREPARED / "val.parquet").to_dict("records")

    def _free():
        gc.collect()
        torch.cuda.empty_cache()

    # Pipeline A ablation (LLM only; no image needed)
    gen_a = pd.read_parquet(GEN / "A.parquet").to_dict("records")
    llm = load_qwen_lm(cfg["models"]["qwen_lm"], gen_cfg)
    for with_expl, tag in ((True, "expl"), (False, "noexpl")):
        rows = ablate_pipeline_a(items, gen_a, llm, prompts, with_expl, null_caption)
        pd.DataFrame(rows).to_parquet(GEN / f"A_ablated_{tag}.parquet", index=False)
    del llm
    _free()

    for key, mid in (("B", cfg["models"]["qwen_vl_3b"]), ("B7", cfg["models"]["qwen_vl_7b"])):
        gen = pd.read_parquet(GEN / f"{key}.parquet").to_dict("records")
        vlm = load_qwen_vl(mid, gen_cfg)
        for with_expl, tag in ((True, "expl"), (False, "noexpl")):
            rows = ablate_pipeline_b(items, gen, vlm, prompts, with_expl)
            pd.DataFrame(rows).to_parquet(GEN / f"{key}_ablated_{tag}.parquet", index=False)
        del vlm
        _free()
    print("wrote ablation arms")


if __name__ == "__main__":  # pragma: no cover
    main()
