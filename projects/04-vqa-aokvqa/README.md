# 04 — Do caption-then-LLM explanations actually describe the image?

**Question.** When a caption-then-LLM pipeline answers a visual question, the LLM's explanation describes a caption — not an image. How often does that diverge from what a direct vision-language model says, and how often can each pipeline reproduce its own answer when the image is removed?

**Answer.** _Coming soon — this project ships in week 5._

**Why it matters.** Caption-then-LLM was a common pre-2024 VQA recipe and is still common in production cost-tier pipelines; understanding where its explanations come unstuck is useful.

<!-- ![hero](assets/hero.png) -->

## Method

A-OKVQA (~25k knowledge-required visual questions). **Filtered to remove the ~15% of items where the human rationale contains the answer string verbatim** (Schwenk et al. 2022) — this would otherwise inflate the consistency probe. Results reported on both filtered and unfiltered subsets for transparency.

- **Pipeline A.** BLIP-2 captions the image; Qwen-2.5-7B-Instruct answers and explains from the caption.
- **Pipeline B.** Qwen-2.5-VL-3B-Instruct answers and explains directly from the image.

A **vision-ablation consistency** probe asks: does the model produce the same answer when the image is replaced by a black tile, given the question and its own previously-generated explanation as context? Measured on a fixed 1k random-stratified subset; a `--full` flag runs the entire set.

The probe is one operation, not a battery. It cannot distinguish a model that ignores the image from one that uses the image but produces image-independent rationales — that limitation is explicit in [`REPORT.md`](REPORT.md).

## Reproduce

```
just setup
just data
just train    # zero-shot pipelines; no fine-tuning
just eval
```

## Limitations

- No fine-tuning of either pipeline.
- No multi-metric faithfulness battery — one probe, with its limits stated.
- No general claim about which pipeline is better; this is a comparison on A-OKVQA only.
