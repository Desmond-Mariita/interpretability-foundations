# 04 -- Do caption-then-LLM explanations actually describe the image?

**Question.** When a caption-then-LLM pipeline answers a visual question, the LLM
explains a *caption*, not an *image*. How often does that diverge from what a direct
vision-language model says, and how much does each pipeline's answer depend on the image
vs. on its own explanation?

The image-dependence is measured with a **vision-ablation probe and a paired baseline**:
visual evidence is removed and each pipeline re-answers twice -- once with its prior
explanation and once without. The headline is `Delta = consistency(with-explanation) -
consistency(no-explanation)`. A positive Delta flags self-rationale recoverability (the
explanation, not the image, drives recovery); Delta near zero means language priors
dominate. See [REPORT.md](REPORT.md) and
[ADR 004](../../docs/decisions/004-vqa-pipelines-and-vision-ablation.md).

<!-- ![hero](assets/hero.png) -->

## Design

**Dataset.** A-OKVQA validation split (~1.1k items, labelled). Full split only for the
headline; no sub-sampling. Items where any human rationale leaks the gold choice text are
flagged; all metrics are reported on both filtered and unfiltered subsets.

**Three pipelines (zero-shot, deterministic):**

- **Pipeline A -- caption-then-LLM.** `Salesforce/blip2-opt-2.7b` captions the image;
  `Qwen/Qwen2.5-7B-Instruct` answers and explains from the caption.
- **Pipeline B -- direct VLM.** `Qwen/Qwen2.5-VL-3B-Instruct` answers and explains
  directly from the image.
- **Pipeline B7 -- size-matched arm (required for the headline).**
  `Qwen/Qwen2.5-VL-7B-Instruct` runs the same direct-VLM protocol as B. Required to bound
  the 7B-vs-3B parameter-count confound in the A-vs-B comparison.

**Probe.** Visual evidence removed; re-answer with and without prior explanation.
`Delta = consistency(with-expl) - consistency(no-expl)` per pipeline, with paired-bootstrap
95% CI. Inter-pipeline divergence reported with a correctness-conditioned 2x2 contingency
for all three pairs (A_vs_B, A_vs_B7, B_vs_B7).

**Pure metric core.** `src/awake/eval/vqa_consistency.py` -- all metric logic is pure
(no I/O, no models) and covered by unit tests.

## Reproduce

Set `P4_PROJECT_ROOT` to point at this project directory, then run the four numbered
scripts in order:

```bash
export P4_PROJECT_ROOT=$PWD/projects/04-vqa-aokvqa

# 1. Prepare data (decode images, build leakage flag, write parquet)
uv run python projects/04-vqa-aokvqa/scripts/00_data.py

# 2. Run pipelines A, B, B7 (one model resident at a time; expect 4-8 h on RTX 3090)
uv run python projects/04-vqa-aokvqa/scripts/10_run_pipelines.py

# 3. Run the two-arm ablation probe (with-expl and no-expl)
uv run python projects/04-vqa-aokvqa/scripts/20_probe.py

# 4. Compute metrics, CIs, and generate the hero figure
uv run python projects/04-vqa-aokvqa/scripts/30_eval.py
```

For the real run, install the `vqa` optional-dependency group first:

```bash
uv sync --extra vqa
```

## Results

See [REPORT.md](REPORT.md) for the full results table and interpretation. The hero figure
is at `assets/hero.png` (committed after the real run).

## Architecture decision record

[ADR 004](../../docs/decisions/004-vqa-pipelines-and-vision-ablation.md) documents five
design decisions: the paired-baseline Delta headline; the Pipeline A null-caption ablation;
the narrowed A-vs-B claim with the required B7 arm; the strict-then-text parsing and
None-as-wrong denominator policy; and determinism with logged model revisions.

## Limitations

See [REPORT.md section 7](REPORT.md) for the full list. Key constraints: zero-shot only;
A-vs-B confounded by model family and modality stack; probe is one family, not a battery;
multiple-choice format only.
