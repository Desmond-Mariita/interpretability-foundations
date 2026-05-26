# Project 4 — `04-vqa-aokvqa`: do caption-then-LLM explanations describe the image?

**Status:** draft for four-way review · 2026-05-26 · author Desmond Mariita · repo stays v0.x

Part of `interpretability-foundations` (P1 tabular, P2 text-faithfulness, P3 multimodal
shipped). Built per `docs/PLAYBOOK.md` (autonomous defaults applied). **No HF Space** for
this project. Hardware: single RTX 3090 (24 GB).

## 1. Question and deliverable

When a **caption-then-LLM** pipeline answers a visual question, its explanation describes a
*caption*, not the *image*. How often does its answer diverge from a **direct vision-language
model**, and — via a **vision-ablation consistency probe** — how often does each pipeline
reproduce its own answer when the image is removed (given the question + its own prior
explanation)? Deliverables: `metrics.json` (per-pipeline A-OKVQA accuracy; inter-pipeline
answer divergence; per-pipeline vision-ablation consistency rate; on filtered **and**
unfiltered subsets, with bootstrap CIs), a hero figure, a notebook (committed with outputs),
`REPORT.md`. No Space, no deploy.

## 2. Data — A-OKVQA (open; code-only)

A-OKVQA (Schwenk et al. 2022), ~25k knowledge-required multiple-choice VQA over COCO images.
Source: HuggingFace `HuggingFaceM4/A-OKVQA` (bundles the COCO image per item as PIL, avoiding
a separate ~19 GB COCO download); fall back to `allenai/aokvqa` + local COCO if that dataset
is unavailable. Per-item fields used: `question`, `choices` (4), `correct_choice_idx`,
`rationales` (3 human rationales), `image`. **Eval on the `validation` split** (labelled,
~1.1k; `test` has no public answers). `train` is not used (zero-shot, no fine-tuning).

- `scripts/00_data.py` — load `validation`; build the **rationale-leakage filter**: drop the
  ~15% of items where any rationale contains the gold answer string verbatim (case/space-
  normalised) — these would inflate the consistency probe (Schwenk et al.). Keep a
  `leakage_flag` per item so results can be reported on **filtered and unfiltered** subsets.
  Build a **fixed, seeded 1k random subset** (or the whole split if smaller) for the probe;
  a `--full` flag uses the entire split. Write `outputs/prepared/val.parquet`
  (`id, question, choices, correct_choice_idx, rationales, leakage_flag, image_path`),
  caching images under `outputs/images/` (never committed).

## 3. Pipelines (zero-shot, one model resident at a time to fit 24 GB)

- **Pipeline A — caption-then-LLM.** `Salesforce/blip2-opt-2.7b` captions the image →
  `Qwen/Qwen2.5-7B-Instruct` receives `(question, the caption, the 4 choices)` and returns a
  chosen-letter **answer** + a one-sentence **explanation**. The image is never shown to the LLM.
- **Pipeline B — direct VLM.** `Qwen/Qwen2.5-VL-3B-Instruct` (already cached) receives
  `(question, image, choices)` and returns **answer** + **explanation** directly.
- Multiple-choice: prompt for the letter A–D; parse robustly (letter, or choice-text match).
- `scripts/10_run_pipelines.py` — generate `(caption?, answer, explanation)` for both
  pipelines over the prepared set; cache to `outputs/gen/{A,B}.parquet`. Orchestrate
  load→generate→free per model so only one large model is resident at once.

## 4. Vision-ablation consistency probe

For each pipeline, re-ask with the **image replaced by a black tile**, giving the model the
question + **its own previously generated explanation** as context, and check whether the
answer is unchanged:
- Pipeline B: feed the black image directly.
- Pipeline A: BLIP-2 captions the black tile (→ an uninformative caption); Qwen re-answers
  from `(question, blank caption, choices, prior explanation)`.
Consistency rate = fraction of items where `ablated_answer == original_answer`. A **high**
rate means the answer didn't depend on the image (the explanation alone carried it) — the
signal of interest. `scripts/20_probe.py` over the fixed subset (`--full` for all);
caches `outputs/gen/{A,B}_ablated.parquet`.
**Stated limitation (REPORT):** one probe, not a battery — it cannot separate "ignores the
image" from "uses the image but produces image-independent rationales."

## 5. Shared core — `src/awake/eval/vqa_consistency.py` (pure, unit-tested)

```python
def normalize_answer(text: str) -> str: ...                      # lowercase, strip, punct
def extract_choice(model_output: str, choices: list[str]) -> int | None:
    """Parse the chosen choice index from a model's free text (letter A-D or choice match)."""
def rationale_leaks_answer(rationales: list[str], answer: str) -> bool:
    """True if any rationale contains the (normalised) answer substring."""
def consistency_rate(original: list[int|None], ablated: list[int|None]) -> float:
    """Fraction of items whose ablated answer equals the original (both parsed)."""
def pipeline_divergence(answers_a: list[int|None], answers_b: list[int|None]) -> float:
    """Fraction of items where the two pipelines' answers differ (both parsed)."""
def accuracy(pred: list[int|None], gold: list[int]) -> float: ...
```
All pure (operate on parsed ints / strings, no models), injected nothing, fully unit-tested.
Reuse `awake.eval.bootstrap` for CIs. Add to `src/awake/eval/__init__.py`.

## 6. Eval + figures

`scripts/11_eval.py` — over filtered + unfiltered subsets: per-pipeline accuracy, inter-
pipeline divergence, per-pipeline vision-ablation consistency rate, each with bootstrap 95%
CIs (`awake.eval.bootstrap`). `metrics.json` schema:
`{"split":"validation","n":..,"n_filtered":..,"subsets":{"unfiltered":{...},"filtered":{...}}}`
with each subset holding `{"accuracy":{A,B}, "divergence", "consistency":{A,B}}` (means+CIs).
Hero figure: per-pipeline vision-ablation **consistency rate** (A vs B), filtered vs
unfiltered, with CIs — the headline visual.

## 7. Tests

- **unit** (`src/awake/eval/vqa_consistency.py`, carry the 90% floor): `extract_choice` (letter,
  choice-text, unparseable→None), `rationale_leaks_answer` (leak/no-leak/normalisation),
  `consistency_rate`/`pipeline_divergence`/`accuracy` on hand cases incl. `None` handling.
- **smoke** (CPU, no model downloads): a **stub generator** returning canned
  answers/explanations; the pipeline-driver + probe logic end-to-end on the stub →
  metrics over tiny synthetic items. Never loads BLIP-2/Qwen or the dataset.
- **slow** (excluded from CI): real BLIP-2 / Qwen-2.5-7B / Qwen-2.5-VL-3B generation on GPU.

## 8. Deps + governance

Add `accelerate`, `qwen-vl-utils`, `pillow` (transformers/torch/datasets already present;
Qwen2.5-VL needs `qwen-vl-utils` + transformers≥4.49 — the installed 5.9 has it). BLIP-2 +
Qwen via transformers (`Blip2ForConditionalGeneration`, `AutoModelForCausalLM`,
`Qwen2_5_VLForConditionalGeneration`). ADR `004-vqa-pipelines-and-vision-ablation.md`: the
two-pipeline choice, the rationale-leakage filter + report-both-subsets, the single-probe
scope + its stated limitation, and the model substitutions if any are forced by the env.
CHANGELOG `[Unreleased]` P4 entry; repo stays v0.x.

## 9. Limitations

No fine-tuning (zero-shot pipelines); a single consistency probe (not a faithfulness battery)
that cannot distinguish image-ignoring from image-independent-rationale; A-OKVQA-only, no
general which-pipeline-is-better claim; multiple-choice answering only (direct-answer A-OKVQA
not scored); free-text answer parsing is heuristic (unparseable answers counted as a miss and
reported).

## 10. Real-run scope (per PLAYBOOK §1)

A-OKVQA validation + bundled images is a modest download; Qwen-2.5-VL-3B is cached; BLIP-2 +
Qwen-2.5-7B (~30 GB) download on first run. 7B-class generation over the subset is the cost.
Build to green-CI + ready, then run on a **right-sized** real subset (the configurable probe
subset; reduce N if full-1k generation is time-prohibitive on the 3090) to produce real,
clearly-labelled-N numbers; document the exact N + `--full` reproduce step. One model resident
at a time. No fabricated numbers.

## 11. Repository layout (new)

```
src/awake/eval/vqa_consistency.py            # NEW pure core (+ __init__ export)
tests/test_eval_vqa_consistency.py
projects/04-vqa-aokvqa/
  configs/{data,pipelines}.yaml
  scripts/{00_data,10_run_pipelines,20_probe,11_eval}.py  _paths.py _models.py _stub.py
  tests/{conftest.py, test_data_smoke.py, test_pipeline_smoke.py}
  notebooks/01-vqa-consistency.py            # committed WITH outputs
  REPORT.md  (README.md exists — updated)  (assets/, outputs/ gitignored)
docs/decisions/004-vqa-pipelines-and-vision-ablation.md
```
