# Project 4 — `04-vqa-aokvqa`: do caption-then-LLM explanations actually describe the image?

**Status:** revised after four-way review (round 1) · 2026-05-26 · author Desmond Mariita · v0.x

Part of `interpretability-foundations`, built per `docs/PLAYBOOK.md`. **No HF Space.**
Hardware: single RTX 3090 (24 GB). §12 records review traceability.

## 1. Question and deliverable

A **caption-then-LLM** VQA pipeline explains a *caption*, not an *image*. Two questions:
(a) how often does its answer **diverge** from a **direct vision-language model**, and (b) how
much does each pipeline's answer actually **depend on the image** vs. on its own explanation?

The image-dependence is measured by a **vision-ablation probe with a paired baseline**: for
each pipeline we re-answer with the visual evidence removed, **once given the model's own prior
explanation and once not**, and report the **consistency gain from the explanation**
`Δ = consistency(with-explanation) − consistency(no-explanation)`. A large `Δ` means the
explanation (not the image) is what lets the model reproduce its answer — the faithfulness red
flag. The raw with-explanation rate is reported too, **honestly labelled "self-rationale
recoverability"** (it conflates image-independence with explanation-copying on its own).

Deliverables: `metrics.json` (per pipeline: A-OKVQA accuracy, parse_rate; the two ablation
consistency rates + their `Δ`; inter-pipeline divergence **conditioned on correctness** + a
2×2 contingency; on **filtered and unfiltered** subsets; cross-pipeline **paired-bootstrap**
deltas), a multi-panel hero figure (consistency `Δ` + accuracy + parse_rate), a notebook
(committed with outputs), `REPORT.md`. No Space, no deploy.

## 2. Data — A-OKVQA (open; code-only)

A-OKVQA (Schwenk et al. 2022), multiple-choice VQA over COCO images. Source HF
`HuggingFaceM4/A-OKVQA` (bundles the image as a PIL object per item; fall back to
`allenai/aokvqa` + local COCO). **Headline split = `validation`** (labelled, ~1.1k) — the
**full** split is the default run (no sub-sampling for the headline; a seeded subset is used
**only** as a compute-fallback/pilot, never silently, and never pool `train+val` for the
headline — train+val is an explicitly-labelled exploratory option only). Fields:
`question, choices` (4), `correct_choice_idx, rationales` (3), `image`.

`scripts/00_data.py`:
- **Decode → save → record path:** HF image columns are PIL objects, not paths; write each to
  `outputs/images/<id>.jpg` and store `image_path` (images never committed).
- **Leakage sensitivity split:** flag items where any human rationale contains the **gold
  choice text** (normalised; **match the choice *text*, never the bare letter** — letters
  A–D false-positive). Report all metrics on **filtered (no-leak) and unfiltered** subsets.
- Write `outputs/prepared/val.parquet` (`id, question, choices, correct_choice_idx,
  rationales, leakage_flag, image_path`). Seed + exact sampled IDs (if a fallback subset is
  used) recorded in `configs/data.yaml` and the parquet metadata.

## 3. Pipelines (zero-shot; deterministic; one model resident at a time)

- **Pipeline A — caption-then-LLM.** `Salesforce/blip2-opt-2.7b` (fp16) captions the image →
  `Qwen/Qwen2.5-7B-Instruct` (fp16) gets `(question, caption, choices)` → answer + one-sentence
  explanation. Image never shown to the LLM.
- **Pipeline B — direct VLM.** `Qwen/Qwen2.5-VL-3B-Instruct` (fp16, cached) gets
  `(question, image, choices)` → answer + explanation.
- **Size-matched sensitivity arm (cached):** also run Pipeline B with
  `Qwen/Qwen2.5-VL-7B-Instruct` to bound the 7B-LLM-vs-3B-VLM capacity confound; report as a
  secondary arm. The **headline A-vs-B claim is explicitly narrowed** to "this BLIP-2→Qwen-7B
  pipeline vs this Qwen-VL pipeline" — not a general which-is-better claim (§9).
- **Determinism + memory:** `do_sample=False`, fixed `max_new_tokens`, `torch_dtype=fp16`,
  `device_map="cuda:0"`; log each model's HF revision into `metrics.json`. Orchestrate
  strictly load → generate-all → `del model; torch.cuda.empty_cache()` so only one large
  model is resident (BLIP-2 ~6 GB, Qwen-7B ~15 GB, Qwen-VL-3B ~7 GB, VL-7B ~16 GB — all fit
  one-at-a-time on 24 GB).
- **Strict answer parsing:** prompt forces a first line `Answer: <A-D>`; parse only that
  (fallback: exact choice-text match; else `None` = unparseable). One main prompt +
  **one alternate-wording sensitivity** prompt for the probe (§4), pre-registered in
  `configs/pipelines.yaml`.

`scripts/10_run_pipelines.py` → `outputs/gen/{A,B,B7}.parquet`
(`id, caption?, answer_idx, explanation, raw_output`).

## 4. Vision-ablation probe (two arms per pipeline)

Visual evidence is removed at each pipeline's entry, and we re-answer **twice**:
- **with-explanation arm:** prompt includes the model's own prior explanation.
- **no-explanation arm (baseline):** prompt has only `(question, choices)` (+ ablated visual).

Visual ablation: **Pipeline B** → replace the image with a black tile (neutral prompt, so the
VLM perceives "no content", not a leading question). **Pipeline A** → replace the caption with
the fixed string `"(no visual information available)"` (the clean caption-pathway ablation);
BLIP-2-on-black-tile is kept only as a **secondary sensitivity**, not the primary.

Per pipeline: `consistency = P(ablated answer == original answer)` for each arm, and the
headline **`Δ = consistency_with_expl − consistency_no_expl`**. `scripts/20_probe.py` →
`outputs/gen/{A,B,B7}_ablated_{expl,noexpl}.parquet`. Limitation (REPORT): even with the
baseline, this is one probe family, not a full faithfulness battery.

## 5. Shared core — `src/awake/eval/vqa_consistency.py` (pure, unit-tested)

```python
def normalize_text(s: str) -> str: ...                       # lower, strip, collapse punct/space
def extract_choice(model_output: str, choices: list[str]) -> int | None:
    """Strict: first-line 'Answer: <A-D>'; else exact normalized choice-text match; else None."""
def explanation_leaks_answer(explanation: str, chosen_choice_text: str) -> bool: ...
def rationale_leaks_answer(rationales: list[str], gold_choice_text: str) -> bool: ...
def parse_rate(parsed: list[int | None]) -> float: ...
def accuracy(pred: list[int | None], gold: list[int]) -> float:
    """Primary policy: None counts as wrong (denominator = all items)."""
def consistency_rate(original: list[int | None], ablated: list[int | None]) -> float:
    """Primary policy: a pair is consistent iff both parse AND are equal; unparseable on
    either side counts as INconsistent (denominator = all items). Paired-parsed-only is a
    documented sensitivity, computed by a separate flag."""
def pipeline_divergence(a: list[int | None], b: list[int | None], gold: list[int]) -> dict:
    """Overall divergence + the 2x2 contingency conditioned on correctness
    (A right/wrong x B right/wrong, agree/disagree)."""
```
All pure (parsed ints / strings; no models, no I/O). Reuse `awake.eval.bootstrap` for CIs and
**paired** deltas. Add to `src/awake/eval/__init__.py`.

## 6. Eval + figures

`scripts/30_eval.py` — for **{unfiltered, filtered}** subsets and **{A, B, B7}** pipelines:
accuracy, parse_rate, the two ablation consistencies + `Δ`, divergence + the correctness
contingency. **Bootstrap 95% CIs** on each; **paired-bootstrap** CIs on the cross-pipeline and
with−no-expl deltas (CIs reported regardless of zero; no post-hoc significance). `metrics.json`:
`{"split":"validation","n":..,"n_filtered":..,"model_revisions":{...},
"subsets":{"unfiltered":{"pipelines":{"A":{accuracy,parse_rate,consistency:{with_expl,no_expl,delta}},"B":{...},"B7":{...}},
"divergence":{overall, contingency:{...}}},"filtered":{...}}, "prompt_variant":"main"}`.
Hero figure: a multi-panel — (i) consistency `Δ` per pipeline (with CIs), (ii) accuracy,
(iii) parse_rate — so a stably-wrong/image-ignoring pipeline can't look good on `Δ` alone.

## 7. Tests

- **unit** (`vqa_consistency.py`, 90% floor): `extract_choice` (first-line letter, choice-text
  fallback, malformed→None); `rationale_leaks_answer`/`explanation_leaks_answer` (leak/no-leak,
  letter-not-matched, normalisation); `consistency_rate` denominator policy incl. `None` on
  either side = inconsistent; `parse_rate`; `accuracy` None=wrong; `pipeline_divergence`
  contingency on a hand-built case.
- **smoke** (CPU, no model downloads, no dataset): a **stub generator** (canned
  answers/explanations) drives `10_run_pipelines`/`20_probe` logic + `30_eval` end-to-end on
  tiny synthetic items. Heavy model imports are **lazy (inside functions)** so importing the
  scripts triggers no download.
- **slow** (excluded from CI): real BLIP-2 / Qwen-2.5-7B / Qwen-2.5-VL-{3B,7B} generation.

## 8. Deps + governance

Add `qwen-vl-utils>=0.0.10`, `accelerate>=0.30` (pillow already present; transformers/torch/
datasets present — transformers 5.9 has Qwen2.5-VL). ADR
`004-vqa-pipelines-and-vision-ablation.md`: the paired-baseline probe (`Δ`) + why raw
consistency is mislabelled without it; Pipeline-A null-caption ablation; the narrowed A-vs-B
claim + size/family/caption confounds; strict parsing + denominator policy; determinism + model
revisions. CHANGELOG `[Unreleased]` P4 entry; repo stays v0.x.

## 9. Limitations

Zero-shot (no fine-tuning); **A-vs-B is confounded** by parameter count (7B+captioner vs 3B/7B
VLM), model family, and modality stack — the headline claim is scoped to these specific
instantiations, with the VL-7B arm bounding the size effect; **BLIP-2 caption quality** is a
confound for Pipeline A's accuracy/divergence (scope claims to the BLIP-2 instantiation); the
probe is one family (with baseline) not a battery; multiple-choice only (direct-answer A-OKVQA
not scored); answer parsing is strict-then-heuristic with unparseables reported via parse_rate;
prompt sensitivity bounded by one alternate-wording arm, not exhaustive.

## 10. Real-run scope (PLAYBOOK §1)

A-OKVQA val + bundled images = modest download; Qwen-VL-3B/7B cached; BLIP-2 + Qwen-2.5-7B
(~30 GB) download on first run. Cost = generation over ~1.1k items × pipelines × (answer +
explanation + 2 ablation arms) with 7B-class models, one resident at a time. Build to
green-CI + ready, then run the **full validation** headline if it completes in reasonable
wall-clock; otherwise fall back to a **seeded, clearly-labelled-N subset** for the headline and
note `--full` to reproduce. The VL-7B arm and BLIP-2-on-black secondary sensitivity run only if
time permits (clearly marked optional). No fabricated numbers; log model revisions.

## 11. Repository layout (new)

```
src/awake/eval/vqa_consistency.py            # NEW pure core (+ __init__ export)
tests/test_eval_vqa_consistency.py
projects/04-vqa-aokvqa/
  configs/{data,pipelines}.yaml              # prompts, seed, max_new_tokens, model ids+revisions
  scripts/{00_data,10_run_pipelines,20_probe,30_eval}.py  _paths.py _models.py _stub.py
  tests/{conftest.py, test_data_smoke.py, test_pipeline_smoke.py}
  notebooks/01-vqa-consistency.py            # committed WITH outputs (no raw dataset dumps)
  REPORT.md  (README.md exists — updated)    (assets/, outputs/ gitignored)
docs/decisions/004-vqa-pipelines-and-vision-ablation.md
```

## 12. Review traceability (round 1 → folded in)

- No-explanation baseline arm + headline `Δ`; relabel raw rate as self-rationale recoverability
  ← all 4 (BLOCKER).
- Pipeline-A null-caption ablation (black-tile = secondary) ← methodology + Gemini + Codex.
- Denominator policy + parse_rate + strict `Answer:` parsing ← engineering + Codex + Gemini.
- Narrow A-vs-B claim + size/family/caption confounds + cached VL-7B size-matched arm ← all.
- Divergence conditioned on correctness + 2x2 contingency ← methodology + Codex.
- Paired-bootstrap cross-pipeline deltas ← methodology + Codex.
- Full validation headline; no silent train+val pooling ← all.
- Leakage filter matches choice **text** (not letter), kept as sensitivity split + model-output
  leakage flag ← engineering + Codex.
- HF image decode→save→path; lazy heavy imports; `30_eval` renumber; seed in config + logged
  IDs; determinism (`do_sample=False`, revisions); `torch_dtype=fp16`/`device_map`;
  `qwen-vl-utils>=0.0.10` ← engineering + Codex.
- Multi-panel hero (Δ + accuracy + parse_rate) ← Codex + Gemini + methodology.
