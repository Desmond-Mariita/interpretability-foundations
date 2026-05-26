# Project 4 — `04-vqa-aokvqa`: do caption-then-LLM explanations actually describe the image?

**Status:** v2.1 — round-2 four-way review passed (no third round; precision patch applied) ·
2026-05-26 · author Desmond Mariita · repo v0.x

Part of `interpretability-foundations`, built per `docs/PLAYBOOK.md`. **No HF Space.**
Hardware: single RTX 3090 (24 GB). §12 records review traceability.

## 1. Question and deliverable

A **caption-then-LLM** VQA pipeline explains a *caption*, not an *image*. Two questions:
(a) how often does its answer **diverge** from a **direct vision-language model**, and (b) how
much does each pipeline's answer actually **depend on the image** vs. on its own explanation?

The image-dependence is measured by a **vision-ablation probe with a paired baseline**: for
each pipeline we re-answer with the visual evidence removed, **once given the model's own prior
explanation and once not**, and report the **consistency gain from the explanation**
`Δ = consistency(with-explanation) − consistency(no-explanation)`. A large positive `Δ` means the
explanation (not the image) is what lets the model reproduce its answer — the faithfulness red
flag. `Δ ≈ 0` means the explanation adds nothing the question alone didn't already supply
(answers are driven by language priors / the question, not recovered from the rationale).
`Δ < 0` (explanation *hurts* recovery) is treated as noise around zero, reported with its CI,
not over-interpreted. The raw with-explanation rate is reported too, **honestly labelled
"self-rationale recoverability"** (it conflates image-independence with explanation-copying on
its own).

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
- **Decode → save → record path (idempotent):** HF image columns are PIL objects, not paths;
  write each to `outputs/images/<id>.jpg` and store `image_path` (images never committed).
  Skip re-decoding when the jpg already exists so reruns are cheap.
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
- **Size-matched arm B7 (REQUIRED, cached):** also run Pipeline B with
  `Qwen/Qwen2.5-VL-7B-Instruct` to bound the 7B-LLM-vs-3B-VLM capacity confound. This is
  **required for the headline** (it is what makes the A-vs-B comparison defensible), not
  optional. The A-vs-B claim is still scoped to "this BLIP-2→Qwen-7B pipeline vs this Qwen-VL
  pipeline" (§9). (~4–8 h wall-clock on the 3090; if it genuinely cannot finish, the
  size-confound is reported as **unmitigated** in §9, not silently dropped.)
- **Determinism + memory:** `do_sample=False`, `max_new_tokens=256` (enough for
  `Answer: <X>` + a one-sentence explanation; too-small truncates the answer line and inflates
  parse failures), `torch_dtype=fp16`, `device_map="cuda:0"`; log each model's HF revision into
  `metrics.json`. Orchestrate strictly load → generate-all → `del model;
  torch.cuda.empty_cache()` so only one large model is resident (BLIP-2 ~6 GB, Qwen-7B ~15 GB,
  Qwen-VL-3B ~7 GB, VL-7B ~16 GB — all fit one-at-a-time on 24 GB).
- **Answer parsing ("strict-then-text"):** the prompt forces a first line `Answer: <A-D>`;
  the **primary** rule parses only that letter. Fallback (counted but distinguished in
  `parse_rate`): the normalised text of **exactly one** choice appears in the output (unique
  substring after `normalize_text`); ambiguous/none → `None` (unparseable). Two prompts in
  `configs/pipelines.yaml`: the **main** prompt and **one alternate-wording** prompt (varies
  the instruction phrasing/format only) run as a separate sensitivity table.

`scripts/10_run_pipelines.py` → `outputs/gen/{A,B,B7}.parquet`
(`id, caption?, answer_idx, explanation, raw_output, parsed_by` [`strict`|`text`|`none`],
`expl_leaks` [bool, from `explanation_leaks_answer(explanation, choices[answer_idx])`]).

## 4. Vision-ablation probe (two arms per pipeline)

Visual evidence is removed at each pipeline's entry, and we re-answer **twice**. Both arms use
the **identical instruction format** (same "give `Answer: <A-D>` then a one-sentence
explanation" wording, same null-caption string for Pipeline A) — they differ in **exactly one
thing**: whether the model's own prior explanation is included.
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
def extract_choice(model_output: str, choices: list[str]) -> tuple[int | None, str]:
    """strict-then-text. Returns (idx_0based|None, parsed_by) where parsed_by in
    {'strict','text','none'}. strict = first-line 'Answer: <A-D>'. text fallback = exactly ONE
    choice's normalized text is a substring of normalize_text(output) (ambiguous/zero → None)."""
def explanation_leaks_answer(explanation: str, chosen_choice_text: str) -> bool: ...
def rationale_leaks_answer(rationales: list[str], gold_choice_text: str) -> bool: ...
def parse_rate(parsed: list[int | None]) -> float: ...        # share that parsed (idx is not None)
def accuracy(pred: list[int | None], gold: list[int]) -> float:
    """Primary policy: None counts as wrong (denominator = all items)."""
def consistency_rate(original: list[int | None], ablated: list[int | None],
                     paired_only: bool = False) -> float:
    """Primary (paired_only=False): consistent iff both parse AND equal; unparseable on either
    side = INconsistent; denominator = all items. paired_only=True: drop pairs where either
    side is None, denominator = pairs where both parsed (documented sensitivity)."""
def pipeline_divergence(a: list[int | None], b: list[int | None], gold: list[int]) -> dict:
    """Returns {'overall': disagree_rate, 'contingency': {...}} where contingency keys are the
    four cells 'both_correct','a_correct_b_wrong','a_wrong_b_correct','both_wrong', each a
    dict {'agree':int,'disagree':int}. The 'both_wrong'+'agree' cell (shared error) is
    reported explicitly. None on either side counts toward disagree."""
```
All pure (parsed ints / strings; no models, no I/O). Reuse `awake.eval.bootstrap` for CIs and
**paired** deltas. Add to `src/awake/eval/__init__.py`. (`extract_choice` returns a tuple so the
parser-source feeds the `parsed_by` parquet column; callers resolve choice text via
`choices[idx]` before the leakage helpers.)

## 6. Eval + figures

`scripts/30_eval.py` — for **{unfiltered, filtered}** subsets and **{A, B, B7}** pipelines:
accuracy, parse_rate (answer arm **and** each ablated arm: `parse_rate.{answer,abl_expl,abl_noexpl}`),
model-output leakage rate (`expl_leak_rate` = mean of `expl_leaks`), the two ablation
consistencies + `Δ`, and pairwise divergence. **Bootstrap 95% CIs** on each; **paired-bootstrap**
CIs on the cross-pipeline and with−no-expl deltas (CIs reported regardless of zero; no post-hoc
significance). The three pipeline pairs are stored under explicit keys — `A_vs_B` (headline),
`A_vs_B7` and `B_vs_B7` (bound the size confound). Valid `metrics.json` example:

```json
{"split": "validation", "n": 1145, "n_filtered": 1012, "prompt_variant": "main",
 "model_revisions": {"blip2": "<sha>", "qwen7b": "<sha>", "qwenvl3b": "<sha>", "qwenvl7b": "<sha>"},
 "b7_completed": true,
 "subsets": {
   "unfiltered": {
     "pipelines": {
       "A":  {"accuracy": 0.0, "parse_rate": {"answer": 0.0, "abl_expl": 0.0, "abl_noexpl": 0.0},
              "expl_leak_rate": 0.0,
              "consistency": {"with_expl": 0.0, "no_expl": 0.0, "delta": 0.0, "delta_ci": [0.0, 0.0]}},
       "B":  {"...": "same shape"},
       "B7": {"...": "same shape"}},
     "divergence": {
       "A_vs_B":  {"overall": 0.0, "overall_ci": [0.0, 0.0],
                   "contingency": {"both_correct": {"agree": 0, "disagree": 0},
                                   "a_correct_b_wrong": {"agree": 0, "disagree": 0},
                                   "a_wrong_b_correct": {"agree": 0, "disagree": 0},
                                   "both_wrong": {"agree": 0, "disagree": 0}}},
       "A_vs_B7": {"...": "same shape"},
       "B_vs_B7": {"...": "same shape"}}},
   "filtered": {"...": "same shape"}}}
```

Hero figure (`assets/hero.png`): a multi-panel — (i) consistency `Δ` per pipeline (with CIs,
with a 0 reference line; panel label "self-rationale recoverability gain (Δ)"), (ii) accuracy,
(iii) parse_rate — so a stably-wrong/image-ignoring pipeline can't look good on `Δ` alone.

## 7. Tests

- **unit** (`vqa_consistency.py`, 90% floor): `extract_choice` (first-line letter, choice-text
  fallback, malformed→None); `rationale_leaks_answer`/`explanation_leaks_answer` (leak/no-leak,
  letter-not-matched, normalisation); `consistency_rate` denominator policy incl. `None` on
  either side = inconsistent; `parse_rate`; `accuracy` None=wrong; `pipeline_divergence`
  contingency on a hand-built case.
- **smoke** (CPU, no model downloads, no dataset): a **stub generator** in `_stub.py`,
  interface `stub_generate(prompt: str, item: dict) -> str` returning a canned
  `"Answer: <X>\n<one-sentence explanation>"` string keyed off the item, drives
  `10_run_pipelines`/`20_probe` logic + `30_eval` end-to-end on tiny synthetic items. The
  generate functions take an injectable `generate` callable (default = real model, stub in
  tests). Heavy model imports are **lazy (inside functions)** so importing the scripts
  triggers no download.
- **slow** (excluded from CI): real BLIP-2 / Qwen-2.5-7B / Qwen-2.5-VL-{3B,7B} generation.

## 8. Deps + governance

Add an **optional-dependency group** `[project.optional-dependencies] vqa = ["qwen-vl-utils>=0.0.10",
"accelerate>=0.30"]` (pillow already present; transformers/torch/datasets present — note the
**effective floor `transformers>=5.9`** for Qwen2.5-VL, satisfied by the installed stack). The
`vqa` group is installed for the slow/real run, not required for unit+smoke CI. ADR
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
green-CI + ready, then run the headline.

**Headline = full `validation`, all three arms {A, B, B7}.** This is the only configuration
that may be reported as the headline. There is no "headline subset": a seeded, clearly-labelled-N
subset is permitted **only** as a pilot/smoke run while developing (never reported as the result),
and pooling `train+val` is an explicitly-labelled exploratory sensitivity, never the headline.
B7 is part of the headline (§3) — if it genuinely cannot finish, the size confound is reported
as **unmitigated** in §9, not silently dropped, and that fact is recorded in `metrics.json`.
The BLIP-2-on-black-tile secondary sensitivity is the only arm that runs "only if time permits"
(clearly marked optional). No fabricated numbers; log model revisions.

## 11. Repository layout (new)

```
src/awake/eval/vqa_consistency.py            # NEW pure core (+ __init__ export)
tests/test_eval_vqa_consistency.py
projects/04-vqa-aokvqa/
  configs/{data,pipelines}.yaml              # prompts, seed, max_new_tokens, model ids+revisions
  scripts/{00_data,10_run_pipelines,20_probe,30_eval}.py  _paths.py _models.py _stub.py
  tests/{conftest.py, test_data_smoke.py, test_pipeline_smoke.py}
  notebooks/01-vqa-consistency.py            # committed WITH outputs (no raw dataset dumps);
                                             #   resolves root via os.environ["P4_PROJECT_ROOT"]
  assets/hero.png                            # committed (the multi-panel figure)
  REPORT.md  (README.md exists — updated)    (outputs/ gitignored; assets/ committed)
docs/decisions/004-vqa-pipelines-and-vision-ablation.md
```

## 12. Review traceability

**Round 2 (v2.1 precision patch — all 4 reviewers: sound, no third round):** resolved §2/§10
headline contradiction (full-val is the *only* headline; subset = pilot only); elevated B7 from
optional to **required** for the headline; wired model-output leakage into the parquet
(`expl_leaks`) + `metrics.json` (`expl_leak_rate`); explicit divergence pair keys
`A_vs_B`/`A_vs_B7`/`B_vs_B7` + concrete contingency cells + valid JSON example; renamed parser
"strict-then-text" with `parsed_by` source + per-arm parse_rate; `extract_choice -> (idx, parsed_by)`
0-based; `consistency_rate(..., paired_only=False)`; `max_new_tokens=256`; identical instruction
format across ablation arms; `stub_generate` interface; `Δ<0`/`Δ≈0` interpretation; `[vqa]`
optional-dep group (`transformers>=5.9` floor); idempotent decode; `P4_PROJECT_ROOT`; committed
`assets/hero.png`.

**Round 1 → folded in:**

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
