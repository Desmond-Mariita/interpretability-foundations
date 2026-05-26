# ADR 004 — VQA pipelines, paired-baseline vision-ablation probe, parsing policy, and determinism

**Status:** Accepted
**Date:** 2026-05-26

## Context

Project 4 (`04-vqa-aokvqa`) runs two zero-shot multiple-choice VQA pipelines on A-OKVQA
and measures the degree to which each pipeline's answer depends on the image vs. on its
own explanation, using a **vision-ablation probe with a paired baseline**. Before writing
pipeline code, a four-way design review identified five architectural decisions carrying
correctness or methodological implications significant enough to document formally.

This ADR records all five decisions.

---

## Decision 1 — Paired-baseline probe: Delta = consistency(with-explanation) - consistency(no-explanation) is the headline, not raw consistency alone

### Context

The raw with-explanation consistency rate (P(ablated answer == original answer) when the
model is given its own prior explanation and the visual evidence is removed) has an obvious
interpretation problem: it confounds image-independence with explanation-copying. A model
that simply copies its own explanation verbatim will produce a high rate even if the
explanation contains no image-derived information. Without a baseline arm, the number is
mislabelled as a faithfulness or image-dependence measure.

Earlier drafts of the probe reported only the with-explanation consistency rate and called
it "image-dependence consistency". The four-way design review unanimously flagged this as a
BLOCKER: the rate is meaningful only relative to what the model does when the explanation
is absent.

### Options considered

1. **Report only the raw with-explanation consistency rate.** Rejected: mislabelled without
   a baseline; conflates image-independence with self-explanation copying. Cannot distinguish
   a model that genuinely relies on the image from one whose explanation is the only cue it
   needs.
2. **Run a single no-explanation arm as the denominator and report the ratio.** Rejected:
   ratios of consistency rates are awkward to interpret and do not carry natural CIs.
3. **Run both arms symmetrically and report the difference Delta = consistency_with_expl -
   consistency_no_expl, with a paired-bootstrap CI.** Selected.

### Decision

- The probe runs two arms per pipeline using identical instruction format (same prompt
  wording, same null-visual representation), differing only in whether the model's own
  prior explanation is included.
- The **with-explanation arm** feeds the question, null visual, and the model's prior
  explanation back as context.
- The **no-explanation arm (baseline)** feeds only the question and null visual.
- The headline metric is **Delta = consistency(with-explanation) - consistency(no-explanation)**
  with a paired-bootstrap 95% CI (2,000 resamples). The raw with-explanation rate is also
  reported, **honestly labelled "self-rationale recoverability"** so readers understand it
  conflates image-independence with explanation-copying without the Delta framing.
- Interpretation: a large positive Delta flags the self-rationale recoverability red flag
  (the explanation, not the image, is what lets the model reproduce its answer); Delta near
  zero means the explanation adds nothing the question alone did not supply (answers are
  driven by language priors, not recovered from the rationale); Delta < 0 is treated as
  noise around zero and reported with its CI without over-interpretation.
- Both consistency rates and Delta are reported on **unfiltered and filtered** (leakage-free)
  subsets for transparency.

### Consequences

- Two ablation arms are required per pipeline per run, doubling the ablation generation
  cost. This is accepted as the price of a defensible probe.
- The Delta framing is propagated to the hero figure (panel 1), REPORT.md, and the notebook.
- The raw with-explanation rate is never promoted as the primary result.

---

## Decision 2 — Pipeline A vision ablation: null-caption string is primary; BLIP-2-on-black-tile is secondary sensitivity only

### Context

Pipeline A passes a BLIP-2 caption to the LLM; the LLM never sees the image directly.
There are two natural ways to ablate the visual evidence for Pipeline A:

- Replace the caption with a fixed null string (e.g., "(no visual information available)")
  so the LLM receives the same prompt structure with no image-derived text.
- Run BLIP-2 on a black (all-zeros) tile and feed that caption to the LLM.

Both remove real visual information. The black-tile approach chains two components: it
tests whether BLIP-2 produces a neutral or misleading caption for an all-zeros image, which
is an independent variable not controlled by the ablation protocol.

### Options considered

1. **Black-tile captioning as the primary ablation (BLIP-2 generates on a 224x224 black image).**
   Introduces an uncontrolled BLIP-2 response to a degenerate input as an additional source
   of variance. What BLIP-2 says about a black tile is a property of BLIP-2 at the
   distributional edge, not a property of the LLM's image-dependence. Rejected as primary.
2. **Fixed null-caption string as the primary ablation.** Clean and controlled: the LLM
   receives the identical prompt structure with a single well-defined replacement string.
   The LLM's response to this string is the probe target. Selected as primary.
3. **Both, with null-caption as primary and black-tile as a secondary sensitivity arm.**
   Provides a cross-check. Selected: the black-tile captioning path is kept and documented
   as a **secondary sensitivity**, not the primary, and is labelled "only if time permits"
   in the real-run plan (Task 18).

### Decision

- Pipeline A's primary ablation replaces the caption with the fixed string
  `"(no visual information available)"` (defined in `configs/pipelines.yaml`).
- BLIP-2-on-black-tile is kept as a **secondary sensitivity arm only**: it is labelled
  optional in the real-run task, never reported as the primary result, and never used to
  override or replace the null-caption primary.
- The null-caption string is normalised (lowercased, punctuation-stripped) before any
  leakage check to ensure it does not accidentally match any choice text.

### Consequences

- Pipeline A ablation is clean and fast: no BLIP-2 inference required for the primary
  probe, only for the optional secondary arm.
- Pipeline B uses a black tile directly (replacing the image fed to the VLM), which is
  the natural equivalent: no caption stage is involved.
- The asymmetry between the two pipelines' ablation methods (null string vs. black tile)
  is noted explicitly in REPORT.md as a limitation of the probe design.

---

## Decision 3 — A-vs-B claim is narrowed; size-matched B7 arm is required for the headline

### Context

Pipeline A is BLIP-2 (captioner) + Qwen2.5-7B-Instruct (LLM), while Pipeline B is
Qwen2.5-VL-3B-Instruct (VLM). A direct A-vs-B comparison is confounded by at least three
factors: total parameter count (~9.7B for A vs ~3B for B), model family (BLIP-2+Qwen vs
Qwen-VL), and modality stack (caption pipeline vs. direct VLM). Any claim that "A is
better/worse than B" without controlling for size is undefendable.

Earlier drafts treated the B7 arm (Qwen2.5-VL-7B-Instruct) as optional. The design review
unanimously elevated it to **required for the headline**: without a size-matched VLM arm,
the size confound is unmitigated and the A-vs-B comparison adds no interpretable signal.

### Options considered

1. **Report A vs B (3B) as the headline without a size-matched arm.** Rejected: the
   7B-LLM-vs-3B-VLM capacity confound is the dominant source of uncertainty in the
   comparison. A headline built on it is misleading.
2. **Run B with Qwen2.5-VL-7B-Instruct as the sole Pipeline B arm (drop 3B).** Rejected:
   still confounds model family and modality stack with size; also discards the 3B arm,
   which is informative for the divergence analysis.
3. **Run both B3 and B7; headline is A vs B7; B3 provides an additional divergence data
   point; all three pairs {A_vs_B, A_vs_B7, B_vs_B7} are reported under explicit keys.**
   Selected.

### Decision

- **B7 (Qwen2.5-VL-7B-Instruct) is a required arm for the headline**, not optional.
- The **narrowed claim** is scoped to these specific instantiations: "BLIP-2-captioned
  Qwen2.5-7B-Instruct vs Qwen2.5-VL-3B-Instruct (and vs Qwen2.5-VL-7B-Instruct)".
  No general claim is made about caption-then-LLM vs. direct VLM as families.
- The remaining confounds -- model family and modality stack -- are explicitly listed
  in REPORT.md Limitations and are not treated as controlled.
- If B7 genuinely cannot finish (e.g., extended OOM), the fact is recorded in `metrics.json`
  as `"b7_completed": false` and the size confound is reported as **unmitigated** in
  REPORT.md, never silently dropped.
- Divergence pairs are stored under the explicit keys `A_vs_B`, `A_vs_B7`, `B_vs_B7`
  in `metrics.json` so no pair is silently omitted or merged.

### Consequences

- The real run requires four models to be run sequentially: BLIP-2, Qwen2.5-7B-Instruct,
  Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct. Wall-clock is approximately 4-8 h on
  the RTX 3090; this is accepted.
- REPORT.md carries an explicit statement that the A-vs-B comparison is confounded by
  family and modality stack regardless of the size control, and that B7 bounds (but does
  not eliminate) the parameter-count confound.

---

## Decision 4 — Strict-then-text parsing; None-as-wrong primary denominator; paired-only as documented sensitivity

### Context

The prompt instructs the model to begin with `Answer: <A-D>` on the first line. In
practice, instruction-following is imperfect: the model sometimes produces the answer
letter only mid-paragraph, or outputs only a choice's text without a letter, or fails to
produce any identifiable choice. The parsing policy affects both the accuracy and the
consistency denominators and has to be fixed before any results are computed.

Two denominator policies for consistency are possible:
- **All-items (primary):** an unparseable answer on either the original or ablated arm
  counts as inconsistent. Denominator = all items.
- **Paired-only (sensitivity):** items where either arm is unparseable are dropped.
  Denominator = items where both arms parsed.

The paired-only policy looks more "fair" (only counting items with a real answer on both
sides) but inflates the consistency rate by excluding the hardest items, and changes the
estimand (consistency among parseable items, not consistency over the full set).

### Options considered

1. **Strict parsing only (first-line letter, no fallback): any other output -> None.**
   Clean but excessively lossy: a model that produces the right answer as a text substring
   on an otherwise well-formatted response would be penalised.
2. **Text fallback only (unique substring match): no structured anchor.**
   Too permissive: a model that outputs a long reasoning trace with multiple choice words
   mentioned can resolve to None (ambiguous), but short outputs with partial matches would
   match incorrectly.
3. **Strict-then-text (primary strict, text fallback only when strict fails; ambiguous text
   -> None): reported via parsed_by field.** Selected.
4. **Paired-only as the primary denominator.**
   Rejected as primary: inflates the rate and changes the estimand. Acceptable as a
   documented sensitivity check.

### Decision

- **Strict-then-text parsing:** primary rule extracts the first `Answer: <A-D>` match
  (first line or first occurrence in the output); used only if the index is valid for the
  choice list. Fallback: if strict fails, normalised text of **exactly one** choice must be
  a unique substring of the normalised output; ambiguous (multiple choices match) or
  zero-match -> `None`. The `parsed_by` field records which path succeeded (`strict`,
  `text`, or `none`).
- **None-as-wrong policy (primary):** an unparseable answer counts as wrong for accuracy
  and as inconsistent for consistency. Denominator = all items (including unparseable).
  This is reported as the headline.
- **Paired-only policy (documented sensitivity):** items where either the original or the
  ablated arm is unparseable are dropped; denominator = both-parsed pairs. This is
  reported in `metrics.json` as a secondary check, never as the headline.
- **Per-arm parse_rate** is reported for each of the three arms (answer, ablated-with-expl,
  ablated-no-expl) so a pathologically low parse rate is visible and interpretable.

### Consequences

- The strict-then-text parser is a pure function (no I/O, no models) locked down by unit
  tests in `tests/test_eval_vqa_consistency.py`.
- The None-as-wrong primary policy is conservative and biases consistency rates downward
  relative to paired-only; both are reported so the reader can assess the impact.
- Prompt wording uses `Answer: <letter>` (explicitly uppercase A-D) to maximise strict-parse
  coverage; `max_new_tokens=256` ensures the answer line is not truncated before generation.

---

## Decision 5 — Determinism via do_sample=False and logged model revisions; one-model-resident memory discipline

### Context

Reproducibility for LLM-based VQA pipelines requires two controls: the generation must
be deterministic (same prompt -> same output across re-runs), and the exact model version
must be recorded (a model that is updated on HuggingFace Hub produces different outputs
even if the model id is unchanged). In addition, the RTX 3090 (24 GB) can hold only one
of the large models in memory at a time (BLIP-2 ~6 GB, Qwen2.5-7B ~15 GB, Qwen-VL-3B
~7 GB, Qwen-VL-7B ~16 GB); holding two concurrently risks OOM.

### Options considered

1. **Greedy decoding (do_sample=False) as the sole determinism mechanism.** Sufficient for
   reproducibility if the model revision is also fixed. Selected.
2. **Sampling with a fixed seed.** Adds variance relative to greedy; the resulting
   answers would differ from a seed-free re-run unless the seed is also pinned. Rejected.
3. **Model revisions not logged (rely on HuggingFace caching).** Fragile: a model updated
   on the Hub would silently change results. Rejected; revisions are logged explicitly.

### Decision

- **Generation is deterministic:** `do_sample=False`, `max_new_tokens=256`, `torch_dtype=fp16`,
  `device_map="cuda:0"` for all models. These settings are fixed in `configs/pipelines.yaml`
  and not exposed as per-run overrides.
- **Model revisions are logged:** `model_revisions.py` resolves each model's HuggingFace
  commit hash at run time via `huggingface_hub.HfApi().model_info(mid).sha` and writes the
  result to `metrics.json["model_revisions"]`. Resolution failure is caught and recorded as
  `"unknown"` so a lookup failure never aborts a run.
- **One-model-resident discipline:** scripts load, generate over all items, then
  explicitly `del model; torch.cuda.empty_cache()` before loading the next model.
  No two large models are held in memory concurrently.

### Consequences

- Results from a re-run with the same model revision are bitwise identical to the
  original run (greedy decoding has no stochastic components).
- If a model is updated on HuggingFace between runs, the revision hash in `metrics.json`
  will differ and flag the discrepancy.
- The one-model-resident pattern means the pipeline cannot be trivially parallelised across
  GPUs without refactoring; this is accepted for a single-GPU research setup.

---

## Summary table

| Decision | Chosen option | Key rationale |
|---|---|---|
| Probe headline | Delta = consistency(with-expl) - consistency(no-expl) | Raw consistency is mislabelled without the no-explanation baseline; Delta is the only defensible headline |
| Pipeline A ablation | Null-caption string primary; BLIP-2-on-black-tile secondary only | Black-tile caption quality is an uncontrolled BLIP-2 variable; null string is a clean, controlled replacement |
| A-vs-B claim | Narrowed to specific instantiations; B7 required for the headline | 7B-LLM-vs-3B-VLM size confound is the dominant uncertainty; B7 bounds it |
| Parsing + denominator | Strict-then-text; None-as-wrong primary; paired-only as documented sensitivity | Primary policy is conservative and unambiguous; paired-only is a useful cross-check, not the headline |
| Determinism + memory | do_sample=False; logged model revisions; one-model-resident sequential loading | Bitwise reproducibility; version traceability; GPU memory budget on 24 GB RTX 3090 |
