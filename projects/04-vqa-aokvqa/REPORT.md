# Project 4 -- Vision-ablation probe for caption-then-LLM vs. direct VLM on A-OKVQA

**Author:** Desmond Mariita.
**Dataset:** A-OKVQA (Schwenk et al. 2022; open; code-only, images not committed).
**Status:** complete -- real run on the full validation split (n=1145; 805 leak-flagged,
340 leakage-free). Generated 2026-05-27 on a single RTX 3090. Model revisions logged in
`metrics.json` (BLIP-2 `59a1ef6c`, Qwen2.5-7B `a09a3545`, Qwen2.5-VL-3B `66285546`,
Qwen2.5-VL-7B `cc594898`).

---

## 1. Question and framing

When a caption-then-LLM pipeline answers a visual question, the LLM's explanation
describes a caption -- not an image. Two questions motivate this project:

(a) How often does the caption-then-LLM pipeline's answer **diverge** from a direct
vision-language model on the same question?

(b) How much does each pipeline's answer actually **depend on the image** vs. on its own
explanation?

The second question is measured by a **vision-ablation probe with a paired baseline**: for
each pipeline we remove the visual evidence and re-answer twice -- once with the model's
own prior explanation and once without -- and report the **consistency gain from the
explanation**:

```
Delta = consistency(with-explanation) - consistency(no-explanation)
```

### How Delta is read

- **Delta > 0 (positive):** the explanation, not the image, is what lets the model
  reproduce its answer -- a self-rationale recoverability red flag. The model is recovering
  its original choice from its own explanation rather than from visual evidence.
- **Delta near 0:** the explanation adds nothing the question alone did not already supply.
  Answers are likely driven by language priors or the question text, not recovered from the
  rationale. The visual ablation has little incremental effect when the explanation is present.
- **Delta < 0 (negative):** the explanation actively hurts recovery relative to the
  no-explanation baseline. Treated as noise around zero; reported with its CI, not
  over-interpreted.

The raw with-explanation rate is also reported, **honestly labelled "self-rationale
recoverability"** to signal that it conflates image-independence with explanation-copying
on its own. See [ADR 004](../../docs/decisions/004-vqa-pipelines-and-vision-ablation.md)
for the full design rationale.

## 2. Data

**Corpus.** A-OKVQA (Schwenk et al. 2022). Multiple-choice VQA requiring outside knowledge,
built over COCO images. Source: `HuggingFaceM4/A-OKVQA` on HuggingFace, which bundles the
COCO image as a PIL object per item.

**Headline split.** `validation` (~1.1k items, labelled). The full validation split is the
only configuration that may be reported as the headline; no sub-sampling is performed.

**Fields per item.** `question`, `choices` (4), `correct_choice_idx`, `rationales` (3),
`image`.

**Leakage sensitivity split.** Items where any human rationale contains the gold choice
text (normalised; choice text matched, not bare letter -- bare letters cause false
positives) are flagged `leakage_flag=True`. All metrics are reported on both the unfiltered
set and the filtered (leakage-free) subset.

**Governance.** A-OKVQA is open but images are not committed. `outputs/` is gitignored.
Only code, configs, `assets/hero.png`, the notebook (with outputs, no raw dataset dumps),
and this `REPORT.md` are committed.

## 3. Pipelines

All pipelines are zero-shot (no fine-tuning). Generation is deterministic (`do_sample=False`,
`max_new_tokens=256`, `torch_dtype=fp16`, `device_map="cuda:0"`).

**Pipeline A -- caption-then-LLM.**
`Salesforce/blip2-opt-2.7b` (fp16) captions the image; `Qwen/Qwen2.5-7B-Instruct` (fp16)
receives `(question, caption, choices)` and produces an answer and one-sentence explanation.
The LLM never sees the image directly.

**Pipeline B -- direct VLM.**
`Qwen/Qwen2.5-VL-3B-Instruct` (fp16) receives `(question, image, choices)` and produces
an answer and one-sentence explanation.

**Pipeline B7 -- size-matched VLM arm (required for the headline).**
`Qwen/Qwen2.5-VL-7B-Instruct` (fp16) runs the same direct-VLM protocol as Pipeline B.
B7 is required to bound the parameter-count confound in the A-vs-B comparison: Pipeline A
uses a ~7B LLM while Pipeline B uses a ~3B VLM; without B7, the size gap is unmitigated.
If B7 cannot complete, this fact is recorded in `metrics.json` as `"b7_completed": false`
and the size confound is reported as unmitigated below (never silently dropped).

**Model revisions.** Each model's HuggingFace commit hash is logged in `metrics.json` at
run time to ensure reproducibility.

## 4. Metric definitions

All metric code lives in `src/awake/eval/vqa_consistency.py` (pure, unit-tested). Bootstrap
95% CIs use `awake.eval.bootstrap` (2,000 resamples, percentile method, fixed seed).

**Accuracy.** Top-1 accuracy on the original (non-ablated) answers. Unparseable outputs
(`None`) count as wrong. Denominator = all items.

**Parse rate.** Fraction of items where the model output was successfully parsed to a
choice index. Reported separately for the answer arm and each ablated arm
(`parse_rate.{answer, abl_expl, abl_noexpl}`).

**Self-rationale recoverability.** `P(ablated answer == original answer)` when the
with-explanation arm is used. Honest label: this rate alone conflates image-independence
with explanation-copying. Denominator = all items (None on either side = inconsistent).

**Consistency (no-explanation baseline).** `P(ablated answer == original answer)` when
only `(question, null-visual, choices)` is used -- no prior explanation. This is the
baseline that makes Delta interpretable.

**Delta.** Consistency(with-explanation) - Consistency(no-explanation). With paired
bootstrap 95% CI.

**Model-output leakage rate.** Fraction of items where the model's own explanation
contains the chosen choice's text verbatim (normalised). High leakage rate means the
explanation is essentially restating the answer, which inflates self-rationale recoverability.

**Inter-pipeline divergence.** Rate at which two pipelines produce different answers
(None on either side counts as disagreement). Reported per pair with a 95% CI and a
2x2 correctness-conditioned contingency: `both_correct`, `a_correct_b_wrong`,
`a_wrong_b_correct`, `both_wrong`, each with `agree` and `disagree` counts.

## 5. Results

Hero figure (`assets/hero.png`): three-panel summary -- (i) Delta per pipeline (with 0
reference line), (ii) accuracy, (iii) parse rate.

![hero](assets/hero.png)

### 5.1 Headline metrics (validation, full split, n=1145)

Accuracy is a point estimate (the eval stores bootstrap CIs for Delta and divergence, not
for accuracy). Parse rate is the answer arm.

| Pipeline | Accuracy | Parse rate | Expl leak rate |
|---|---|---|---|
| A (BLIP-2 + Qwen-7B) | 0.628 | 0.963 | 0.812 |
| B (Qwen-VL-3B) | 0.831 | 0.997 | 0.252 |
| B7 (Qwen-VL-7B) | 0.873 | 1.000 | 0.866 |

The two direct VLMs outperform the caption-then-LLM pipeline by ~20-25 accuracy points: the
caption is a lossy bottleneck. B7's near-perfect parse rate and B7 > B accuracy are the
expected capacity gains within the VLM family.

### 5.2 Vision-ablation probe (Delta per pipeline)

| Pipeline | Self-rationale recov. (with-expl) | Consistency (no-expl baseline) | Delta (95% CI) |
|---|---|---|---|
| A (BLIP-2 + Qwen-7B) | 0.924 | 0.544 | **0.380** [0.351, 0.409] |
| B (Qwen-VL-3B) | 0.666 | 0.522 | **0.144** [0.115, 0.170] |
| B7 (Qwen-VL-7B) | 0.986 | 0.500 | **0.486** [0.456, 0.516] |

Every Delta is positive with a CI well clear of zero: across all three pipelines, the
model's own explanation is a substantial cue for reproducing its answer once the image is
removed. The effect is **largest for the strongest model (B7, Delta=0.486)** and smallest
for the 3B VLM (B, Delta=0.144) -- i.e. the more capable model leans *more* on its
self-rationale, the central faithfulness red flag this probe was built to detect. Note all
three no-explanation baselines sit near 0.50, so a naive reading of the raw with-explanation
rate (0.99 for B7) would badly overstate image-independence; the paired baseline is what
makes that visible.

### 5.3 Inter-pipeline divergence

| Pair | Divergence (95% CI) | both_correct (agree/dis) | both_wrong (agree/dis) |
|---|---|---|---|
| A vs B | 0.375 [0.347, 0.403] | 649 / 0 | 67 / 57 |
| A vs B7 | 0.364 [0.336, 0.393] | 672 / 0 | 56 / 42 |
| B vs B7 | 0.139 [0.119, 0.160] | 905 / 0 | 81 / 18 |

The caption pipeline disagrees with each VLM on ~37% of items, but the two VLMs disagree on
only ~14% -- the modality stack (caption vs. direct vision), not raw capacity, drives most
of the A-vs-B divergence, since the size-matched B-vs-B7 pair diverges far less. When A and
B both answer wrong they more often pick *different* wrong answers (57 disagree vs 67 agree),
suggesting the two architectures fail for different reasons.

### 5.4 Filtered subset (leakage-free, n=340)

| Pipeline | Accuracy | Delta (95% CI) |
|---|---|---|
| A | 0.641 | 0.362 [0.309, 0.418] |
| B | 0.847 | 0.159 [0.112, 0.209] |
| B7 | 0.882 | 0.465 [0.409, 0.521] |

Removing the 805 items whose human rationales leak the gold answer text barely moves the
picture: the Delta ordering (B7 > A > B) and magnitudes are essentially unchanged, so the
result is not an artifact of dataset-side answer leakage.

### 5.5 B7 completion status

`b7_completed: true`. The size-matched 7B VLM arm ran on the full split, so the
parameter-count confound in the A-vs-B comparison is bounded (see Discussion).

## 6. Discussion

**The explanation, not the image, carries the answer once vision is ablated -- most of all
for the strongest model.** All three pipelines show a large positive Delta (A 0.38, B 0.14,
B7 0.49): re-answering with the image removed succeeds far more often when the model's own
prior explanation is in the prompt than when it is not. Because the no-explanation baselines
all hover near 0.50 (chance-like for a 4-way choice after some language-prior signal), the
headline-grabbing with-explanation rates -- 0.92-0.99 -- would, read alone, suggest these
pipelines barely need the image. The paired baseline corrects that illusion: the *gain* from
the explanation is what is large, which is the self-rationale-recoverability red flag.

**Capacity makes faithfulness worse here, not better.** The size-matched B7 (7B VLM) has the
highest accuracy (0.873) *and* the highest Delta (0.486), while the smaller B (3B VLM) has
both the lowest accuracy and the lowest Delta. So within the VLM family, the more capable
model is the one whose answer is most recoverable from its own rationale -- a caution against
assuming stronger models give more image-grounded explanations.

**Explanation leakage tracks Delta.** B7 and A, the high-Delta pipelines, also restate the
chosen answer text in their explanations most often (leak rate 0.87 and 0.81 vs. B's 0.25).
This is mechanistically consistent: an explanation that contains the answer makes the answer
trivially recoverable when the image is gone. The model-output leak rate and Delta are
measuring two faces of the same behavior.

**Divergence is driven by the modality stack, not capacity.** A diverges from both VLMs on
~37% of items, but the two VLMs diverge on only ~14%. Since B and B7 differ only in size,
the much larger A-vs-VLM divergence is attributable to the caption-then-LLM architecture
(a lossy text bottleneck) rather than the parameter gap. The correctness contingency
reinforces this: A and B never disagree when both are correct (649/0) but frequently pick
different wrong answers when both fail (57 disagree vs. 67 agree), i.e. they fail for
different reasons.

**Net.** On A-OKVQA, the caption-then-LLM pipeline is both less accurate and more divergent
than a direct VLM, and *every* pipeline -- especially the strongest -- shows that its
self-explanation, not the image, is the dominant cue for reproducing its own answer under
vision ablation. The filtered-subset replication rules out dataset answer-leakage as the
cause.

## 7. Limitations

**Zero-shot only.** No fine-tuning of either pipeline. Results are specific to the
zero-shot regime; fine-tuned models may show qualitatively different Delta patterns.

**A-vs-B confounded by parameter count, model family, and modality stack.** Pipeline A
uses a ~7B LLM (after a ~2.7B captioner) while Pipeline B uses a ~3B VLM. The B7 arm
bounds the parameter-count confound but does not remove it entirely; the model-family
(BLIP-2+Qwen vs Qwen-VL) and modality-stack (caption pipeline vs. direct VLM) confounds
remain uncontrolled. The headline claim is scoped to these specific instantiations.

**BLIP-2 caption quality is a confound for Pipeline A.** If BLIP-2 produces a misleading
or low-quality caption, Pipeline A's accuracy and divergence reflect that, not the LLM's
visual reasoning. Results are scoped to the BLIP-2 instantiation.

**Probe is one family, not a battery.** The paired-baseline ablation probe is one design
in a broad space of faithfulness evaluations. It cannot distinguish all causal pathways
between image evidence, explanation, and answer. A full faithfulness battery (e.g.,
counterfactual image swaps, attention analysis, causal tracing) would be needed for a
comprehensive picture.

**Pipeline A ablation asymmetry.** Pipeline A replaces the caption with a null string while
Pipeline B replaces the image with a black tile. These are the natural ablations for each
architecture but they are not strictly comparable: the null string is a lexical intervention
while the black tile is a visual intervention.

**Multiple-choice only.** A-OKVQA direct-answer splits are not scored; all results are
restricted to the four-choice multiple-choice format.

**Answer parsing is strict-then-heuristic.** The strict parser (first-line `Answer: <A-D>`)
is the primary; a text-substring fallback handles non-strict-compliant outputs. Unparseable
outputs count as wrong/inconsistent. Parse rate is reported per arm to make failures visible.

**Single prompt wording (headline).** The headline run used only the `main` prompt
(`metrics.json` records `"prompt_variant": "main"`). An alternate wording is pre-registered
in `configs/pipelines.yaml` as a sensitivity arm but was **not executed** for this report;
absolute accuracy and consistency rates could shift under different prompt formulations, and
the alternate-prompt sensitivity remains future work.

**Headline is the full split.** All numbers above are the full `validation` split (n=1145);
no subsampling was used (`metrics.json` records `n: 1145`). A seeded 8-item pilot was run
first only to validate the pipeline end-to-end and is not reported here.

## 8. References

- Schwenk, D., Khandelwal, A., Clark, C., Marino, K., & Mottaghi, R. (2022).
  *A-OKVQA: A benchmark for visual question answering using world knowledge.*
  ECCV 2022.
- Salesforce Research. (2023). *BLIP-2: Bootstrapping language-image pre-training with
  frozen image encoders and large language models.*
- Qwen Team. (2024). *Qwen2.5-VL technical report.*
- Qwen Team. (2024). *Qwen2.5 technical report.*
