# Project 4 -- Vision-ablation probe for caption-then-LLM vs. direct VLM on A-OKVQA

**Author:** Desmond Mariita.
**Dataset:** A-OKVQA (Schwenk et al. 2022; open; code-only, images not committed).
**Status:** complete -- real run on the full validation split (n=1145; 805 leak-flagged,
340 leakage-free). Generated 2026-05-27 on a single RTX 3090.

**Provenance note (added 2026-09-22, claims/provenance repair).** `outputs/metrics.json`
from the real run was gitignored and is not preserved in the repository. The authoritative
surviving artifacts are the executed notebook (`notebooks/01-vqa-consistency.ipynb` / `.html`,
outputs intact) and `assets/hero.png`. Every headline value below is copied from the
notebook's recorded stdout (see also `assets/metrics_snapshot.json`). The divergence
contingency counts and the filtered-subset accuracies were copied from `metrics.json` into
the original report at run time; they are marked accordingly and cannot be re-derived from
surviving artifacts. Model revisions logged at run time: BLIP-2 `59a1ef6c`, Qwen2.5-7B
`a09a3545`, Qwen2.5-VL-3B `66285546`, Qwen2.5-VL-7B `cc594898`.

---

## 1. Question and framing

When a caption-then-LLM pipeline answers a visual question, the LLM's explanation
describes a caption -- not an image. This project asks **four separate questions**, which
the results keep apart rather than collapsing into one "faithfulness" notion:

| # | Question | Metric |
|---|---|---|
| 1 | Original task performance: how accurate is each answer model on A-OKVQA? | accuracy, parse rate |
| 2 | Explanation-conditioned recoverability: how much does supplying the generated explanation change answer recovery when visual input is removed? | Delta (with vs. without explanation), with CI |
| 3 | Answer leakage: how often does the generated explanation explicitly restate the answer? | explanation leak rate |
| 4 | Human-rationale similarity/plausibility: how similar is the generated explanation to human rationales? | **not measured** (see §4.4) |

### 1.1 The central estimand, defined narrowly

The probe's key quantity is:

> **incremental answer recoverability from the supplied explanation under null visual input**

For each pipeline the visual evidence is removed and the model re-answers twice: once with
its own prior explanation supplied and once without. Delta is the difference in the rate at
which the ablated answer reproduces the original answer:

```
Delta = consistency(with-explanation) - consistency(no-explanation)
```

Delta is **not automatically** any of the following:

- faithfulness of the original answer;
- proof that the original model used text instead of the image;
- proof of hidden reasoning in the explanation;
- proof that the explanation is causally faithful to the answer.

A faithful explanation may legitimately preserve visual evidence in textual form, so
"the answer is recoverable from the explanation" does not mean "the answer did not come
from the image". A small Delta is also ambiguous (§7). The raw with-explanation rate is
reported, honestly labelled **self-rationale recoverability**, to signal that it conflates
image-independence with explanation-copying on its own (ADR 004).

### How Delta is read (narrowed)

- **Delta > 0:** supplying the explanation increases answer recovery under null visual
  input. This flags self-rationale recoverability -- the answer is reproducible from the
  explanation -- without asserting why the original answer was chosen.
- **Delta near 0:** the explanation adds nothing measurable on top of the question alone;
  recovery is driven by language priors or the question text.
- **Delta < 0:** treated as noise around zero; reported with its CI, not over-interpreted.

## 2. Data

**Corpus.** A-OKVQA (Schwenk et al. 2022). Multiple-choice VQA requiring outside knowledge,
built over COCO images. Source: `HuggingFaceM4/A-OKVQA` on HuggingFace, which bundles the
COCO image as a PIL object per item.

**Headline split.** `validation` (n=1145, labelled). The full validation split is the only
configuration reported; no sub-sampling.

**Fields per item.** `question`, `choices` (4), `correct_choice_idx`, `rationales` (3),
`image`.

**Leakage-free subset.** An item is flagged `leakage_flag=True` iff any of its three human
rationales contains the gold choice **text** verbatim (normalised; bare letters are not
matched -- they false-positive). The filter is:

- **deterministic** (a pure function of the dataset row; no model involved),
- **dataset-side only** (human rationale vs. gold answer; never model output),
- computed **once** in `scripts/00_data.py` and applied identically to all pipelines
  (805 flagged, 340 leakage-free).

Two caveats are stated up front. (i) Removing explicit gold-answer strings does **not**
remove all forms of answer leakage: paraphrase leakage, distractor leakage, and the
model's own generated-explanation leakage remain; the last is measured separately by the
explanation leak rate. (ii) The filter is **not a random subsample**: items whose gold
answer is short or easily restated are more likely to be flagged, so the filtered subset
is selected on item type. Its metrics are a sensitivity check, not an independent
population estimate.

**Governance.** A-OKVQA is open but images are not committed. `outputs/` is gitignored.
Only code, configs, `assets/`, the notebook (with outputs, no raw dataset dumps), and this
`REPORT.md` are committed.

## 3. Pipelines

All pipelines are zero-shot (no fine-tuning). Generation is deterministic
(`do_sample=False`, `max_new_tokens=256`, `torch_dtype=fp16`, `device_map="cuda:0"`).

**Pipeline A -- caption-then-LLM.** `Salesforce/blip2-opt-2.7b` (fp16) captions the image;
`Qwen/Qwen2.5-7B-Instruct` (fp16) receives `(question, caption, choices)` and produces an
answer and one-sentence explanation. The LLM never sees the image directly.

**Pipeline B -- direct VLM.** `Qwen/Qwen2.5-VL-3B-Instruct` (fp16) receives
`(question, image, choices)` and produces an answer and one-sentence explanation.

**Pipeline B7 -- size-matched VLM arm (required for the headline).**
`Qwen/Qwen2.5-VL-7B-Instruct` (fp16) runs the same direct-VLM protocol as B, to bound the
parameter-count confound in the A-vs-B comparison. B7 completed on the full split
(`b7_completed: true`); had it not, the size confound would have been reported as
unmitigated.

**Model revisions.** Each model's HuggingFace commit hash was logged in `metrics.json` at
run time (reproduced in the header above).

## 4. Metric definitions

All metric code lives in `src/awake/eval/vqa_consistency.py` (pure, unit-tested). Bootstrap
95% CIs use `awake.eval.bootstrap` (2,000 resamples, percentile method, seed 0).

### 4.1 Original task performance

- **Accuracy.** Top-1 accuracy on the original (non-ablated) answers. Unparseable outputs
  (`None`) count as wrong. Denominator = all items.
- **Parse rate.** Fraction of items whose output parsed to a choice index, reported
  separately for the answer arm and each ablated arm.

### 4.2 Explanation-conditioned recoverability

- **Consistency (with-explanation).** P(ablated answer == original answer) when the
  ablated prompt includes the model's own prior explanation. Labelled **self-rationale
  recoverability**; denominator = all items (None on either side = inconsistent).
- **Consistency (no-explanation baseline).** The same rate without the explanation. This
  is the empirical baseline that makes Delta interpretable -- not a chance level. Uniform
  guessing among four choices would give ~0.25, but the models' outputs are not uniform
  random, so no chance null is claimed; the baseline is the measured no-explanation rate.
- **Delta.** with-explanation minus no-explanation, with a paired-bootstrap 95% CI (the
  per-item consistency vectors it was computed from are not preserved; the recorded CI
  bounds are reproduced as logged).

### 4.3 Answer leakage

- **Explanation leak rate.** Fraction of items where the model's own explanation contains
  the chosen choice's text verbatim (normalised). A high leak rate means the explanation
  restates the answer, a trivial channel that inflates self-rationale recoverability.

### 4.4 Human-rationale similarity/plausibility

**Not measured.** Human rationales are used only to build the leakage-free split (§2). No
similarity/plausibility score is computed, and the recoverability delta must not be read
as one.

### 4.5 Inter-pipeline divergence

Rate at which two pipelines produce different answers (None on either side counts as
disagreement), reported per pair with a 95% CI and a 2x2 correctness-conditioned
contingency.

## 5. Results

Hero figure (`assets/hero.png`): four panels -- (i) recoverability Delta per pipeline with
paired-bootstrap 95% CI and a 0 reference line, (ii) answer accuracy, (iii) explanation
answer-leak rate, (iv) answer parse rate. The panels are separate constructs and should not
be read as one scale.

![hero](assets/hero.png)

### 5.1 Headline metrics (validation, full split, n=1145)

| Pipeline | Accuracy | Parse rate | Expl leak rate |
|---|---|---|---|
| A (BLIP-2 + Qwen-7B) | 0.628 | 0.963 | 0.812 |
| B (Qwen-VL-3B) | 0.831 | 0.997 | 0.252 |
| B7 (Qwen-VL-7B) | 0.873 | 1.000 | 0.866 |

The two direct VLMs answer more accurately than the caption-then-LLM pipeline by ~20-25
points under this setup: for these specific instantiations the caption is a lossy
bottleneck. Accuracy and leak rate are point estimates (the run computed bootstrap CIs
only for Delta and divergence); parse rate is the answer arm.

### 5.2 Vision-ablation probe (Delta per pipeline)

| Pipeline | Self-rationale recov. (with-expl) | Consistency (no-expl baseline) | Delta (95% CI) |
|---|---|---|---|
| A (BLIP-2 + Qwen-7B) | 0.924 | 0.544 | **0.380** [0.351, 0.409] |
| B (Qwen-VL-3B) | 0.666 | 0.522 | **0.144** [0.115, 0.170] |
| B7 (Qwen-VL-7B) | 0.986 | 0.500 | **0.486** [0.456, 0.516] |

Every Delta is positive with a CI clear of zero: supplying each model's own explanation
increases answer recovery once the image is removed. The no-explanation baselines sit near
0.50 (A 0.544, B 0.522, B7 0.500) -- an empirical prior-driven rate, not a chance level --
so the raw with-explanation rates (0.67-0.99; B is far below A and B7) would overstate
image-independence on their own; the paired baseline is what makes the gain visible. The paired-parsed-only sensitivity (in the run's `metrics.json`, not
re-derivable from surviving artifacts: A 0.397, B 0.144, B7 0.486) was near-identical, so
the None-as-inconsistent headline policy is not an artifact of unparseable outputs.

### 5.3 Inter-pipeline divergence

| Pair | Divergence (95% CI) | both_correct (agree/dis) | both_wrong (agree/dis) |
|---|---|---|---|
| A vs B | 0.375 [0.347, 0.403] | 649 / 0 | 67 / 57 |
| A vs B7 | 0.364 [0.336, 0.393] | 672 / 0 | 56 / 42 |
| B vs B7 | 0.139 [0.119, 0.160] | 905 / 0 | 81 / 18 |

The caption pipeline disagrees with each VLM on ~37% of items, while the two VLMs disagree
on only ~14%. This pattern is consistent with the architecture/modality-stack difference
contributing substantially to the A-vs-VLM divergence, but the design does not isolate
that factor from model-family and other differences (B7 bounds only the parameter-count
confound). The contingency counts were recorded from `metrics.json` at run
time (the notebook does not print them); they are internally consistent with the divergence
rates and accuracies above, but they cannot be re-derived from surviving artifacts. When A
and B are both wrong they pick the **same** wrong answer on 67 of 124 such items and
different wrong answers on 57 -- near parity, so shared error modes are common and only a
minority of shared failures pick distinct wrong answers.

### 5.4 Filtered subset (leakage-free, n=340)

| Pipeline | Accuracy | Delta (95% CI) |
|---|---|---|
| A | 0.641 | 0.362 [0.309, 0.418] |
| B | 0.847 | 0.159 [0.112, 0.209] |
| B7 | 0.882 | 0.465 [0.409, 0.521] |

(Filtered accuracies are recorded from the run's `metrics.json` via the original report;
the notebook prints only the filtered Deltas.) The same Delta ordering (B7 > A > B) and
similar magnitudes persist after filtering the 805 items whose human rationales contain
the gold answer text verbatim. This reduces concern that the observed pattern is
explained solely by that specific form of dataset-side leakage, but it does not rule out
other leakage channels (paraphrase, distractors, model-generated answers) or selection
effects. The subset-selection caveats of §2 apply.

### 5.5 B7 completion status

`b7_completed: true`. The size-matched 7B VLM arm ran on the full split, so the
parameter-count confound in the A-vs-B comparison is bounded (see §7).

## 6. Discussion

**Supplying the explanation increases answer recovery under null visual input -- most of
all for B7.** All three pipelines show a positive Delta (A 0.38, B 0.14, B7 0.49): with
visual input removed, re-answering succeeds substantially more often when the model's own
prior explanation is supplied than when it is not. That is *incremental answer
recoverability from the supplied explanation under null visual input* -- evidence that the
explanations carry enough of each model's own answer signal to reproduce it. It is not
evidence that the original answers were driven by text instead of the image: a faithful
explanation may encode visual evidence textually, and the probe cannot distinguish
recovered visual content from recovered model bias.

**Capacity: descriptive pattern only, no general claim.** Under this setup, B7 (7B VLM)
showed higher answer accuracy and a larger recoverability delta than B (3B VLM)
(0.873 vs 0.831; 0.486 vs 0.144). No general capacity effect is inferred: the two models
differ in more than parameter count, and this design is not a controlled capacity
comparison.

**Explanation leakage tracks Delta.** B7 and A, the high-Delta pipelines, also restate the
chosen answer text in their explanations most often (leak rate 0.87 and 0.81 vs B's 0.25).
Part of the recoverability gain is therefore plausibly the trivial restating-the-answer
channel rather than recovered reasoning. The correlation is descriptive; it does not
quantify how much of each Delta is leakage-driven.

**Divergence pattern.** A diverges from both direct-VLM variants much more often (~37%)
than B and B7 diverge from each other (~14%). This pattern is consistent with the
architecture/modality-stack difference contributing substantially, but the design does not
isolate that factor from model-family and other differences.

**Net.** On A-OKVQA, the caption-then-LLM pipeline is less accurate and more divergent
than the direct VLMs under this setup, and every pipeline -- most strongly B7 -- shows a
positive explanation-conditioned recoverability delta under null visual input. The
filtered-subset replication reduces concern that dataset-side answer leakage alone
explains the pattern; the generated-explanation leak channel and other leakage forms
remain (§4.3, §5.4).

## 7. Limitations

**Delta is recoverability, not faithfulness.** The probe measures incremental answer
recoverability from the supplied explanation under null visual input. It does not measure
faithfulness of the original answer, does not show the original model used text rather
than the image, and does not demonstrate hidden reasoning or causal faithfulness. A
faithful explanation may legitimately preserve visual evidence in textual form.

**A small Delta is ambiguous.** A near-zero or small Delta can mean the explanation is
uninformative, that the model ignores its explanation, that the no-explanation baseline is
already near the with-explanation ceiling, or a mix of these. Low Delta alone neither
condemns nor vindicates the explanation.

**Zero-shot only.** No fine-tuning of either pipeline. Results are specific to the
zero-shot regime.

**A-vs-B confounded by parameter count, model family, and modality stack.** Pipeline A
uses a ~7B LLM (after a ~2.7B captioner) while Pipeline B uses a ~3B VLM. The B7 arm
bounds the parameter-count confound but does not remove it; the model-family and
modality-stack confounds remain uncontrolled. All claims are scoped to these specific
instantiations.

**BLIP-2 caption quality is a confound for Pipeline A.** If BLIP-2 produces a misleading
caption, Pipeline A's accuracy and divergence reflect that, not the LLM's visual reasoning.

**Probe is one family, not a battery.** The paired-baseline ablation probe is one design in
a broad space of faithfulness evaluations. A full battery (counterfactual image swaps,
attention analysis, causal tracing) would be needed for a comprehensive picture; see §9
for the later thesis work that pursues this.

**Pipeline A ablation asymmetry.** Pipeline A replaces the caption with a null string while
Pipeline B replaces the image with a black tile. These are the natural ablations for each
architecture but are not strictly comparable.

**Multiple-choice only.** A-OKVQA direct-answer splits are not scored.

**Answer parsing is strict-then-heuristic.** The strict parser (first-line
`Answer: <A-D>`) is primary; a text-substring fallback handles non-strict outputs.
Unparseable outputs count as wrong/inconsistent; parse rate is reported per arm.

**Leakage-free subset is selected, not random.** The filter removes items whose human
rationales restate the gold answer text; it removes neither paraphrase/indirect leakage nor
generated-explanation leakage, and it selects on item type (§2).

**Single prompt wording (headline).** The headline run used only the `main` prompt
(`prompt_variant: "main"`). The alternate wording in `configs/pipelines.yaml` was not
executed; prompt sensitivity remains future work.

**Headline is the full split.** All numbers above are the full `validation` split
(n=1145); no subsampling was used. A seeded 8-item pilot validated the pipeline only and
is not reported.

**Provenance gap.** `outputs/metrics.json` and the per-item parquets were not committed.
Headline values survive in the executed notebook and are reproduced here; the divergence
contingency counts, filtered accuracies, and paired-only sensitivity survive only as
written in the original report and cannot be re-derived (§8). Delta CIs cannot be
recomputed from surviving artifacts.

## 8. Provenance

**Recovered (surviving record):** model ids (`configs/pipelines.yaml`) and HF commit
revisions (notebook output); dataset id + split + counts (notebook output); generation
parameters (`configs/pipelines.yaml`: greedy decoding, 256 max tokens, fp16, cuda:0);
bootstrap settings (2,000 resamples, seed 0, in `30_eval.py`); prompt text (configs);
environment (single RTX 3090; dependency versions pinned in the committed `uv.lock`);
artifacts: executed notebook (ipynb + html), `assets/hero.png`, `assets/metrics_snapshot.json`.

**Not preserved:** `outputs/metrics.json` (gitignored); per-item generation and ablation
parquets; per-item consistency vectors; the HuggingFace dataset revision at download time
(`load_dataset` was called without a pinned revision).

**Not invented.** No historical model or dataset revision beyond the logged values is
claimed. Future reruns should pin the dataset revision and commit `metrics.json` (or a
snapshot) so headline numbers remain re-derivable.

## 9. References and further reading

- Schwenk, D., Khandelwal, A., Clark, C., Marino, K., & Mottaghi, R. (2022).
  *A-OKVQA: A benchmark for visual question answering using world knowledge.* ECCV 2022.
- Salesforce Research. (2023). *BLIP-2: Bootstrapping language-image pre-training with
  frozen image encoders and large language models.*
- Qwen Team. (2024). *Qwen2.5-VL technical report.*
- Qwen Team. (2024). *Qwen2.5 technical report.*
- [ADR 004](../../docs/decisions/004-vqa-pipelines-and-vision-ablation.md) -- design
  decisions (read together with this corrected estimand language).
- [Spec v2.1](../../docs/superpowers/specs/2026-05-26-vqa-aokvqa-design.md).
- **Later related work.** A separate MSc thesis by the same author conducts a stronger
  behavioural-faithfulness study. Project 4 is **not** part of that thesis, no thesis
  results are imported here, and thesis evidence is not used to retroactively validate P4;
  the thesis is the recommended follow-up reading for the faithfulness questions this
  probe only partially addresses.
