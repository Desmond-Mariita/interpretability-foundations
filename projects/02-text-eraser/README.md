# 02 — Which text-classification explainer is actually faithful?

**Question.** When LIME, KernelSHAP, Integrated Gradients, and Attention rollout disagree about which tokens drove a classifier's decision, which is closest to what the model actually used?

**Answer.** _Coming soon — this project ships in week 3._

**Why it matters.** Confident-looking explanations are easy to produce and easy to trust by default; the ERASER suite gives us human-annotated rationales to check them against.

<!-- ![hero](assets/hero.png) -->

## Method

Fine-tune `microsoft/deberta-v3-base` on the ERASER Movies sentiment task. For each prediction, run four explainers: LIME, KernelSHAP, Integrated Gradients, Attention rollout. Measure **faithfulness** with sufficiency, comprehensiveness, and AOPC (Samek et al. 2017); **plausibility** with token-IoU and token-F1 against human rationales. Report a 2D scatter of faithfulness vs. plausibility per explainer, with bootstrap confidence intervals.

Attention as an explanation is contested in the literature (Jain & Wallace 2019; Bibal et al. 2022) — included for completeness, framed accordingly. See [`REPORT.md`](REPORT.md).

## Reproduce

```
just setup
just data
just train
just eval
```

## Limitations

- No new explainer is proposed; no claim of a universal best.
- Results are specific to DeBERTa-v3-base on ERASER Movies; generalisation is not asserted.
- Bootstrap CIs quantify sampling noise, not model-selection noise.
