# ERASER Movies: perturbation faithfulness versus rationale plausibility

Do explanations identify words whose masking changes a sentiment classifier's decision,
and do those words overlap human rationales? These are separate questions.

**Status: code repair validated offline; scientific rerun pending.** Previous comparative
results are superseded because attribution alignment, LIME's visible input, and target
scalars were inconsistent. No current explainer ranking is established.
See [repair report](REPAIR_REPORT.md), [protocol](REPORT.md), and the explicitly
[superseded record](results/superseded-v1/README.md).

The repaired comparison uses random word rankings, Gradient x Input, Integrated
Gradients and positional word-mask LIME. All operate on one frozen token sequence,
with complete visible whitespace words mapped by character overlap. Gradient methods
and LIME target the original predicted-class **logit**. Perturbation metrics measure
changes in that same class's **probability**. Human-rationale overlap measures plausibility.

## Reproduce

Restore the original fine-tuned model, saved tokenizer, `model_sha256.txt` and
`train_meta.json` into `outputs/model/`. No retraining is part of this repair.
Follow the staged commands in [REPAIR_REPORT.md](REPAIR_REPORT.md#reproduction).
Prepared rows, token/word caches and original data remain private and gitignored.
Only a verified full run may publish aggregate metrics.

Installing `explain-shap` does **not** add SHAP to this evaluation. Its legacy adapter
has not been validated against the repaired contract and is excluded.

The old executed notebooks and plot are archived with superseded results. They are
not evidence for the repaired protocol. Tutorial production is outside this repair.
