# ERASER Movies: perturbation faithfulness versus rationale plausibility

Do explanations identify words whose masking changes a sentiment classifier's decision,
and do those words overlap human rationales? These are separate questions.

**Status (2026-09-22): corrected full real-data rerun completed and published.** Previous
comparative results are superseded because attribution alignment, LIME's visible input,
and target scalars were inconsistent. Under the corrected v2 protocol, Integrated
Gradients produced the largest measured perturbation-faithfulness effect among the
evaluated methods, with statistical support at the Bonferroni-corrected level.
See [repair report](REPAIR_REPORT.md), [protocol](REPORT.md), and the explicitly
[superseded record](results/superseded-v1/README.md).

The repaired comparison uses random word rankings, Gradient x Input, Integrated
Gradients and positional word-mask LIME. All operate on one frozen token sequence,
with complete visible whitespace words mapped by character overlap. Gradient methods
and LIME target the original predicted-class **logit**. Perturbation metrics measure
changes in that same class's **probability**. Human-rationale overlap measures plausibility.

## Reproduce

The original v1 checkpoint is unavailable; the authoritative results use the fresh-v2
checkpoint trained under the frozen protocol (commit `6e3652f`) and verified by the
permanent pilot and full-run audits. Follow the staged commands in
[REPAIR_REPORT.md](REPAIR_REPORT.md#reproduction).
Prepared rows, token/word caches and original data remain private and gitignored.
Only a verified full run may publish aggregate metrics.

Installing `explain-shap` does **not** add SHAP to this evaluation. Its legacy adapter
has not been validated against the repaired contract and is excluded.

The old executed notebooks and plot are archived with superseded results. They are
not evidence for the repaired protocol. Tutorial production is outside this repair.
