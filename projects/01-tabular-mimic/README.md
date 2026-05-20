# 01 — Interpretable models for ICU mortality risk

**Question.** How much accuracy do we trade away when we restrict ourselves to intrinsically interpretable models on a real critical-care dataset, and what do post-hoc explanations add when we don't?

**Answer.** _Coming soon — this project ships in week 2._

**Why it matters.** Clinicians making rapid triage decisions need risk estimates they can interrogate, not just trust.

<!-- ![hero](assets/hero.png) -->

## Method

MIMIC-IV v3.0, first-24h vitals + labs + demographics → 24h in-hospital mortality. Four models compared head-to-head: Decision Tree, Explainable Boosting Machine (EBM), LightGBM, L2 Logistic Regression. Train/test split is **grouped on `subject_id`** to prevent patient leakage. Calibration curves, Brier score, AUROC/AUPRC reported on a held-out test split, alongside one per-model explanation (decision path / shape function / SHAP / coefficient × value).

See [`REPORT.md`](REPORT.md) for the full methodology and results.

## Reproduce

```
just setup
just data    # prints the MIMIC-IV access + cohort instructions; no data committed
just train
just eval
```

## Limitations

- No fairness or slice analysis in v1.0 (deferred to v1.1).
- No counterfactual generation in v1.0 (deferred).
- Findings are illustrative of trade-offs on this cohort; not a clinical claim.
