# 01 — Interpretable models for ICU mortality risk

**Question.** How much accuracy do we trade away when we restrict ourselves to intrinsically interpretable models on a real critical-care dataset, and what do post-hoc explanations add when we don't?

**Answer.** _Modeling results land later in week 2._ Cohort and feature pipeline are ready: 68,770 first ICU stays from 54,964 adult patients, 13.0% in-hospital mortality base rate, 18 vital + lab features (4 aggregates each over the first 24h window) plus demographics.

**Why it matters.** Clinicians making rapid triage decisions need risk estimates they can interrogate, not just trust.

<!-- ![hero](assets/hero.png) -->

## Method

MIMIC-IV v3.1 (spec named v3.0; v3.1 is forward-compatible). Cohort: adult patients on their first ICU stay per hospital admission with either ≥24h of ICU LOS or an in-hospital death inside the first 24h. Features: vitals from `chartevents` and labs from `labevents`, both filtered to the first 24h after `intime`, aggregated as first / min / max / mean. Itemid lists and physiologic clipping bounds mirror the MIT-LCP `firstday_*` concept views.

Train/test split is **grouped on `subject_id`** to prevent patient leakage: 15% of subjects held out for test (10,197 stays), with 5-fold `GroupKFold` CV on the remainder. Four models compared head-to-head — Decision Tree, Explainable Boosting Machine (EBM), LightGBM, L2 Logistic Regression — with calibration curves, Brier score, AUROC/AUPRC, and one per-model explanation (decision path / shape function / SHAP / coefficient × value).

See [`REPORT.md`](REPORT.md) for the full methodology and results, and [`cohort_stats.json`](cohort_stats.json) for the cohort manifest.

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
