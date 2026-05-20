# 01 — Interpretable models for ICU mortality risk

**Question.** How much accuracy do we trade away when we restrict ourselves to intrinsically interpretable models on a real critical-care dataset, and what do post-hoc explanations add when we don't?

**Answer.** On 68,770 ICU stays from MIMIC-IV, **the glassbox EBM lands within 1.0 AUROC point of LightGBM** (0.879 vs. 0.889 on a held-out test set of 10,197 stays) — and **the EBM is the best-calibrated of the four models** by Brier score (0.075 vs. 0.091 for LightGBM). The intrinsically interpretable L2 Logistic and Decision Tree pay a meaningful accuracy cost (0.850 and 0.788 respectively) but for very different reasons: LR can't model interactions, the depth-5 tree can't fit the signal density.

**Why it matters.** Clinicians making rapid triage decisions need risk estimates they can interrogate, not just trust.

![hero](assets/frontier.png)

## Headline numbers

| Model | CV AUROC | Test AUROC | Test AUPRC | Brier | Interpretability |
|---|---|---|---|---|---|
| LightGBM | 0.875 ± 0.004 | **0.889** | **0.632** | 0.091 | Post-hoc (TreeSHAP) |
| EBM | 0.868 ± 0.003 | 0.879 | 0.608 | **0.075** | Glassbox (shape funcs) |
| L2 Logistic | 0.842 ± 0.003 | 0.850 | 0.515 | 0.159 | Intrinsic (coefficients) |
| Decision Tree | 0.784 ± 0.004 | 0.788 | 0.411 | 0.184 | Intrinsic (decision path) |

Calibration tells the second story — see [`assets/calibration.png`](assets/calibration.png). EBM tracks the reliability diagonal closely; LightGBM is mildly under-confident; the class-balanced LR and DT systematically *over*-predict mortality at the high end, which inflates their Brier despite reasonable AUROC.

## Top-feature agreement

The four models broadly agree on what matters at 24h — see [`feature_agreement.csv`](feature_agreement.csv) for the full ranks across all four. The biggest cross-model agreement is **first careunit** (top-5 in all four), **age**, the **BUN** family (kidney function), **admission type**, and **bicarbonate** (acid-base status). Clinically plausible.

## Method

MIMIC-IV v3.1 (spec named v3.0; v3.1 is forward-compatible). Cohort: adult patients on their first ICU stay per hospital admission with either ≥24h of ICU LOS or an in-hospital death inside the first 24h (N = 68,770 stays from 54,964 subjects; 13.0% mortality base rate). Features: vitals from `chartevents` and labs from `labevents`, both filtered to the first 24h after `intime`, aggregated as first / min / max / mean. Itemid lists and physiologic clipping bounds mirror the MIT-LCP `firstday_*` concept views.

Train/test split is **grouped on `subject_id`** to prevent patient leakage: 15% of subjects held out for test (10,197 stays), with 5-fold `GroupKFold` CV on the remainder.

See [`REPORT.md`](REPORT.md) for the full methodology and discussion, [`cohort_stats.json`](cohort_stats.json) for the cohort manifest, and [`metrics.json`](metrics.json) for the per-fold and per-model numbers.

## Per-model explanations

Each model family is read with the explanation form that fits it natively — see [`notebooks/01-model-explanations.html`](notebooks/01-model-explanations.html) (pre-rendered, opens in any browser):

- **L2 Logistic** — standardised coefficient bar chart.
- **Decision Tree** — full depth-5 `plot_tree` + the decision path for one held-out positive.
- **EBM** — shape functions (Δ log-odds vs. feature value) for the top six terms.
- **LightGBM** — gain importance for the global view + a TreeSHAP waterfall for the same held-out example.

`.ipynb` source is in [`notebooks/01-model-explanations.ipynb`](notebooks/01-model-explanations.ipynb); re-run with `just notebook`.

## Reproduce

Requires PhysioNet MIMIC-IV credentials and the data at `~/Physionet/mimic-iv-3.1/` (override via `MIMIC_PATH`).

```
just setup        # from the repo root
just data         # verifies MIMIC modules; downloads nothing
just cohort       # ~2 s   — builds cohort.parquet + cohort_stats.json
just features     # ~4 min — first-24h vitals + labs from chartevents/labevents
just splits       # ~1 s   — subject-grouped 5-fold CV + 15% test holdout
just train        # ~5 min — trains all four models (EBM dominates the runtime)
just eval         # ~3 s   — writes assets/{calibration,roc_curves,frontier}.png and feature_agreement.csv
```

## Limitations

- No fairness or slice analysis in v1.0 (deferred to v1.1).
- No counterfactual generation in v1.0 (deferred).
- The `class_weight='balanced'` policy on LR/DT is honest about handling imbalance but worsens their calibration. A v1.1 follow-up could compare with cost-sensitive thresholds or Platt scaling.
- Findings are illustrative of trade-offs on this cohort; not a clinical claim.

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
