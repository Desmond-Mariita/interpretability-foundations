# Project 1 — Interpretable models for ICU mortality risk

**Author:** Desmond Mariita.
**Dataset:** MIMIC-IV v3.1 (PhysioNet, credentialed access).
**Status:** v1.0 (week 2 of the `interpretability-foundations` sprint).

---

## 1. Question and framing

How much accuracy do interpretable models cost on a real critical-care
dataset, and what do post-hoc explanations add when an opaque model is
chosen anyway? The classroom version of this story compares an off-the-shelf
black box against an off-the-shelf linear baseline. The version that
actually appears at the bedside has three positions worth taking seriously:

- **Intrinsically interpretable.** Linear models and shallow trees. The
  decision rule is the model.
- **Glassbox additive.** Explainable Boosting Machines (EBM): an additive
  model with per-feature shape functions and a small number of pairwise
  interactions, fittable in seconds-to-minutes.
- **Black box + post-hoc.** Gradient-boosted trees with SHAP or similar.
  Reads as a single risk score; the explanation is computed after the fact.

This project places one representative of each tier on the same MIMIC-IV
ICU mortality task and looks at the cost in accuracy *and* calibration.

## 2. Cohort

Built directly from the PhysioNet CSVs via DuckDB (no Postgres staging).
The cohort SQL mirrors the
[MIT-LCP `mimic-code`](https://github.com/MIT-LCP/mimic-code) `firstday_*`
concept views.

Inclusion filters:

- Adult (≥ 18 at admission, age computed as
  `anchor_age + (admittime.year - anchor_year)`).
- First ICU stay per hospital admission (`ROW_NUMBER() OVER
  (PARTITION BY hadm_id ORDER BY intime) = 1`).
- LOS in ICU ≥ 24 h, **or** an in-hospital death inside the first 24 h —
  early deaths are included so they are not silently dropped from the
  positive class; for those stays the observation window ends at the
  death time.

| | |
|---|---|
| N stays | 68,770 |
| N unique subjects | 54,964 |
| In-hospital mortality | 13.03 % |
| Median age (IQR) | 67 (55–78) |
| Gender split | 56.5 % M, 43.5 % F |
| Median LOS in ICU | 2.35 days |

Full manifest: [`cohort_stats.json`](cohort_stats.json).

### 2.1 Temporal setup

This is a **retrospective** modeling experiment, not a prospective
hour-24 prediction:

- **Anchor.** The ICU admission time (`intime`).
- **Feature window.** `[intime, intime + 24 h)` — every vital and lab value
  is filtered to this window (labs joined on `subject_id + hadm_id` carry
  the same time filter, since `labevents` has no `stay_id`).
- **Outcome.** In-hospital mortality during that admission
  (`hospital_expire_flag`), at any horizon up to discharge. The label is
  retrospective: it is not known at hour 24.
- **Early deaths.** Stays with `deathtime ≤ intime + 24 h` are included as
  positives. They have no measurements after death, so their observation
  window is truncated at the death time and no feature is ever recorded
  after the outcome being predicted.
- **No post-outcome features.** For every stay, all features precede the
  outcome. `deathtime`, `outtime`, and `los_icu_days` are excluded from the
  modeling frame entirely.

The setup therefore supports claims about *predicting in-hospital mortality
from first-24h ICU features in a retrospective cohort*; it does not support
claims about prospective deployment at an hour-24 decision point.

## 3. Features

First-24 h window, vitals from `chartevents` (joined on `stay_id`) and labs
from `labevents` (joined on `subject_id + hadm_id` — `labevents` has no
`stay_id`). Items, units, and physiologic clipping bounds are taken from
the `firstday_vitals` / `firstday_labs` concepts so the engineering choices
are auditable.

- **Vitals (7).** Heart rate, SBP, DBP, MBP, respiratory rate, temperature
  (F → C normalisation), SpO₂. Invasive and non-invasive BP itemids are
  merged into one series per modality.
- **Labs (11).** Hematocrit, hemoglobin, platelet, WBC, BUN, creatinine,
  glucose, sodium, potassium, chloride, bicarbonate.
- **Aggregates.** `first`, `min`, `max`, `mean` per feature within the
  24 h window — 18 features × 4 aggregates = 72 numeric columns.
- **Demographics.** `age`, `gender`, `admission_type`, `first_careunit`.

Total feature matrix: 68,769 rows (one cohort stay had zero in-window
measurements and is dropped by the inner join) × 76 columns pre-encoding.

## 4. Splits

Subject-grouped split, **not** stay-grouped:

- 15 % of *subjects* held out for test (10,197 stays from 8,245 subjects).
- 5-fold `GroupKFold` on the remaining 85 % (58,573 cohort stays; the
  modeling frame trains on 58,572 after the feature inner join drops the
  one stay with zero in-window measurements, see §3).
- Deterministic via `awake.utils.seed_everything(1337)`.

A smoke test in `tests/test_cohort_smoke.py` checks the patient-leakage
invariant on synthetic data: no subject appears in both train and test.

## 5. Models

One default configuration per family. Hyperparameters are picked to be
defensible defaults, not the result of an outer search — the CV is used
for honest performance estimation, not selection.

| Family | Library | Imbalance handling | Native categorical / NaN |
|---|---|---|---|
| L2 Logistic | scikit-learn `LogisticRegression` | `class_weight='balanced'` | One-hot + median impute |
| Decision Tree | scikit-learn `DecisionTreeClassifier(max_depth=5)` | `class_weight='balanced'` | Ordinal encode; NaN passes through |
| EBM | `interpret-core ExplainableBoostingClassifier` (10 pairwise interactions) | None (defaults handle it) | Native |
| LightGBM | `lightgbm.LGBMClassifier` (n_estimators=600, num_leaves=63) | `scale_pos_weight = neg/pos ≈ 6.7` | Native |

Configs in [`configs/models.yaml`](configs/models.yaml).

## 6. Results

### 6.1 Headline metrics

| Model | CV AUROC | Test AUROC | Test AUPRC | Brier | Log-loss |
|---|---|---|---|---|---|
| LightGBM | 0.875 ± 0.004 | **0.889** | **0.632** | 0.091 | 0.290 |
| EBM | 0.868 ± 0.003 | 0.879 | 0.608 | **0.075** | 0.251 |
| L2 Logistic | 0.842 ± 0.003 | 0.850 | 0.515 | 0.159 | 0.485 |
| Decision Tree | 0.784 ± 0.004 | 0.788 | 0.411 | 0.184 | 0.546 |

Test AUROC / AUPRC / Brier / log-loss are the test-set values from
[`metrics.json`](metrics.json). `±` on CV AUROC is the standard deviation
across the 5 CV folds — descriptive fold-to-fold variability, not a
confidence interval.

Full per-fold breakdown in [`metrics.json`](metrics.json).

### 6.2 What the numbers say

- **EBM is within 1.0 AUROC point of LightGBM** and within 2.4 AUPRC
  points. The accuracy cost of going glassbox at the additive-model tier
  is small on this task.
- **EBM achieves the lowest Brier score of any of the four** (0.075 vs
  0.091 for LightGBM), i.e. the best overall squared probabilistic error on
  the test set. Brier score is not a pure calibration measure — it also
  rewards discrimination and matching the base rate — so this is read
  alongside the reliability diagram in §6.3, not as a calibration claim on
  its own. LightGBM's higher Brier despite higher AUROC is consistent with
  `scale_pos_weight` shifting its predicted probabilities upward (visible
  as over-prediction in §6.3): the rankings are right, the probability
  magnitudes are biased.
- **L2 Logistic loses ~3 AUROC points to EBM**, almost entirely because it
  cannot model interactions across the 18 vital / lab signals. Doubling
  features wouldn't close that gap; non-linearity would.
- **Decision Tree at depth 5 underfits.** The shallow tree was chosen so
  the *decision path* stays human-readable; that constraint costs ~10
  AUROC points relative to LightGBM. A "fairer" comparison would be a
  shallow EBM vs. a shallow tree, but the spec's framing is "how much do
  intrinsically interpretable models cost," and a depth-5 tree *is* what
  reads as intrinsically interpretable in practice.

### 6.3 Calibration

See [`assets/calibration.png`](assets/calibration.png). The figure plots,
per model, the mean predicted probability against the empirical positive
rate in 10 probability bins on the test set (≈1,020 stays per bin). No ECE
or other summary calibration metric was computed; the assessment below is
qualitative, from the figure.

- **EBM** tracks the diagonal closely in most bins, with modest deviations
  in the mid-range (bins 6–7 sit ~0.05 above the diagonal, bin 8 ~0.1
  below). It is the closest of the four to the diagonal overall.
- **LightGBM** sits below the diagonal across the whole range — it
  *over*-predicts mortality everywhere, with the widest gap at the high end
  (mean predicted ≈ 0.93 → empirical ≈ 0.84). This upward shift of the
  probabilities is the expected signature of `scale_pos_weight`.
- **L2 Logistic** and **Decision Tree** both over-predict mortality at
  high predicted probabilities (mean predicted ≈ 0.91 → empirical ≈ 0.69
  for LR and ≈ 0.73 for DT). This is the cost of `class_weight='balanced'`
  for a downstream user who reads the predicted probability as a risk;
  their *rank ordering* is fine.

All four sets of probabilities are raw model outputs: LR/DT use
`class_weight='balanced'` and LightGBM uses `scale_pos_weight`, both of
which bias the predicted probabilities, and no post-hoc calibration was
applied. The reliability diagram — not Brier score alone — is the basis
for the calibration statements above.

Both ranking (AUROC/AUPRC) and probability quality (Brier, log-loss, the
reliability diagram) are reported, because a downstream user reading a
predicted probability as a risk needs both.

### 6.4 What the four models agree on

[`feature_agreement.csv`](feature_agreement.csv) shows ranks per model
(1 = most important) for the union of every model's top-10 features.

Strong cross-model agreement on:

- **First careunit** — top-5 in all four. The cohort spans medical,
  surgical, cardiac, and trauma units; baseline mortality varies widely
  across them.
- **Age** — top-3 for EBM and LightGBM.
- **BUN family** (`bun_first`, `bun_max`, `bun_min`) — kidney function as
  a mortality proxy is consistently picked up.
- **Admission type** — emergency vs. elective.
- **Bicarbonate** — acid-base status.

These are all features clinicians would expect at the top, which is a
weak sanity check that the four very different model families are
picking up real signal rather than spurious correlations.

## 7. Limitations

- **No fairness or slice analysis** in v1.0. The spec defers this to v1.1
  to keep the scope honest about what's been measured.
- **No counterfactual generation** (DiCE, etc.) in v1.0; deferred.
- **Fold-to-fold variability only.** The CV `±` values are fold standard
  deviations; no confidence intervals or significance tests were computed.
- **One configuration per family.** Light tuning could move EBM and
  LightGBM by another 0.5–1 AUROC each, but the spec deliberately framed
  this as a *comparison at sensible defaults* rather than a tuning race.
- **`class_weight='balanced'` worsens calibration on LR / DT**, and
  `scale_pos_weight` shifts LightGBM's probabilities upward; no post-hoc
  calibration was applied to any model. Documented here rather than papered
  over; a v1.1 follow-up could compare against post-hoc Platt scaling or
  isotonic regression (evaluated strictly on test).
- **The cohort is not "all ICU admissions".** Stays with <24h of ICU
  observation are included only when an in-hospital death occurred within
  24h of ICU admission; stays that left the ICU within 24h and died later
  in hospital are excluded. This mirrors MIT-LCP firstday-style cohorts but
  is still a selection choice worth keeping in mind when comparing against
  other MIMIC mortality numbers.
- **One stay is dropped** from the modeling frame (the inner join with
  features). It is plausibly an immediate-death case with no in-window
  measurements; not investigated.

## 8. Reproducibility

Deterministic from seed: `seed_everything(1337)` ([
`awake.utils.seed_everything`](../../src/awake/utils/seeding.py)) is
threaded through both `02_make_splits.py` and `10_train.py`. The cohort
SQL is parameterless and produces byte-identical outputs run-to-run.
LightGBM is set to `deterministic=True`.

End-to-end pipeline runtime on the development host (Ryzen / RTX 3090,
DuckDB single-machine):

| Step | Time | Output |
|---|---|---|
| `just cohort` | ~2 s | `cohort.parquet`, `cohort_stats.json` |
| `just features` | ~4 min | `features.parquet` (68,769 × 73) |
| `just splits` | ~1 s | `splits.parquet` |
| `just train` | ~5 min | `metrics.json`, `outputs/models/*.pkl`, MLflow run |
| `just eval` | ~3 s | `assets/{calibration,roc_curves,frontier}.png`, `feature_agreement.csv` |

EBM dominates training time (≈ 3 min of the 5). LightGBM finishes the
CV in 18 s. The data engineering step is bound by the one-shot scan
of `chartevents.csv.gz` (11.6 M relevant events).

## 9. References

- Johnson, A. E. W., Bulgarelli, L., Shen, L., et al. (2023).
  *MIMIC-IV, a freely accessible electronic health record dataset.*
  Scientific Data 10, 1.
- Nori, H., Jenkins, S., Koch, P., Caruana, R. (2019).
  *InterpretML: A Unified Framework for Machine Learning Interpretability.*
  arXiv:1909.09223.
- Lundberg, S. M., Lee, S.-I. (2017).
  *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- Niculescu-Mizil, A., Caruana, R. (2005).
  *Predicting good probabilities with supervised learning.* ICML.
