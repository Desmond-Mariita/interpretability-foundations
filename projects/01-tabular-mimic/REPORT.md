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
- LOS in ICU ≥ 24 h, **or** an in-hospital death inside the first 24 h
  (so the first-24 h feature window is always well-defined).

| | |
|---|---|
| N stays | 68,770 |
| N unique subjects | 54,964 |
| In-hospital mortality | 13.03 % |
| Median age (IQR) | 67 (55–78) |
| Gender split | 56.5 % M, 43.5 % F |
| Median LOS in ICU | 2.35 days |

Full manifest: [`cohort_stats.json`](cohort_stats.json).

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
- 5-fold `GroupKFold` on the remaining 85 % (58,573 stays).
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
| LightGBM | 0.875 ± 0.004 | **0.889** | **0.632** | 0.091 | 0.302 |
| EBM | 0.868 ± 0.003 | 0.879 | 0.608 | **0.075** | 0.247 |
| L2 Logistic | 0.842 ± 0.003 | 0.850 | 0.515 | 0.159 | 0.473 |
| Decision Tree | 0.784 ± 0.004 | 0.788 | 0.411 | 0.184 | 0.527 |

Full per-fold breakdown in [`metrics.json`](metrics.json).

### 6.2 What the numbers say

- **EBM is within 1.0 AUROC point of LightGBM** and within 2.4 AUPRC
  points. The accuracy cost of going glassbox at the additive-model tier
  is small on this task.
- **EBM has the best Brier score of any of the four.** LightGBM's higher
  Brier despite higher AUROC is the calibration signature of
  `scale_pos_weight` — the rankings are right, the magnitudes are off.
- **L2 Logistic loses 4 AUROC points to EBM**, almost entirely because it
  cannot model interactions across the 18 vital / lab signals. Doubling
  features wouldn't close that gap; non-linearity would.
- **Decision Tree at depth 5 underfits.** The shallow tree was chosen so
  the *decision path* stays human-readable; that constraint costs ~10
  AUROC points relative to LightGBM. A "fairer" comparison would be a
  shallow EBM vs. a shallow tree, but the spec's framing is "how much do
  intrinsically interpretable models cost," and a depth-5 tree *is* what
  reads as intrinsically interpretable in practice.

### 6.3 Calibration

See [`assets/calibration.png`](assets/calibration.png).

- **EBM** tracks the diagonal closely across all 10 bins.
- **LightGBM** is mildly under-confident at the high end (predicts 0.95 →
  empirical ~0.88), which is the expected signature of `scale_pos_weight`.
- **L2 Logistic** and **Decision Tree** both over-predict mortality at
  high predicted probabilities (predict 0.95 → empirical ~0.72). This is
  the cost of `class_weight='balanced'` for a downstream user who reads
  the predicted probability as a risk; their *rank ordering* is fine.

Calibration matters more than AUROC for this kind of bedside-decision
framing, which is why this project reports both.

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
- **One configuration per family.** Light tuning could move EBM and
  LightGBM by another 0.5–1 AUROC each, but the spec deliberately framed
  this as a *comparison at sensible defaults* rather than a tuning race.
- **`class_weight='balanced'` worsens calibration on LR / DT.** Documented
  here rather than papered over; a v1.1 follow-up could compare against
  post-hoc Platt scaling or isotonic regression.
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
