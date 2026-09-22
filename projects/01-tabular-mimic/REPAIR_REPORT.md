# P1 Repair Report — Scientific Integrity and Presentation

**Branch:** `fix/p1-mimic-scientific-integrity`
**Date:** 2026-09-22
**Scope:** Audit and repair of `projects/01-tabular-mimic` only. No retraining
performed.

---

## 1. Audit findings

| Concern | Verdict | Evidence |
|---|---|---|
| Brier treated as calibration | **CONFIRMED (wording)** | `README.md` and repo-root `README.md`: "best-calibrated … by Brier score"; `REPORT.md` §6.2: "calibration signature" |
| Class weighting confounds probabilities | **PARTIAL** | LR/DT use `class_weight='balanced'`, LightGBM uses `scale_pos_weight ≈ 6.7` (`configs/models.yaml`, `scripts/_models.py`); no post-hoc calibration anywhere; disclosed in Limitations but ignored by the headline claim |
| Reliability wording too strong / mislabeled | **CONFIRMED (one mislabel)** | Curves reconstructed from `assets/calibration.png`: LightGBM sits *below* the diagonal everywhere (top bin: predicted ≈ 0.93, empirical ≈ 0.84 → **over**-prediction) yet `REPORT.md` §6.3 said "under-confident"; "tracks the diagonal closely across all 10 bins" overstates EBM (bins 6–8 deviate ≈ ±0.05–0.12); LR/DT "over-predict at the high end" is correct (≈0.91 → 0.69/0.73) |
| "24 h" framing | **PARTIAL (wording only)** | Construction is sound (see §2); no "prospective" claims existed, but the temporal setup was never stated explicitly and "the first-24h feature window is always well-defined" was imprecise for early deaths |
| Log-loss mismatch | **CONFIRMED** | `REPORT.md` §6.1 log-loss column (0.302 / 0.247 / 0.473 / 0.527) matches **no** value in `metrics.json` (test: 0.290 / 0.251 / 0.485 / 0.546; CV means differ as well) |
| Fold SD presented as CI | **NOT CONFIRMED** | No "confidence"/"CI" language found anywhere in P1; the `±` values were unlabeled rather than mislabeled |

Additional defects found and repaired:

- `README.md` contained two duplicate "Reproduce" and two duplicate
  "Limitations" sections with divergent content.
- `REPORT.md` §6.2: "L2 Logistic loses **4** AUROC points to EBM" — the
  test-set gap is 2.9 points (~3).
- `REPORT.md` §4 said the train pool is 58,573 stays; the modeling frame
  trains on 58,572 (the feature inner join drops one cohort stay).
- `metrics.json` `cohort_size` (68,769) is the modeling-frame size, not the
  cohort size (68,770); now explained in the docs.
- The cohort silently excludes stays that leave the ICU within 24 h and die
  later in hospital; now documented as a limitation.
- No ECE or summary calibration metric exists; §6.3 now says so explicitly
  instead of implying stronger calibration evidence.

## 2. Temporal-validity assessment

Exact construction, traced from `scripts/00_build_cohort.py` and
`scripts/01_extract_features.py`:

```text
raw MIMIC CSVs
    → first ICU stay per hadm_id (ROW_NUMBER() OVER (PARTITION BY hadm_id))
    → adult (≥18), and (LOS ≥ 24 h OR deathtime ≤ intime + 24 h)
    → features: vitals (chartevents, join stay_id) and labs
      (labevents, join subject_id + hadm_id), charttime ∈ [intime, intime + 24 h)
    → target = admissions.hospital_expire_flag (any in-hospital death,
      unbounded horizon)
    → subject-grouped 15% test holdout + 5-fold GroupKFold
    → model input
```

1. One sample = one ICU stay (first ICU stay of a hospital admission).
2. Unit is the stay; 68,770 stays from 54,964 subjects, so subjects can
   contribute multiple stays.
3. Splits are grouped on `subject_id` — verified in code
   (`scripts/02_make_splits.py`) and by the smoke-test leakage assertion.
4. Anchor: ICU `intime`.
5. Latest feature time: `intime + 24 h`, or earlier for stays that end
   earlier (death or ICU discharge).
6. Outcome: `hospital_expire_flag` (in-hospital mortality during that
   admission).
7. Death **can** occur before the window ends; those stays are explicitly
   included (early-death clause) and their observation window is truncated
   at the death time.
8. No feature is ever recorded after the outcome: early deaths have no
   post-death measurements, and `deathtime`/`outtime`/`los_icu_days` are
   excluded from the modeling frame (`scripts/_data.py` `COHORT_DROP_COLS`).
9. Public framing is "predict in-hospital mortality from first-24h ICU
   features", which the construction supports as a **retrospective**
   experiment. It would not support a prospective hour-24 deployment claim;
   the docs now say so (§2.1 of `REPORT.md`, README "Temporal setup").

**Verdict:** no temporal leakage, no post-outcome features, no immortal-time
exclusion of early deaths. The issue was framing/wording, now repaired.

## 3. Split / leakage assessment

- Grouping key: `subject_id` (test holdout and GroupKFold both at subject
  level). No subject appears in more than one split — enforced in code and
  asserted in `tests/test_cohort_smoke.py`.
- Preprocessing (median imputation, standardisation, one-hot/ordinal
  encoding) is fit inside each CV fold's training partition only
  (`scripts/_models.py` pipelines, `scripts/10_train.py` `_cv_scores`).
- No feature engineering uses labels or held-out statistics.
- No model selection touches the test set: CV is for estimation, the final
  refit uses the full train pool, and the test set is scored once.
- Test-set size 10,197 stays (8,245 subjects) — consistent with
  `metrics.json` (`n_test`) and the docs.

**Verdict:** subject-grouping correctly enforced; this is now surfaced
explicitly in the README as a strength.

## 4. Calibration / Brier assessment

- Brier scores (test): LightGBM 0.091, EBM 0.075, LR 0.159, DT 0.184 —
  unchanged; the values were always correct.
- Class weighting: LR/DT `class_weight='balanced'`; LightGBM
  `scale_pos_weight` computed from the training pool (≈6.7); EBM none.
- No post-hoc calibration was performed on any model.
- The reliability diagram (10 bins, ≈1,020 test stays per bin) is the only
  direct calibration evidence; no ECE was computed. The reconstructed
  curves show: EBM closest to the diagonal (modest mid-range deviations);
  LightGBM below the diagonal across the range (over-prediction, widest at
  the high end); LR/DT strongly over-predict at the high end.
- Wording repaired: Brier is presented as overall probabilistic error, not
  calibration; "best-calibrated" claims removed; the LightGBM
  "under-confident" mislabel corrected to over-prediction.

## 5. Metric inconsistencies found

| Location | Old value | Authoritative value (metrics.json, test) | Action |
|---|---|---|---|
| `REPORT.md` §6.1 log-loss LightGBM | 0.302 | 0.290 | Corrected |
| `REPORT.md` §6.1 log-loss EBM | 0.247 | 0.251 | Corrected |
| `REPORT.md` §6.1 log-loss L2 Logistic | 0.473 | 0.485 | Corrected |
| `REPORT.md` §6.1 log-loss Decision Tree | 0.527 | 0.546 | Corrected |
| `REPORT.md` §6.2 "4 AUROC points to EBM" | 4 | 2.9 (≈3) | Corrected |
| `REPORT.md` §4 train-pool stays | 58,573 | 58,572 (modeling frame) | Clarified |

The stale log-loss values match neither the test nor the CV values in
`metrics.json`; they predate the final run recorded in the artifact. The
prose now quotes the authoritative artifact; a regression test
(`tests/test_claims_consistency.py`) pins every headline number to
`metrics.json`.

## 6. Uncertainty-language assessment

Only fold mean ± SD exists (5 GroupKFold folds, `ddof=1`). No bootstrap CIs
or significance tests exist. The `±` values are now labeled in both
`README.md` and `REPORT.md` as "standard deviation across the 5 CV folds —
descriptive fold-to-fold variability, not a confidence interval", and a
limitation bullet states no inferential statistics were computed. No CI was
fabricated.

## 7. Claims withdrawn / narrowed

- ~~"the EBM is the best-calibrated of the four models by Brier score"~~
  → "the EBM achieves the lowest Brier score … and tracks the reliability
  diagonal most closely of the four" (qualitative, figure-based).
- ~~"LightGBM is mildly under-confident at the high end"~~ → LightGBM
  over-predicts across the range (below-diagonal curve), widest at the high
  end, the expected signature of `scale_pos_weight`.
- ~~"EBM tracks the diagonal closely across all 10 bins"~~ → "closely in
  most bins, with modest deviations in the mid-range".
- ~~Implicit "well-defined 24 h window" for early deaths~~ → early deaths
  are included with the observation window truncated at death.
- Brier is no longer offered as stand-alone proof of calibration.

## 8. Authoritative metrics

`projects/01-tabular-mimic/metrics.json` is the authoritative
machine-readable source (test-set values):

| Model | AUROC | AUPRC | Brier | Log-loss |
|---|---|---|---|---|
| LightGBM | 0.889 | 0.632 | 0.091 | 0.290 |
| EBM | 0.879 | 0.608 | 0.075 | 0.251 |
| L2 Logistic | 0.850 | 0.515 | 0.159 | 0.485 |
| Decision Tree | 0.788 | 0.411 | 0.184 | 0.546 |

CV AUROC: LightGBM 0.875 ± 0.004, EBM 0.868 ± 0.003, LR 0.842 ± 0.003,
DT 0.784 ± 0.004 (fold SD). Cohort: 68,770 stays / 54,964 subjects /
13.0% in-hospital mortality; modeling frame 68,769 stays (58,572 train /
10,197 test). No metric value was changed; only prose was corrected.

## 9. Provenance status

- `metrics.json`, `cohort_stats.json`, `feature_agreement.csv`, and the
  three `assets/*.png` figures were committed together with the original
  REPORT in commit `3afdbbe` ("P1 modeling: 4-way comparison + headline
  figures + REPORT") and are internally consistent.
- The stale REPORT log-loss column matches nothing in the artifact and
  predates the final run; its exact origin could not be traced (no
  surviving outputs/ or MLflow runs on disk).
- MIMIC data, model pickles, and per-stay predictions are not committed
  (code-only policy, ADR 001). The calibration figures therefore could not
  be regenerated; the corrected §6.3 numbers were reconstructed from the
  committed `calibration.png` itself and are quoted as approximations.

## 10. Files changed

- `projects/01-tabular-mimic/README.md` — claim repairs, probability
  caveat, temporal-setup statement, `±` labeling, de-duplicated
  Reproduce/Limitations, early-transfer limitation.
- `projects/01-tabular-mimic/REPORT.md` — log-loss column corrected to
  `metrics.json`; §2.1 temporal setup added; §4 count clarification; §6.2 /
  §6.3 calibration wording repaired; limitations extended.
- `README.md` (repo root) — P1 portfolio row: "best-calibrated" → "lowest
  Brier" (one cell; the only root-doc edit).
- `projects/01-tabular-mimic/tests/test_claims_consistency.py` — new
  consistency guards (see §11).
- `projects/01-tabular-mimic/REPAIR_REPORT.md` — this report.

No code, config, figure, notebook, or artifact files were changed.

## 11. Tests

New `tests/test_claims_consistency.py` (6 unit tests, no data required):

- README headline table equals `metrics.json` (rounded to 3 dp).
- REPORT headline table (incl. log-loss) equals `metrics.json`.
- `metrics.json` schema invariants (per-fold blocks, test blocks,
  `n_train + n_test == cohort_size`, metric ranges).
- Doc stay counts agree with `cohort_stats.json` / `metrics.json`.
- Claim-language guards: no "best-calibrated" / "under-confident" in the
  P1 docs or the repo-root README; required repaired wording present.
- No duplicated README sections.

Existing smoke tests (subject-disjoint split, cohort filters, model-family
fits) still pass unchanged.

## 12. Rerun required?

**No.** The experiment itself was valid; the defects were interpretive and
presentational. `metrics.json` was already authoritative and internally
consistent, so no pipeline step was re-run. Regenerating predictions or ECE
would require the MIMIC data and full retraining; skipped deliberately so
the committed artifact remains the single source of truth.

## 13. Unresolved limitations

- No ECE/summary calibration metric (would need surviving predictions or a
  rerun); calibration evidence remains qualitative (reliability diagram).
- The cohort excludes stays that left the ICU within 24 h and died later
  in hospital (documented, not changed).
- One stay with zero in-window measurements is dropped from the modeling
  frame (documented, not investigated).
- Fold SD only — no inferential statistics (documented, not added).
- Raw probabilities are biased by imbalance handling; no post-hoc
  calibration (documented, deferred to v1.1).

## 14. Branch / commit / PR

- Branch: `fix/p1-mimic-scientific-integrity` (based on `origin/main`).
- Draft PR opened against `main`; not merged.
- See the PR description for the full final-report block.
