# Changelog

All notable changes to this repository are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses
[SemVer](https://semver.org).

## [Unreleased]

### Added

#### Project 2 — `02-text-eraser`
- **Question.** Which of LIME, Integrated Gradients, Gradient×Input, and SHAP
  PartitionExplainer is most faithful to a fine-tuned sentiment classifier, and do the
  faithful explainers also agree with human rationales (plausibility)?  Benchmarked on
  the ERASER Movies corpus.
- **Pipeline.** `scripts/00_fetch_data.py` (canonical tarball download + sha256
  verification) → `01_prepare.py` (frozen 512-subword visible sequence + truncation
  coverage + comparison-doc dropping) → `10_train.py` (fine-tune with warmup + gradient
  clipping) → `20_explain.py` (attribution cache with checkpoint-hash invalidation) →
  `30_eval.py` (metrics.json + hero scatter).
- **Metric suite.** ERASER-exact comprehensiveness + sufficiency at dataset rationale
  budget `k_d`; AOPC (bins 0–50%); token F1 + AUPRC against human rationale masks;
  paired bootstrap 95% CIs (2 000 resamples); Bonferroni-corrected pairwise tests.
  Mask-replacement erasure (preserve positions, replace with `[MASK]`) documented as a
  deliberate deviation from ERASER literal removal.
- **Explainers.** LIME, Integrated Gradients, Gradient×Input (replaces attention rollout),
  SHAP PartitionExplainer (optional extra `[explain-shap]`), random baseline floor.
- **Model.** `roberta-base` (test accuracy 0.925). The spec named `deberta-v3-base`, but
  DeBERTa-v3 diverges to NaN under this env's transformers 5.9 / torch 2.12 / CUDA 13
  stack (library bug); RoBERTa trains cleanly and nothing in the method is
  DeBERTa-specific. See ADR 002.
- **Results.** On 199 ERASER Movies test reviews: Integrated Gradients is the only faithful
  explainer (comprehensiveness 0.52, AOPC 0.34 vs. ~0.02–0.06 for the others; p < 0.001),
  while all explainers barely exceed the random floor on plausibility (AUPRC 0.30–0.33) —
  faithfulness and plausibility do not coincide. See `projects/02-text-eraser/metrics.json`.
- **Documentation.** `projects/02-text-eraser/REPORT.md` (9-section methodology +
  limitations + references); `projects/02-text-eraser/README.md` (question / method /
  reproduce / limitations); `notebooks/01-explainer-comparison.py` (jupytext
  py:percent source — faithfulness table, hero scatter, token heatmaps).
- **ADR.** `docs/decisions/002-eraser-data-and-truncation.md` — five decisions:
  code-only ERASER download, frozen visible-sequence truncation contract,
  mask-replacement erasure, `shap` optional extra with `numba>=0.59`, and the
  RoBERTa-base model substitution.
- Repo stays v0.x, unpromoted per §15.

## [0.1.0] — 2026-05-20

First milestone release. Project 1 complete end-to-end on the real MIMIC-IV
cohort, with reproducible CI and the engineering scaffolding the remaining
four projects will plug into. Per spec §15, the repo remains unpromoted at
v0.x: no pinned-repo flag, no external announcement until v1.0.0.

### Added

#### Repository scaffolding
- `src/awake/` shared library (`utils.seeding`, `viz.style`, `eval/`
  placeholder), five `projects/` directories, `apps/hatefulmemes-space/`
  Gradio Space, CI + deploy-space workflows.
- `legacy/v1/awake-main/` — verbatim 2023 MSc coursework, preserved frozen.
- Code-quality rules: ruff D rules with `convention="google"`, `pytest-cov`
  with `--cov-fail-under=90` on `src/awake/`. Mirrors keragita
  INV-012 / INV-013.
- `docs/decisions/001-mimic-code-only.md` — data-governance ADR with the
  bound safety-sweep pattern list (§15 references this file).
- `.gitleaks.toml` — allowlists `uv.lock` and `legacy/` paths from the
  default rules (lockfile hashes trigger a Square-token false positive
  on entropy).

#### Project 1 — `01-tabular-mimic`
- **Data pipeline.** DuckDB-driven cohort + feature + split scripts on
  MIMIC-IV v3.1 (`hosp/` + `icu/` `.csv.gz` files read in place). Cohort
  filters mirror MIT-LCP `firstday_*` concepts: adult, first ICU stay per
  admission, ≥24h LOS or in-window death. 68,770 stays from 54,964
  subjects; 13.03% in-hospital mortality. Patient-level leakage prevented
  by a subject-grouped 15% test holdout + 5-fold `GroupKFold` on the
  remainder.
- **Models.** L2 Logistic, Decision Tree (depth-5), EBM (10 interactions),
  LightGBM. One default config per family; CV used for honest performance
  estimation, not selection. Class imbalance handled per-family
  (`class_weight='balanced'` for LR / DT; `scale_pos_weight≈6.7` for
  LightGBM; EBM defaults).
- **Results.** EBM lands within 1.0 AUROC point of LightGBM (0.879 vs
  0.889 on the test set) and beats it on Brier (0.075 vs 0.091). The
  intrinsically interpretable L2 Logistic and Decision Tree pay 4 and 10
  AUROC points respectively.
- **Artifacts** at the project root: `cohort_stats.json`, `metrics.json`,
  `feature_agreement.csv`. Hero figures in `assets/`:
  `frontier.png`, `calibration.png`, `roc_curves.png`.
- **Per-model explanations** in `notebooks/01-model-explanations.html`
  (pre-rendered): coefficient bar (LR), depth-5 `plot_tree` + decision
  path (DT), shape functions for top-6 features (EBM), gain importance
  + TreeSHAP waterfall (LightGBM).
- **REPORT.md** — 9-section methodology + results + limitations writeup.

#### Tooling
- `jupyter`, `jupytext`, `nbconvert` added to dev deps.
- `just notebook` recipe: jupytext `.py` → `.ipynb` → execute →
  `.html` → `nbstripout`.
- `lightgbm`, `interpret-core` added to runtime deps. (Note: `interpret`
  pulls a `shap[numba[llvmlite-0.36]]` cascade that breaks on Python
  3.11; `interpret-core` exposes the same `ExplainableBoostingClassifier`
  without the cascade.)

### Pre-promotion status
Per spec §15, the repo is publicly visible at v0.1.0 but **not pinned, not
promoted, not announced**. Pinning waits for v1.0.0 and the full §15
acceptance checklist.
