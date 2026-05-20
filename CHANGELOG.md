# Changelog

All notable changes to this repository are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses
[SemVer](https://semver.org).

## [Unreleased]

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
