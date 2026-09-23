# Changelog

All notable changes to this repository are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses
[SemVer](https://semver.org).

## [Unreleased]

No unreleased scientific changes. Tutorial videos are a separate presentation layer and do not alter the frozen project evidence.

## [1.0.0] — 2026-09-23

Final portfolio release: five completed projects, corrected scientific claims, reproducible quality gates, and a shared research-engineering standard.

### Portfolio closeout

- Reconciled root-level claims with the final frozen P1–P5 evidence.
- Added `docs/PORTFOLIO_SUMMARY.md` as the authoritative cross-project summary.
- Added publication-consistency guards so superseded root-level claims do not silently return.
- Established the repository-wide quality baseline in PR #6: frozen dependency sync, full pre-commit CI, Gitleaks, explicit test markers, scheduled quality checks, and future-project standards.
- Superseded pre-repair root summaries are intentionally omitted from the release changelog; project repair reports and Git history preserve that development history.
- Promoted the repository from the earlier v0.x milestone state to the completed five-project portfolio.

### Final project state

- **P1 — Tabular MIMIC:** EBM test AUROC 0.879 vs LightGBM 0.889; EBM Brier 0.075 vs 0.091. Brier is overall probabilistic error, not stand-alone calibration evidence.
- **P2 — ERASER:** corrected v2 protocol is authoritative. Integrated Gradients has the largest measured perturbation-faithfulness effect among evaluated methods (comprehensiveness 0.205; AOPC 0.168; rationale-overlap AUPRC 0.339).
- **P3 — Hateful Memes:** fused/image/text AUROC 0.711/0.692/0.575. Paired fused−image difference +0.019 [−0.013,+0.054]; fused−text +0.136 [+0.095,+0.177]. Image has the larger average absolute Shapley contribution under the specified game.
- **P4 — A-OKVQA:** Delta is incremental answer recoverability from the supplied explanation under null visual input: A/B/B7 = 0.380/0.144/0.486. It is not a grounding, correctness, or original-answer-faithfulness metric.
- **P5 — Pythia:** the decoded noun-number direction shows direction-specific causal contribution under the tested intervention (E 4.84 → 0.12 across causal depth). No uniqueness, necessity, complete mediation, full grammar-circuit, or grammatical-understanding claim is made.

### Final engineering state

- P1–P5 merged and frozen.
- All repository tests explicitly classified as unit, smoke, or slow.
- Unit/smoke CI passes on the merged five-project portfolio.
- Shared-library coverage remains above the 90% gate.
- `CONTRIBUTING.md` and `docs/project-template/` define the mandatory baseline for future projects.

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
