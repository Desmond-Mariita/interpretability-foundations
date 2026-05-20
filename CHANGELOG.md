# Changelog

All notable changes to this repository are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses
[SemVer](https://semver.org).

## [Unreleased]

### Added
- Repository skeleton: `src/awake/` shared library, five `projects/` directories,
  `apps/hatefulmemes-space/` Gradio scaffold, CI workflows, ADR-001.
- `legacy/v1/awake-main/` — verbatim 2023 MSc coursework, preserved frozen.
- Code-quality rules: ruff D rules with `convention="google"`, `pytest-cov`
  with `--cov-fail-under=90`. Mirrors keragita INV-012 / INV-013.
- **Project 1 — `01-tabular-mimic` v0.1.0.** Cohort + feature + split
  pipeline on MIMIC-IV v3.1 (68,770 stays, 13.0% in-hospital mortality);
  four-model comparison (L2 Logistic, Decision Tree, EBM, LightGBM) with
  subject-grouped 5-fold CV + 15% test holdout. EBM lands within 1.0
  AUROC point of LightGBM with the best Brier score. README, REPORT,
  metrics, feature-agreement table, and three headline figures
  (`frontier.png`, `calibration.png`, `roc_curves.png`).
