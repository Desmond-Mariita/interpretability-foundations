# interpretability-foundations

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](./.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue)](./pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

A five-project research portfolio on model interpretability and explanation evaluation.

The portfolio progresses from **prediction → attribution → faithfulness → representation → mechanism**. Its common question is not whether a model can produce a plausible explanation, but what evidence a model uses, what an explanation measures, and when an interpretable representation is causally connected to behaviour.

> **Status.** All five projects are complete, reproducible, and covered by the shared repository quality baseline. The authoritative portfolio summary is in [`docs/PORTFOLIO_SUMMARY.md`](docs/PORTFOLIO_SUMMARY.md).

![hero](projects/01-tabular-mimic/assets/frontier.png)

## Projects

| # | Project | Research question | Final headline result |
|---|---|---|---|
| 1 | [`01-tabular-mimic`](projects/01-tabular-mimic) | How much predictive performance is traded for intrinsic interpretability on ICU mortality risk? | On the held-out test set, EBM reached **AUROC 0.879** versus **0.889** for LightGBM and had the lowest **Brier score 0.075** versus 0.091 for LightGBM. Brier is reported as overall probabilistic error, not a stand-alone calibration claim. |
| 2 | [`02-text-eraser`](projects/02-text-eraser) | Which explainer produces the largest measured perturbation-faithfulness effect, and how does that relate to rationale plausibility? | Under the corrected v2 protocol, **Integrated Gradients** had the largest measured comprehensiveness (**0.205**) and AOPC (**0.168**) among the evaluated methods. Its rationale-overlap AUPRC was **0.339**. Faithfulness and plausibility are reported as distinct constructs. |
| 3 | [`03-multimodal-hatefulmemes`](projects/03-multimodal-hatefulmemes) | How much do image and text contribute to a fused classifier? | Fused AUROC was **0.711**, image-only **0.692**, text-only **0.575**. The paired fused−image AUROC difference was **+0.019 [−0.013, +0.054]**; fused−text was **+0.136 [+0.095, +0.177]**. Mean absolute Shapley contribution was **0.842 image vs 0.675 text**. |
| 4 | [`04-vqa-aokvqa`](projects/04-vqa-aokvqa) | How much does a supplied explanation increase recovery of a model's original answer after visual input is removed? | Incremental answer recoverability under null visual input was **+0.380** for the caption→LLM pipeline, **+0.144** for Qwen2.5-VL-3B, and **+0.486** for Qwen2.5-VL-7B. This is a recoverability measure, not proof of original-answer faithfulness or grounding. |
| 5 | [`05-mechanistic-pythia`](projects/05-mechanistic-pythia) | Does the linearly decoded noun-number direction causally contribute to subject–verb agreement behaviour? | Opposite-number direction patches shifted verb-number preference toward the donor across all 13 causal points (**E = 4.84 → 0.12**). Paired contrasts against same-number and norm-matched random controls were positive at every causal point, supporting direction-specific causal contribution under the tested intervention. |

## Portfolio progression

| Stage | Project | What becomes stronger |
|---|---|---|
| Prediction | P1 | Compare interpretable and black-box predictors without overclaiming calibration. |
| Attribution | P3 | Decompose a multimodal prediction into modality-level contributions. |
| Faithfulness | P2 / P4 | Test explanation behaviour under perturbation or ablation rather than relying on plausibility alone. |
| Representation | P5 v1.0 | Ask what information is linearly decodable across depth. |
| Mechanism | P5 v1.1 | Intervene on a decoded direction and measure the resulting behavioural change. |

The progression is methodological, not a claim that later projects solve every limitation of earlier ones.

## How to read the repository

- **Portfolio reviewers:** start with [`docs/PORTFOLIO_SUMMARY.md`](docs/PORTFOLIO_SUMMARY.md), then the individual project READMEs.
- **Engineers:** start with [`src/awake/`](src/awake/) and [`CONTRIBUTING.md`](CONTRIBUTING.md).
- **Scientific reviewers:** inspect the project REPORTs, ADRs in [`docs/decisions/`](docs/decisions/), and machine-readable metrics artifacts.
- **Reproducers:** run `just setup`, then follow the per-project reproduction instructions.

## Research-engineering baseline

Repository-wide policy is defined in [`CONTRIBUTING.md`](CONTRIBUTING.md). In particular:

- Python 3.11 + `uv.lock` define the supported environment.
- CI runs the full pre-commit suite, Gitleaks, explicit test-marker enforcement, and the unit/smoke suite.
- Shared-library coverage has a hard 90% threshold.
- Every future project must document its research question, estimand, data/model identity, frozen configuration, authoritative metrics artifact, controls, uncertainty procedure, provenance, claim boundary, and limitations.
- Passing tests establish code contracts; they do not by themselves establish scientific truth.

## Data governance

Projects using gated or licensed data are code-only where required. Restricted raw data and derived restricted embeddings are not committed. See the project ADRs in [`docs/decisions/`](docs/decisions/).

## Repository layout

```text
src/awake/             shared evaluation and plotting utilities
projects/              five completed research projects
apps/                  public demo application(s)
docs/decisions/        architecture / scientific decision records
docs/project-template/ required baseline for future projects
legacy/v1/             preserved historical coursework
```

## Scope boundary

This repository is a **completed five-project portfolio**. New research directions should be developed as separate milestones rather than by reopening frozen results without a material defect.

Tutorial videos are a separate presentation layer and are not evidence for the scientific claims in the repository.
