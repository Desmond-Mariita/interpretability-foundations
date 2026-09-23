# Portfolio summary

This document is the authoritative cross-project summary for the completed `interpretability-foundations` portfolio. Project-level READMEs, REPORTs, ADRs, and machine-readable metrics remain authoritative for implementation details.

## Methodological progression

The portfolio progresses from **prediction → attribution → faithfulness → representation → mechanism**. The stages are connected by a shared concern: whether an interpretable output reflects the evidence or internal representation that actually matters to model behaviour.

| Project | Stage | Data / model | Estimand or target quantity | Final result | Claim boundary |
|---|---|---|---|---|---|
| P1 — Tabular MIMIC | Prediction | MIMIC-IV; LR, DT, EBM, LightGBM | Held-out predictive performance and probabilistic error | EBM AUROC 0.879 vs LightGBM 0.889; EBM Brier 0.075 vs 0.091 | Brier is not treated as stand-alone calibration evidence; findings are cohort-specific, not clinical claims |
| P2 — ERASER | Faithfulness / plausibility | ERASER Movies; RoBERTa classifier | Perturbation effect under a corrected visible-word masking protocol; rationale overlap measured separately | Integrated Gradients: comprehensiveness 0.205, AOPC 0.168, AUPRC 0.339 | Largest measured perturbation-faithfulness effect among evaluated methods; no generic claim that an explainer is universally faithful |
| P3 — Hateful Memes | Attribution | Meta Hateful Memes; frozen CLIP-ViT-L/14 + LightGBM | Predictive AUROC plus 2-player interventional modality Shapley | Fused/image/text AUROC 0.711/0.692/0.575; fused−image +0.019 [−0.013,+0.054]; fused−text +0.136 [+0.095,+0.177]; mean|phi| image/text 0.842/0.675 | Image has larger average absolute contribution under this Shapley setup; paired fused−image interval includes zero |
| P4 — A-OKVQA | Behavioural explanation probe | A-OKVQA; caption→LLM and Qwen2.5-VL 3B/7B pipelines | Incremental answer recoverability from a supplied explanation under null visual input | Delta A/B/B7 = 0.380/0.144/0.486 | Not a measure of original-answer faithfulness, grounding, hidden reasoning, correctness, or a controlled capacity effect |
| P5 — Pythia | Representation → mechanism | Pythia-160M; UD English-EWT | v1.0 layerwise probe selectivity; v1.1 donor-directed logit shift after replacing the projection on the decoded noun-number direction | Primary causal effect E = 4.84 at embedding → 0.12 at block_11; paired number−same and number−random contrasts positive at every causal point | Supports direction-specific causal contribution under this intervention; not uniqueness, necessity, complete mediation, a full grammar circuit, or grammatical understanding |

## Authoritative artifacts

| Project | Primary human-readable source | Machine-readable evidence |
|---|---|---|
| P1 | `projects/01-tabular-mimic/README.md`, `REPORT.md` | `projects/01-tabular-mimic/metrics.json` |
| P2 | `projects/02-text-eraser/README.md`, `REPORT.md`, `REPAIR_REPORT.md` | `projects/02-text-eraser/metrics.json` |
| P3 | `projects/03-multimodal-hatefulmemes/README.md`, `REPORT.md` | `projects/03-multimodal-hatefulmemes/metrics.json` |
| P4 | `projects/04-vqa-aokvqa/README.md`, `REPORT.md` | `projects/04-vqa-aokvqa/assets/metrics_snapshot.json` |
| P5 | `projects/05-mechanistic-pythia/README.md`, `V11_CAUSAL_REPORT.md` | `projects/05-mechanistic-pythia/assets/causal_metrics.json` |

## Shared standards

- Keep estimands explicit.
- Keep plausibility separate from behavioural or causal faithfulness.
- Decodability is not evidence of causal use.
- A behavioural intervention is not automatically evidence for a unique hidden mechanism.
- Confidence-interval overlap is not used as a hypothesis test.
- Nonsignificance is not interpreted as equivalence.
- Tests validate software and analysis contracts; they do not certify scientific truth.
- Frozen scientific artifacts are preserved byte-for-byte unless a material defect requires a new version.

## Reproduction and governance

`CONTRIBUTING.md` defines the repository-wide engineering standard. `docs/project-template/` defines the minimum contract for future projects. Restricted data are not committed where licensing or access controls prohibit redistribution.

## Scope

This document summarizes the completed five-project portfolio. Tutorial videos are a separate communication layer and must explain the frozen evidence without changing or extending the scientific claims.
