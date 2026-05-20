# 03 — Modality attribution for fused image+text classification

**Question.** When a multimodal classifier gets something right or wrong, how much of the signal came from the image and how much from the text?

**Answer.** _Coming soon — this project ships in week 4 and powers the live demo._

**Why it matters.** Multimodal models blend image and text features in ways that aren't easily inspected; modality attribution gives a first cut at where the decision came from.

<!-- ![hero](assets/hero.png) -->

## Method

Frozen CLIP-ViT-L/14 encodes image and text. A LightGBM head trains on concatenated embeddings. Unimodal baselines (image-only, text-only) are reported alongside. Modality attribution is computed as a **2-player Shapley game** over the (image-features, text-features) modality groups — not per-dimension SHAP aggregated to modalities — to avoid the additivity pathologies under correlated features documented by Janzing et al. (2020). See [`docs/decisions/004-modality-shapley-over-shap-aggregation.md`](../../docs/decisions/004-modality-shapley-over-shap-aggregation.md).

Reported on the **dev-unseen** split (dev-seen has documented confounds where unimodal text reaches ~58–60% AUROC).

## Reproduce

```
just setup
just data    # documents Hateful Memes access steps
just train
just eval
```

## Limitations

- No end-to-end CLIP fine-tuning; backbone is frozen by design.
- No image-region-level attribution; granularity stops at the modality.
- The HuggingFace Space uses **CLIP-ViT-B/32** (CPU latency budget). Reproduce locally with L/14 for the headline numbers.
