---
title: Modality Attribution Demo
emoji: 🪞
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# Modality attribution demo

Live demo for Project 3 of [`interpretability-foundations`](https://github.com/Desmond-Mariita/interpretability-foundations).
Upload a meme image + caption; the app encodes both with **CLIP-ViT-B/32**, classifies the
concatenated embedding with a LightGBM head, and reports the prediction, class confidence,
**per-modality (image vs text) contribution** via a 2-player interventional Shapley game, and
the top caption tokens via leave-one-out occlusion.

## Required configuration

This Space loads its trained head from the HuggingFace **Model Hub** at runtime (the Hateful
Memes dataset and its embeddings are never redistributed — only the model is, per the dataset
licence). Set this **Space variable** (Settings → Variables and secrets):

- `HF_MODEL_REPO` — your public Model Hub repo holding `fused.txt` + `head_meta.json`
  (e.g. `your-username/p3-hatefulmemes-head`). Keep the repo public so no token is needed.

## Caveats

- The demo uses **CLIP-ViT-B/32** and a **generic (non-Hateful-Memes) background** for the
  Shapley baseline, so its attribution numbers are **illustrative and not numerically
  comparable** to the CLIP-ViT-L/14 headline in the project's `REPORT.md`.
- Built by the repo's `deploy-space` GitHub Action, which vendors `awake.eval.modality_shapley`
  + `awake.eval.text_occlusion` into `_vendored/`.

See [`projects/03-multimodal-hatefulmemes/`](../../projects/03-multimodal-hatefulmemes) for the full method.
