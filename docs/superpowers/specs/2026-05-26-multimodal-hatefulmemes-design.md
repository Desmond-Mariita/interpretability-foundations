# Project 3 — `03-multimodal-hatefulmemes`: modality attribution for fused image+text classification

**Status:** approved design (post four-way review + data/licence verification), 2026-05-26
**Author:** Desmond Mariita · **Repo:** `interpretability-foundations` · stays v0.x

> Revised after a four-way design review (internal methodology + engineering subagents;
> external Codex + Gemini) and after verifying the actual dataset on disk. §11 records
> review traceability.

## 1. Question and deliverable

When a fused image+text classifier calls a meme hateful or benign, how much of the signal
came from the image vs. the text? Deliverables: `metrics.json` (AUROC/AUPRC/accuracy for
fused + image-only + text-only on **dev**, with bootstrap CIs), per-example + aggregate
modality attribution, a hero figure, a notebook (committed **with outputs**), `REPORT.md`,
and a live Gradio HuggingFace Space (CLIP-ViT-B/32).

## 2. Data — verified on disk (code-only, licence-bound)

The dataset is the **Meta Hateful Memes** corpus (Getty-sourced images). Verified the local
copy (`~/Downloads/archive.zip`, a Kaggle mirror): it is the **phase-1 release**
(`LICENSE.txt` dated 2020-06-14) with `train.jsonl` (8,499; labels 5,450 benign / 3,050
hateful), `dev.jsonl` (499; **balanced 250/250**), `test.jsonl` (999; **unlabelled**), and
`img/` (10,000 PNGs). Records: `{id, img, label, text}`.

**Licence (Facebook HM Dataset Agreement) governs everything and applies regardless of the
Kaggle origin.** Key clauses: §6.1 no derivative works; §6.2 no distributing/copying/hosting
the dataset (≤100 images only, with Getty attribution, for academic publication); §6.5 no
incorporating it into another product; **§2 the Participant retains IP in models/algorithms
derived from use, usable academically and commercially.** Consequences encoded throughout:
- The dataset and any **CLIP embeddings of the memes are never committed or hosted** (they
  are derivative works of the dataset).
- The **trained LightGBM head and CLIP weights may be published** (§2 model carve-out).
- The HF Space therefore loads the head from the user's HF Model Hub and uses a **generic,
  non-Hateful-Memes background** for its interventional baseline (no HM-derived artifact
  goes public). See ADR 003.

**Split decision (reconciliation):** the project README assumed phase-2 `dev_seen`/
`dev_unseen`; this release has none. **We report on `dev` (499, balanced).** `test` is
unlabelled → used only for qualitative demo examples, never for metrics.

- `scripts/00_data.py` — verify the dataset at `DATA_PATH` (default `~/.cache/hateful_memes`)
  and, if only the archive is present, extract it there; never downloads, never commits.
  Prints licence-acceptance + access instructions when missing (P1/MIMIC pattern).

## 3. Encode + models (config-driven backbone)

Two CLIP backbones are first-class, selected by config, never sharing a cache:
- `configs/clip_l14.yaml` (headline) and `configs/clip_b32.yaml` (Space). Each sets the
  CLIP model id and an embedding subdir `outputs/embeddings/<variant>/`.
- `scripts/01_encode.py --config <variant>` — frozen CLIP encodes image + text →
  `outputs/embeddings/<variant>/{split}.parquet` (`id, label, img_emb, txt_emb`). The
  expensive step; cached once per variant.
- `scripts/10_train.py --config <variant>` — LightGBM **fused** head on `[img_emb|txt_emb]`
  + **image-only** + **text-only** heads. `scale_pos_weight` set from the train ratio
  (~1.79). All hyperparameters fixed via train-only 5-fold CV; **dev is never used for
  selection.** Saves to `outputs/models/<variant>/`.
- `scripts/15_background.py --config <variant>` — sample a seeded, class-stratified
  background set (default N=200) of train embeddings → `outputs/models/<variant>/background.npz`
  (used by attribution; for the Space, a **generic** background is built separately, §6).

## 4. Shared core — `src/awake/eval/`

### `modality_shapley.py` (pure, unit-tested) — split exact math from value estimation
```python
def shapley_2player(v_empty: float, v_a: float, v_b: float, v_ab: float) -> tuple[float, float]:
    """Exact 2-player Shapley: phi_a = ½[(v_a - v_empty)+(v_ab - v_b)], phi_b symmetric."""

def interventional_values(
    img_emb: np.ndarray, txt_emb: np.ndarray,
    value_fn: Callable[[np.ndarray], float],   # concat([img,txt]) -> scalar (LOGIT)
    img_background: np.ndarray, txt_background: np.ndarray,  # (N, D)
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Background-averaged v(∅), v({img}), v({txt}), v({img,txt}). Absent modality is
    replaced by each background row and averaged (marginal/interventional)."""

def modality_shapley(...) -> dict[str, float]:
    """Compose the two: returns {'image': phi_img, 'text': phi_txt}."""
```
- **Value function is the LightGBM raw margin (log-odds of the hateful class)**, not the
  probability — Shapley additivity holds on the unbounded margin; the sigmoid would compress
  contributions near 0/1 and bias the modality share. The Space maps back to probability for
  display only.
- **Primary estimand: interventional background-averaged** (the single headline). The
  single mean-embedding baseline is an **ablation**; `metrics.json` reports the gap. `v(∅)`
  = both modalities drawn from background. No "causal" language — this is marginal/
  interventional reliance.
- RNG injected; tests: efficiency (φ_img+φ_txt = v_ab−v_empty), symmetry, a hand-computed
  toy game, and determinism under a fixed seed.

### `text_occlusion.py` (pure, unit-tested)
```python
def occlusion_importance(
    tokens: list[str], score_fn: Callable[[list[str]], float], top_k: int = 5,
) -> list[tuple[str, float]]:
    """Leave-one-out: importance[i] = score(all) - score(all without token i)."""
```
Used by the notebook and the Space for "top tokens." Pure (takes a `score_fn`), unit-tested
on a toy additive scorer.

Both modules are added to `src/awake/eval/__init__.py`'s exports.

## 5. Eval + attribution

- `scripts/11_eval.py --config <variant>` — AUROC, **AUPRC**, accuracy@0.5 (dev is
  balanced) for fused / image-only / text-only on **dev**. **Paired bootstrap 95% CIs
  (2,000 resamples) on the fused−unimodal AUROC differences** (n=499 is small; point
  estimates alone are insufficient). dev-seen/confound framing dropped (not in this release).
- `scripts/20_attribute.py --config <variant>` — per-example (φ_image, φ_text) via
  `modality_shapley` over cached dev embeddings + the fused head's raw-margin `value_fn` and
  the seeded background. **Primary aggregate: all dev examples**; secondary stratifications
  by correctness and by gold label reported separately (clearly labelled). `metrics.json` +
  hero figure (image-vs-text signed-share distribution, faceted by label/correctness).

## 6. Gradio Space (`apps/hatefulmemes-space`)

Replace the stub `predict()` with the real model, **licence-safe**:
- Load **CLIP-ViT-B/32** + the **LightGBM B/32 head pulled from the user's HF Model Hub**
  (`hf_hub_download`) at module import (one-time cold-start; CPU). Nothing HM-derived is
  committed to the public Space.
- Interventional background = a **small generic, non-Hateful-Memes image+text set** shipped
  with the Space (public-domain/synthetic), so the Shapley baseline carries no dataset
  derivative.
- Output: label, class confidence (probability, mapped from the margin), **modality bars**
  (Shapley on the margin), and **top tokens** (`awake.eval.text_occlusion`).
- `requirements.txt`: pin **`torch==2.x+cpu`** via the CPU index, `transformers`,
  `lightgbm`, `huggingface_hub`, `gradio`, `pillow`.
- `deploy-space.yml`: vendor `awake.eval.modality_shapley` + `awake.eval.text_occlusion`
  (+ `awake.viz`) instead of `plausibility.py`; assemble the subtree **without committing
  `_vendored/` onto `main`** (build in a scratch worktree). Deploy on push to `main`;
  secrets `HF_TOKEN`/`HF_USER`/`HF_SPACE` set by the user. **UI note:** the demo uses B/32
  and is not the L/14 headline.

`just export-space-artifacts` runs the B/32 encode→train→head, and uploads the head to the
HF Model Hub (user-owned) — it does **not** copy HM-derived embeddings into the repo.

## 7. Tests

- **unit** (`src/awake/eval/`, carry the 90% floor): `shapley_2player` efficiency/symmetry/
  toy; `interventional_values` determinism on tiny arrays; `text_occlusion` on a toy scorer.
- **smoke** (CPU, **no gated data, no CLIP**): LightGBM head + modality_shapley on tiny
  **random** embeddings; Gradio `build_interface()` + `predict()` shape check on a dummy
  image with a stubbed model. Smoke must never import the real dataset or download CLIP.
- **slow** (excluded from CI): real CLIP-L/14 encode + full train/eval/attribute on GPU.

## 8. Deps + governance

Reuse `torch`/`transformers`/`lightgbm`; add `pillow`, `huggingface_hub` (already transitive
via transformers). CLIP via HF `CLIPModel`/`CLIPProcessor`. **ADR
`docs/decisions/003-hateful-memes-licence-and-modality-shapley.md`** (renumbered from the
README's `004` to close the 003 gap; the README link is updated to 003) covering: the HM
licence analysis (model publishable, dataset/embeddings not), the modality-game-over-SHAP
choice **justified by parsimony + dimensionality (2^D coalitions intractable), not by
"additivity pathologies"**, the interventional baseline + its off-manifold caveat, and the
logit value-function choice. CHANGELOG `[Unreleased]` P3 entry; repo stays v0.x.

## 9. Limitations

Frozen backbone (no CLIP fine-tune); modality-level granularity (no image regions); Space
uses B/32 on CPU and does **not** reproduce the L/14 headline; interventional/factorized
backgrounds create off-manifold image+text pairs that a tree model scores at arbitrary
leaves (inherent to marginal Shapley on trees) — disclosed, with the mean-baseline ablation
as a cross-check; dev is the only labelled eval split here (499, balanced); single
dataset/backbone, no generalisation claim.

## 10. Repository layout (new/changed)

```
src/awake/eval/modality_shapley.py        # NEW (pure, unit-tested)
src/awake/eval/text_occlusion.py          # NEW (pure, unit-tested)
src/awake/eval/__init__.py                # export the two
projects/03-multimodal-hatefulmemes/
  configs/{clip_l14,clip_b32,train}.yaml
  scripts/{00_data,01_encode,10_train,15_background,11_eval,20_attribute}.py
  scripts/_paths.py _models.py _stub.py
  tests/{conftest,test_*_smoke}.py
  notebooks/01-modality-attribution.py     # committed WITH outputs
  REPORT.md  README.md  (assets/, outputs/ gitignored)
apps/hatefulmemes-space/app.py            # real model, licence-safe
apps/hatefulmemes-space/requirements.txt  # +torch(cpu)/transformers/lightgbm/hf_hub
.github/workflows/deploy-space.yml        # vendor modality_shapley+text_occlusion; no _vendored on main
docs/decisions/003-hateful-memes-licence-and-modality-shapley.md
```

## 11. Review traceability

- Logit value function ← all 4 reviewers (BLOCKER).
- Config-driven L/14 vs B/32 + `export-space-artifacts` recipe ← engineering + Gemini
  (BLOCKER) + Codex.
- No-leakage protocol (train-CV selection, dev = final eval) + bootstrap CIs on
  fused−unimodal ← methodology + Codex.
- Interventional = primary estimand, mean = ablation, off-manifold caveat, drop causal
  wording, define v(∅) ← methodology (BLOCKER) + Codex + Gemini.
- Janzing reframe (parsimony/dimensionality, not additivity pathologies) ← methodology +
  Codex.
- text_occlusion built properly with tests ← user decision (reviewers flagged scope).
- Licence-safe Space (model from HF Hub, generic background, nothing HM-derived committed)
  ← Codex (BLOCKER) + on-disk licence verification; user asked to confirm — confirmed.
- Pure `modality_shapley` API with injected RNG + math/estimator split; torch CPU pin +
  import-time model load; smoke never touches gated data; seeding config;
  `awake.eval.__init__` export; `00_data` is verify+extract; ADR renumber 004→003 ←
  engineering + Codex.
- Notebook committed WITH outputs ← user preference (overrides one reviewer's note).
- Report on `dev` (not dev-unseen); test unlabelled ← on-disk data verification.
