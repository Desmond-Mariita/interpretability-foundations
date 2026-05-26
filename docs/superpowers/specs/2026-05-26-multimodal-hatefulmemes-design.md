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
hateful), `dev.jsonl` (**500; balanced 250/250** — the file's last line has no trailing
newline, so `wc -l` reports 499), `test.jsonl` (999; **unlabelled**), and `img/`
(10,000 PNGs). Records: `{id, img, label, text}`.

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
- **Public-artifact rule (enforced everywhere):** committed notebook outputs, `REPORT.md`,
  `metrics.json`, hero figures, and the Space contain **no raw HM images or meme text** —
  only aggregate numbers, attribution scores, and IDs. Qualitative "example" displays use
  **synthetic stand-in** image+caption pairs (not real memes); the ≤100-image academic
  exception is not exercised, so no Getty-attributed images are committed.

**Split decision (reconciliation):** the project README assumed phase-2 `dev_seen`/
`dev_unseen`; this release has none. **We report on `dev` (499, balanced).** `test` is
unlabelled → used only for qualitative demo examples, never for metrics.

- `scripts/00_data.py` — verify the dataset at `DATA_PATH` (default `~/.cache/hateful_memes`)
  and, if only the archive is present, extract it there; never downloads, never commits.
  Prints licence-acceptance + access instructions when missing (P1/MIMIC pattern).

## 3. Encode + models (config-driven backbone)

Two CLIP backbones are first-class, selected by config, never sharing a cache:
- `configs/clip_l14.yaml` (headline) and `configs/clip_b32.yaml` (Space), plus
  `configs/train.yaml`. Config keys (the `--config` contract): `clip_model_id`,
  `embedding_subdir` (`outputs/embeddings/<variant>/`); `train.yaml`: `random_state`,
  `n_background`, `background_seed`, `lgbm` params.
- `scripts/01_encode.py --config <variant>` — frozen CLIP encodes image + text →
  `outputs/embeddings/<variant>/{split}.parquet` (`id, label, img_emb, txt_emb`). The
  expensive step; cached once per variant.
- `scripts/10_train.py --config <variant>` — LightGBM **fused** head on `[img_emb|txt_emb]`
  + **image-only** + **text-only** heads. `scale_pos_weight` set from the train ratio
  (~1.79). All hyperparameters fixed via train-only 5-fold CV; **dev is never used for
  selection.** Saves to `outputs/models/<variant>/`.
- `scripts/15_background.py --config <variant>` — sample a seeded **empirical** (unstratified,
  reflects the observed train class prior) background set (default N=200) of train embeddings
  → `outputs/models/<variant>/background.npz`. A class-stratified/balanced background is a
  sensitivity ablation, not the primary. (The Space uses a separate **generic** background, §6.)

## 4. Shared core — `src/awake/eval/`

### `modality_shapley.py` (pure, unit-tested) — split exact math from value estimation
```python
def shapley_2player(v_empty: float, v_a: float, v_b: float, v_ab: float) -> tuple[float, float]:
    """Exact 2-player Shapley: phi_a = ½[(v_a - v_empty)+(v_ab - v_b)], phi_b symmetric."""

def interventional_values(
    img_emb: np.ndarray, txt_emb: np.ndarray,          # the actual example (D,) each
    value_fn: Callable[[np.ndarray], np.ndarray],      # (M, 2D) batch -> (M,) LOGITS
    img_background: np.ndarray, txt_background: np.ndarray,  # (N, D) each, row-paired
) -> tuple[float, float, float, float]:
    """Exact interventional coalition values, deterministic (no sampling).

    With background rows j = 1..N (paired: img_background[j] goes with txt_background[j]):
      v(img,txt) = value_fn([img_emb, txt_emb])                      # point score, NOT averaged
      v(img)     = mean_j value_fn([img_emb,        txt_background[j]])  # absent text from bg
      v(txt)     = mean_j value_fn([img_background[j], txt_emb])        # absent image from bg
      v(∅)       = mean_j value_fn([img_background[j], txt_background[j]])  # paired bg rows
    Each mean is a single vectorised value_fn call on a tiled (N, 2D) batch — so the whole
    attribution is ~4 batched LightGBM calls per example, not N python calls.
    """

def modality_shapley(...) -> dict[str, float]:
    """Compose interventional_values + shapley_2player: {'image': phi_img, 'text': phi_txt}."""
```
- **Value function is the LightGBM raw margin (log-odds of the hateful class)**, not the
  probability — Shapley additivity holds on the unbounded margin; the sigmoid would compress
  contributions near 0/1 and bias the modality share. The Space maps back to probability for
  display only.
- **`v({img,txt})` is the point score on the actual example (never background-averaged);**
  only absent modalities are replaced. The estimator is **deterministic** (averages over all
  N background rows, paired) — there is **no `rng`** in the public API, so results are
  exactly reproducible given a fixed background set.
- **Estimand: interventional (background-averaged), empirical-train background as primary**
  (reflects the observed train distribution). The single mean-embedding baseline and a
  class-stratified background are **ablations**; `metrics.json` reports the gap. This is
  *interventional reliance* — not a causal claim.
- Tests: `shapley_2player` efficiency (φ_img+φ_txt = v_ab−v_empty) + symmetry on a toy game;
  an **end-to-end** test that feeds a known linear `value_fn` through `interventional_values`
  → `shapley_2player` and checks the sum against an independently computed v_ab−v_empty;
  a check that `v({img,txt})` equals the direct `value_fn` call (not the background mean).

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

- `scripts/11_eval.py --config <variant>` — for fused / image-only / text-only on **dev**:
  AUROC, **AUPRC**, accuracy@0.5 (dev is balanced), each with a **bootstrap 95% CI** (2,000
  resamples), plus **paired bootstrap 95% CIs on the fused−unimodal AUROC differences**. CIs
  are reported regardless of whether they straddle zero; **no post-hoc significance threshold**
  is applied (at n=500 the AUROC-difference half-width is ~±0.03–0.05, so a small true gap may
  be inconclusive — stated, not hidden). `metrics.json` schema:
  `{"split":"dev","n":500,"models":{<name>:{"auroc":{m,lo,hi},"auprc":{...},"acc":{...}}},
  "auroc_diffs":{"fused_vs_image":{mean,lo,hi},"fused_vs_text":{...}},
  "attribution":{...},"background":{"type":"empirical_train","n":200,"seed":...}}`.
- `scripts/20_attribute.py --config <variant>` — per-example (φ_image, φ_text) via
  `modality_shapley` over cached dev embeddings + the fused head's raw-margin `value_fn` and
  the empirical background. **Full-set aggregate over all dev examples is the headline**;
  correctness- and gold-label-stratified breakdowns reported separately (clearly labelled).
  `metrics.json` always logs raw `φ_image`, `φ_text`, and `total = v_ab − v_∅` per example.
  Hero figure plots the **signed image share** `s = φ_image / (|φ_image| + |φ_text| + ε)`
  (ε = 1e-9), ∈ [−1, 1]; examples with `|φ_image|+|φ_text| < ε` (negligible total
  attribution) are excluded from the share plot and counted in a footnote, but their raw φ
  still appear in `metrics.json`.

## 6. Gradio Space (`apps/hatefulmemes-space`)

Replace the stub `predict()` with the real model, **licence-safe**:
- A **cached lazy loader** (`_load()` with a module-level `None` guard, called on first
  `predict`, *not* at import) loads **CLIP-ViT-B/32** + the **LightGBM B/32 head from the
  user's HF Model Hub** (`hf_hub_download`) + the generic background. Lazy + injectable so the
  smoke test stubs it and never downloads CLIP. A **sidecar `head_meta.json`** (CLIP model id,
  processor id, embedding dims, class map `1=hateful`, lightgbm version, config + commit hash)
  is downloaded with the head and **asserted to match** before inference.
- Interventional background = a **committed generic, non-Hateful-Memes set** (~50 CC0/public-
  domain images + generic captions) **pre-encoded to `generic_background.npz`** (~50×512
  float32, well under the 2 MB pre-commit limit) by `scripts/build_generic_background.py`; the
  Space loads the `.npz` (it does not re-encode raw images at runtime). No HM-derived data.
- Output: label, class confidence (probability, mapped from the margin), **modality bars**
  (Shapley on the margin), and **top tokens** (`awake.eval.text_occlusion`; its `score_fn`
  re-joins occluded tokens with spaces before the CLIP tokenizer).
- `requirements.txt`: pin **`torch==2.x+cpu`** via the CPU index, `transformers`,
  `lightgbm`, `huggingface_hub`, `gradio`, `pillow`.
- `deploy-space.yml`: vendor `awake.eval.modality_shapley` + `awake.eval.text_occlusion`
  (+ `awake.viz`) instead of `plausibility.py`; assemble the subtree **without committing
  `_vendored/` onto `main`** (build in a scratch worktree). Deploy on push to `main`;
  secrets `HF_TOKEN`/`HF_USER`/`HF_SPACE` set by the user. **UI + README disclaimer:** the
  demo uses B/32 **and a generic background**, so its attributions are illustrative and not
  numerically comparable to the L/14 headline in `REPORT.md`.

`just export-space-artifacts` runs the B/32 encode→train→head, and uploads the head to the
HF Model Hub (user-owned) — it does **not** copy HM-derived embeddings into the repo.

## 7. Tests

- **unit** (`src/awake/eval/`, carry the 90% floor): `shapley_2player` efficiency/symmetry/
  toy; `interventional_values` determinism on tiny arrays; `text_occlusion` on a toy scorer.
- **smoke** (CPU, **no gated data, no CLIP**): LightGBM head + modality_shapley on tiny
  **random** embeddings; the Gradio test **injects a stub loader** (monkeypatches `_load`)
  so `build_interface()` + `predict()` run a shape check on a dummy image without any
  `hf_hub_download` or CLIP download. Smoke never imports the real dataset.
- **slow** (excluded from CI): real CLIP-L/14 encode + full train/eval/attribute on GPU.

## 8. Deps + governance

Reuse `torch`/`transformers`/`lightgbm`; add `pillow`, `huggingface_hub` (already transitive
via transformers). CLIP via HF `CLIPModel`/`CLIPProcessor`. **ADR
`docs/decisions/003-hateful-memes-licence-and-modality-shapley.md`** (renumbered from the
README's `004` to close the 003 gap; the README link is updated to 003) covering: the HM
licence analysis (model publishable, dataset/embeddings not), the modality-game-over-SHAP
choice justified **primarily by parsimony** (the research question is modality-level, two
numbers per example) — dimensionality makes raw-feature SHAP impractical as a cross-check,
and group-SHAP over the two modalities is a valid alternative that we simply don't pursue;
**not** justified by "additivity pathologies" (the original framing overstated Janzing
et al.). Plus the interventional baseline + its off-manifold caveat and the logit
value-function choice. CHANGELOG `[Unreleased]` P3 entry; repo stays v0.x.

## 9. Limitations

Frozen backbone (no CLIP fine-tune); modality-level granularity (no image regions); Space
uses B/32 on CPU and does **not** reproduce the L/14 headline; interventional/factorized
backgrounds create off-manifold image+text pairs that a tree model scores at arbitrary
leaves (inherent to marginal Shapley on trees) — disclosed, with the mean-baseline ablation
as a cross-check; the empirical background carries the train class prior (~64% benign), so
`v(∅)` and absolute φ magnitudes are prior-dependent — relative image-vs-text *share* is the
robust quantity (the balanced-background ablation quantifies the shift); probability
calibration is not guaranteed for the unimodal heads (margin/Shapley results are unaffected,
but displayed probabilities are indicative); dev is the only labelled eval split here (500,
balanced); single dataset/backbone, no generalisation claim.

## 10. Repository layout (new/changed)

```
src/awake/eval/modality_shapley.py        # NEW (pure, unit-tested)
src/awake/eval/text_occlusion.py          # NEW (pure, unit-tested)
src/awake/eval/__init__.py                # export the two
projects/03-multimodal-hatefulmemes/
  configs/{clip_l14,clip_b32,train}.yaml
  scripts/{00_data,01_encode,10_train,15_background,11_eval,20_attribute}.py
  scripts/build_generic_background.py       # CC0 images+captions -> generic_background.npz
  scripts/_paths.py _models.py _stub.py
  justfile                                  # data/train/eval/attribute/export-space-artifacts
  tests/{conftest,test_*_smoke}.py
  notebooks/01-modality-attribution.ipynb   # committed WITH outputs (no raw HM content)
  REPORT.md  README.md  (assets/, outputs/ gitignored)
apps/hatefulmemes-space/app.py            # real model, lazy loader, licence-safe
apps/hatefulmemes-space/generic_background.npz   # committed, non-HM
apps/hatefulmemes-space/requirements.txt  # +torch(cpu index)/transformers/lightgbm/hf_hub
.github/workflows/deploy-space.yml        # vendor modality_shapley+text_occlusion; build subtree in a worktree (no _vendored on main)
.gitignore                                # + apps/hatefulmemes-space/_vendored/
docs/decisions/003-hateful-memes-licence-and-modality-shapley.md
projects/03-multimodal-hatefulmemes/README.md  # updated: dev (not dev-unseen), ADR 003, parsimony framing
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

### v2 re-review (second round) — additional fixes folded in
- Estimator pinned exactly: `v({img,txt})` = point score; absent modalities background-
  averaged; `v(∅)` over paired rows; deterministic (no `rng`); `value_fn` vectorised
  `(M,2D)→(M,)` ← Codex (BLOCKER) + methodology (B-1/B-2) + Gemini (perf).
- Public-artifact rule: no raw HM images/text in committed outputs/REPORT/Space; synthetic
  qualitative examples ← Codex (BLOCKER).
- Empirical train background = primary (was class-stratified); prior caveat in §9 ← Codex +
  methodology (M-1).
- Per-metric bootstrap CIs + explicit `metrics.json` schema; "report CIs regardless, no
  post-hoc significance" ← Codex + methodology (M-3).
- Hero share formula `φ_img/(|φ_img|+|φ_txt|+ε)` with exclusion footnote + raw-φ logging ←
  methodology (M-2) + Codex.
- Head/CLIP sidecar metadata + assert; lazy/stubbable Space loader; generic background
  pre-encoded `.npz` + build script; `_vendored` gitignore + worktree deploy ← Codex +
  engineering.
- dev count corrected 499→500 (newline-less last line) ← Codex.
- Soften dimensionality claim (parsimony primary; group-SHAP a valid unused alternative);
  drop "marginal/interventional" conflation ← methodology + Codex.
- Stale P3 README (dev-unseen, ADR 004, old Janzing framing) to be updated by the plan's
  docs task ← engineering.
