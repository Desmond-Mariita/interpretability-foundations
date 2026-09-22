# Project 3 — Modality attribution for fused image+text classification

**Author:** Desmond Mariita.
**Dataset:** Meta Hateful Memes phase-1 (Meta HM Dataset Agreement; gated, code-only).
**Status:** complete — end-to-end run on CLIP-ViT-L/14 (see §5; metrics in `metrics.json`).
Statistical-repair note: see [REPAIR_REPORT.md](REPAIR_REPORT.md) (2026-09-22) for the
corrections to the paired-comparison estimand and the Shapley share semantics.

---

## 1. Question and framing

When a fused image+text classifier calls a meme hateful or benign, how much of the signal
came from the image vs. the text? This is a modality-level attribution question with a
concrete answer: two numbers per example (φ_image, φ_text) and a population-level
aggregate. The question is actionable — it flags whether failures are image-driven or
text-driven and reveals which modality the model exploits when the two conflict.

We frame this as a **2-player interventional Shapley game** (image embedding and text
embedding as the two players, raw margin as the value function, empirical train background
as the interventional estimand). Three LightGBM heads — fused, image-only, text-only —
are trained on frozen CLIP-ViT-L/14 embeddings. Standard classifier metrics (AUROC, AUPRC,
accuracy) plus per-example modality attribution are reported on **dev** (500 examples,
balanced 250/250).

## 2. Data

**Corpus.** Meta Hateful Memes phase-1 (Kiela et al. 2020). Getty-sourced PNG images
paired with meme captions; binary hateful/benign labels. Verified on disk from a Kaggle
mirror of the phase-1 release (`LICENSE.txt` dated 2020-06-14).

**Splits and counts.**

| Split | N | Labels |
|---|---|---|
| train | 8,499 | 5,450 benign / 3,050 hateful (~64% / ~36%) |
| dev | 500 | 250 benign / 250 hateful (balanced) |
| test | 999 | unlabelled |

The `dev.jsonl` file's last line has no trailing newline, so `wc -l` reports 499; the
actual record count is 500. Each record: `{id, img, label, text}`.

**Licence.** The Meta HM Dataset Agreement (Facebook HM Dataset Agreement) governs
everything and applies regardless of the Kaggle origin.

Key constraints encoded throughout the pipeline:

- **§6.1 / §6.2** — no derivative works; no distributing, copying, or hosting the dataset.
  CLIP embeddings of the memes are derivative works and are therefore never committed or
  hosted. The trained LightGBM head is not a derivative work (§2 model carve-out) and
  may be published.
- **§2** — the Participant retains IP in models and algorithms derived from use; usable
  for academic and commercial purposes.
- **Public-artifact rule:** `metrics.json`, `REPORT.md`, committed notebook outputs,
  and the HuggingFace Space contain no raw HM images or meme text — only aggregate
  numbers, attribution scores, and identifiers. Qualitative "example" displays use
  **synthetic stand-in** image+caption pairs (not real memes).

See [ADR 003](../../docs/decisions/003-hateful-memes-licence-and-modality-shapley.md)
for the full licence analysis.

**Data access.** `scripts/00_data.py` verifies the dataset at `DATA_PATH`
(default `~/.cache/hateful_memes`) and, if only the archive is present, extracts it
there. It never downloads and never commits anything. Running `just data` prints
licence-acceptance and access instructions when the dataset is absent.

## 3. Model

**Backbone.** Frozen **CLIP-ViT-L/14** (headline; `openai/clip-vit-large-patch14`).
Image and text are each encoded to 1024-dimensional normalised embeddings; no CLIP weights
are updated. `scripts/01_encode.py --config configs/clip_l14.yaml` writes
`outputs/embeddings/clip_l14/{train,dev,test}.parquet` (`id, label, img_emb, txt_emb`).

The HuggingFace Space uses **CLIP-ViT-B/32** (512-dimensional; CPU latency budget). B/32
is a separate config (`configs/clip_b32.yaml`); it writes to a separate embedding
subdirectory and trains a separate head. B/32 results are **not numerically comparable**
to the L/14 headline reported here.

**Classifier heads.** Three LightGBM (`lgb.LGBMClassifier`) heads trained by
`scripts/10_train.py --config configs/clip_l14.yaml`:

| Head | Input | Dimension |
|---|---|---|
| Fused | `[img_emb \| txt_emb]` concatenated | 2048 |
| Image-only | `img_emb` | 1024 |
| Text-only | `txt_emb` | 1024 |

Hyperparameters are **fixed in `configs/train.yaml`** (400 trees, lr 0.05, 31 leaves) and
identical across the three heads; no hyperparameter search or cross-validation was run.
Dev is never used for selection; it is the final evaluation split only. Class imbalance
is handled via `scale_pos_weight` set from the train label ratio (~1.79). Models are
saved to `outputs/models/clip_l14/`.

**Background.** `scripts/15_background.py` samples a seeded empirical background of N = 200
unstratified train rows → `outputs/models/clip_l14/background.npz`. The background seed
and N are set in `configs/train.yaml`. Mean-embedding and balanced-background sensitivity
analyses are specified in ADR 003 but **have not been executed** (future work; they would
require a re-run of the pipeline).

## 4. Metric definitions

Shared pure metric code lives in `src/awake/eval/`; the project-specific corrected
estimands live in `scripts/_stats.py` (P3-local, unit-tested).

### 4.1 Classifier metrics (per head, on dev)

- **AUROC** — area under the ROC curve (threshold-free ranking metric; chance = 0.5).
- **AUPRC** — area under the precision–recall curve (accounts for class imbalance;
  dev is balanced, but AUPRC is still informative for the operating-point distribution).
- **Accuracy at 0.5** — binary accuracy at the default probability threshold; dev is
  balanced, so the chance baseline is 0.5.
- All three metrics reported with **bootstrap 95% confidence intervals** (2,000 resamples,
  percentile method, fixed seed).
- **Paired AUROC differences (fused − unimodal)** are the intended comparison. The
  corrected estimand (`scripts/_stats.py::paired_auroc_bootstrap`) resamples **example
  indices as paired units**, recomputes the fused AUROC and the unimodal AUROC **inside
  each resample**, and records their difference; the 95% CI is the percentile interval
  of that paired-difference distribution. A resample whose label vector contains a
  single class is degenerate (AUROC undefined) and is counted in `n_valid`-terms and
  skipped — never silently coerced. Significance is read from the paired CI, not from
  overlap of the marginal CIs (overlapping marginal CIs are not a test).
- **Provenance caveat.** The original full run did not cache per-example dev scores, so
  the paired bootstrap CIs cannot be recomputed from the committed artifacts. The exact
  **point differences** are derivable from the committed AUROC means (+0.019 fused−image;
  +0.136 fused−text) and are recorded in `metrics.json`; the paired CIs are `null` until
  `just eval` is re-run. The previous `auroc_diffs` entries were paired mean differences
  over per-example *probability* arrays (mislabeled as AUROC differences); they are
  preserved under `metrics.json → provenance → removed_prob_array_diffs` for traceability
  and are used in no claim.

### 4.2 2-player interventional modality Shapley

**Players.** Image embedding (φ_image) and text embedding (φ_text).

**Value function.** The **LightGBM raw margin** (log-odds of the hateful class,
`predict(raw_score=True)`). Shapley additivity holds on the unbounded margin; the sigmoid
probability would compress contributions near 0 and 1 and bias the modality share.
Probability is mapped back from the margin for display only.

**Coalition values.** Let `img_emb` and `txt_emb` be the embeddings of the example being
attributed, and let `(img_bg[j], txt_bg[j])` for j = 1…N be the paired background rows:

```
v({img,txt}) = value_fn([img_emb,     txt_emb])                 # point score, not averaged
v({img})     = (1/N) sum_j value_fn([img_emb,     txt_bg[j]])   # absent text replaced from bg
v({txt})     = (1/N) sum_j value_fn([img_bg[j],   txt_emb])     # absent image replaced from bg
v(empty)     = (1/N) sum_j value_fn([img_bg[j],   txt_bg[j]])   # paired bg rows
```

Each mean is a single vectorised `value_fn` call over an (N, 2D) batch — four batched
LightGBM calls per example in total.

**Shapley formula (exact, 2-player):**

```
phi_image = 0.5 * [(v({img}) - v(empty)) + (v({img,txt}) - v({txt}))]
phi_text  = 0.5 * [(v({txt}) - v(empty)) + (v({img,txt}) - v({img}))]
```

Efficiency: `phi_image + phi_text = v({img,txt}) - v(empty)`.

**Background.** Primary: empirical train background (N = 200, seeded). The primary
reflects the observed train prior (~64% benign), so `v(∅)` and absolute φ magnitudes are
prior-dependent. Per-example shares normalise magnitudes. Mean-embedding and balanced
(class-stratified) background sensitivity analyses are specified in ADR 003 but have
**not been run** — no ablation numbers exist and none are claimed.

**Off-manifold caveat.** Factorised (image, text) pairs produced by background replacement
are out-of-distribution for the fused LightGBM head: a real image embedding is paired
with a text embedding from a different example. Tree models route these inputs to whatever
leaf they reach — inherent to marginal Shapley on trees.

**Estimand.** This is **interventional reliance** — a measure of how much the model
relies on each modality when the other is marginalised out. It is not a causal claim,
not a claim of semantic complementarity, and not a claim about what the model "really
understands".

**Per-example quantities (sign and magnitude reported separately).**

- **Signed contribution** `phi_image`, `phi_text` — direction of each modality's push on
  the raw margin. The sign is a *direction*, not a dominance measure: `phi_image = −10`,
  `phi_text = +1` means the image dominates in **magnitude** while its signed share is
  negative.
- **Signed image share** `s = φ_image / (|φ_image| + |φ_text| + ε)`, ε = 1e-9, in [−1, 1].
  Summarises *direction* across examples. Defined as 0 when the total magnitude is
  negligible (direction undefined).
- **Image magnitude share** `m = |φ_image| / (|φ_image| + |φ_text|)`, in [0, 1], with
  `m = 0.5` when both contributions are ~0 (no modality dominates). This is the
  dominance-appropriate quantity: `m > 0.5` means the image carries the larger absolute
  contribution on that example.
- **Interaction** `I = v({img,txt}) − v({img}) − v({txt}) + v(∅)` — the non-additivity
  of the margin under the chosen interventional game, and nothing more. It is not
  "semantic synergy" and not causal interaction.

The hero figure shows both histograms (signed share; magnitude share) when generated by
the repaired `scripts/20_attribute.py`. Per-example rows (phi, shares, interaction) are
written by `just attribute` to `outputs/attribution/<variant>/dev_attribution.json` (a
local run artifact); only aggregates enter the committed `metrics.json`.

## 5. Results

Run on the balanced 500-example dev split (frozen CLIP-ViT-L/14, all numbers from
`metrics.json`).

### 5.1 Classifier metrics (dev, n = 500, balanced)

| Head | AUROC (95% CI) | AUPRC | Accuracy |
|---|---|---|---|
| Fused (CLIP-L/14) | 0.711 [0.669, 0.756] | 0.672 | 0.606 |
| Image-only | 0.692 [0.648, 0.736] | 0.691 | 0.622 |
| Text-only | 0.575 [0.524, 0.626] | 0.566 | 0.548 |

Text-only is barely above chance; image-only carries most of the signal. The exact AUROC
point differences are **fused − image = +0.019** and **fused − text = +0.136**
(`metrics.json → auroc_diffs → point_diff`). The paired bootstrap CIs for these
differences are `null` in the committed file (per-example scores were not cached) and
populate on re-run. The marginal CIs overlap for fused vs image-only; **this overlap is
not a significance test**, so no claim about whether fusion "significantly" beats the
image head is made here. (The previous `auroc_diffs` entries were paired mean
differences over probability arrays — a different quantity — and have been removed from
the results; see §4.1 and REPAIR_REPORT.md.)

### 5.2 Modality attribution (dev aggregate)

Mean |φ| (raw-margin interventional Shapley, empirical train background, N=200):
**image 0.842 vs text 0.675** — on average the image modality carries the larger
*absolute* margin contribution, consistent with the unimodal AUROCs. The aggregate
magnitude share of the mean absolutes is **image 0.555 / text 0.445**
(`magnitude_share_of_mean_abs`). The mean signed image share is **−0.034**
(`signed_share_mean`): across examples the image's signed contribution is not
systematically positive — sign (direction) and magnitude are separate summaries, and
the near-zero mean signed share is not evidence of "balanced reliance". The mean signed
φ values and the mean per-example magnitude share require per-example data and are
`null` until re-run. The committed hero figure
(`assets/modality_attribution.png`) is the signed-share histogram from the original run;
a re-run of `just attribute` regenerates it as a two-panel (signed + magnitude) figure.

![modality attribution](assets/modality_attribution.png)

### 5.3 Interaction (defined, not yet computed)

The non-additivity contrast `I = v({img,txt}) − v({img}) − v({txt}) + v(∅)` is
implemented in `scripts/20_attribute.py` (per example, with mean / mean-absolute /
bootstrap interval over examples) and covered by unit tests on synthetic coalition
games. Real-data values are `null` in `metrics.json`: computing them requires the
coalition scores of the original run, which were not cached. They populate on re-run.

## 6. Discussion

On this frozen CLIP-L/14 + LightGBM setup, **the image modality carries the larger
average absolute margin contribution**: image-only AUROC (0.692) is close to fused
(0.711) while text-only (0.575) is barely above chance, and mean|φ| is 0.84 (image) vs
0.68 (text) with an aggregate magnitude share of ~0.56. This matches the well-documented
difficulty of Hateful Memes for unimodal text and the modest gains of simple late fusion
over a strong image encoder.

The near-zero mean signed share (−0.034) means the image's *signed* contribution is not
systematically positive across examples while its *magnitude* dominates on average — a
pattern the per-example φ values make inspectable after re-run. What the Shapley numbers
do **not** establish, and what this report does not claim:

- they do not show semantic complementarity between modalities;
- they do not show modality independence or dependence;
- they do not establish causal necessity of either modality;
- a mean magnitude share near 0.5 would not, by itself, demonstrate "balanced reliance"
  on both modalities.

The fused-vs-unimodal AUROC gaps (point differences above; paired CIs on re-run) quantify
whether fusion helps predictively. Even a consistently positive paired difference would
be evidence about *predictive* gain at n = 500, not about semantic complementarity. The
interaction contrast I, when computed, is interpreted strictly as non-additivity of the
margin under the specified interventional game.

## 7. Limitations

- **Frozen backbone.** No CLIP fine-tuning; the image and text representations are fixed
  at what CLIP learned from its own pre-training corpus. A fine-tuned CLIP would likely
  change both the absolute metrics and the modality attribution.
- **Modality-level granularity.** Attribution stops at the modality. No image-region-level
  attribution (e.g., GradCAM, DINO patch attention) is computed. A qualitative
  token-occlusion sketch appears in the notebook for the text side, but it is not a
  comprehensive second-level attribution.
- **Space uses B/32 and a generic background; not numerically comparable.** The HuggingFace
  Space uses CLIP-ViT-B/32 on CPU with a generic non-HM background. Both the backbone and
  the background differ from the headline; the Space attributions are illustrative only.
- **Interventional / factorised backgrounds are off-manifold for the tree model.** Marginal
  Shapley replaces absent features with background draws, creating (image, text) pairs that
  no real meme would produce. Tree models score these at arbitrary leaves.
- **Prior-dependent magnitudes.** The empirical background carries the train prior (~64%
  benign); `v(empty)` and absolute φ magnitudes reflect this prior. The per-example
  shares (s, m) normalise magnitudes; mean-baseline and balanced-background sensitivity
  analyses are specified in ADR 003 but **have not been executed**.
- **Unavailable run artifacts.** Per-example dev scores and coalition values from the
  original run were not cached, so the paired AUROC CIs, mean signed φ, mean magnitude
  share, and the interaction aggregate are `null` in `metrics.json` and require a
  re-run of the pipeline (gated dataset). Point differences and mean-absolute aggregates
  in this report trace exactly to the committed `metrics.json` values.
- **Probability calibration not guaranteed for unimodal heads.** Displayed probabilities
  for the image-only and text-only heads are indicative. Margin/Shapley results are
  unaffected.
- **Dev only.** Dev (500, balanced) is the only labelled evaluation split. `test.jsonl`
  is unlabelled and not used for metrics — it appears only as a source of qualitative demo
  examples in the Space (displayed with synthetic captions, never with raw meme content).
- **Single dataset, single backbone family.** Findings are specific to CLIP L/14 on
  Hateful Memes phase-1. Generalisation to other multimodal datasets or backbone families
  is not asserted.

## 8. References

- Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model
  predictions.* NeurIPS 2017.
- Janzing, D., Minorics, L., & Blöbaum, P. (2020). *Feature relevance quantification
  in explainable AI: A causal problem.* AISTATS 2020.
- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
  Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). *Learning
  transferable visual models from natural language supervision.* ICML 2021.
- Kiela, D., Firooz, H., Mohan, A., Goswami, V., Singh, A., Ringshia, P., &
  Testuggine, D. (2020). *The hateful memes challenge: Detecting hate speech in
  multimodal memes.* NeurIPS 2020.
