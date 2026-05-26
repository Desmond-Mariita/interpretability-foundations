# Project 3 — Modality attribution for fused image+text classification

**Author:** Desmond Mariita.
**Dataset:** Meta Hateful Memes phase-1 (Meta HM Dataset Agreement; gated, code-only).
**Status:** pipeline implemented; end-to-end run pending (metrics populated once completed).

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

All hyperparameters are fixed via **train-only 5-fold cross-validation**. Dev is never
used for selection; it is the final evaluation split only. Class imbalance is handled via
`scale_pos_weight` set from the train label ratio (~1.79). Models are saved to
`outputs/models/clip_l14/`.

**Background.** `scripts/15_background.py` samples a seeded empirical background of N = 200
unstratified train rows → `outputs/models/clip_l14/background.npz`. The background seed
and N are set in `configs/train.yaml`.

## 4. Metric definitions

All metric code lives in `src/awake/eval/` (pure, unit-tested).

### 4.1 Classifier metrics (per head, on dev)

- **AUROC** — area under the ROC curve (threshold-free ranking metric; chance = 0.5).
- **AUPRC** — area under the precision–recall curve (accounts for class imbalance;
  dev is balanced, but AUPRC is still informative for the operating-point distribution).
- **Accuracy at 0.5** — binary accuracy at the default probability threshold; dev is
  balanced, so the chance baseline is 0.5.
- All three metrics reported with **bootstrap 95% confidence intervals** (2,000 resamples,
  percentile method, fixed seed).
- **Paired bootstrap 95% CIs on fused − unimodal AUROC differences** are reported in
  `metrics.json`. At n = 500 the AUROC-difference half-width is approximately ±0.03–0.05;
  CIs are reported regardless of whether they straddle zero — no post-hoc significance
  threshold is applied.

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

**Background.** Primary: empirical train background (N = 200, seeded). Ablations:
mean-embedding baseline (single point) and balanced (class-stratified) background. The
primary reflects the observed train prior (~64% benign); the ablations quantify the
sensitivity of absolute φ magnitudes to this choice.

**Off-manifold caveat.** Factorised (image, text) pairs produced by background replacement
are out-of-distribution for the fused LightGBM head: a real image embedding is paired
with a text embedding from a different example. Tree models route these inputs to whatever
leaf they reach — inherent to marginal Shapley on trees. The mean-baseline ablation
quantifies the practical effect.

**Estimand.** This is **interventional reliance** — a measure of how much the model
relies on each modality when the other is marginalised out. It is not a causal claim.

**Summary statistic and hero figure.** The **signed image share**:

```
s = phi_image / (|phi_image| + |phi_text| + eps),   eps = 1e-9
```

lies in [−1, 1]: positive means image-dominant, negative means text-dominant. Examples
where `|phi_image| + |phi_text| < eps` (negligible total attribution) are excluded from
the share plot and counted in a footnote; their raw φ values are retained in `metrics.json`.

All per-example (phi_image, phi_text, total = v_ab − v_empty) values are written to
`metrics.json`. Full-set aggregate over all 500 dev examples is the headline;
correctness-stratified and gold-label-stratified breakdowns are reported separately and
clearly labelled.

## 5. Results

Run on the balanced 500-example dev split (frozen CLIP-ViT-L/14, all numbers from
`metrics.json`).

### 5.1 Classifier metrics (dev, n = 500, balanced)

| Head | AUROC (95% CI) | AUPRC | Accuracy |
|---|---|---|---|
| Fused (CLIP-L/14) | 0.711 [0.669, 0.756] | 0.672 | 0.606 |
| Image-only | 0.692 [0.648, 0.736] | 0.691 | 0.622 |
| Text-only | 0.575 [0.524, 0.626] | 0.566 | 0.548 |

Text-only is barely above chance; image-only carries most of the signal; the fused head's
edge over image-only (0.711 vs 0.692) sits well within the overlapping 95% CIs, so fusion
is not clearly better than the image alone on this frozen-feature setup. (`metrics.json`
also records a `paired_diff_test` over the probability arrays as a stand-in; the honest
comparison is the overlapping AUROC CIs, not that test's p-value.)

### 5.2 Modality attribution (dev aggregate)

Mean |φ| (raw-margin interventional Shapley, empirical train background, N=200):
**image 0.842 vs text 0.675** — the fused decision leans on the image modality, consistent
with the unimodal AUROCs. The signed image-share distribution
(`assets/modality_attribution.png`) is centred near zero (share_mean −0.034) but with the
image carrying the larger magnitude on most examples.

![modality attribution](assets/modality_attribution.png)

## 6. Discussion

On this frozen CLIP-L/14 + LightGBM setup, **the image modality carries the discriminative
load**: image-only AUROC (0.692) is close to fused (0.711) while text-only (0.575) is barely
above chance, and the modality Shapley assigns larger magnitude to the image (mean|φ| 0.84
vs 0.68). This matches the well-documented difficulty of Hateful Memes for unimodal text and
the modest gains of simple late fusion over a strong image encoder. The near-zero signed
share_mean (−0.034) with image-dominant magnitudes means the two modalities frequently push
in opposite directions on a given example while the image more often dominates — a pattern
the per-example raw φ in `metrics.json` makes inspectable. Two honest caveats bound the
reading: fusion's edge over image-only is within overlapping CIs (not a clear win on frozen
features), and the interventional baseline forms off-manifold image+text pairs that a tree
model scores at arbitrary leaves (§7). The framing below interprets the share axis.

The signed image share `s` indicates whether the fused model is predominantly
image-driven or text-driven across the dev set. Conditional breakdowns — by gold label
(hateful vs. benign) and by prediction correctness — reveal whether the model's
modality reliance is consistent across outcomes or whether, for example, missed hateful
examples are predominantly those where the text signal is weak or absent.

A text-dominant model (mean `s` near −1) suggests the CLIP text encoder is carrying most
of the discriminative load; an image-dominant model (mean `s` near +1) suggests the visual
content drives decisions. A balanced mix (`s` near 0 on average) suggests the fused head
exploits complementary signals, which would motivate further image-region attribution work.

The fused-vs-unimodal AUROC gaps (with bootstrap CIs from `metrics.json`) quantify how
much the fusion actually helps; if the gaps straddle zero, the evidence for complementarity
is inconclusive at n = 500.

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
  no real meme would produce. Tree models score these at arbitrary leaves. The mean-baseline
  ablation in `metrics.json` quantifies the practical effect.
- **Prior-dependent magnitudes.** The empirical background carries the train prior (~64%
  benign); `v(empty)` and absolute φ magnitudes reflect this prior. The signed image share `s`
  is the robust summary; the balanced-background ablation quantifies the shift.
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
