# ADR 003 — Hateful Memes licence constraints, modality Shapley game, interventional baseline, and logit value function

**Status:** Accepted
**Date:** 2026-05-26

## Context

Project 3 (`03-multimodal-hatefulmemes`) trains a fused image+text classifier on the Meta
Hateful Memes corpus and attributes each prediction to the image vs. text modality using a
2-player Shapley game. Before writing pipeline code, a four-way design review identified
four architectural decisions carrying governance, correctness, or methodological implications
significant enough to document formally.

This ADR records all four decisions. The 003 numbering closes the gap: the stale
project README previously linked `docs/decisions/004-modality-shapley-over-shap-aggregation.md`,
which was never created. That file is superseded by this ADR; the README link has been
updated to 003.

---

## Decision 1 — Dataset is code-only; model weights may be published; CLIP embeddings of the memes may not

### Context

The Hateful Memes corpus is distributed under the Facebook HM Dataset Agreement
(Getty-sourced images). Key clauses:

- **§6.1** prohibits creating derivative works of the dataset.
- **§6.2** prohibits distributing, copying, or hosting the dataset (a narrow academic
  exception of ≤100 images with Getty attribution is not exercised here).
- **§6.5** prohibits incorporating the dataset into another product.
- **§2** explicitly preserves the Participant's IP in models and algorithms derived from
  use of the dataset; these may be used and published for academic and commercial purposes.

The dataset is accessed via a Kaggle mirror of the phase-1 release; it remains governed
by the HM Dataset Agreement regardless of origin.

CLIP embeddings computed from the memes are derivative works of the dataset (§6.1). They
may not be committed to the repository or hosted publicly. The trained LightGBM head and
CLIP model weights are not derivative works of the dataset and may be published.

The HuggingFace Space cannot include any HM-derived data in its committed artifacts.

### Options considered

1. **Commit a small sample of embeddings for fast CI smoke tests.** Rejected: embeddings
   are §6.1 derivative works of the dataset; committing them violates the licence
   regardless of sample size.
2. **Use the ≤100-image academic exception for qualitative display in the notebook and
   REPORT.** Rejected: the exception requires Getty attribution and is cumbersome to
   apply consistently. Synthetic stand-in examples achieve the same illustrative purpose
   without activating the exception.
3. **Code-only dataset access; publish trained head; use generic non-HM background in
   the Space.** Selected.

### Decision

- The dataset and any CLIP embeddings derived from it are never committed or hosted.
  `projects/03-multimodal-hatefulmemes/outputs/` and `data/` are gitignored.
- `scripts/00_data.py` verifies the dataset at `DATA_PATH` (default
  `~/.cache/hateful_memes`) and, if only the archive is present, extracts it there. It
  downloads nothing; it prints licence-acceptance and access instructions when the dataset
  is absent, following the P1 MIMIC pattern.
- The trained LightGBM head is published to HuggingFace Model Hub (user-owned repository);
  it contains no HM images, no HM text, and no HM-derived embeddings.
- The HuggingFace Space uses a **generic, non-HM background** — approximately 50 CC0 /
  public-domain images with generic captions, pre-encoded to `generic_background.npz` by
  `scripts/build_generic_background.py` and committed to `apps/hatefulmemes-space/`. No
  HM-derived artifact is committed or uploaded.
- **Public-artifact rule (enforced everywhere):** committed notebook outputs, `REPORT.md`,
  `metrics.json`, hero figures, and the Space contain no raw HM images or meme text — only
  aggregate numbers, attribution scores, and identifiers. Qualitative displays use
  **synthetic stand-in** image+caption pairs (not real memes); the ≤100-image academic
  exception is not exercised.

### Consequences

- Reproducers must obtain the Hateful Memes dataset independently (Meta HM Agreement) and
  place it at `DATA_PATH` before running `just data`. This is the intended cost of a
  gated corpus.
- The Space attributions are **illustrative only** — they use a generic, non-HM background
  that does not reflect the train distribution. The UI and README carry an explicit
  disclaimer to this effect.
- No HM-derived content can enter the public git history. `gitleaks` and the bound
  safety-sweep pattern list (ADR 001) continue to run.

---

## Decision 2 — 2-player modality Shapley game over per-dimension feature SHAP

### Context

The research question is: _how much of the classifier's output came from the image
modality vs. the text modality?_ This is a modality-level question — it calls for two
attribution numbers per example, not one per embedding dimension. Per-dimension SHAP
aggregated to modality groups is technically valid but poorly matched to the question.

The design review raised a Janzing et al. (2020) concern about "additivity pathologies"
for SHAP on highly correlated features. After review, the team concluded that framing was
an overstatement: SHAP's additivity property holds exactly on the linear margin; any
non-linearity effects are a property of the model, not a flaw in SHAP. Janzing et al.'s
critique concerns a specific inconsistency in the SHAP framework when features are
correlated, but it is not the primary reason for choosing the 2-player game here.

### Options considered

1. **Per-dimension SHAP, aggregated to modalities (e.g., sum |φ_i| over image dims, sum
   over text dims).** This would produce modality-level numbers but would require running
   SHAP over 768 (B/32) or 1024 (L/14) image dimensions plus the same for text — a
   dimension count that makes full Shapley impractical (KernelSHAP scales exponentially
   in the number of players; TreeSHAP for LightGBM is polynomial but still operates on
   ~1500–2000 features). The aggregation step (sum of absolute values, or signed values)
   adds a choice that is not grounded in the original question.
2. **Group-SHAP over the two modality groups.** A valid formulation: treat the image
   embedding as one "player" and the text embedding as another, then compute Shapley
   values for each group using the exact 2-player formula. This is methodologically
   equivalent to the chosen approach and is a valid alternative that is simply not pursued
   here.
3. **Exact 2-player interventional Shapley game.** Selected.

### Decision

- The image and text modalities are treated as the two players of a coalitional game.
  The exact 2-player Shapley formula is applied:

  ```
  φ_image = ½ [(v({img}) − v(∅)) + (v({img,txt}) − v({txt}))]
  φ_text  = ½ [(v({txt}) − v(∅)) + (v({img,txt}) − v({img}))]
  ```

- This choice is justified **primarily by parsimony**: the question has two modalities
  and calls for two numbers. The 2-player game gives exactly that, with a closed-form
  solution that requires no approximation.
- Dimensionality makes raw per-dimension SHAP impractical as a cross-check at these
  embedding sizes (~1500–2000 features). This reinforces the parsimony argument but is
  not the primary reason.
- Group-SHAP over the two modalities is a valid alternative not pursued here; it would
  be equivalent under the same value function and background.
- The framing that the 2-player game avoids "additivity pathologies" (the earlier README
  language) is **dropped**: it overstated the Janzing et al. (2020) critique and implied
  a flaw in SHAP that is not the actual basis for the design choice.

### Consequences

- Two attribution numbers per example (φ_image, φ_text) are the headline output.
- The hero figure plots the **signed image share**
  `s = φ_image / (|φ_image| + |φ_text| + ε)`, ε = 1e-9, which lies in [−1, 1].
  Examples with `|φ_image| + |φ_text| < ε` (negligible total attribution) are excluded
  from the share plot and counted in a footnote; their raw φ values still appear in
  `metrics.json`.
- The exact formula requires only four coalition evaluations per example, making the
  full-dev attribution fast (§ Decision 3 details the evaluation).

---

## Decision 3 — Interventional background-averaged baseline; empirical train background as primary; off-manifold caveat disclosed

### Context

To evaluate the coalition values v(∅), v({img}), v({txt}), and v({img,txt}), absent
modalities must be marginalised out. Two families of approach exist:

- **Conditional expectation:** integrate absent features over their conditional
  distribution given the present features. Theoretically preferable but intractable
  without a generative model.
- **Interventional (marginal) expectation:** integrate over the marginal distribution
  of absent features, ignoring the correlation between image and text. Tractable given
  a background set.

An interventional approach with a background of real train-distribution examples pairs
each absent modality with a drawn background row, producing factorised (image, text)
pairs that are off-manifold — a real image is paired with a text embedding from a
different example, and vice versa. For a tree model, off-manifold inputs land at whatever
leaf the tree routes them to; this is inherent to marginal Shapley on trees and is the
same trade-off SHAP's TreeExplainer makes by default.

A single mean-embedding baseline collapses the background to one point; it is equivalent
to replacing absent features with the mean embedding of the training set. This is a
simpler but less stable estimator.

### Options considered

1. **Mean-embedding baseline (single background point).** Simple; always deterministic.
   Equivalent to the "zero baseline" or "mean baseline" common in attribution literature.
   Chosen as an **ablation**, not the primary.
2. **Class-stratified background (balanced benign/hateful).** Would remove the prior
   from v(∅) and make absolute φ magnitudes prior-independent. Chosen as a **sensitivity
   ablation**; reported in `metrics.json` alongside the primary.
3. **Empirical (unstratified) train background, seeded, N = 200.** Selected as primary.
   Reflects the observed train class prior (~64% benign / 36% hateful) and is the most
   natural interventional estimand for the observed data distribution.

### Decision

- The **empirical train background** (N = 200, unstratified, seeded via `background_seed`
  in `configs/train.yaml`) is the primary interventional baseline. It is saved to
  `outputs/models/<variant>/background.npz` by `scripts/15_background.py`.
- Coalition values are computed as:

  ```
  v({img,txt}) = value_fn([img_emb,  txt_emb])               # point score, not averaged
  v({img})     = mean_j value_fn([img_emb,      txt_bg[j]])   # absent text replaced from bg
  v({txt})     = mean_j value_fn([img_bg[j],    txt_emb])     # absent image replaced from bg
  v(∅)         = mean_j value_fn([img_bg[j],    txt_bg[j]])   # paired background rows
  ```

  where rows j are paired (img_bg[j] and txt_bg[j] come from the same training example),
  and each mean is a single vectorised call over an (N, 2D) batch.

- `v({img,txt})` is the **point score on the actual example** (never background-averaged);
  only the absent modalities in v({img}) and v({txt}) are replaced.
- The estimator is **deterministic** (all N background rows are used, no sampling); given
  a fixed background set, results are exactly reproducible.
- The **off-manifold caveat** is disclosed in the REPORT Limitations section: factorised
  (image, text) pairs are out-of-distribution for the fused LightGBM head. Tree models
  score off-manifold inputs at whatever leaf the tree routes them to. This is the standard
  marginal-Shapley trade-off; the mean-baseline ablation quantifies its practical effect.
- The **mean-baseline ablation** (replace absent modality with the embedding mean of the
  background, collapsing to a single background point) is reported in `metrics.json` as a
  cross-check.
- The **balanced-background ablation** (class-stratified background) is also reported to
  quantify the shift caused by the empirical class prior in absolute φ magnitudes.
- This is **interventional reliance** — a measure of how much the model relies on each
  modality when the other is marginalised out. It is **not** a causal claim.

### Consequences

- Attribution results are prior-dependent: `v(∅)` and absolute φ magnitudes reflect the
  ~64% benign train prior. The signed image share `s` is the robust summary quantity
  (the balanced-background ablation quantifies the shift).
- The off-manifold caveat is a stated limitation, not a deficiency of the chosen estimand.
  All marginal-Shapley approaches on tree models share this property.
- The deterministic estimator makes results exactly reproducible from a fixed seed.

---

## Decision 4 — Logit (raw margin) as the Shapley value function; probability for UI only

### Context

The Shapley value function must map a coalition input to a scalar. Two natural choices for
a binary classifier are the **predicted probability** (sigmoid of the margin) and the
**raw margin** (log-odds of the hateful class, i.e., the pre-sigmoid logit).

Shapley additivity requires that φ_image + φ_text = v({img,txt}) − v(∅). This holds
exactly for both the margin and the probability. However, the probability's sigmoid
nonlinearity compresses contributions near 0 and 1: a large shift in the margin from 1.5
to 3.0 maps to a small shift in probability (0.82 → 0.95), while the same shift from
−0.2 to 1.5 maps to a large shift (0.45 → 0.82). This compression means that the
probability-based Shapley values reflect both the raw modality contribution and the
operating point on the sigmoid curve — confounding the two.

The logit/margin does not compress; a unit shift in φ_image represents the same
contribution to the decision regardless of where on the margin scale the example sits.

### Options considered

1. **Probability (sigmoid of margin) as the value function.** Produces directly
   interpretable numbers (in [0, 1]) but introduces operating-point compression.
   Used for UI display only.
2. **Raw margin (logit) as the value function.** Unbounded; additivity holds without
   sigmoid compression. Selected.

### Decision

- The value function is the **LightGBM raw margin** (the log-odds of the hateful class,
  before sigmoid), i.e., `value_fn([emb]) = model.predict(emb, raw_score=True)`.
- All four coalition values (v(∅), v({img}), v({txt}), v({img,txt})) are computed in
  raw-margin space; φ_image and φ_text are in raw-margin units.
- The Space and notebook map the raw margin back to a probability via sigmoid **for
  display only**; the attribution bars are based on the raw-margin Shapley values.

### Consequences

- Shapley additivity holds exactly on the unbounded margin without sigmoid compression.
- Absolute φ magnitudes are in log-odds units, which are less immediately interpretable
  to non-technical readers. The signed image share `s` (a normalised ratio) is the
  primary quantity for communication; it is dimensionless and does not carry the unit
  issue.
- Probability calibration of the unimodal heads is not guaranteed; displayed probabilities
  for the image-only and text-only heads are indicative. The margin/Shapley results are
  unaffected by this.

---

## Summary table

| Decision | Chosen option | Key rationale |
|---|---|---|
| Dataset governance | Code-only; commit model head, not embeddings; generic Space background | HM Agreement §6.1/6.2 forbids derivative works and hosting; §2 permits model publication |
| Modality attribution | Exact 2-player interventional Shapley game | Parsimony: the question is modality-level (two numbers); dimensionality makes per-dimension SHAP impractical as a cross-check |
| Baseline | Empirical train background (N=200, primary) + mean-baseline + balanced-background ablations | Reflects observed train distribution; mean and balanced variants quantify sensitivity |
| Value function | Raw margin (logit) | Additivity on the unbounded margin; no sigmoid compression near 0/1; probability mapped back for display only |
