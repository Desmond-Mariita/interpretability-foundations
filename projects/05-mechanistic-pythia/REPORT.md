# Project 5 -- At which layer does a property become linearly decodable from Pythia-160M?

**Author:** Desmond Mariita.
**Dataset:** UD English-EWT (Silveira et al. 2014; CC BY-SA 4.0; code-only, `.conllu` not committed).
**Status:** skeleton -- real run pending (Task 15). All result cells below are explicitly
labelled "(filled by the real run)" and contain no fabricated numbers.

---

## 1. Question and framing

**At which layer of a small open transformer does a given binary linguistic property become
linearly decodable from the residual stream -- beyond what a probe can recover from word
identity alone?**

We answer with **per-layer linear probes** on **Pythia-160M** (EleutherAI; GPT-NeoX
architecture; 12 blocks; `d_model` = 768), each paired with a **Hewitt & Liang (2019)
control task** (random per-type labels). The headline per (property, layer) is
**selectivity = balanced_accuracy(probe) - balanced_accuracy(control)**.

### How to read selectivity

Selectivity is **necessary, but not sufficient**, evidence of genuine encoding. A positive
selectivity shows the property is *more linearly recoverable than an arbitrary word-type
code* at that layer. It does **not** show the model uses the feature causally. Confounds
that can produce positive selectivity without genuine encoding include:

- **Lexical identity memorisation** -- the probe may simply be reading the identity of
  the word type (nouns tend to be the same words across sentences).
- **Suffix orthography** -- especially for `noun_number`: the plural morpheme `-s` is
  often the last subword of a plural noun at every depth (layer 0 = token embeddings,
  no transformer computation). High `noun_number` selectivity at shallow depths is
  expected to reflect orthography, not transformer computation.
- **Token frequency and position** -- frequent tokens and sentence-initial tokens have
  distinct residual distributions.
- **Type-cluster structure** -- Ravichander et al. (2021) bound individual-type
  memorisation but not cluster-level memorisation.

Linear decodability is a description of *linear availability by depth*, not a causal claim.
See [ADR 005](../../docs/decisions/005-probing-pythia-and-control-tasks.md) for the full
design rationale.

## 2. Data

**Corpus.** Universal Dependencies English-EWT (Silveira et al. 2014). A gold-standard
Universal Dependencies treebank over English web text, with UPOS labels and morphological
features. Source: `UniversalDependencies/UD_English-EWT`, tag `r2.14` (fetched and
SHA-256-verified by `scripts/00_data.py`).

**Licence.** CC BY-SA 4.0.

**Attribution (CC BY-SA 4.0 requirement).**
Silveira, N., Dozat, T., de Marneffe, M.-C., Bowman, S., Connor, M., Bauer, J., &
Manning, C. D. (2014). A Gold Standard Dependency Corpus for English. In Proceedings of
the Ninth International Conference on Language Resources and Evaluation (LREC-2014).
Licensed under CC BY-SA 4.0.

**Splits.** Standard UD-EWT train / dev / test splits. Approximate sizes: train ~12.5k
sentences, dev ~2k sentences, test ~2.1k sentences.

**Fields per token.** FORM (surface), UPOS, `Number=` feature from FEATS, `SpaceAfter=`
from MISC (used for surface string reconstruction).

**Governance.** The `.conllu` files and all derived `outputs/` artefacts (parquet,
activations) are gitignored and never committed. Only code, configs, `assets/` figures,
the notebook (with outputs; no UD text in notebook outputs), and this `REPORT.md` are
committed.

## 3. Method

### Model

**Pythia-160M** (`EleutherAI/pythia-160m`, GPT-NeoX architecture). 12 transformer blocks,
`d_model` = 768. Loaded frozen (`eval()`, no grad) with a pinned `revision` commit SHA
(recorded in `configs/probe.yaml` and in `metrics.json`).

### Depth axis: 13 points via forward hooks

Activations are extracted at **13 points on the depth axis** using forward hooks (not the
HF `output_hidden_states` tuple, which mislabels GPT-NeoX: index 0 = embedding input; last
= post-`final_layer_norm`). The 13 points are:

- `embedding` -- output of `model.gpt_neox.embed_in` (token + position; no transformer
  computation).
- `block_0` through `block_11` -- output of each `GPTNeoXLayer` (i.e. `resid_post` for
  that block).

An additional point **`ln_f`** (output of `model.gpt_neox.final_layer_norm`, what the
unembedding reads) is captured separately and reported as an extra point, not on the
depth axis. See ADR 005 Decision 1.

Residuals are stored float16, upcast to float64 before StandardScaler/LR.

### Properties (3 binary, token-level)

| Property | Definition | Base rate (approx; measured at run) |
|---|---|---|
| `is_noun` | `upos == "NOUN"` (common noun) | ~0.17 |
| `is_verb` | `upos == "VERB"` | ~0.11 |
| `noun_number` | Among NOUN tokens with a `Number` feature: `Plur=1`, `Sing=0` | measured at run |

`noun_number` is conditional on noun-hood being encoded. If train n falls below the
configured minimum (3,000 tokens), it is flagged as underpowered in `metrics.json`.

### Probes and control tasks

**Pre-probe standardisation.** Per-(property, point) `StandardScaler` fitted on train
residuals; applied to test. `C` chosen once per property on the dev split (grid
{0.01, 0.1, 1.0}); held constant across all points for that property (comparability
tradeoff -- reported selectivity is a conservative estimate at extreme depths).

**Probe.** `LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs")`,
trained on (train, stratified cap 60k tokens) and scored on the full test split.

**Control task (Hewitt & Liang 2019).** Random per-word-type binary label map, built over
the train+dev+test type union (so no test token is ever unseen). Positive token share is
matched to the property's train base rate (greedy seeded-random assignment by train token
frequency). Repeated over K=5 seeds; control balanced accuracy is averaged over seeds.

**Selectivity.** `balanced_accuracy(probe) - mean_K balanced_accuracy(control_K)` per point.

### Metrics and CIs

- **Primary:** balanced accuracy (mean of per-class recall; chance = 0.5 regardless of
  prevalence). Robust to the `is_verb` / `is_noun` class imbalance.
- **Secondary:** raw accuracy, AUROC, majority-class baseline.
- **CIs:** paired sentence-cluster bootstrap (2,000 resamples). Sentences are the resampling
  unit (tokens within a sentence are correlated). Probe and control are recomputed on the
  same resampled sentences (paired), giving correct variance estimates for selectivity.
- **Emergence.** Peak = argmax selectivity over the 13 depth points. Earliest emergence =
  earliest depth point whose selectivity CI overlaps the peak's CI (avoids over-reading a
  noisy argmax; `ln_f` excluded from peak search).

### Word-to-token alignment

Full sentence tokenised once with `return_offsets_mapping=True`. Each UD word's character
span is matched to subword tokens by an **overlap** test (`tok_end > ws AND tok_start < we`,
required for byte-level BPE). The **last** overlapping subword's residual is used. Dropped
words are counted in `metrics.json`.

## 4. Results

> **All cells in this section are placeholders, labelled "(filled by the real run)".**
> No numbers have been fabricated. The structure below mirrors the expected `metrics.json`
> schema (see spec section 7 and ADR 005). Numbers will be filled in after Task 15 completes.

### Run metadata (filled by the real run)

| Field | Value |
|---|---|
| model | EleutherAI/pythia-160m |
| model_revision | (filled by the real run) |
| tokenizer_revision | (filled by the real run) |
| transformers version | (filled by the real run) |
| torch version | (filled by the real run) |
| chosen C: is_noun | (filled by the real run) |
| chosen C: is_verb | (filled by the real run) |
| chosen C: noun_number | (filled by the real run) |

### 4.1 `is_noun` -- probe vs. control balanced accuracy by depth

Figure: `assets/probe_is_noun.png` (generated by `scripts/30_eval.py`).

![probe_is_noun](assets/probe_is_noun.png)

| Point | Balanced acc (probe) | 95% CI | Ctrl balanced acc | Selectivity | Selectivity 95% CI |
|---|---|---|---|---|---|
| embedding | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_0 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_1 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_2 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_3 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_4 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_5 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_6 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_7 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_8 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_9 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_10 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_11 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| ln_f (extra) | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |

**train_n:** (filled by the real run) | **test_n:** (filled by the real run)
**base_rate:** (filled by the real run) | **majority_baseline:** (filled by the real run)
**OOV token rate:** (filled by the real run)
**Peak emergence point:** (filled by the real run)
**Earliest within peak CI:** (filled by the real run)

### 4.2 `is_verb` -- probe vs. control balanced accuracy by depth

Figure: `assets/probe_is_verb.png` (generated by `scripts/30_eval.py`).

![probe_is_verb](assets/probe_is_verb.png)

| Point | Balanced acc (probe) | 95% CI | Ctrl balanced acc | Selectivity | Selectivity 95% CI |
|---|---|---|---|---|---|
| embedding | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_0 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_1 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_2 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_3 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_4 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_5 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_6 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_7 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_8 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_9 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_10 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_11 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| ln_f (extra) | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |

**train_n:** (filled by the real run) | **test_n:** (filled by the real run)
**base_rate:** (filled by the real run) | **majority_baseline:** (filled by the real run)
**OOV token rate:** (filled by the real run)
**Peak emergence point:** (filled by the real run)
**Earliest within peak CI:** (filled by the real run)

### 4.3 `noun_number` -- probe vs. control balanced accuracy by depth

Note: high balanced accuracy at `embedding` / `block_0` is expected to reflect
**plural-suffix orthography** (the `-s` plural morpheme is often the last subword of a
plural noun at every depth), not transformer computation. The emergence story for
`noun_number` is the depth at which selectivity rises *above* the orthographic baseline,
not the absolute level at shallow depths.

Figure: `assets/probe_noun_number.png` (generated by `scripts/30_eval.py`).

![probe_noun_number](assets/probe_noun_number.png)

| Point | Balanced acc (probe) | 95% CI | Ctrl balanced acc | Selectivity | Selectivity 95% CI |
|---|---|---|---|---|---|
| embedding | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_0 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_1 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_2 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_3 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_4 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_5 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_6 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_7 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_8 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_9 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_10 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| block_11 | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |
| ln_f (extra) | (filled by the real run) | (filled) | (filled) | (filled) | (filled) |

**train_n:** (filled by the real run) | **test_n:** (filled by the real run)
**base_rate:** (filled by the real run) | **majority_baseline:** (filled by the real run)
**OOV token rate:** (filled by the real run)
**Underpowered flag:** (filled by the real run -- flagged if train_n < 3000)
**Peak emergence point:** (filled by the real run)
**Earliest within peak CI:** (filled by the real run)

### 4.4 Hero figure: selectivity by depth across all three properties

Figure: `assets/hero.png` (generated by `scripts/30_eval.py`).

![hero](assets/hero.png)

The hero figure overlays the three properties' selectivity-by-depth curves (with
cluster-bootstrap CI bands) and annotates the peak and earliest-emergence points for each.
The `ln_f` extra point is drawn off-axis as a marker.

## 5. Emergence summary (filled by the real run)

> The narrative below is a placeholder structure. The actual emergence story, including
> which properties show a clear rise vs. a gradual ramp, and whether `noun_number` and
> `is_noun` show correlated emergence depths, will be written after the real run.

| Property | Peak point | Earliest within peak CI | Notes |
|---|---|---|---|
| is_noun | (filled by the real run) | (filled by the real run) | |
| is_verb | (filled by the real run) | (filled by the real run) | |
| noun_number | (filled by the real run) | (filled by the real run) | orthographic baseline at layer 0 |

## 6. Limitations

The limitations below are drawn directly from spec section 12 and apply to the results
regardless of what the real run produces.

**Selectivity is necessary, not sufficient.** Beating a random type-control shows linear
recoverability beyond an arbitrary type code, but lexical identity, **suffix orthography**
(especially plural `-s` for `noun_number`), **token frequency**, **position**, and
**type-cluster structure** (H&L bounds individual-type but not cluster memorisation;
Ravichander et al. 2021) can all produce positive selectivity. Linear decodability is not
evidence the model *uses* the feature.

**Last-subword pooling.** Last-subword pooling entangles emergence with subword count and,
for `noun_number`, with the plural morpheme. `first`/`mean` pooling is a config option; a
`last`-vs-`first` sensitivity check is reported if time permits, else flagged as future work.
The single-vs-multi-subword split per property is reported in `metrics.json`.

**Pre-LN / `ln_f`.** The 13 depth points are pre-final-LN residual states; `ln_f` (what
the unembedding reads) is reported separately. Layer 0 = token embeddings (no transformer
computation).

**`noun_number` is conditional and sample-limited.** Underpowered runs (train n below the
configured minimum of 3,000) are flagged in `metrics.json`, not silently dropped.

**Single cross-layer `C`.** `C` is chosen once per property on dev and held constant across
points for fair comparison; it may slightly underfit at extreme depths, so reported
selectivity is a conservative estimate at those points. This is a deliberate comparability
tradeoff, not a bug.

**One model, one size, one domain.** Pythia-160M only -- **no scaling claim**; one English
web-text domain (UD-EWT); **linear probes only**. Activation patching and SAE inspection are
explicitly **deferred** (README scope v1.1). Results do not generalise to other architectures,
sizes, or languages without further verification.

## 7. References

- Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control tasks.
  In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing
  (EMNLP-2019).
- Ravichander, A., Belinkov, Y., & Hovy, E. (2021). Probing the probing paradigm: Does
  probing accuracy entail the presence of linguistic knowledge? In Proceedings of the 16th
  Conference of the European Chapter of the Association for Computational Linguistics
  (EACL-2021).
- Silveira, N., Dozat, T., de Marneffe, M.-C., Bowman, S., Connor, M., Bauer, J., &
  Manning, C. D. (2014). A Gold Standard Dependency Corpus for English. In Proceedings of
  the Ninth International Conference on Language Resources and Evaluation (LREC-2014).
  Licensed under CC BY-SA 4.0.
- Black, S., Biderman, S., Hallahan, E., et al. (2022). GPT-NeoX-20B: An open-source
  autoregressive language model. arXiv:2204.06745.
- Biderman, S., Schoelkopf, H., Anthony, Q., et al. (2023). Pythia: A suite for analyzing
  large language models across training and scaling. In Proceedings of the 40th International
  Conference on Machine Learning (ICML-2023).
