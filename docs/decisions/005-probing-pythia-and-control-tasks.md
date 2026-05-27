# ADR 005 -- Layerwise probing of Pythia-160M: extraction, alignment, metrics, and scope

**Status:** Accepted
**Date:** 2026-05-27

## Context

Project 5 (`05-mechanistic-pythia`) measures at which layer of Pythia-160M a binary
linguistic property becomes linearly decodable from the residual stream, using per-layer
linear probes paired with Hewitt & Liang (2019) control tasks. Before writing any code,
a four-way design review identified six architectural decisions carrying correctness or
methodological implications significant enough to document formally.

This ADR records all six decisions.

---

## Decision 1 -- Residual-stream extraction via forward hooks (not the `hidden_states` tuple)

### Context

HuggingFace Transformers exposes `output_hidden_states=True` on `GPTNeoXForCausalLM`, which
returns a tuple of hidden states. For a casual reader this looks like the 13-point depth
axis we need. It is subtly wrong for two reasons:

1. **Index 0 is the embedding *input***, not the embedding output: GPT-NeoX passes the raw
   token embedding into `layers[0]`, so `hidden_states[0]` is the pre-`embed_in` tensor that
   goes into block 0, not the `embed_in` output that is the true "embedding" point on the
   residual stream.
2. **The last entry is post-final-layernorm**, not `block_11` `resid_post`: GPT-NeoX applies
   `final_layer_norm` (attribute `gpt_neox.final_layer_norm`) before the unembedding head, so
   `hidden_states[-1]` is the normed tensor the unembedding reads, not the raw residual at
   the exit of block 11. Treating it as block 11 silently mislabels the deepest depth point.

Additionally, RoPE positional information is applied *inside* the attention mechanism (not
added to the residual stream), so none of the `hidden_states` entries contain a positional
term that would distort the residual norm.

### Options considered

1. **Use `output_hidden_states` tuple, skip index 0, call the last entry `block_11`.**
   Rejected: silently wrong -- the last entry is post-`ln_f`, a different representation.
2. **Use `output_hidden_states` tuple, reindex to correct for the off-by-one.**
   Feasible but fragile: the tuple layout is an internal transformer implementation detail
   that could change between `transformers` releases.
3. **Register forward hooks on the exact model attributes we care about.**
   Each hook captures one named point; the label is ours, not inferred from tuple position.
   Selected.

### Decision

- We **register forward hooks** directly on `model.gpt_neox.embed_in` (the embedding
  output) and on each `model.gpt_neox.layers[i]` for i = 0..11 (each block's output,
  i.e. `resid_post` for that block). This gives the correct 13-point depth axis:
  `["embedding", "block_0", "block_1", ..., "block_11"]`.
- We additionally hook `model.gpt_neox.final_layer_norm` to capture the **`ln_f` point**
  (what the unembedding head reads) as a **separately labelled extra point**, clearly
  distinguished from the 13-point depth axis. This makes the pre-LN / post-LN distinction
  explicit and lets the reader compare the raw residual (block_11) against the normed
  representation (`ln_f`) without conflating them.
- In `transformers` 5.9 a `GPTNeoXLayer` forward hook receives the hidden-states tensor
  directly. Defensively, if a future version returns a tuple, we take element 0.
- The `output_hidden_states` tuple is never used; the hook-based approach is the only
  extraction path.

### Consequences

- Per-point activations are unambiguously labelled and correct regardless of the tuple
  layout in future `transformers` versions.
- The `ln_f` point appears in `metrics.json` under `extra_points` (not `points`), so
  no downstream analysis accidentally treats it as a depth axis point.
- `emergence_point` explicitly excludes `ln_f` from the peak search.

---

## Decision 2 -- Offset-overlap word-to-token alignment

### Context

Pythia-160M uses a byte-level BPE tokenizer (GPT-NeoX Fast tokenizer). Byte-level BPE
attaches the leading space to the token rather than to the preceding token: `"world"` in
`"hello world"` is tokenized as `"Gworld"` (with the space prepended), whose character
offset span starts one position before the word boundary. A containment test (`tok_start >= ws
and tok_end <= we`) would silently fail to match `"Gworld"` against the word `"world"` and
drop the second word in every multi-word sentence.

Encoding each word separately (feeding `["hello", "world"]` individually to the tokenizer)
is also incorrect for byte-level BPE: the tokenizer's context changes the tokenisation of
each token when it is surrounded by neighbours (e.g., the leading-space byte is conditional
on context).

### Options considered

1. **Containment test (`tok_start >= ws and tok_end <= we`): last contained subword.**
   Rejected: silently drops every non-first word because byte-level BPE attaches the leading
   space to the token, so `tok_start < ws` for all non-first words.
2. **Encode each word separately and concatenate activations.**
   Rejected: the leading-space context changes the tokenisation; the resulting subwords
   do not correspond to positions in the whole-sentence encoding.
3. **Tokenise the full sentence once (with `return_offsets_mapping=True`) and use an
   OVERLAP test: `tok_end > ws and tok_start < we`. Take the LAST overlapping subword.**
   Handles byte-level BPE correctly (the leading-space token overlaps but is not contained).
   Selected.

### Decision

- For each sentence, the surface string is reconstructed from `words` + `space_after`
  (using the `SpaceAfter=No` field from CoNLL-U `MISC`), then tokenised as a whole sentence
  with `return_offsets_mapping=True`.
- Each UD word's character span `[ws, we)` is matched to the subword tokens whose offset
  span **overlaps** it: `tok_end > ws AND tok_start < we`. The residual at the **last**
  overlapping subword is taken as the word's representation.
- Words that fail to align (tokenizer normalisation edge cases) are dropped and their count
  is logged in `metrics.json` under `dropped_alignment`.
- The overlap predicate is the sole alignment logic; no containment fallback is used.

### Consequences

- Every non-first word in a sentence aligns correctly under byte-level BPE.
- The `last`-subword choice is a deliberate tradeoff: for plural nouns (`dogs`) the last
  subword is often the plural morpheme, so `noun_number` probe scores at shallow layers
  may reflect orthography rather than transformer computation. This is called out in
  `metrics.json` and in REPORT.md.

---

## Decision 3 -- Primary metric (balanced accuracy), control task, StandardScaler, and CIs

### Context

Three decisions about the evaluation protocol are tightly coupled: the choice of metric
affects the control design, which affects the CI strategy.

**Metric.** `is_noun` has a base rate near 0.17 and `is_verb` near 0.11 in UD-EWT; raw
accuracy at these class imbalances is dominated by the majority-class predictor and gives
a misleading picture of probe quality. A metric that is 0.5 at chance regardless of
prevalence is required.

**Standardisation.** Residual-stream L2 norm grows with depth in a pre-LN model. A
`LogisticRegression` with L2 penalty applied to non-standardised residuals would see
growing norms as growing "confidence" signals, conflating emergence with norm growth.

**Control task.** Hewitt & Liang (2019) define selectivity as the gap between probe
accuracy and control accuracy (the same probe trained on random per-type labels). The
control base rate must match the probe's base rate, and the control labels must cover
every token that will ever be scored to avoid test-time key errors.

**CIs.** Tokens within a sentence are correlated. Standard bootstrap (resampling tokens)
underestimates variance. Selectivity is a paired difference (probe and control are scored
on the same test set), so standard paired bootstrap also underestimates variance if it
ignores sentence-level clustering.

### Options considered

For each sub-decision:

**Metric:**
1. **Raw accuracy (primary).** Rejected: majority-class predictor scores 0.83 on `is_noun`
   by always predicting "not noun"; emergence is undetectable.
2. **AUROC only.** Good for imbalance; no intuitive "chance = 0.5" reference point for
   readers; kept as a secondary metric, not the primary.
3. **Balanced accuracy (mean of per-class recall, chance = 0.5).** Selected as primary.

**Standardisation:**
1. **No standardisation.** Rejected: norm growth with depth is a systematic confound.
2. **Global z-score (across all depths).** Does not correct for the per-depth norm
   difference; rejected.
3. **Per-(property, point) StandardScaler fitted on train, applied to dev/test.** Correct
   per-depth comparison; `C` chosen once per property on dev then held constant across
   points so cross-layer comparison is fair. Selected.

**Control task:**
1. **Random per-token labels (not per-type).** Rejected: a probe with access to word
   identity (via subword overlap) will not match random per-token labels; the control
   would understate what identity alone can recover.
2. **Per-type labels, positive share matched by type count (not token count).** Rejected
   under imbalance: "the" is a single type but accounts for ~5% of all tokens; a
   type-count match misrepresents the token-level prevalence that the probe sees.
3. **Per-type labels, positive share matched in TOKEN space (greedy seeded-random
   assignment until cumulative train-token mass of label-1 types reaches the base rate),
   map built over train+dev+test union, repeated over K=5 seeds.** Selected.

**CIs:**
1. **Standard (token-level) bootstrap.** Underestimates variance (within-sentence
   correlation).
2. **Cluster bootstrap (resample sentences), unpaired.** Correct unit; but selectivity
   is a paired difference.
3. **Paired sentence-cluster bootstrap (same resampled sentences for probe and control,
   take the distribution of their difference).** Correct variance estimate for selectivity;
   earliest-within-CI emergence point instead of raw argmax. Selected.

### Decision

- **Primary metric:** balanced accuracy (mean of per-class recall, chance = 0.5 regardless
  of prevalence). Raw accuracy and majority-class baseline are secondary context; AUROC is
  logged.
- **Standardisation:** per-(property, point) `StandardScaler` fitted on train residuals
  (upcast from float16 to float64 before the scaler). `C` chosen once per property on dev
  via a small grid {0.01, 0.1, 1.0}; held constant across all points for that property.
- **Control task:** token-rate-matched, multi-seed (K=5) Hewitt & Liang control. The
  per-type label map is built over the train+dev+test type union; positive token share
  is matched to the property's train base rate. Control balanced accuracy is averaged over
  seeds; seed spread is reported.
- **Selectivity:** `balanced_accuracy(probe) - mean_k balanced_accuracy(control_k)`.
- **CIs:** paired sentence-cluster bootstrap (2,000 resamples, seed fixed in
  `configs/probe.yaml`). Emergence is reported as the earliest depth point whose
  selectivity CI overlaps the peak's CI (not raw argmax).

### Consequences

- All three decisions are wired into the pure `awake.eval.probing` core and the pure
  `awake.eval.bootstrap` additions, which are locked by unit tests at >=90% coverage.
- The single cross-layer `C` is a deliberate comparability tradeoff: it may slightly
  underfit at extreme depths, making reported selectivity a conservative estimate.

---

## Decision 4 -- "Selectivity is necessary, not sufficient": honest framing

### Context

Selectivity (probe beats control) is often misread as evidence that the model "encodes" a
feature in a causally meaningful way. Several confounds can produce positive selectivity
without genuine encoding: lexical identity memorisation, suffix orthography (e.g., the
plural `-s` for `noun_number`), token frequency, position, and type-cluster structure
(Ravichander et al. 2021 bound individual-type but not cluster memorisation). Linear
decodability is not evidence the model *uses* the feature.

### Decision

- All reports, figures, and the notebook carry an explicit **"How to read selectivity"
  note**: selectivity is **necessary but not sufficient** evidence of genuine encoding.
  A positive selectivity shows the property is more linearly recoverable than an arbitrary
  word-type code at that layer; it does not show the model uses the feature causally.
- The note lists the specific confounds: lexical identity, suffix orthography (especially
  `noun_number` layer 0), token frequency, position, and type-cluster structure.
- The `noun_number` figure caption calls out explicitly that high embedding/layer-0 scores
  reflect plural-suffix orthography, not transformer computation.
- No selectivity result is described as "the model knows X" or "the model uses X".

### Consequences

- REPORT.md section 1 and section 7 (Limitations) carry the honest framing.
- ADR references the Ravichander et al. (2021) type-cluster caveat as an explicit scope
  limitation.

---

## Decision 5 -- UD English-EWT (CC BY-SA 4.0) over Penn Treebank

### Context

The natural first-choice syntactic corpus for English probing is the Penn Treebank (PTB /
WSJ). PTB is licensed through the LDC and is not openly redistributable; using it would
make this project non-reproducible for readers without an LDC licence and would introduce a
gated-data governance issue similar to the one we deliberately avoided in Projects 1 and 3.

UD English-EWT is a full Universal Dependencies treebank over English web text, carrying
UPOS labels and morphological features (including `Number=` for `noun_number`). It is
released under **CC BY-SA 4.0** and is openly available from the UniversalDependencies
GitHub organisation at a pinned release tag.

### Options considered

1. **Penn Treebank (PTB/WSJ).** Rejected: LDC licence; not openly redistributable;
   inconsistent with the repo's open-science governance.
2. **OntoNotes 5.0.** LDC licence; same issue as PTB. Rejected.
3. **UD English-EWT (CC BY-SA 4.0).** Openly licensed, openly available, pinned release
   tag, SHA-256 verified download, UPOS + morphological feature annotations that directly
   support all three target properties. Selected.

### Decision

- **UD English-EWT, release tag `r2.14`**, fetched by `scripts/00_data.py` via stdlib
  `urllib.request` from the pinned GitHub raw URL, with SHA-256 verification recorded in
  `configs/data.yaml`.
- **Attribution (CC BY-SA 4.0 requirement):** Silveira, N., Dozat, T., de Marneffe, M.-C.,
  Bowman, S., Connor, M., Bauer, J., & Manning, C. D. (2014). *A Gold Standard Dependency
  Corpus for English.* Proceedings of the Ninth International Conference on Language
  Resources and Evaluation (LREC-2014). Licensed under CC BY-SA 4.0.
- The `.conllu` files and all derived `outputs/` artefacts (parquet, activations) are
  gitignored and never committed.
- Only code, configs, `assets/` figures, the notebook (with outputs, no UD text in
  notebook outputs), and `REPORT.md` are committed.

### Consequences

- The project is fully reproducible without any institutional data access.
- Attribution is propagated to REPORT.md and this ADR; no UD text appears in committed
  artefacts or notebook outputs.

---

## Decision 6 -- Scope: probing only; activation patching and SAE inspection deferred

### Context

A natural extension of layerwise probing is causal verification: do activation patches
at the emergence layer change downstream behaviour in a property-consistent way? A further
extension is inspecting pretrained SAE features (sparse autoencoders, e.g. via `sae-lens`)
to see whether the features that drive probe accuracy correspond to interpretable SAE
directions.

Both extensions are valuable but out of scope for a single project sprint, and both
introduce non-trivial additional design choices (patch protocol, SAE registry version,
`sae-lens` API).

### Decision

- **v1.0 scope is probing only**: per-layer linear probes + Hewitt & Liang control tasks,
  on Pythia-160M, for three binary properties from UD English-EWT.
- **Deferred to v1.1** (explicitly noted in README.md):
  - Activation patching on GPT-2-small (where the IOI literature is native; Wang et al.
    2022) to test causal claims at the emergence layer.
  - Pretrained-SAE feature inspection via `sae-lens` (GPT-2-small residual SAEs across
    all layers are available per the `sae-lens` registry).
- The deferral is not a limitation of the probing methodology; it is a scope boundary.

### Consequences

- No causal claims are made in REPORT.md: selectivity is reported as linear decodability,
  not as evidence of causal use.
- The README.md Scope section retains the v1.1 deferral note as written in the existing
  stub.

---

## Summary table

| Decision | Chosen option | Key rationale |
|---|---|---|
| Extraction | Forward hooks on `embed_in`, `layers[i]`, `final_layer_norm` | `hidden_states` tuple mislabels: index 0 = embedding input; last = post-`ln_f`; hooks are unambiguous |
| Alignment | Offset-overlap (last overlapping subword), full-sentence tokenisation | Containment silently drops non-first words under byte-level BPE; separate encoding is also wrong |
| Metrics + CI | Balanced accuracy primary; per-layer StandardScaler; token-rate-matched K=5 control; paired sentence-cluster bootstrap | Norm growth confound; class imbalance; correct CI unit; paired selectivity difference |
| Selectivity framing | Necessary-not-sufficient, explicit confound list | Avoid over-claiming linear decodability as causal encoding |
| Dataset | UD English-EWT CC BY-SA 4.0 (attributed) | PTB/OntoNotes require LDC licence; UD-EWT is open, pinned, verified |
| Scope | Probing only; activation patching + SAE deferred to v1.1 | Sprint scope; causal extensions add non-trivial design choices |
