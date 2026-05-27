# Project 5 — `05-mechanistic-pythia`: at which layer does a property become linearly decodable?

**Status:** v0.x draft for four-way review · 2026-05-27 · author Desmond Mariita

Part of `interpretability-foundations`, built per `docs/PLAYBOOK.md`. **No HF Space.**
Hardware: CPU is sufficient (Pythia-160M); the RTX 3090 only speeds activation extraction.
§12 records review traceability.

## 1. Question and deliverable

**At which layer of a small open transformer does a given binary linguistic property become
linearly decodable from the residual stream?** We answer with **per-layer linear probes** on
**Pythia-160M**, and — crucially — report a **Hewitt & Liang (2019) control task** beside every
probe so the reader can tell whether a high probe accuracy reflects *model structure* or just the
*probe's own capacity to memorise word types*.

The headline per (property, layer) is **selectivity = probe_accuracy − control_accuracy**. A probe
that only memorises will match its control (selectivity ≈ 0); a probe reading a genuinely
linearly-encoded property will sit well above its control. The depth at which selectivity peaks is
the "where it emerges" answer.

Deliverables: `metrics.json` (per property × per layer: probe accuracy, control accuracy,
selectivity, majority-class baseline; bootstrap 95% CIs), **one figure per property** (probe vs
control accuracy, plus selectivity, as a function of depth) + a combined hero, a notebook
(committed **with outputs**, including a **self-authored worked example**), `REPORT.md`, an ADR.
**No Space, no deploy.**

## 2. Data — Universal Dependencies English-EWT (open; code-only)

UD English-EWT (Silveira et al. 2014), **CC BY-SA 4.0** — openly licensed, unlike PTB/WSJ. The HF
script loader is dead (`datasets` 5.x dropped dataset scripts), so we fetch the three `.conllu`
files directly from a **pinned UD release tag** (e.g. `r2.14`) at
`raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/<tag>/en_ewt-ud-{train,dev,test}.conllu`
and parse them (CoNLL-U is line-oriented and trivial to parse: tab-separated
`ID FORM LEMMA UPOS XPOS FEATS …`; skip blank lines, `#` comments, and multiword/empty tokens whose
`ID` contains `-`/`.`). The release tag is pinned in `configs/data.yaml` for reproducibility.

`scripts/00_data.py`:
- Fetch (idempotent — skip if the `.conllu` already exists) and parse to
  `outputs/prepared/{train,dev,test}.parquet` with columns `sent_id, words` (list[str]),
  `upos` (list[str]), `number` (list[str], `Sing`/`Plur`/`""` from the `Number=` FEAT).
- **Never commit** the `.conllu` or the parquet (raw text); `outputs/` is gitignored. Only derived
  metrics, figures, and a self-authored worked example are committed.

## 3. Model and activation extraction (frozen; deterministic)

`EleutherAI/pythia-160m` (12 layers, `d_model` 768) via HF `AutoModelForCausalLM`, `eval()`, no
grad. Activations come from `output_hidden_states=True`: `hidden_states` is a tuple of **13**
tensors — `hidden_states[0]` = token+position embeddings, `hidden_states[i]` = residual stream
**after** block `i` (`resid_post`) for `i = 1..12`. We probe **all 13** points (call them layers
`0..12`) so "emergence with depth" is visible from the embedding onward. No `transformer_lens`
dependency (it is unverified on this transformers-5.9 stack); raw `hidden_states` *is* the residual
stream.

**Tokenisation alignment (the one real wrinkle).** Pythia uses byte-level BPE; UD gives whitespace
words. For each sentence we tokenise with the Pythia tokeniser tracking word boundaries (encode each
UD word separately and concatenate, recording each word's subword span), then take the residual at
the **last subword** of each word (standard practice) as that word's representation at each layer.
Determinism: `do_sample` is irrelevant (no generation — a single forward pass); fixed model
revision logged to `metrics.json`. `scripts/10_extract.py` → `outputs/acts/<split>.npz`
(`X[layer]`: `(n_words, 768)` float16 per layer; aligned arrays `words, upos, number, sent_id`).

## 4. Properties (3 binary, token-level)

Defined on UD tokens (one label per word, aligned to the last-subword residual):
- **is_noun** — `upos == "NOUN"` (common noun; base rate ≈ 0.17).
- **is_verb** — `upos == "VERB"` (≈ 0.11).
- **noun_number** — among tokens with `upos == "NOUN"` **and** a `Number` feature: `Plur` = 1,
  `Sing` = 0 (a morphological property *distinct from POS identity*; the probe set is restricted to
  nouns, so this asks "where is grammatical number on a noun encoded"). Base rate ≈ 0.3 plural.

Each property yields a train/dev/test design matrix per layer: rows = qualifying tokens, features =
the 768-d residual at that layer, label = the binary property.

## 5. Probe, control task, and selectivity

**Probe.** `sklearn.linear_model.LogisticRegression` (linear; `max_iter` high, `C` from config,
fixed `random_state`) fit on **train** residuals at a layer, evaluated on **test**. Per property ×
per layer we report **accuracy** (and the **majority-class baseline** for context).

**Control task (Hewitt & Liang 2019).** Assign each **word type** (lowercased surface form) a
*random* binary label, drawn once with a fixed seed and matched to the property's empirical base
rate, over the **union of word types across all splits** (so test has no unseen types). Train the
**same** probe to predict this control label from the same residuals; its test accuracy is the
**control accuracy** — how well the representation + linear probe can fit an arbitrary type→label
map, i.e. probe capacity / type memorisation.

**Selectivity = probe_accuracy − control_accuracy**, per property × per layer. This is the headline:
it isolates real linear encoding from memorisation. We report all three (probe, control,
selectivity) by layer, with **bootstrap 95% CIs** (`awake.eval.bootstrap`, resampling test tokens).

## 6. Shared pure core — `src/awake/eval/probing.py` (pure, unit-tested, 90% floor)

The methodology's pure logic lives here (no models, no sklearn, no I/O); the sklearn fit and
activation extraction stay in scripts (slow-tested).

```python
def assign_control_labels(word_types: list[str], base_rate: float, seed: int) -> dict[str, int]:
    """Hewitt-Liang control: a deterministic random binary label per word type, with the share
    of 1s approximately ``base_rate``. Same (types, base_rate, seed) -> same mapping."""
def control_vector(words: list[str], type_to_label: dict[str, int]) -> list[int]:
    """Map a token's (lowercased) word to its control label; every word's type must be present."""
def base_rate(labels: list[int]) -> float:
    """Share of positive labels (0.0 for empty)."""
def accuracy(pred: list[int], gold: list[int]) -> float:
    """Top-1 accuracy; raises ValueError on length mismatch; 0.0 for empty."""
def majority_baseline(labels: list[int]) -> float:
    """Accuracy of always predicting the training-majority class (= max(p, 1-p))."""
def selectivity(probe_acc: float, control_acc: float) -> float:
    """probe_acc - control_acc."""
def best_layer(selectivity_by_layer: dict[int, float]) -> int:
    """Layer index with the highest selectivity (lowest index on ties)."""
```
Reuse `awake.eval.bootstrap` (`bootstrap_ci`) for CIs. Add the new names to
`src/awake/eval/__init__.py`.

## 7. Eval + figures

`scripts/20_probe.py` — for each property, for each layer 0..12: fit the probe (train) + the control
probe (train), score both on test, record accuracy/control/selectivity/majority + per-token correct
vectors for bootstrap. `scripts/30_eval.py` — assemble `metrics.json`:
`{"model":"EleutherAI/pythia-160m","model_revision":<sha>,"n_layers":13,
"splits":{"train":n,"test":n},"properties":{"is_noun":{"base_rate":..,"majority":..,
"layers":[{"layer":0,"probe_acc":..,"probe_ci":[lo,hi],"control_acc":..,"control_ci":[lo,hi],
"selectivity":..,"selectivity_ci":[lo,hi]}, …13]}, "is_verb":{…}, "noun_number":{…}}}`.

Figures (`assets/`): **one per property** — `probe_<property>.png`: probe accuracy, control
accuracy, and majority baseline vs layer (with CI bands) on a left axis, selectivity on a twin axis;
plus a combined **`hero.png`** overlaying the three properties' selectivity-by-depth curves with
their peak layers marked.

## 8. Tests

- **unit** (`probing.py`, 90% floor): `assign_control_labels` (determinism; base-rate≈ matched
  within tolerance; same type→same label; different seed→different map); `control_vector` (maps
  words, KeyError/handled on unseen type); `accuracy` (value, length-mismatch raises, empty);
  `base_rate`; `majority_baseline` (= max(p,1−p)); `selectivity`; `best_layer` (argmax, tie→lowest).
- **smoke** (CPU, no model download, no dataset): a **stub** with tiny synthetic residual matrices
  (e.g. a feature that linearly encodes the label at "deep" layers only) drives `20_probe`/`30_eval`
  end-to-end via an **injectable probe-fit callable** (default = sklearn LR; stub = a trivial
  fitter), asserting the metrics.json shape and that selectivity rises with the planted signal.
  Heavy imports (torch/transformers) are **lazy inside functions** so importing the scripts triggers
  no download.
- **slow** (excluded from CI): real Pythia-160m extraction + sklearn probes on real UD data.

## 9. Deps + governance

Pythia/UD need no new heavy deps: `transformers`, `torch`, `datasets`, `scikit-learn`, `numpy`,
`pandas`/`pyarrow`, `matplotlib` are all present (P2/P3 use sklearn + transformers). Add **no** new
optional-dep group unless extraction needs one. ADR `005-probing-pythia-and-control-tasks.md`: why
per-layer linear probing + the **control-task selectivity** framing (probe accuracy alone is not
evidence of encoding); raw `hidden_states` as the residual stream (no transformer_lens); UD-EWT
(open) over PTB; the last-subword alignment choice; scope = probing only (defer activation patching
+ SAE per the README). CHANGELOG `[Unreleased]` P5 entry; repo stays v0.x.

## 10. Real-run scope (PLAYBOOK §1)

UD-EWT download is tiny (~6 MB of `.conllu`); Pythia-160m (~320 MB) downloads on first run.
Extraction = one forward pass over ~12k train + ~2k test sentences (fast on CPU, seconds on GPU);
probing = 13 layers × 3 properties × 2 (probe + control) sklearn LR fits on ≤~200k × 768 matrices
(minutes). Full train/test is the headline (no subsampling needed at this scale). Build to green CI
+ ready, then run end-to-end; log the model revision; no fabricated numbers.

## 11. Repository layout (new)

```
src/awake/eval/probing.py                    # NEW pure core (+ __init__ export)
tests/test_eval_probing.py
projects/05-mechanistic-pythia/
  configs/{data,probe}.yaml                  # UD release tag; model id; properties; LR C/max_iter; seed
  scripts/{00_data,10_extract,20_probe,30_eval}.py  _paths.py _models.py _stub.py __init__.py
  tests/{conftest.py, test_data_smoke.py, test_probe_smoke.py}
  notebooks/01-layerwise-probing.py          # committed WITH outputs; self-authored worked example
  assets/{probe_is_noun,probe_is_verb,probe_noun_number,hero}.png
  REPORT.md  (README.md exists — updated)    (outputs/ gitignored; assets/ committed)
docs/decisions/005-probing-pythia-and-control-tasks.md
```

The notebook's **worked example** (per the portfolio convention that every notebook show a concrete
instance) runs a **self-authored sentence** (not UD text) through Pythia + the trained `is_noun`
probe, printing each token's predicted-noun probability at the best layer, and tracing one token's
noun-probability across all 13 layers — making "emergence with depth" concrete and licence-safe.

## 12. Limitations

Probing shows what is **linearly decodable**, not what the model **uses** causally (the control task
bounds memorisation but not causal relevance). Three properties are a **sample, not a survey**.
One model, one size (Pythia-160m) — no scaling claim. Last-subword alignment is one defensible
choice among several (first-subword / mean would differ). UD-EWT is one English domain (web text).
Linear probes only (no MLP probes). Activation patching and SAE feature inspection are explicitly
**deferred** (README scope v1.1).
