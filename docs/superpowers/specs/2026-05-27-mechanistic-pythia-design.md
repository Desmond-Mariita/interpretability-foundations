# Project 5 — `05-mechanistic-pythia`: at which layer does a property become linearly decodable?

**Status:** v2 — rewritten after four-way review round 1 (3× ACCEPT-WITH-CHANGES, 1× REWORK;
all blockers/majors folded in) · 2026-05-27 · author Desmond Mariita. Awaiting round-2 review.

Part of `interpretability-foundations`, built per `docs/PLAYBOOK.md`. **No HF Space.**
Hardware: CPU is sufficient (Pythia-160M); the RTX 3090 only speeds activation extraction.
§13 records review traceability.

## 1. Question and deliverable

**At which layer of a small open transformer does a given binary linguistic property become
linearly decodable from the residual stream — beyond what a probe can recover from word identity
alone?** We answer with **per-layer linear probes** on **Pythia-160M**, each paired with a
**Hewitt & Liang (2019) control task** (random per-type labels). The headline per (property, layer)
is **selectivity = balanced_accuracy(probe) − balanced_accuracy(control)**.

**What selectivity does and does not show (honest framing).** A probe beating its control shows the
property is *more linearly recoverable than an arbitrary word-type code* at that layer — it is
**necessary but not sufficient** evidence of genuine encoding. Lexical identity, suffix orthography,
token frequency, and tokenisation artefacts can still drive positive selectivity, and linear
decodability is **not** evidence the model *uses* the feature causally. The depth at which
selectivity rises is a description of *linear availability by depth*, not a causal claim.

Deliverables: `metrics.json` (per property × per probe-point: balanced accuracy + raw accuracy for
probe and control, selectivity, majority baseline, all with **sentence-cluster bootstrap 95% CIs**;
selectivity averaged over **K control seeds**; peak-point with its bootstrap uncertainty;
train→test type-overlap/OOV rates), **one figure per property** + a combined hero, a notebook
(committed **with outputs**, with a **self-authored worked example** — never UD text), `REPORT.md`,
an ADR. **No Space, no deploy.**

## 2. Data — Universal Dependencies English-EWT (open; code-only)

UD English-EWT (Silveira et al. 2014), **CC BY-SA 4.0** (explicit attribution in REPORT + ADR) —
openly licensed, unlike PTB/WSJ. The `datasets` script loader is dead (5.x dropped scripts), so
`scripts/00_data.py` fetches the three `.conllu` files from a **pinned UD release tag** (e.g.
`r2.14`) via stdlib `urllib.request` (no new dep) and **verifies a SHA-256** recorded in
`configs/data.yaml` (fail loudly on mismatch; idempotent — skip if the verified file exists).

**Faithful CoNLL-U parse** (the alignment foundation — §3): for each sentence keep the comment
`# text = …` (the gold surface string) and, per token row (skip `#` comments and multiword/empty
rows whose `ID` contains `-`/`.`): `FORM, UPOS, FEATS, MISC`. Record `SpaceAfter=No` from `MISC` so
the surface string can be reconstructed exactly (handles `do`/`n't`, punctuation, no-space joins).
Write `outputs/prepared/{train,dev,test}.parquet` with an **explicit pyarrow schema**
(`sent_id: str`, `text: str`, `words: list<str>`, `upos: list<str>`, `number: list<str>`,
`space_after: list<bool>`) so empty strings never coerce to null. **Never commit** the `.conllu` or
parquet (raw text); `outputs/` is gitignored.

## 3. Model and activation extraction (frozen; deterministic)

`EleutherAI/pythia-160m` (`GPTNeoXForCausalLM`; 12 layers, `d_model` 768) loaded with a **pinned
`revision`** (commit SHA in `configs/probe.yaml`), `eval()`, no grad. Log the resolved model SHA,
**tokenizer SHA**, and `transformers`/`torch` versions to `metrics.json`.

**Residual-stream points via forward hooks (not the `hidden_states` tuple).** The HF
`output_hidden_states` tuple for GPT-NeoX is subtly mislabelled for our purpose: index 0 is the
embedding *input*, RoPE is applied inside attention (not added to the residual), and the **last**
entry is **post-final-layernorm**, not block-12's `resid_post`. To get clean, correctly-labelled
points we **register forward hooks** on `model.gpt_neox.layers[i]` capturing each block's output
(`resid_post`, i = 0..11 → 12 points) plus the **embedding output** (`resid_pre` of block 0 → 1
point), giving the **13-point depth axis** `["embedding", "block_0", …, "block_11"]`. We
additionally capture **`ln_f`** (the final layernorm output the unembedding actually reads) as a
**separate, clearly-labelled extra point** (not on the depth axis), so the pre-/post-LN distinction
is explicit rather than hidden.

**Token↔word alignment (full-sentence, offset-based).** For each sentence: reconstruct the surface
string from `words` + `space_after` (cross-checked against `# text =`), tokenise the **whole
sentence once** with the Pythia **Fast** tokeniser using `return_offsets_mapping=True`, and map each
UD word's character span to the subword tokens whose offsets fall inside it; take the residual at
the **last subword** of each word as that word's representation. (Encoding words separately is
incorrect for byte-level BPE — leading-space bytes and cross-word merges shift positions.) Words
that fail to align (rare; e.g. tokenizer normalisation) are dropped and counted in `metrics.json`.

`scripts/10_extract.py` streams the corpus and writes **per-point** arrays to
`outputs/acts/<split>/<point>.npy` (float16, `(n_words, 768)`) plus an aligned
`outputs/acts/<split>/meta.parquet` (`word, upos, number, sent_id`, row-aligned). Per-point `.npy`
(not one big `.npz`) keeps each ~`n_words×768×2` array memmappable; full train activations are
~3–4 GiB across 14 points, loaded one point at a time.

## 4. Properties (3 binary, token-level)

Defined on UD tokens (one label per word, aligned to its last-subword residual):
- **is_noun** — `upos == "NOUN"` (common noun; train base rate measured at run, ≈ 0.17).
- **is_verb** — `upos == "VERB"` (≈ 0.11; imbalanced — see §5 balanced-accuracy choice).
- **noun_number** — among tokens with `upos == "NOUN"` **and** a `Number` feature: `Plur` = 1,
  `Sing` = 0. The exact train n and base rate are **measured and logged at run time** (and asserted
  to be ≥ a configurable minimum, e.g. 3000 train tokens, else the property is reported as
  underpowered, not silently). This property is **conditional on the model encoding noun-hood**
  (the subset is UD-annotated nouns) and the last subword of plural nouns often *is* the plural
  morpheme — so high **layer-0/embedding** noun_number scores reflect subword orthography, not
  transformer computation; this is called out in §12 and in the figure caption.

## 5. Probe, control task, metric, and CIs

**Pre-probe standardisation (required).** Residual-stream L2 norm grows with depth in a pre-LN
model; with L2-regularised logistic regression this would confound "emergence" with norm growth.
So per (property, point) we fit a `sklearn.preprocessing.StandardScaler` on **train** residuals and
apply it to dev/test before the probe.

**Probe.** `LogisticRegression(class_weight="balanced", max_iter=2000, random_state=…)`; the
inverse-reg strength `C` is chosen **once per property on the dev split** (small grid, e.g.
{0.01, 0.1, 1.0}) and then fixed across all points for that property (so cross-layer comparison
holds `C` constant). Probes are fit on **train**, scored on **test**.

**Primary metric = balanced accuracy** (mean of per-class recall): robust to the is_verb/is_noun
imbalance (chance = 0.5 regardless of prevalence). Raw accuracy and the **majority-class baseline**
(train-majority class scored on test) are kept as secondary context; AUROC is logged too.

**Control task (Hewitt & Liang 2019), token-base-rate matched, multi-seed.** Build a random
per-**word-type** label map (using the **exact surface form**, case-sensitive — the natural key for
a byte-BPE model) over the **union of train+test types**, with the positive share matched in
**token space**: greedily assign label 1 to types (in seeded-random order) accumulating their
**train token frequency** until the cumulative positive *token* mass ≈ the property's train token
base rate (the realised token-level control base rate is logged). The same standardise→
LogisticRegression probe is fit on train control labels and scored on test. The probe is trained
only on train tokens, so test-only types are genuinely unseen (it must guess) — we **report the
train→test type-overlap and OOV token rate**, and report selectivity on **all-test** and on the
**seen-type subset** separately. To avoid a single random draw biasing the result, the control is
repeated over **K seeds** (e.g. K = 5) and control balanced-accuracy is **averaged over seeds**
(seed spread reported).

**Selectivity = balanced_accuracy(probe) − mean_k balanced_accuracy(control_k)** per point.

**Confidence intervals — paired sentence-cluster bootstrap.** Tokens within a sentence are not
independent, and selectivity is a paired difference. We resample **sentences** with replacement
(cluster bootstrap), recompute balanced accuracy for probe and control on the **same** resample, and
take the distribution of their difference for the selectivity CI (paired). CIs are reported for
probe, control, and selectivity at every point. **Peak-point uncertainty:** rather than over-reading
a noisy `argmax`, we report the bootstrap distribution of the argmax point and the **earliest point
whose selectivity CI overlaps the peak's** as the "emergence" summary.

## 6. Shared pure core — `src/awake/eval/probing.py` + `bootstrap.py` additions (pure, 90% floor)

Pure logic (no models, no sklearn, no I/O); the standardise+LR fit and activation extraction stay in
scripts (slow-tested).

```python
# probing.py
def assign_control_labels(train_words: list[str], base_rate: float, seed: int) -> dict[str, int]:
    """H&L control: deterministic random per-type binary label, with the positive share matched
    in TOKEN space (weighted by each type's train frequency) to ~base_rate. Same args -> same map.
    Keyed on exact surface form. Returns a label for every type present in train_words."""
def control_vector(words: list[str], type_to_label: dict[str, int]) -> list[int]:
    """Map each token's exact-surface-form type to its control label. Raises KeyError on a type
    absent from the map (a caller error: the map must be built over the train+test union)."""
def balanced_accuracy(pred: list[int], gold: list[int]) -> float:
    """Mean of per-class recall (chance = 0.5); 0.0 for empty; ValueError on length mismatch.
    A class with no gold instances is omitted from the mean."""
def accuracy(pred: list[int], gold: list[int]) -> float: ...          # raw; None/len rules as P4
def majority_class(train_labels: list[int]) -> int: ...                # 1 if >half positive else 0
def base_rate(labels: list[int]) -> float: ...
def selectivity(probe_metric: float, control_metric: float) -> float: ...
def type_overlap(train_words: list[str], test_words: list[str]) -> dict:
    """{'seen_type_token_rate': .., 'oov_type_token_rate': ..} for the test set vs train types."""
def emergence_point(sel_by_point: dict[str, float], sel_ci_by_point: dict[str, tuple]) -> dict:
    """{'peak': <point>, 'earliest_within_peak_ci': <point>} (lowest-index tie-break)."""

# bootstrap.py (additions)
def cluster_bootstrap_ci(y_true, y_pred, groups, metric_fn, n_resamples=2000, alpha=.05, seed=0): ...
    # resample GROUPS (sentences) with replacement; recompute metric_fn(y_true', y_pred') -> (lo,mean,hi)
def paired_cluster_bootstrap(y_true, pred_a, pred_b, groups, metric_fn, ...): ...
    # CI of metric_fn(a)-metric_fn(b) on the SAME resampled groups (paired)
```
Reuse existing `bootstrap_ci` only for non-clustered cases. Export all new names from
`src/awake/eval/__init__.py`.

## 7. Eval + figures

`scripts/20_probe.py` — per property, per point: standardise→fit probe (train) + K control probes
(train), score on test, store per-token `(gold, pred)` for probe and each control plus `sent_id`
for clustering. `scripts/30_eval.py` — assemble `metrics.json`:
`{"model":"EleutherAI/pythia-160m","model_revision":<sha>,"tokenizer_revision":<sha>,
"versions":{"transformers":..,"torch":..},"points":["embedding","block_0",…,"block_11"],
"extra_points":["ln_f"],"control_seeds":[…],
"properties":{"is_noun":{"train_n":..,"test_n":..,"base_rate":..,"majority_baseline":..,
"type_overlap":{…},"dropped_alignment":..,
"points":[{"point":"embedding","balanced_acc":..,"balanced_acc_ci":[lo,hi],"raw_acc":..,
"auroc":..,"control_balanced_acc":..,"control_ci":[lo,hi],"control_seed_spread":..,
"selectivity":..,"selectivity_ci":[lo,hi]}, … 13 + ln_f],
"emergence":{"peak":"block_k","earliest_within_peak_ci":"block_j"}}, "is_verb":{…},
"noun_number":{…}}}`. All accuracies are **balanced** unless prefixed `raw_`.

Figures (`assets/`): **one per property** — `probe_<property>.png`: probe balanced-acc, mean control
balanced-acc, and the 0.5 chance line vs the 13-point depth axis (with cluster-bootstrap CI bands),
selectivity on a twin axis, the `ln_f` point drawn off-axis as a marker, peak/earliest-emergence
annotated; plus `hero.png` overlaying the three properties' selectivity-by-depth curves with peaks
marked. Captions note the layer-0 = embedding (orthography, not computation) caveat.

## 8. Tests

- **unit** (`probing.py` + new `bootstrap.py` fns, 90% floor): `assign_control_labels`
  (determinism; **token-level** positive share within tolerance of base_rate; same type→same label;
  different seed→different map); `control_vector` (maps; **raises KeyError** on a type absent from
  the union map); `balanced_accuracy` (per-class-recall value, imbalanced case, empty-class omitted,
  length-mismatch raises); `accuracy`; `majority_class`; `base_rate`; `selectivity`; `type_overlap`
  (seen/OOV token rates on a hand case); `emergence_point` (peak + earliest-within-CI, tie-break);
  `cluster_bootstrap_ci`/`paired_cluster_bootstrap` (grouped resampling is deterministic under seed;
  a planted between-group difference yields a CI excluding 0; resampling groups not rows).
- **smoke** (CPU, no model download, no dataset): a **stub** producing tiny synthetic per-point
  residual matrices where the label is linearly encoded only at "deep" points, driving
  `20_probe`/`30_eval` via an **injectable probe-fit callable** (default = standardise+sklearn LR;
  stub = a trivial linear fitter), asserting the metrics.json shape and that
  **selectivity(deep point) − selectivity(embedding) ≥ a margin** (not full monotonicity). Heavy
  imports (torch/transformers) are **lazy inside functions**.
- **slow** (excluded from CI): real Pythia-160m hook extraction + sklearn probes on real UD data.

## 9. Deps + governance

No new heavy deps (`transformers`, `torch`, `datasets`, `scikit-learn`, `numpy`, `pandas`/`pyarrow`,
`matplotlib` all present; fetch uses stdlib `urllib`). ADR `005-probing-pythia-and-control-tasks.md`:
hook-based resid_post extraction (and why not the `hidden_states` tuple) + the separate `ln_f`
point; offset-based word alignment; **balanced-accuracy + token-rate-matched multi-seed control +
StandardScaler + paired sentence-cluster bootstrap**; the honest "necessary-not-sufficient"
selectivity framing; UD-EWT (CC BY-SA, attributed) over PTB; scope = probing only (defer activation
patching + SAE per the README). CHANGELOG `[Unreleased]` P5 entry; repo stays v0.x.

## 10. Real-run scope (PLAYBOOK §1)

UD-EWT download ~6 MB; Pythia-160m (~320 MB) on first run. Extraction = one forward pass over
~12.5k train + ~2k test sentences (minutes on CPU, faster on GPU). **Activation size is real:**
~200k train tokens × 768 × 2 bytes × 14 points ≈ 3.7 GiB — stored **per-point** and loaded one point
at a time. Probing: to keep the **78+ sklearn fits** (13 points × 3 props × (1 probe + K controls))
tractable (~1–2 h CPU worst case), train probes on a **seeded cap** of train tokens (configurable,
e.g. 60k, class-stratified) — the headline test set is full; `n_jobs=-1` where the solver allows.
Full test is the headline. Log all revisions/versions; no fabricated numbers.

## 11. Repository layout (new)

```
src/awake/eval/probing.py                    # NEW pure core (+ __init__ export)
src/awake/eval/bootstrap.py                  # + cluster_bootstrap_ci, paired_cluster_bootstrap
tests/test_eval_probing.py                   # + tests for the new bootstrap fns
projects/05-mechanistic-pythia/
  configs/{data,probe}.yaml                   # UD tag+SHA; model+tokenizer revision; properties;
                                              #   C grid; control seeds K; train cap; alignment=last
  scripts/{00_data,10_extract,20_probe,30_eval}.py  _paths.py _models.py _stub.py __init__.py
  tests/{conftest.py, test_data_smoke.py, test_probe_smoke.py}   # conftest = P4 isolation fixture
  notebooks/01-layerwise-probing.py           # committed WITH outputs; self-authored worked example
  assets/{probe_is_noun,probe_is_verb,probe_noun_number,hero}.png
  REPORT.md  (README.md exists — updated)     (outputs/ gitignored; assets/ committed)
docs/decisions/005-probing-pythia-and-control-tasks.md
```

`_paths.py` resolves the project root via `P5_PROJECT_ROOT` (then `__file__`, then cwd). The
project `conftest.py` copies P4's autouse `_isolate_project_scripts` fixture, evicting
`{_paths,_models,_stub,00_data,10_extract,20_probe,30_eval}` from `sys.modules` (bare-module imports
collide across projects). The notebook's **worked example** runs a **self-authored sentence** (never
UD text) through Pythia + the trained `is_noun` probe: per-token predicted-noun probability at the
peak point, and one token's noun-probability traced across all 13 points — making emergence concrete
and licence-safe.

## 12. Limitations

- **Selectivity is necessary, not sufficient,** for encoding: beating a random type-control shows
  linear recoverability beyond an arbitrary type code, but lexical identity, **suffix orthography**
  (esp. plural `-s` for noun_number), **token frequency**, **position**, and **type-cluster
  structure** (H&L bounds individual-type but not cluster memorisation; Ravichander et al. 2021) can
  all produce positive selectivity. Linear decodability ≠ the model *uses* the feature.
- **Last-subword pooling** is one choice; it entangles emergence with subword count and (for
  noun_number) the plural morpheme. `first`/`mean` pooling is a config option; a `last`-vs-`first`
  sensitivity check is reported if time permits, else flagged as future work. We report the
  single-vs-multi-subword split per property.
- **Pre-LN / `ln_f`:** the 13 depth points are pre-final-LN residual states; `ln_f` (what the
  unembedding reads) is reported separately. Layer-0 = token embeddings (no transformer computation).
- **noun_number** is conditional on noun-hood being encoded and is sample-limited; underpowered runs
  are flagged, not hidden.
- One model, one size (Pythia-160m) — **no scaling claim**; one English web-text domain (UD-EWT);
  **linear** probes only. Activation patching and SAE inspection are explicitly **deferred** (README
  scope v1.1).

## 13. Review traceability (round 1 → folded into v2)

Round 1 four-way review (2 sonnet + Codex + Gemini): 3× ACCEPT-WITH-CHANGES, 1× REWORK (Codex,
web-verified against the GPT-NeoX source). All blockers/majors folded in:
- **Hook-based resid_post extraction** + separate `ln_f` point; corrected the `hidden_states`-tuple
  mislabelling (index 0 = embedding input; last = post-`ln_f`; RoPE not in residual) ← Codex BLOCKER.
- **Offset-based full-sentence alignment** with `SpaceAfter=No` surface reconstruction (not
  encode-separately, not whitespace split) ← Codex + engineering BLOCKER.
- **Token-base-rate-matched, multi-seed control** over the train+test union; **report OOV/seen-type
  rates**; selectivity on all-test and seen-type subsets ← methodology + Gemini + Codex.
- **Per-layer StandardScaler** + `C`/`class_weight` chosen on **dev** ← Gemini BLOCKER + Codex +
  methodology (residual-norm growth confound).
- **Balanced accuracy** primary (raw acc + majority baseline + AUROC secondary) ← all four.
- **Paired sentence-cluster bootstrap** for CIs; **peak-point uncertainty** (earliest-within-CI, not
  raw argmax) ← all four (CI) + Codex/methodology (argmax).
- **Toned-down selectivity claims** ("necessary not sufficient"; lexical/suffix/frequency/position/
  type-cluster confounds) ← methodology + Codex.
- **Last-subword pooling** sensitivity + single/multi-subword split + plural-suffix/layer-0 caveat ←
  methodology + Gemini + Codex.
- **Realistic compute/storage** (3.7 GiB; per-point `.npy`; train-token cap; `n_jobs`) ← Gemini +
  Codex.
- Engineering: `control_vector` KeyError contract; `majority_class` train→test; `.conllu` SHA +
  pinned model/tokenizer revisions + logged `transformers`/`torch` versions; explicit pyarrow
  string-list schema; stdlib `urllib`; conftest isolation; `P5_PROJECT_ROOT`; float16→float64 upcast
  noted; precise smoke assertion; `best_layer`/`emergence_point` tie-break; UD CC BY-SA attribution;
  notebook never prints UD text ← engineering + Codex + Gemini.

Round 2 (this v2) re-runs the full four-way review to confirm the §3/§5 rework is correct.
