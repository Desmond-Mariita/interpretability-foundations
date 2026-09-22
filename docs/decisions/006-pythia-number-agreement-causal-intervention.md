# ADR 006 -- Pythia-160M noun-number causal intervention (v1.1): scope, design freeze, and gates

**Status:** Accepted (pre-registered)
**Date:** 2026-09-22

## Context

Project 5 v1.0 showed that `noun_number` is strongly linearly decodable from Pythia-160M's
residual stream -- probe balanced accuracy 0.983 at the embedding, selectivity rising to
0.353 at `block_11`. ADR 005 Decision 6 deferred causal verification to v1.1 and framed that
deferral around GPT-2-small ("where the IOI literature is native").

Decodability, even with high selectivity, says nothing about whether the model *uses* the
decoded direction for behaviour. The central unresolved question is:

> **Does replacing only the noun-number coordinate of a subject representation with the
> coordinate from an opposite-number donor shift Pythia-160M's next-token verb preference
> toward the donor number?**

This ADR pre-registers the v1.1 causal-intervention extension. It supersedes the ADR 005
deferral note (same-model design, see Decision 1); ADR 005 itself is a historical record and
is not rewritten.

---

## Decision 1 -- Stay on Pythia-160M (do not follow the ADR 005 GPT-2 deferral)

### Context

ADR 005 deferred activation patching to GPT-2-small because the IOI literature is native
there. But the v1.0 probe -- scaler, direction, depth axis, alignment convention, data --
exists *for Pythia-160M*. Moving to GPT-2 would require re-deriving a probe and would test a
different representation than the one v1.0 established.

### Decision

The v1.1 intervention stays on **Pythia-160M** (pinned revision `50f5173d`, GPT-NeoX, 12
blocks, `d_model=768`) and tests the causal relevance of the *exact* representation v1.0
probed:

```text
same model
same property (noun_number)
same residual points (embedding + block_0..block_11)
same probe protocol (train split, StandardScaler, LR, C=0.01)
        ↓
causal intervention
```

### Consequences

- The result speaks directly to the v1.0 decodability finding; the layerwise selectivity
  curve and the layerwise causal-effect curve are directly comparable.
- The ADR 005 v1.1 deferral note (GPT-2-small / sae-lens) is superseded for the causal
  extension only; SAE inspection remains out of scope.
- No model-family/size switch is permitted to obtain a positive result (stop condition 5).

## Decision 2 -- Target property: `noun_number` only

`noun_number` is the only v1.1 target. It has the strongest v1.0 signal (selectivity 0.218
at embedding, 0.353 at block_11), it is behaviourally grounded (subject--verb agreement is
an observable next-token behaviour), and it has the clearest confound story
(plural-suffix orthography at layer 0), which is exactly why the causal test matters:
decodability is high, but does the model *use* the direction?

`is_noun` / `is_verb` probing is not extended in v1.1.

## Decision 3 -- Research question and pre-registered hypotheses

**Primary RQ.** Does replacing only the noun-number coordinate of a subject representation
with the coordinate from an opposite-number donor shift Pythia-160M's next-token verb
preference toward the donor number?

- **H1 (behavioural competence prerequisite).** On un-intervened stimuli, subject number
  affects the singular-vs-plural verb logit contrast in the grammatically expected
  direction. (This is the competence gate, Decision 4.)
- **H2 (direction-only causal effect).** Replacing the recipient's projection on the
  decoded number direction with the projection from an opposite-number donor shifts the
  verb logit contrast toward the donor number.
- **H3 (specificity).** The opposite-number direction patch produces a stronger
  donor-directed shift than no-op, same-number donor transfer, and norm-matched
  random-direction transfer.
- **H4 (full-residual positive control).** Replacing the full subject-position residual
  with an opposite-number donor residual provides a broader positive-control intervention
  on the matched-tokenization subset.
- **H5 (decodability != causal effect).** The layerwise causal-effect curve is allowed to
  differ from the selectivity curve; there is no assumption that they peak at the same
  layer.

Null results are acceptable and are not a stop condition.

## Decision 4 -- Stimulus design, splits, and the competence gate

### Stimuli

Deterministic, versioned, minimal matched singular/plural pairs. Template families:

- `simple`: `The {SUBJECT}`
- `near`: `The {SUBJECT} near the park` (neutral intervening context)
- `attractor`: `The {SUBJECT} near the {ATTRACTOR}` with attractor set
  `{park, parks, river, rivers, gate, gates, fence, fences}` (fixed surface + explicit
  number annotation; both same-number and opposite-number attractor conditions arise
  naturally since each subject pair spans both numbers; the attractor surface is fixed
  across the singular/plural subject pair).

The prompt stem ends immediately before the target verb. Target verb pairs (singular,
plural): `(is, are)`, `(was, were)`, `(has, have)`, `(does, do)`. Each verb must be exactly
one tokenizer token including leading-space convention under the pinned tokenizer; a pair
failing this check is excluded under a deterministic rule and the exclusion is recorded in
the run manifest (no mixed scoring definitions).

For each item, `M = logit(plural_verb) - logit(singular_verb)` from next-token logits at the
final stem position; `y = +1` for plural subject, `-1` for singular; the grammatical
agreement margin is `A = y * M`.

The subject is aligned to its **last overlapping subword** (the v1.0 convention, ADR 005
Decision 2), and the intervention targets that exact token position. Subject positions and
subword counts are recorded per stimulus.

### Noun lexicon (derived from UD English-EWT train, `r2.14`)

Per lemma, from NOUN tokens annotated `Number=Sing` / `Number=Plur`:

1. observed singular and plural surface sets (letter-only lowercase surfaces
   `^[a-z]+$`, else excluded);
2. chosen surfaces = most frequent surface per number (tie-break: prefer surface ==
   lemma, then lexicographic);
3. require both surfaces observed with at least the configured minimum frequency
   (`min_surface_freq`, see `configs/causal.yaml`);
4. exclude degenerate pairs (`plural_surface == singular_surface`, e.g. zero-plurals);
5. stratum: `suffix_transparent` iff `plural == singular + "s"` or
   `plural == singular + "es"`; else `nontransparent`.

The stratum rule is a surface heuristic, not a linguistic morphology claim; the strata are
named accordingly (no claim of "regular vs irregular").

### Splits (anti-overfitting discipline)

- `pilot`: the first 5 lemmas (sorted) across all templates/attractors -- code debugging,
  tokenization validation, hook validation ONLY; never analysed.
- `dev` / `test`: remaining lemmas split 70/30 by a seeded RNG over the sorted lemma list;
  **noun lemmas are disjoint** between dev and test.
- **Template-family holdout:** dev uses `simple` + `near` only; the `attractor` family is
  held out of dev and appears only in test.
- Split seed, lemma lists, template IDs, and per-stimulus SHA-256 hashes are persisted in
  `outputs/stimuli/` and the split hashes are committed in the design-freeze manifest
  (Decision 6) **before** any model evaluation on dev or test.

The pilot may be used only for plumbing. Dev may be used for the competence gate and for
verifying the intervention implementation. The test split is isolated before model
evaluation and is run **once** (confirmatory); no tuning based on test outcomes. No
template/search is tuned until the result becomes positive.

### Competence gate (H1) -- pre-registered, evaluated on dev before any intervention

Do not assume Pythia-160M has sufficient subject--verb agreement behaviour. On dev baseline
(un-intervened) items:

- **Gate 1 (paired subject-number effect).** Per (lemma, verb) pair, over dev stimuli,
  `d(lemma, verb) = mean(M(plural subject) - M(singular subject))` (templates averaged).
  Statistic = mean of `d` over all (lemma, verb) rows. Pass: **lemma-cluster bootstrap 95%
  CI strictly above 0** (2,000 resamples, lemmas resampled with replacement, seed fixed in
  config).
- **Gate 2 (directional cluster accuracy).** Fraction of (lemma, verb) rows with
  `d > 0`. Pass: **lemma-cluster bootstrap 95% CI lower bound > 0.5** (same resampling).

The cluster is the noun lemma because the same lemma appears in multiple
templates/verb pairs. The exact gate (statistics, resampling, thresholds) is defined in
`configs/causal.yaml` before dev evaluation.

**Mandatory stop condition.** If the gate fails: STOP the scientific run. Do not switch
model, search prompts/templates, loosen the gate, or cherry-pick verbs. The boundary result
is reported as: *Pythia-160M did not demonstrate the prerequisite agreement behaviour under
the pre-registered stimulus design, so a causal number intervention is not interpretable.*

## Decision 5 -- Probe-direction recovery, intervention, and controls

### Direction recovery (train split only)

For each layer/point, the `noun_number` probe is refitted on the v1.0 train split under the
v1.0 protocol (StandardScaler per point; `LogisticRegression(class_weight="balanced",
max_iter=2000, C=0.01)` -- the v1.0-chosen C, re-verified on dev; train cap 60k, seed 0).

In standardized coordinates `z = (x - mean) / scale`, the classifier is `w_z . z + b`.
The equivalent residual-space normal is:

```text
w_x = w_z / scale        (componentwise)
u   = w_x / ||w_x||
```

Class ordering is verified so that the sign of `u` is explicitly aligned:
**positive projection = plural direction** (LR `classes_ = [0, 1]` with `0=Sing, 1=Plur`;
the sign is flipped if the fit produces the reverse ordering).

Persisted per layer: scaler mean/scale, `w_z`, `b`, `w_x`, `u`, class mapping, `C`,
training-data identity (split + subset + cap seed), and a SHA-256 hash of the direction
artifact. A unit test asserts that for arbitrary residual vectors the standardized
classifier score and the residual-space score agree up to the transformed intercept
(the raw standardized coefficient is never used directly as a residual-space direction).

A probe-reconstruction gate: the refit probe must reproduce the v1.0 recorded
`noun_number` balanced accuracies (embedding 0.983, block_11 0.977) within a documented
tolerance; failure is stop condition 3.

### Intervention

For a recipient residual `h_r` and a matched donor residual `h_d` at the **subject
position** and layer `l`, with unit direction `u_l`:

```text
a_r = u_l . h_r      a_d = u_l . h_d
h'_r = h_r + (a_d - a_r) * u_l
```

Only the coordinate along the decoded number direction is replaced; the orthogonal
component is unchanged. Only the designated subject-position residual is modified in each
patched forward pass; all other positions are untouched; next-token verb logits are
computed normally.

**Donors.** Opposite-number donor = the matched counterpart stimulus (same lemma, template,
attractor, opposite subject number) -- the matched singular/plural pair.

### Layers

`embedding, block_0, ..., block_11` (13 causal points), plus `ln_f` included ONLY as a
**terminal/no-downstream-mixing control** (patching the subject position after the final
layernorm has no route to the prediction position; an effect there would indicate a bug).

Causal-architecture sanity point (documented): a patch at the subject position after the
final block (`block_11`) has no attention layer left to propagate it to a later token
position, so `block_11` is a near-zero sanity control for downstream agreement effects --
lack of effect there is not absence of number information.

### Controls

- **C0 no-op.** Hook infrastructure active, no residual change. Must reproduce baseline
  logits within numerical tolerance (max |delta| < 1e-6).
- **C1 same-number donor direction patch.** The number-direction coordinate is replaced
  using a donor with the same grammatical number. Deterministic matching: cyclic pairing of
  lemmas within (split, template, attractor condition, subject number). Controls for
  projection-replacement magnitude and donor variability without reversing number.
- **C2 norm-matched random-direction patch.** Random unit directions (fixed seeds),
  orthogonalized to `u_l`, applied with per-example norm equal to the primary patch's
  delta norm: `||delta_random|| = ||delta_number||`. K=3 seeds.
- **C3 full subject-position residual replacement** (`h'_r = h_d`), restricted to the
  documented **matched-tokenization subset** (stimulus pairs whose singular and plural
  subjects have equal subword counts and equal subject positions); never silently compared
  across non-equivalent positions.

### Primary causal outcome

Per item, layer, condition: `M = logit(plural_verb) - logit(singular_verb)`. For a donor
with number label `y_d` in {-1, +1}, the donor-directed shift is
`E = y_d * (M_patched - M_baseline)`. Positive `E` = the intervention shifted the verb
preference toward the donor's grammatical number.

**Primary estimand:** per layer, paired mean donor-directed shift under the opposite-number
direction patch, with 95% **lemma-cluster bootstrap** CIs (resampling unit = lemma, not
individual repeated template rows).

### Secondary outcomes

Fraction of examples shifting in donor-consistent direction; verb-preference flip rate;
effect under same-number donor; effect under random-direction controls; effect under
full-residual replacement; effects by simple vs intervening templates, attractor
conditions, and noun strata; baseline agreement margin by stratum. Subgroup results are
descriptive unless sample size supports more.

### Decodability vs causality comparison

Layerwise `noun_number` selectivity (v1.0) is plotted against the layerwise causal effect.
Interpretation rules: *layers with stronger linear selectivity need not be the layers where
manipulating the decoded direction most affects agreement behaviour*; a layerwise
correlation, if computed, is descriptive only (few layers, adjacent layers not
independent); no claim that a correlation proves the probe found the model's
"grammatical-number circuit".

## Decision 6 -- Design-freeze discipline and artifact provenance

Before the confirmatory test run:

1. commit the implementation/config/stimulus generator;
2. write a design-freeze manifest (`outputs/stimuli/design_freeze.json` + a committed
   summary section in `V11_CAUSAL_REPORT.md`) containing: git SHA; model revision;
   tokenizer revision; UD revision; stimulus-generation config; lemma split hash; template
   split hash; target verb pairs (post-validation); random seeds; intervention definitions;
   competence gate; primary metric; bootstrap unit;
3. record this as the confirmatory design.

After the freeze: no parameter/template changes based on test results; any necessary bug
fix invalidates the test run and creates a new explicitly versioned design freeze.

Artifacts persisted locally under `outputs/` (gitignored): stimulus table, split table,
probe-direction artifacts, baseline logits, per-example intervention results, bootstrap
inputs, run manifest. Committed public artifact:
`projects/05-mechanistic-pythia/assets/causal_metrics.json` (authoritative aggregates,
model/data revisions, design-freeze SHA, stimulus hash, direction hashes, config hash, run
ID). A manifest with SHA-256 hashes covers all publication-critical local outputs.

## Decision 7 -- Claim boundaries and failure-mode interpretation

Allowed final language (positive): *Replacing only the subject representation's projection
on the decoded noun-number direction with the projection from an opposite-number donor
shifted Pythia-160M's verb-number preference toward the donor at specific layers, relative
to no-op, same-number, and norm-matched random-direction controls. This is evidence that
the decoded direction makes a causal contribution to agreement behaviour under the tested
intervention.*

Allowed final language (null): *Noun number is highly linearly decodable, but manipulating
the decoded direction did not produce a direction-specific change in agreement behaviour
under the tested intervention. This separates decodability from causal use.*

Never written: "we found the grammar circuit"; "this proves the representation is the
mechanism"; "the model understands number"; "probe selectivity proves causal use";
"CI overlap proves layers are equivalent"; "nonsignificant = no effect."

Failure-mode interpretation matrix (used explicitly in the v1.1 report):

| Observation | Conclusion |
|---|---|
| Competence fails | prerequisite behaviour not established; causal intervention uninterpretable |
| Competence passes; full residual works; number-direction patch null | number is decodable, but the decoded linear direction is not sufficient evidence of causal contribution to this agreement behaviour (strong, acceptable result) |
| Number-direction and random-direction effects similar | intervention is not direction-specific; causal number claim unsupported |
| Number-direction > controls and shifts toward donor | evidence that the decoded number direction makes a causal contribution to agreement behaviour under this intervention (no uniqueness/necessity/mediation/grammar-circuit claims) |
| Full residual null too | investigate downstream route to the prediction position, subject position, and other positions/signals; do not force a positive interpretation |

## Decision 8 -- Stop conditions

Stop and report to the Gatekeeper without switching scope if any of:

1. Pythia-160M fails the pre-registered agreement competence gate.
2. Subject positions cannot be aligned robustly enough for intervention.
3. The existing `noun_number` probe cannot be reconstructed faithfully enough to derive a
   direction.
4. A bug requires changing the confirmatory design after test results were seen.
5. The experiment would require changing model family/size to obtain interpretable
   behaviour.
6. The full-residual and no-op controls reveal the intervention hook is not behaving
   causally as intended.
7. Any outcome/artifact is untraceable to the frozen design.

A negative result is **not** a stop condition.

---

## Summary table

| Decision | Chosen option | Key rationale |
|---|---|---|
| Model scope | Stay on Pythia-160M | Tests the exact representation v1.0 probed; ADR 005 GPT-2 deferral superseded for causal extension only |
| Target | `noun_number` only | Strongest v1.0 signal; behaviourally grounded; clearest confound story |
| Stimuli | 3 template families; 4 verb pairs; UD-train-derived noun lexicon; 2 surface strata | Minimal matched pairs; deterministic; single-token verbs enforced |
| Splits | pilot / dev / test; lemma-disjoint; attractor family held out of dev | Anti-overfitting; test isolated pre-evaluation |
| Competence gate | dev baseline: paired subject-number effect CI > 0 + directional cluster accuracy CI_lo > 0.5 (lemma-cluster bootstrap) | Prerequisite behaviour established before intervention is interpretable |
| Direction | Train-split refit; `w_x = w_z/scale`, `u = w_x/||w_x||`; positive = plural | Equivalent residual-space direction; sign aligned explicitly |
| Intervention | Direction-only coordinate replacement at subject position; 13 causal layers + `ln_f` terminal control | Minimal, specific, downstream-route documented |
| Controls | C0 no-op; C1 same-number; C2 norm-matched orthogonal random (K=3); C3 full-residual (matched-tokenization subset) | Specificity vs magnitude/donor-variability/position confounds |
| Primary estimand | Per-layer paired mean donor-directed shift, lemma-cluster bootstrap CI | Correct resampling unit; directly comparable to selectivity |
| Claims | Directional causal-contribution language only; failure-mode matrix | No circuit/mechanism/understanding over-claims |
