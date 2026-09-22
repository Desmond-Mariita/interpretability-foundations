# P5 v1.1 -- Causal number-agreement intervention: does the decoded noun-number direction drive behaviour?

**Author:** Desmond Mariita.
**Model:** Pythia-160M (`EleutherAI/pythia-160m`), pinned revision `50f5173d`.
**Dataset:** UD English-EWT `r2.14` (CC BY-SA 4.0) for the noun lexicon; self-constructed
controlled stimuli for the intervention (no UD text committed).
**Status:** <STATUS_PLACEHOLDER>.
**Design:** pre-registered in
[ADR 006](../../docs/decisions/006-pythia-number-agreement-causal-intervention.md) and
`configs/causal.yaml`; every number below is read from
`outputs/causal_metrics.json` / `assets/causal_metrics.json` (no fabrication).

---

## 1. Motivation: from decodability to causal use

v1.0 established that `noun_number` is strongly linearly decodable from Pythia-160M's
residual stream: probe balanced accuracy 0.983 at the embedding layer, rising selectivity
to 0.353 at `block_11`. But decodability -- even with a Hewitt & Liang control task -- does
not show that the model **uses** the decoded direction for behaviour. The unresolved
question is:

> **Does replacing only the noun-number coordinate of a subject representation with the
> coordinate from an opposite-number donor shift Pythia-160M's next-token verb preference
> toward the donor number?**

This is a causal contribution question about a representation, not a claim that we have
identified the unique mechanism or circuit for grammatical number.

## 2. Existing-v1 claim cleanup (bounded repair)

Before the extension, the v1.0 docs were audited and narrowly repaired (README.md,
REPORT.md, notebook, figure titles):

- **"Emergence" language.** The v1.0 probe is already near ceiling at the embedding layer
  (`noun_number` 0.983), so peak selectivity is not first availability. The v1.0
  "emergence" prose was replaced with: *peak selectivity*, *earliest depth statistically
  indistinguishable from the peak* (the descriptive CI-overlap rule), and *selectivity
  increases with depth* -- no claim that the property first appears at block 7/9/10.
- **Falling control accuracy.** The v1.0 sentence "the deeper residual stream carries
  progressively less raw word-type identity" was narrowed: a falling control score shows
  that **this particular linear control task** becomes harder to decode at depth; it does
  not prove that total lexical identity information has disappeared from the residual
  stream.
- **CI-overlap rule.** Marked explicitly as descriptive in REPORT/README/notebook: CI
  overlap is not a hypothesis test and is not proof of a transition point.
- **Probe accuracy vs selectivity.** Sharpened: raw probe accuracy is near ceiling at
  shallow depth, so the increasing selectivity is driven substantially by decreasing
  control accuracy -- which makes the causal intervention the decisive test.

No v1.0 result number was changed, and the v1.0 experiment was not rewritten into
something it was not.

## 3. Design freeze

The confirmatory design was frozen in `outputs/stimuli/design_freeze.json` (recorded in
the v1.1 commit): git SHA `<FREEZE_GIT_SHA>`, manifest SHA-256 `<FREEZE_MANIFEST_SHA>`,
stimulus-table SHA-256 `<STIMULUS_TABLE_SHA>`, per-layer direction hashes, config hashes,
splits, verb pairs, seeds, and the gate definition. No parameter/template change was made
after the freeze; a bug fix would have invalidated the run and required a new freeze.

## 4. Competence gate (H1)

Before any intervention, the model had to demonstrate subject--verb agreement behaviour
on **un-intervened dev stimuli**. Pre-registered statistics (dev split; per (lemma, verb)
row, `d = mean(M(plural subject) - M(singular subject))`; lemma-cluster bootstrap,
2,000 resamples, seed 0):

| Gate | Definition | Result | Pass |
|---|---|---|---|
| Gate 1 | paired subject-number effect: mean `d`, CI strictly above 0 | <GATE1_RESULT> | <GATE1_PASS> |
| Gate 2 | directional cluster accuracy: share of rows with `d > 0`, CI lower bound above 0.5 | <GATE2_RESULT> | <GATE2_PASS> |

<GATE_OUTCOME_TEXT>

## 5. Stimulus construction

- **Templates.** `simple` (`The {subject}`), `near` (`The {subject} near the park`, neutral
  intervening context), `attractor` (`The {subject} near the {attractor}`, attractors
  `{park, parks, river, rivers, gate, gates, fence, fences}` with explicit number
  annotations). The prompt stem ends immediately before the target verb; the attractor
  surface is fixed across the matched singular/plural pair.
- **Verbs.** Target pairs (singular, plural): `(is, are)`, `(was, were)`, `(has, have)`,
  `(does, do)`. Tokenizer validation: every form is exactly one tokenizer token
  (leading-space convention); **no pair was excluded**.
- **Noun lexicon.** Derived deterministically from UD English-EWT train: NOUN tokens with
  `Number=Sing/Plur`, letter-only lowercase surfaces, most frequent surface per number,
  minimum frequency 3 per surface, degenerate pairs (plural == singular) excluded.
  **460 lemmas**: `suffix_transparent` 422 (plural == singular + s/es), `nontransparent`
  38 (a surface heuristic, not a linguistic morphology claim).
- **Subject alignment.** The intervention targets the subject's **last overlapping
  subword** (the v1.0 alignment convention, ADR 005 Decision 2). Subject positions and
  subword counts are recorded per stimulus.

## 6. Splits (anti-overfitting)

Pilot (first 5 sorted lemmas, all template families -- plumbing only), dev (200 lemmas,
`simple` + `near`), test (90 lemmas, `simple` + `near` + `attractor`). Lemma sets are
disjoint; the attractor template family is held out of dev. Stimulus counts: pilot 100,
dev 800, test 1,800 rows (2,700 total). Split seed 0; per-stimulus SHA-256 ids; table hash
committed in the freeze manifest. The test split was isolated before model evaluation and
run exactly once (confirmatory); the pilot was used only for tokenization/hook debugging.

## 7. Probe-direction recovery (train split only)

The `noun_number` probe was refitted per depth point under the v1.0 protocol (train split,
per-point StandardScaler, balanced LogisticRegression, C = 0.01 re-verified on dev, cap
60k, seed 0). Standardized coefficients were converted to the residual-space normal
`w_x = w_z / scale` and unit direction `u = w_x / ||w_x||`; class ordering was verified so
that **positive projection = plural** (a flipped fit would have been sign-flipped; none
was). A unit test proves the standardized score and the residual-space score agree up to
the transformed intercept. Reconstruction gate: the refit reproduces the v1.0 recorded
balanced accuracies (embedding 0.983, block_11 0.977) within 0.002 -- <RECON_OK_TEXT>.

## 8. Intervention and controls

For recipient residual `h_r` and donor residual `h_d` at the subject position and layer
`l`: `h'_r = h_r + (u_l . h_d - u_l . h_r) * u_l` -- only the coordinate along the decoded
number direction is replaced. Layers: `embedding`..`block_11` (13 causal points); `ln_f`
included only as a terminal/no-downstream-mixing control; `block_11` doubles as a
near-zero sanity point (no attention layer remains to propagate the patch to the
prediction position -- lack of effect there is not absence of number information).

- **C0 no-op:** hooks active, no change. Reproduced baseline logits within 1e-6
  (max |diff| observed: <NOOP_MAX>).
- **C1 same-number donor:** coordinate replaced with a same-number donor (deterministic
  cyclic lemma pairing within template/attractor/number strata).
- **C2 random direction:** unit vector orthogonal to `u_l` (fixed seeds), norm-matched
  per example to the primary patch (`||delta_random|| = ||delta_number||`), K=3 seeds.
- **C3 full-residual replacement:** `h'_r = h_d` on the matched-tokenization subset only
  (pairs whose singular/plural subjects have equal subword counts).

Scoring convention: for each item, `M = logit(plural_verb) - logit(singular_verb)`; the
donor-directed shift is `E = y_d * (M_patched - M_baseline)`, with `y_d` the donor's
number label for `number`/`full_residual`, the recipient's number for `same_number`, and
the opposite of the recipient's number for `random` (the specificity yardstick -- a
norm-matched non-number perturbation is scored against the direction the primary patch
targets).

## 9. Primary and secondary outcomes

Primary estimand: per layer, paired mean donor-directed shift under the opposite-number
direction patch, with 95% lemma-cluster bootstrap CIs (unit = lemma).

<PRIMARY_RESULTS_PLACEHOLDER>

Secondary: donor-consistent shift rates, verb-preference flip rate, same-number /
random-direction / full-residual control effects, and subgroup effects (template family,
attractor condition, noun stratum).

<SECONDARY_RESULTS_PLACEHOLDER>

## 10. Results

<RESULTS_PLACEHOLDER>

![causal by layer](assets/fig_causal_by_layer.png)

## 11. Decodability vs causal contribution

![selectivity vs causal](assets/fig_selectivity_vs_causal.png)

<SELECTIVITY_VS_CAUSAL_TEXT>

## 12. Limitations

- One model, one size (Pythia-160M), one domain; no scaling claim.
- The direction is a *decoded linear* direction; the result says nothing about uniqueness,
  necessity, complete mediation, or "the" grammatical-number circuit.
- The intervention replaces a coordinate at a single position (the subject's last
  subword); other positions (attractors, sentence start) are untouched.
- Layerwise comparisons are descriptive: adjacent layers are not independent and the
  number of layers is small.
- The noun strata are surface heuristics, not linguistic morphology classes.

## 13. Exact claim boundary

<CLAIM_BOUNDARY_PLACEHOLDER>

## 14. Provenance

- Design freeze: `outputs/stimuli/design_freeze.json` (git SHA, hashes).
- Stimuli: `outputs/stimuli/stimuli.parquet`, `split_manifest.json`,
  `tokenizer_validation.json` (gitignored; hashes committed).
- Directions: `outputs/probe/noun_number_direction/` (per-layer npz + manifest.json).
- Baseline + intervention rows: `outputs/stimuli/baseline/`, `interventions/` (gitignored).
- Authoritative aggregates (committed): `assets/causal_metrics.json`.
- v1.0 reproduction: `outputs/metrics.json` (gitignored) matches the REPORT.md tables.
