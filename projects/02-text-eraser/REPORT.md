# ERASER scientific protocol — repaired implementation

**2026-09-22: corrected full real-data rerun completed and published.** The fresh-v2
checkpoint, prepared data, all four attribution caches and evaluation outputs passed
identity, hash, cache and numerical-convergence verification; IG converges at
n_steps=800 (see [REPAIR_REPORT.md](REPAIR_REPORT.md)). Previous results used an invalid
alignment/frozen-input contract and mismatched target scalars; [historical artifacts](results/superseded-v1/README.md)
are superseded. Corrected headline metrics are reported below.

## Canonical input and target

Keep original review text and whitespace rationale indices in private prepared rows.
Tokenize once with the checkpoint's saved fast tokenizer, truncating at 512 subwords
including special tokens. The retained IDs and attention mask define the intact
classifier input. The visible text is the original prefix ending at the greatest
retained nonspecial character offset. Downstream methods never retokenize perturbations.

Canonical words are complete `\S+` spans, with deterministic positional IDs and
original whitespace-word indices. Character overlap maps every retained subword to
one word or fixed context; ambiguous overlaps and identity/length mismatches raise.
Punctuation remains part of its whitespace word; repeated words have distinct IDs.
All subwords of a word are masked together. Special tokens and whitespace-only tokens
stay fixed. An incompletely retained final word stays in the classifier input as fixed
context, but is excluded from attribution features, budgets and plausibility scoring.
Gold masks must match the original word count and are selected by canonical indices.

The original intact argmax fixes the target class. Both gradient methods differentiate
its logit, using original input IDs (including original positional embedding behavior).
Subword scores sum over embedding dimensions, then aggregate by maximum absolute
subword magnitude per complete word. This inherited magnitude ranking loses sign
and is length-sensitive; it is not a decomposition of the output.

IG replaces editable word tokens with the model's mask token as its reference,
preserving special/partial tokens, attention and positions. Signed completeness residuals
are recorded **before** magnitude aggregation alongside Captum's delta and endpoint
logit difference. Large residuals require inspecting convergence before interpreting rankings.

LIME samples positional binary word masks over these same IDs and fits the fixed
class logit with the configured weighted linear surrogate (absolute coefficient ranking).
This is a word-mask LIME variant, not default text-deletion LIME. Repeated occurrences
are separate features; every returned feature index must match exactly. Local weighted
R-squared is recorded. Random scores are uniform per word with reproducible input-specific seeds.

## Evaluation and uncertainty

Top `ceil(0.2 * number_of_complete_words)` words define the rationale, with stable
position-based tie breaking. Comprehensiveness masks those words; sufficiency masks
the other editable words. Fixed partial context remains in both. AOPC averages drops
at the configured word budgets. These compare original-class probabilities, not logits.
Mask interventions can be out of distribution and are not hidden-mechanism evidence.

Visible-word AUPRC and top-budget F1 measure human-rationale agreement, not faithfulness.
Undefined AUPRC examples are excluded explicitly with valid counts reported. Example
bootstrap intervals are conditional on this checkpoint and explainer seed. All six
pairwise comprehensiveness comparisons include random, with Bonferroni correction.
Nonsignificance is not equivalence. Classifier accuracy/F1/calibration use intact inputs.

## Corrected full-run results (2026-09-22)

Full test split (199 reviews) under `eraser-visible-words-v2`, fresh-v2 checkpoint,
evaluation at commit `b5701ec`. All word-score caches passed manifest identity,
fingerprint, hash and cache verification; IG completeness residuals passed the recorded
tolerance on all 199 examples (worst ratio 0.356). LIME surrogate fits were finite
(weighted R² 0.228–0.915). Means with example-bootstrap 95% CIs:

| Method | Comprehensiveness | AOPC | AUPRC | Sufficiency |
|---|---:|---:|---:|---:|
| IG (n_steps=800) | 0.2052 [0.153, 0.258] | 0.1678 [0.142, 0.191] | 0.3391 [0.314, 0.365] | 0.4337 [0.366, 0.498] |
| Gradient × Input | 0.0687 [0.037, 0.103] | 0.1068 [0.088, 0.127] | 0.3353 [0.310, 0.360] | 0.4497 [0.383, 0.514] |
| LIME | 0.1114 [0.070, 0.155] | 0.1476 [0.123, 0.172] | 0.2845 [0.261, 0.310] | 0.4480 [0.381, 0.513] |
| Random | 0.0366 [0.012, 0.062] | 0.0703 [0.054, 0.088] | 0.2664 [0.242, 0.292] | 0.4639 [0.398, 0.528] |

Pairwise comprehensiveness (paired example bootstrap, Bonferroni α = 0.0083): IG
significantly outperforms random (+0.169, p = 0.0005), gradient × input (+0.137,
p = 0.0005) and LIME (+0.094, p = 0.0015); LIME significantly outperforms random
(+0.075, p = 0.002). Gradient × input vs LIME (p = 0.030) and gradient × input vs
random (p = 0.096) are not significant at the corrected level.

Ordering: IG leads on perturbation sensitivity and plausibility, and its lead is now
statistically supported — though corrected magnitudes are far below the withdrawn
historical values and cannot be compared with them. All methods retain high sufficiency
(0.43–0.46): keeping only the rationale preserves most original-class probability.
AUPRC is defined on 195/199 examples; four are excluded with valid counts recorded.
Bootstrap intervals are conditional on this checkpoint and explainer seed.

## Provenance and publication

A new versioned cache directory excludes old outputs. Metadata binds each word score
to its input fingerprint, class, method and run manifest. The manifest records git commit,
checkpoint/tokenizer content hashes, dataset checksum, prepared input checksum, config,
seeds, sample count and installed model-library versions. The historical upstream Hub
revision was not recorded; saved checkpoint content is the reproducible identity.
A completion marker hashes all input/attribution artifacts. Pilot runs cannot publish
headlines. No public rows, reviews or private model weights are committed.

Scientific interpretation awaits the original checkpoint, a small inspected real-data
pilot, and then the full affected attribution/evaluation run. Synthetic validation
establishes implementation behavior only; it cannot establish which explainer wins.
