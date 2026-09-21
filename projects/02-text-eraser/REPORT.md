# ERASER scientific protocol — repaired implementation

**2026-09-21: code repair complete; scientific rerun pending.** The original trained
checkpoint could not be located locally or on manyee. No corrected headline metrics
or ordering are reported. Previous results used an invalid alignment/frozen-input
contract and mismatched target scalars; [historical artifacts](results/superseded-v1/README.md)
are superseded. See [REPAIR_REPORT.md](REPAIR_REPORT.md) for implementation and validation.

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
