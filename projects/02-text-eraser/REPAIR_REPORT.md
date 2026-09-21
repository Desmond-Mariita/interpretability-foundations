# Phase 1A — ERASER scientific repair

**Status (2026-09-21): code repair complete and validated offline; scientific rerun pending.**
No corrected scientific results are claimed. The original trained checkpoint is unavailable.

## Findings verified before modification

- **Confirmed:** `30_eval.py` resized mismatched attribution vectors with `np.resize`,
  repeating/truncating LIME word scores into unrelated subword positions.
- **Confirmed:** LIME received the original full review and retokenized deletions;
  previously truncated tail text could enter its prediction window.
- **Confirmed:** IG differentiated softmax probability; Gradient x Input differentiated
  a logit. Both now use the original predicted-class logit.
- **Additional:** raw tokenizer word IDs were treated as whitespace rationale indices;
  random scores used unstable Python hashes; IG's pad reference altered special tokens
  while retaining the original attention mask. Completeness was not recorded.
- **False documentation claim:** installing the SHAP extra did not automatically include
  SHAP in the hardcoded comparison. SHAP remains excluded, without claiming validation.
  The original frozen-input ADR described intent, not achieved behavior for all methods.

## Corrected contract

`eraser-visible-words-v2` freezes the saved tokenizer's original truncated IDs, offsets,
attention mask and special-token mask. The exact visible prefix ends at the greatest
retained nonspecial character offset. Canonical features are complete whitespace-word
character spans; overlap maps subwords explicitly. Punctuation belongs to its word,
repeated occurrences retain distinct positional IDs, and multi-subword words are masked
as a unit. Ambiguous overlap, lengths, identities or stale versions raise errors.

A partially retained final word stays fixed in the intact token sequence, but is excluded
from features, word budgets and clipped gold rationales. No hidden words are padded into
evaluation. No explainer retokenizes a perturbation or can reveal the hidden tail.
Gradient methods aggregate maximum absolute subword scores after summing embedding
components. LIME fits original-class logits using canonical binary word masks and ranks
absolute coefficients. Random rankings are deterministic uniform word scores.

The intact argmax fixes the class for attribution and every perturbation. Special tokens,
attention and positions remain fixed. IG uses mask-token replacements for editable words
and records signed completeness residuals before magnitude aggregation. Evaluation measures
probability drops for that fixed class, using ceil-rounded complete-word budgets; rationale
AUPRC/F1 separately measure plausibility. See [REPORT.md](REPORT.md) for the full protocol.

## Files changed

- P2 `scripts/{01_prepare,20_explain,30_eval,_contract,_explainers,_model_adapter,_paths}.py`
  and `configs/explainers.yaml`: preparation, provenance, word-mask explanations/evaluation.
- `src/awake/eval/visible_words.py`, `attribution.py`, and
  `explainers/{_canonical,gradient_x_input,integrated_gradients,lime_text,random_baseline}.py`:
  reusable mapping and fixed-target core methods.
- `tests/test_eval_visible_words.py`, `tests/test_eval_explainers.py`,
  P2 `tests/test_explain_eval_smoke.py`: scientific-boundary tests.
- P2 README, REPORT, metrics status, justfile, this report, and ADR 002: truthful status/contract.
- Previous metrics, report, plot and notebook artifacts archived (formatting normalized by commit hooks) in
  `results/superseded-v1/`, with an explicit invalid-results notice. No history deleted.

## Validation

104 shared/P2 tests pass, including all requested punctuation, repeated-word,
multi-subword, truncation, partial-word, special-token, LIME-mapping, aggregation,
gold-clipping, hidden-tail, fixed-target and mismatch cases. They include an analytic
linear classifier, IG completeness checks, a tiny randomly initialized RoBERTa, and all
four methods through persisted Parquet caches and evaluation. No model download is needed.
Stale metadata, wrong run identity, missing/duplicate/renamed words and wrong classes fail.
Shared/P2 coverage: **98.05%**. The full repository unit/smoke suite also passes:
**130 tests**, **96.94%** shared-code coverage. Ruff lint/format and mypy pass.

## Rerun scope and blocker

Regenerate prepared mappings and **all four** attribution caches; recompute all faithfulness,
plausibility, uncertainty and current plots. Keep the original classifier: no retraining was
performed or is required by this repair. Original IDs and classifier behavior are preserved.

Neither the original `outputs/model/` nor its hash/tokenizer backups were found in the local
workspace/home search or the searched home and mounted-data locations on manyee. SSH access
works; the manyee environment reference was read. No GPU experiments were started.
The attempted two-example CPU pilot failed explicitly with
`original trained checkpoint/hash missing; do not retrain silently` before generating outputs.
The real-data pilot and full run are therefore pending, not passed.

Versioned cache paths, input fingerprints, strict metadata, a hashed completion marker and
separate pilot directories prevent old caches from masquerading as repaired outputs.
Full publication requires a complete full-test run. Manifests record commit, verified saved
model/tokenizer hashes, dataset version/checksum, prepared checksum, scalar/contract,
explainer configuration, seed and sample count. The historical upstream Hub revision is
unknown; it is not invented. The checkpoint must pass its existing training hash check.

## Full-split IG convergence finding (2026-09-22)

The initial v2 IG configuration used 50 integration steps. Full-split completeness
diagnostics showed under-convergence on 61% of examples. A prespecified convergence sweep
on the worst residual cases showed stable convergence by 800 steps. The IG configuration
was therefore revised to n_steps=800, triggering a new run identity and full recomputation
of all explainer outputs under the repository's provenance contract.

The superseded n_steps=50 full-run artifacts and diagnostics are archived at
`outputs/attributions/v2-nsteps50-superseded/`; the run's identity, hash and cache checks
all passed, so only the IG numerical-convergence gate failed. LIME surrogate fits were
finite with weighted R² in [0.228, 0.915] across the split. The real-data pilot audit
(`pilot_audit.json`) passed before the full run.

## Old versus corrected headline metrics

These historical means are **invalid comparisons**, retained solely to identify what was
withdrawn (199 test reviews). The changed protocol precludes interpreting differences as
an isolated method improvement.

| Method | Old comprehensiveness | Old AOPC | Corrected values |
|---|---:|---:|---|
| IG | 0.52017 | 0.33997 | Pending |
| Gradient x Input | 0.02667 | 0.03311 | Pending |
| LIME | 0.01516 | 0.03713 | Pending |
| Random | 0.05613 | 0.04652 | Pending |

**Ordering:** unknown. There is no evidence yet that the historical IG lead survives.
**Current interpretation:** offline tests validate mapping/target mechanics only. No method
is currently established as more perturbation-faithful on this dataset under the repaired
protocol. Nonsignificance is not equivalence, and perturbation effects do not establish
hidden causal computation.

## Remaining limitations

Masking can be out of distribution; max-absolute aggregation loses sign and is sensitive
to word length; partial-word context remains available even in sufficiency interventions.
IG convergence and LIME surrogate fit must be inspected on the real pilot/full run.
One model and explainer seed do not capture training/attribution variability. No baseline
sensitivity study, SHAP validation or broader attribution benchmark is included.

## Reproduction

From the repository root, after restoring the original saved classifier/tokenizer/hash into
`projects/02-text-eraser/outputs/model/`, use the existing project environment:

```bash
uv sync --locked --extra dev
uv run --locked pytest tests projects/02-text-eraser/tests -m 'unit or smoke'
uv run --locked python projects/02-text-eraser/scripts/00_fetch_data.py
uv run --locked python projects/02-text-eraser/scripts/01_prepare.py
uv run --locked python projects/02-text-eraser/scripts/20_explain.py --device cuda --limit 2 --run-dir projects/02-text-eraser/outputs/attributions/pilot-v2
uv run --locked python projects/02-text-eraser/scripts/30_eval.py --device cuda --run-dir projects/02-text-eraser/outputs/attributions/pilot-v2
```

Inspect the pilot's private `subsample.parquet`, word caches, manifest, completeness residuals
and surrogate-fit diagnostics before proceeding. On manyee, follow its environment reference
and use its existing Toolbx/project environment. Do not stop other GPU jobs. Only then run:

```bash
uv run --locked python projects/02-text-eraser/scripts/20_explain.py --device cuda
uv run --locked python projects/02-text-eraser/scripts/30_eval.py --device cuda --publish
```

These commands reuse the original model; there is deliberately no training command.
The preparation command in this repair requires the saved tokenizer. Real-data results must
replace this pending status only after inspection and a successful full run.

## Branch and commits

Branch: `fix/p2-eraser-alignment-contract` (not merged).
Original audited commit: `e02ba47f9a2023c7d9ed12009a8047bcf8b578ac`.
Implementation/test commit: `17e67059dae951c77169cee370876bbd18a5160b`.
This report and superseded-results record are in the following documentation commit;
use `git log -2 --oneline` for both. The pre-existing untracked roadmap was not included.
