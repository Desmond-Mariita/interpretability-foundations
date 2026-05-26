# ADR 002 — ERASER data governance, truncation contract, mask-replacement erasure, and shap-optional

**Status:** Accepted
**Date:** 2026-05-26

## Context

Project 2 (`02-text-eraser`) benchmarks four text-attribution explainers on the ERASER
Movies sentiment dataset.  Before writing any pipeline code, the team ran a four-way
design review (two internal methodology agents and two external CLIs) and identified
four architectural decisions that carry governance, correctness, or reproducibility
implications significant enough to document formally.

This ADR records all four decisions in a single document to keep the decision trail
compact.

---

## Decision 1 — Code-only ERASER Movies download

### Context

The ERASER Movies corpus is publicly available (no credentialing required), but the
repository is public on GitHub and the data files are large.  Committing them would
bloat the history, invite accidental downstream re-hosting, and conflict with the
code-only governance posture established for Project 1 (ADR 001).

### Options considered

1. **Commit a small cached sample** for fast CI smoke tests.  Rejected: any committed
   sample would need to be kept in sync with the canonical tarball, adding maintenance
   burden and complicating the "nothing committed" guarantee.
2. **Use a synthetic substitute** for all tests.  Rejected: a wholly synthetic dataset
   would not exercise the ERASER-specific quirks (comparison-document annotations, span
   format) that the pipeline must handle correctly.
3. **Code-only download with sha256 verification.**  Selected.

### Decision

- `scripts/00_fetch_data.py` downloads the canonical ERASER Movies tarball to
  `~/.cache/eraser/movies` (or the path in `ERASER_DATA_PATH`).
- The sha256 of the tarball is verified after the first fetch and on every subsequent
  cache hit.  If the hash does not match, the script exits with a non-zero status.
- No data, derived rows, or example values are committed.
  `projects/02-text-eraser/data/` and `outputs/` are gitignored.
- `just data` prints the download instructions and invokes `00_fetch_data.py`; it
  downloads from the public URL, not from any repository asset.

### Consequences

- Reproducers with network access can fetch the data with a single `just data` command.
- The sha256 check makes data-integrity problems loud rather than silent.
- The gitignore policy keeps the public history free of large binaries and derived artefacts.

---

## Decision 2 — 512-subword frozen visible-sequence truncation contract

### Context

ERASER Movies reviews average approximately 770 whitespace tokens.  DeBERTa-v3-base caps
at 512 subwords.  A naive "tokenize at inference time" approach allows hidden tail content
to slide into the visible window when tokens are erased from the prefix: erasing N tokens
from position k shifts positions k+1 onward left, and the model silently sees tail tokens
that were absent from the original input.  This leakage breaks the faithfulness metrics
(comprehensiveness and sufficiency compare modified to unmodified inputs; if the inputs are
not comparable, the delta is meaningless).

The design review unanimously flagged this as a BLOCKER.

### Options considered

1. **Tokenize + truncate at every forward pass.**  Rejected: the leakage problem above;
   also slow.
2. **Pad erased positions to keep length constant (within the original tokenization).**
   The mask-replacement approach (Decision 3) implements this; it solves the leakage
   problem but is documented as an approximation.
3. **Freeze the visible sequence once at prepare time.**  Selected: single tokenization
   with `return_offsets_mapping=True`, truncation to 512 subwords, stored per example.
   All downstream operations are defined over this frozen sequence.

### Decision

- `scripts/01_prepare.py` tokenizes each review once, truncates to 512 subwords, and
  stores the frozen `token_ids_visible`, `tokens_visible`, and `offsets` per example in
  `outputs/prepared/{train,val,test}.parquet`.
- All attribution, erasure, and plausibility code in `src/awake/eval/` operates only on
  these frozen arrays.  No pipeline stage re-tokenizes an erased string.
- The gold human-rationale mask is **clipped to the visible window** before any
  plausibility computation.  Tokens from the tail of truncated reviews are excluded.
- `truncation_coverage` = (gold rationale tokens in the visible window) ÷ (total gold
  rationale tokens in the full review) is computed per example and stored alongside the
  prepared data.
- `scripts/30_eval.py` reports plausibility on the full eval subsample and separately on
  the **high-coverage stratum** (`coverage >= 0.8`).  The headline number uses the
  high-coverage stratum.

### Consequences

- Faithfulness deltas are guaranteed to compare identical-length, identical-position
  token sequences; no tail leakage.
- The coverage diagnostic makes the truncation bias quantifiable; the high-coverage
  stratum gives a defensible headline number.
- Prepare time is a one-off cost; all subsequent pipeline stages are fast.

---

## Decision 3 — Mask-replacement erasure as a deliberate deviation from literal token-removal

### Context

ERASER's reference implementation removes selected tokens by deletion, contracting the
sequence.  For a model trained on full-length sequences (and specifically for
DeBERTa-v3-base, whose disentangled positional attention is sensitive to absolute
positions), deletion shifts every positional embedding downstream of the removal point
and creates out-of-distribution inputs.  The design review (Gemini, MAJOR finding) raised
this as a correctness concern.

### Options considered

1. **Literal deletion (ERASER default).**  Rejected for DeBERTa: deletion shifts
   positional embeddings and produces inputs the model was never trained to handle.  The
   erasure delta would confound position-distribution shift with the information-content
   change we want to measure.
2. **Replace with the embedding mean or a learned "background" token.**  Considered.
   Rejected: requires a second forward pass to compute the mean embedding; adds
   complexity; still not the model's training distribution.
3. **Replace with `[MASK]`, preserving positions.**  Selected: `[MASK]` is in the
   model's vocabulary and was seen during pre-training; sequence length is unchanged;
   positional embeddings are undisturbed.

### Decision

- `src/awake/eval/erasure.py` implements `erase(token_ids, keep_mask, mask_token_id)
  -> token_ids`: replaces positions where `keep_mask == False` with `mask_token_id`.
  The function never changes sequence length.
- This deviation from ERASER's literal-removal protocol is recorded in the REPORT
  (§4.1 "Erasure implementation note" and §8 Limitations) and in the notebook.
- Known bias: `[MASK]`-replacement biases comprehensiveness **upward** (the model may
  still attend to `[MASK]` positions and update its prediction, inflating the apparent
  importance of erased tokens) and sufficiency **downward** (retaining only the
  "important" tokens while masking everything else forces the model to process many
  `[MASK]` tokens, which it was not fine-tuned to handle in high density).  This bias
  is stated explicitly in the report's Limitations section.

### Consequences

- Erasure does not create positional-embedding OOD inputs.
- The known direction of bias is documented; downstream readers can interpret the numbers
  with appropriate caution.
- Results are not directly comparable to ERASER leaderboard entries that use literal
  deletion.

---

## Decision 4 — `shap` as an optional extra; `shap_partition.py` excluded from the CI coverage denominator

### Context

`shap` pulls a `numba` / `llvmlite` dependency cascade.  The versions of `numba` that
`shap<0.44` specifies (`numba==0.53.1`, `llvmlite==0.36.0`) do not publish Python 3.11
wheels.  The lockfile, resolved on Python 3.11, attempted to pull these old versions and
failed.  A Project 1 analogue (`interpret` vs. `interpret-core`) established the
pattern: isolate problematic dependency cascades in optional extras.

The design review (Codex + Gemini + engineering agent) identified this as a MAJOR
reproducibility issue.

### Options considered

1. **Pin `shap` at a 3.11-compatible version** and add it to the main dependencies.
   Considered.  At the time of the pin analysis, the first `shap` release with
   first-class Python 3.11 wheels required `numba>=0.59` and `llvmlite>=0.42` (released
   mid-2023); the uv resolve confirmed this resolves cleanly.  However, adding `shap`
   unconditionally makes the default install heavier and blocks users on Python <= 3.10
   from the new pin.
2. **Drop PartitionSHAP entirely.**  Rejected: PartitionSHAP is a methodologically
   significant explainer (stable on 512-token inputs; the recommended replacement for
   KernelSHAP on long texts) and the spec lists it as a first-class, if optional,
   explainer.
3. **Optional extra with a pinned lower bound on numba/llvmlite.**  Selected.

### Decision

- `shap`, `numba>=0.59`, and `llvmlite>=0.42` are listed under
  `[project.optional-dependencies] explain-shap` in `pyproject.toml`.
- The uv lockfile resolves the extra cleanly on Python 3.11 and the resolution is
  recorded in this ADR (locked at the time of the first successful resolve).
- `scripts/_explainers.py` imports `shap_partition.py` inside a `try/except ImportError`
  block; if the extra is not installed, the PartitionSHAP explainer is silently skipped
  and a log line is emitted.
- `shap_partition.py` is **not counted in the CI coverage denominator**: the CI
  environment does not install `[explain-shap]`, so the file is never imported during
  `pytest --cov`.  Coverage is not penalised for uncovered lines in an uninstalled
  optional module.
- The slow test that exercises `shap_partition.py` end-to-end is marked with
  `@pytest.mark.slow` and excluded from the CI test run.

### Consequences

- Default install remains light; GPU CI does not time out on numba compilation.
- Users who want PartitionSHAP can install it with `uv sync --extra explain-shap`.
- The coverage floor (>= 90% on `src/awake/`) is not contaminated by an optionally
  importable module.

---

## Summary table

| Decision | Chosen option | Key rationale |
|---|---|---|
| ERASER data access | Code-only download + sha256 | Keeps history clean; no credentialing required |
| Truncation | Freeze visible sequence at prepare time | Prevents tail-leakage in faithfulness metrics |
| Erasure | `[MASK]`-replacement, preserving positions | Avoids OOD positional embeddings on DeBERTa |
| `shap` dependency | Optional extra `[explain-shap]`, `numba>=0.59` | Python 3.11 wheel availability |
