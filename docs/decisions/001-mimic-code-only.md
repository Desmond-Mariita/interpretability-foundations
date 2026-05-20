# ADR 001 — MIMIC is code-only; bound safety-sweep pattern list

**Status:** Accepted
**Date:** 2026-05-20

## Context

Project 1 uses MIMIC-IV v3.0, a dataset of de-identified critical-care records distributed under a
PhysioNet data-use agreement. PhysioNet permits redistribution of code that operates on MIMIC, but
not of the data, derived rows, or example values. The portfolio is public on GitHub. We need a
policy that lets the project be reproducible without ever placing protected content in the public
git history.

A secondary concern: as the repo grows, the risk of accidentally committing model weights, API
tokens, or local-environment secrets grows with it. The same sweep that protects against PHI
leakage should also catch these.

## Options considered

1. **Commit a small synthetic substitute alongside the real-data pipeline.** Rejected: maintaining
   a parallel "demo" path adds work and invites confusion about which numbers are real.
2. **Use the MIMIC-IV demo subset (~100 patients, openly licensed) as the public reproduction
   path.** Considered as a fallback. Documented in §12 of the design spec as the contingency if
   the full-cohort pipeline overruns its time-box.
3. **Code-only: no data, no derivatives, no example rows committed; reproduction requires
   independent PhysioNet credentialing.** Selected.

## Decision

- No MIMIC data, derivatives, or example rows are committed. `projects/01-tabular-mimic/data/` is
  gitignored.
- The project README documents the PhysioNet access steps; `just data` prints the same instructions
  but downloads nothing.
- A pre-commit `gitleaks` hook runs on every commit.
- Before each `v0.x` → `v0.x+1` release, the maintainer runs the **bound safety-sweep pattern list
  below** against `git log --all -p` and confirms no matches.

### Bound safety-sweep pattern list

The following patterns are checked against the full history before any release tag and before the
v1.0.0 promotion (§15 of the design spec). The list is closed: additions go through an ADR update,
not an ad-hoc edit.

| Pattern | Regex |
|---|---|
| MIMIC `subject_id` | `\b[0-9]{5,8}\b` (context-sensitive — flag when adjacent to MIMIC field names) |
| MIMIC `hadm_id` | `\b2[0-9]{7}\b` |
| MIMIC `stay_id` | `\b3[0-9]{7}\b` |
| Date-of-death column | `\b(deathtime|dod|dod_hosp|dod_ssn)\b` (column names, in case of accidental row pasting) |
| OpenAI key | `sk-[A-Za-z0-9]{20,}` |
| Anthropic key | `sk-ant-[A-Za-z0-9_-]{20,}` |
| HuggingFace token | `hf_[A-Za-z0-9]{30,}` |
| AWS access key id | `AKIA[0-9A-Z]{16}` |
| Generic high-entropy literal | covered by `gitleaks` default rules |

Sweep command (run from repo root):

```bash
git log --all -p | grep -E -nC0 \
  -e 'sk-[A-Za-z0-9]{20,}' \
  -e 'sk-ant-[A-Za-z0-9_-]{20,}' \
  -e 'hf_[A-Za-z0-9]{30,}' \
  -e 'AKIA[0-9A-Z]{16}' \
  -e '\b(deathtime|dod|dod_hosp|dod_ssn)\b' \
  -e '\b2[0-9]{7}\b' \
  -e '\b3[0-9]{7}\b'
```

A non-zero exit (matches found) blocks the release until the matches are reviewed; false positives
are documented in a `safety-sweep.notes` file at the repo root, not added to this list.

## Consequences

- Reproducers must obtain MIMIC-IV credentials independently. This is the intended cost.
- The full-cohort numbers reported in `REPORT.md` are not reproducible from a clean clone alone;
  the demo subset is provided as a public reproduction path if the full-cohort pipeline is gated.
- The pattern list is finite and explicit, which makes the v1.0 promotion check mechanical rather
  than judgemental.
