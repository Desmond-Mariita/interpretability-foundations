# REPAIR_REPORT — P4 (04-vqa-aokvqa) claims, provenance, and presentation repair

**Date:** 2026-09-22 · **Branch:** `fix/p4-aokvqa-claims-provenance` · **Scope:** P4 only.
No GPU rerun was performed or required. No P2/P3/root/tutorial/protein files were touched.

---

## 1. Audit findings

Evidence audited: README, REPORT, scripts (`00_data`, `10_run_pipelines`, `20_probe`,
`30_eval`, `_models`, `_paths`, `_prompts`, `_stub`), `src/awake/eval/vqa_consistency.py`,
ADR 004, spec v2.1, configs, tests, committed artifacts (executed notebook ipynb/html,
`assets/hero.png`), git history. `outputs/metrics.json` and the per-item parquets are
gitignored and were never committed (verified on `manyee` and `envy`).

| # | Concern | Verdict | Evidence |
|---|---|---|---|
| 1 | README under-sells the completed run | **CONFIRMED** | README "Results" section was one sentence pointing at REPORT; no headline numbers, no leak rate, no 30-second answer. |
| 2 | "Recoverability" can be overinterpreted | **CONFIRMED** | REPORT §6: "The explanation, not the image, carries the answer once vision is ablated", "the more capable model leans *more* on its self-rationale"; README: "the explanation, not the image, drives recovery"; notebook: "explanation is the primary cue". |
| 3 | No-vision delta does not directly measure original-answer faithfulness | **CONFIRMED** | REPORT called Delta "the central faithfulness red flag" (§5.2) and "Capacity makes faithfulness worse" (§6); no statement that Delta ≠ faithfulness of the original answer. |
| 4 | Faithful explanation may preserve visual evidence textually | **CONFIRMED (missing)** | Nowhere acknowledged. |
| 5 | Small delta ambiguity | **CONFIRMED (missing)** | B's delta (0.144) interpreted only as "leans less on its self-rationale"; alternative explanations unstated. |
| 6 | "Consistency" ≠ gold-answer accuracy | **NOT CONFIRMED** | Metric names and labels consistently kept consistency and accuracy separate in code, REPORT, and notebook. |
| 7 | 0.25 chance-null baseline | **PARTIALLY CONFIRMED** | REPORT §6 called the ~0.50 no-explanation baselines "chance-like for a 4-way choice after some language-prior signal" — muddled (uniform chance is 0.25, and outputs are not uniform random). No explicit 0.25 comparison found. |
| 8 | Human-rationale overlap ≠ generated-answer leakage | **PARTIALLY CONFIRMED** | REPORT §5.4 correctly scoped the filter to *dataset-side* leakage, but the notebook called the filtered subset "a cleaner signal" without the residual-leakage caveat; no statement that paraphrase/indirect leakage remains. |
| 9 | Capacity/model-size claims too strong | **CONFIRMED** | REPORT §6 heading "Capacity makes faithfulness worse here, not better"; §5.1 "expected capacity gains within the VLM family". |
| 10 | Reversed / inconsistent count claims | **CONFIRMED (2 instances)** | (a) REPORT §5.3 prose: "they more often pick *different* wrong answers (57 disagree vs 67 agree)" — reversed: the counts say the same wrong answer is picked on 67 and different on 57. Counts verified internally consistent with the divergence rates and accuracies (A_vs_B disagree total 429 = 0.3747×1145, etc.), so the prose was wrong. (b) REPORT §6: "with-explanation rates -- 0.92-0.99" omits B's 0.666; the actual range is 0.67–0.99. |
| 11 | Model/data/provenance presentation | **CONFIRMED** | `metrics.json` uncommitted (gitignored); dataset HF revision not recorded; no surviving-artifact statement; contingency counts/filtered accuracies not flagged as run-recorded-only. |
| 12 | Completed experiment not surfaced | **CONFIRMED** | README delegated all results to REPORT; project read as unfinished. |

## 2. Corrected estimand language

The key quantity is now defined and used consistently as:

> **incremental answer recoverability from the supplied explanation under null visual input**

It is stated explicitly (REPORT §1.1, README §9, notebook) that Delta is **not**:
faithfulness of the original answer, proof the original model used text rather than the
image, proof of hidden reasoning, or causal faithfulness. A faithful explanation may
legitimately encode visual evidence in text; a small Delta is ambiguous. The four
questions (task performance / explanation-conditioned recoverability / answer leakage /
human-rationale similarity) are separated; human-rationale similarity is marked **not
measured** in this project.

## 3. Verified sample counts

From the executed notebook's recorded stdout: split `validation`, **n = 1145** full;
**805 leak-flagged**, **340 leakage-free** (805 + 340 = 1145 ✓). No subsampling;
`prompt_variant: main`; `b7_completed: true`.

## 4. Authoritative metrics (notebook-verified)

Unfiltered (n=1145): accuracy A 0.6279 / B 0.8306 / B7 0.8734; parse rate (answer arm)
0.9633 / 0.9974 / 1.0000; explanation leak rate 0.8122 / 0.2524 / 0.8664; consistency
with-expl 0.9240 / 0.6664 / 0.9860; no-expl 0.5441 / 0.5223 / 0.4996; **Delta** 0.3799
[0.3511, 0.4087] / 0.1441 [0.1153, 0.1703] / 0.4865 [0.4559, 0.5162]; divergence
A_vs_B 0.3747, A_vs_B7 0.3642, B_vs_B7 0.1389 (CIs in REPORT). Filtered (n=340) deltas:
A 0.3618 [0.3088, 0.4176], B 0.1588 [0.1118, 0.2088], B7 0.4647 [0.4088, 0.5206].
Model revisions (HF shas): blip2 `59a1ef6c…`, qwen_lm `a09a3545…`, qwen_vl_3b `66285546…`,
qwen_vl_7b `cc594898…`. All reproduced in `assets/metrics_snapshot.json` with provenance
metadata, and cross-checked against REPORT tables by `test_publication_consistency.py`.

Run-recorded only (from `metrics.json` at run time, **not** in the notebook, not
re-derivable): divergence contingency counts (5.3), filtered accuracies (5.4), and the
paired-parsed-only sensitivity (5.2). The report marks these as such.

## 5. Leakage-free subset definition (verified)

Flag = any of the item's 3 human rationales contains the gold choice **text** verbatim
(normalised substring; bare letters not matched). Deterministic pure function
(`rationale_leaks_answer` → `explanation_leaks_answer`), dataset-side only (gold text vs.
human rationale — never model output), computed once in `scripts/00_data.py`, applied
identically to all pipelines via `leakage_flag`. Documented caveats: removes neither
paraphrase/indirect leakage nor generated-explanation leakage (the separate
`expl_leak_rate` metric), and is selected on item type, not a random subsample.

## 6. Provenance recovered vs. unavailable

**Recovered:** model ids (configs) + HF revisions (notebook output); dataset id + split +
counts; generation parameters (`do_sample=False`, `max_new_tokens=256`, fp16, cuda:0);
bootstrap settings (2,000 resamples, seed 0); prompt text (configs); environment (RTX 3090;
deps pinned in `uv.lock`); artifacts (executed ipynb/html, hero.png, snapshot).

**Unavailable (stated, not invented):** `outputs/metrics.json`; per-item parquets;
per-item consistency vectors (Delta CIs cannot be recomputed); the HF dataset revision at
download time. Future reruns should pin the dataset revision and commit the metrics.

## 7. Claims withdrawn or narrowed

- "Capacity makes faithfulness worse" → descriptive only: B7 showed higher accuracy and a
  larger delta than B **under this setup**; no general capacity effect inferred.
- "The explanation, not the image, carries/drives the answer" → narrowed to the estimand
  above.
- "More often pick different wrong answers" → corrected to match the counts (67 same /
  57 different; near parity).
- "With-explanation rates 0.92–0.99" → corrected range 0.67–0.99.
- "Chance-like" baseline description → removed; baseline stated as empirical; the 0.25
  uniform-guess figure appears only to be disclaimed as the null.
- "Expected capacity gains within the VLM family" → reworded descriptively.

## 8. Files changed (P4 only)

- `README.md` — restructured: research question, 30-second answer, hero result, why it
  matters, design, controls/baselines, results, supports / does-NOT-support, reproduce,
  limitations, further reading + thesis pointer.
- `REPORT.md` — corrected estimand (§1.1), four-question separation, fixed 5.3 prose and
  5.2 range, narrowed capacity/discussion, leakage-subset caveats, provenance section,
  no-chance-baseline language, thesis pointer.
- `assets/metrics_snapshot.json` — **new**: authoritative notebook-verified aggregates +
  provenance metadata.
- `assets/hero.png` — regenerated: four panels (delta±CI, accuracy, leak rate, parse
  rate), one construct per panel, point estimates without fabricated error bars, rendered
  from the snapshot (no models).
- `scripts/30_eval.py` — new pure `render_hero_figure()`; `--from-snapshot` mode.
- `notebooks/01-vqa-consistency.py` / `.ipynb` / `.html` — wording corrected everywhere;
  recorded stdout numbers and per-cell outputs preserved verbatim (only the two
  interpretation wordings and the embedded hero image changed; verified by diff); HTML
  regenerated via nbconvert without re-execution.
- `tests/test_publication_consistency.py` — **new**: REPORT↔snapshot cross-check, count
  consistency, model-label mapping, snapshot schema, estimand/chance-baseline guards,
  leakage-subset construction semantics.
- `tests/test_eval_smoke.py` — added hero-renderer smoke test.
- `REPAIR_REPORT.md` — this file.

## 9. Tests / lint / type status

`pytest -m "unit or smoke"` (full repo): **143→158 passed** (final run reported below);
P4-only: 15 passed. `ruff check .` and `ruff format --check .` clean; `mypy src/awake/`
clean. Coverage ≥90% maintained (root CI floor).

## 10. Rerun required?

**No.** Every repaired headline value traces to the executed notebook, which is committed.
A GPU rerun is not needed for this repair.

## 11. Remaining limitations

- ADR 004 (`docs/decisions/`, root-level) still contains the design-time parenthetical
  "the explanation, not the image, is what lets the model reproduce its answer"; it was
  left untouched per the P4-only scope and should be read together with REPORT §1.1.
- Root README's P4 portfolio row says "the image or the model's *own rationale* drives
  answer recovery"; left untouched (root file, out of scope).
- Divergence contingency counts, filtered accuracies, and the paired-only sensitivity are
  run-recorded but not independently verifiable from surviving artifacts.
- Delta CIs cannot be recomputed (per-item vectors not preserved).
- The dataset's HF revision at download time was not recorded; a future rerun should pin it.
- The leakage filter catches verbatim gold-choice text only; paraphrase/indirect leakage
  and generated-explanation leakage remain (documented).
- Hero figure shows point estimates; CIs exist only for the paired-bootstrap delta.

## 12. Branch / commit / PR

- Branch: `fix/p4-aokvqa-claims-provenance` (from `origin/main` @ 830a7e9)
- Commits and Draft PR: see final status report of the implementing session.
- PR is left **unmerged** for Gemini/Codex review.
