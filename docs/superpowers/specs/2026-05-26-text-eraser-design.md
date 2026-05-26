# Project 2 — `02-text-eraser`: faithfulness vs. plausibility of text-classification explainers

**Status:** approved design (post multi-reviewer revision), 2026-05-26
**Author:** Desmond Mariita
**Repo:** `interpretability-foundations` · stays v0.x, unpromoted per §15

> This spec was revised after a four-way design review (two internal Sonnet
> subagents — methodology + engineering — and two external CLIs, Codex and Gemini,
> the latter cross-checking the ERASER reference implementation). The consensus
> findings are folded in below; §11 records what each fix traces back to.

## 1. Question and deliverable

**Question.** When LIME, SHAP, Integrated Gradients, and Gradient×Input disagree about
which tokens drove a sentiment classifier's decision, which explanations are *faithful*
(reflect what the model actually used) and which are merely *plausible* (agree with human
rationales)? Do the two properties coincide?

**Deliverable.**
- Hero figure `assets/faithfulness_plausibility.png`: 2D scatter of faithfulness
  (x) vs. plausibility (y) per explainer, with paired bootstrap CIs, and a **random
  baseline** plotted as the floor.
- `metrics.json`: per-explainer metrics, CIs, pairwise significance, and classifier
  diagnostics.
- Pre-rendered `notebooks/01-explainer-comparison.html`: qualitative token-heatmap
  side-by-side on 1–2 examples + the headline scatter.
- `REPORT.md`: methodology, results, limitations.

## 2. Model

Fine-tune `microsoft/deberta-v3-base` (binary sentiment head) on ERASER Movies with the
HF `Trainer`. Single RTX 3090.

**Truncation contract (the central correctness fix).** Reviews average ~770 whitespace
tokens; DeBERTa-v3-base caps at 512 subwords. We define a single **model-visible
sequence** per example at prepare time and freeze it:

- Tokenize once with `return_offsets_mapping=True`, truncate to 512 subwords.
- All downstream work — explanation, erasure, plausibility — operates **only** on this
  frozen visible sequence. We never re-tokenize an erased *string* (that would let hidden
  tail text slide into the window and silently change the input under evaluation).
- The gold human-rationale mask is **clipped to the visible window** before any
  plausibility computation.
- `01_prepare.py` records per-example `truncation_coverage` = fraction of the original
  human-rationale tokens that survive truncation. Plausibility is reported on the full
  eval subsample **and** on the high-coverage stratum (`coverage ≥ 0.8`); the headline
  number uses the high-coverage stratum to avoid length-correlated bias.

Classifier diagnostics (§6) are reported so erasure deltas are interpretable.

## 3. Data pipeline (code-only; nothing committed)

Original ERASER Movies tarball (the authentic data-prep path; see ADR 002).

- `scripts/00_fetch_data.py` — download the canonical `.tar` to `DATA_PATH`
  (default `~/.cache/eraser/movies`), verify checksum, extract. No data committed.
- `scripts/01_prepare.py` — join `{train,val,test}.jsonl` annotations to the
  whitespace-tokenized files under `docs/`; reconstruct each review + label; build a
  word-level human-rationale binary mask from the `evidences` spans; build and freeze the
  512-subword visible sequence + offset mapping; compute `truncation_coverage`.
  **Comparison/multi-document annotations** (the ERASER Movies quirk where evidence spans
  reference a second document) are detected and **dropped**, with the dropped count
  reported in `outputs/prepare_stats.json`. Writes
  `outputs/prepared/{train,val,test}.parquet`.

## 4. Shared core — `src/awake/eval/` (pure, tokenizer-agnostic, unit-tested)

All metric code is pure and takes plain arrays + a `predict_fn`, never a live tokenizer,
so it carries the 90% coverage floor cheaply.

- `attribution.py`
  - `TokenAttribution` dataclass: `tokens`, `offsets` (`list[tuple[int,int]]`),
    `scores` (np.ndarray), `visible_mask`, `predicted_class: int`,
    `class_scores: np.ndarray`.
  - `Explainer` Protocol: `attribute(example) -> TokenAttribution`.
  - `ModelAdapter` Protocol: wraps the fine-tuned model + tokenizer and exposes
    `predict_proba(token_ids_batch) -> np.ndarray` and the frozen visible tokenization —
    this is what carries predicted-class bookkeeping and visible-prefix info that a bare
    `attribute(text)` cannot.
- `erasure.py` — `erase(token_ids, keep_mask, mask_token_id) -> token_ids`.
  **Erasure is replacement with `[MASK]`, preserving sequence length and positions**
  (deleting tokens shifts positional embeddings → OOD inputs). This is a documented,
  deliberate deviation from ERASER's literal token-removal; recorded in ADR 002 and the
  REPORT.
- `faithfulness.py` — scored on the **original predicted class j**, fixed per example:
  - `comprehensiveness = p_j(x) - p_j(x \ top-k rationale)`
  - `sufficiency = p_j(x) - p_j(only top-k rationale)`
  - `k` for the base metrics comes from a **dataset rationale budget `k_d`** (median
    human-rationale length on train/val), *not* per-example gold length.
  - `aopc(...)` is a **separate** aggregate: mean over bins `{0,1,5,10,20,50}%` of the
    per-step probability drop (DeYoung et al. 2020 definition; includes the 0% point).
- `plausibility.py` — takes `offsets` + word-level gold mask + subword scores (no
  tokenizer dependency):
  - subword→word aggregation rule: **max of |score| over the subwords of a word**
    (a word is "selected" if any of its pieces is); LIME, which is natively
    whitespace-level, uses the identity path.
  - reports **token precision / recall / F1** at budget `k_d`, and **AUPRC** over
    continuous scores (ERASER's soft-score ranking metric).
  - `token_iou` is computed but **labelled a custom metric**, not claimed as the ERASER
    metric (ERASER's IoU is span-level with a >0.5 partial-match rule). ERASER
    partial-match F1 is optional/stretch.
- `bootstrap.py` — paired bootstrap over examples with **full metric recomputation**
  (not resampling of precomputed per-example scalars): `n_resamples=2000`,
  `method="percentile"`, `alpha=0.05`, explicit `seed` arg. Provides per-explainer CIs
  and **pairwise paired-difference tests** between explainers; Bonferroni correction over
  the 6 explainer pairs.

## 5. Explainers — adapters in `src/awake/eval/explainers/` (so they count toward coverage)

Each implements the `Explainer` Protocol over a `ModelAdapter`. Thin drivers in
`scripts/_explainers.py` only wire config → adapter.

- `lime_text.py` — `lime.lime_text.LimeTextExplainer` (whitespace-level; identity
  alignment path).
- `shap_partition.py` — SHAP **`PartitionExplainer`** with a text masker (stable on
  512-token inputs; replaces KernelSHAP). `shap` is an **optional dependency extra**;
  the uv resolve is verified early and the resolution recorded in ADR 002.
- `integrated_gradients.py` — `captum.LayerIntegratedGradients` on the embedding layer.
- `gradient_x_input.py` — gradient×embedding (replaces attention rollout; valid on
  DeBERTa's disentangled attention, which breaks standard rollout).
- `random_baseline.py` — `RandomExplainer`, uniform random scores; the floor reference in
  every figure and table.

## 6. Eval, classifier diagnostics, and figures

- `scripts/10_train.py` — fine-tune, save checkpoint to `outputs/model/` (gitignored),
  log a SHA-256 of the checkpoint. Seeded via `awake.utils.seeding`.
- `scripts/20_explain.py` — run all explainers (+ random) over a test subsample
  (`N≈200`, configurable), **stratified by label and by truncation-coverage stratum**.
  Cache attributions (see schema below).
- `scripts/30_eval.py` — load cache; compute faithfulness + plausibility + paired CIs +
  pairwise significance; emit `metrics.json` (including the explainer seeds), the hero
  scatter, and per-metric bars. **Classifier diagnostics** reported alongside:
  test accuracy / macro-F1 and calibration **ECE**; if ECE > 0.1, apply temperature
  scaling before erasure-based metrics. The eval subset is **all test predictions**
  (faithfulness uses predicted-class prob regardless of correctness); a correct-only
  slice is reported as a secondary cut.

**Attribution cache schema.** One parquet per explainer:
`(example_id, token_idx, token_str, score: float32)`, with parquet key/value metadata
`{model_sha256, explainer_name, explainer_version, seed, n_samples}`. `20_explain.py`
refuses a cached file whose `model_sha256` ≠ the current checkpoint (no silent stale
reuse). Monte-Carlo explainers (LIME, PartitionSHAP) log fixed seeds; their intra-example
variance is *not* propagated by the example bootstrap — stated as a limitation.

## 7. Configs and justfile

`configs/{data,model,explainers}.yaml`. `explainers.yaml` carries every tunable
(`lime.num_samples`, `shap.max_evals`, `ig.n_steps`, `subsample_n`, `k_d`,
`bootstrap.n_resamples`, `coverage_threshold`). Recipes mirror P1:
`just data | prepare | train | explain | eval | notebook | test`, with `just pipeline`
chaining `data → prepare`.

## 8. Tests

- **unit** (`src/awake/eval/`, carry the coverage floor):
  - faithfulness sign/magnitude on a toy `predict_fn` (erasing the only important token
    must drop predicted-class prob; comprehensiveness > 0, sufficiency ≥ 0);
  - AOPC against a hand-computed example;
  - plausibility P/R/F1 + AUPRC on hand-built masks; subword→word max-aggregation on a
    known multi-subword word; LIME identity path;
  - span clipping to the visible window; hidden-tail-leakage regression (erased string
    must not introduce new visible tokens); overlapping spans; empty visible rationale;
  - paired bootstrap determinism under fixed seed.
- **smoke** (CI budget, CPU): a **tiny stub model** (2-layer, ~64-hidden,
  HF-`Trainer`-compatible) on tiny synthetic 2-class text; 1-step train; run
  `RandomExplainer` + `Gradient×Input` end-to-end → metrics. LIME / SHAP / IG on the real
  DeBERTa are marked **slow** (import + CPU forward blow the 8-min budget).
- **slow** (excluded from CI): real fine-tune + full explainer sweep on GPU.

## 9. Dependencies and governance

- Runtime: `transformers`, `torch`, `datasets`, `captum`, `lime`. `shap` and its
  numba/llvmlite cascade go in an **optional extra** (`[project.optional-dependencies]
  explain-shap`), mirroring P1's `interpret`→`interpret-core` lesson; the uv resolve on
  the CPU CI image is verified before any explainer code is written. `accelerate` only if
  mixed precision is enabled (`model.yaml: fp16`); otherwise left transitive.
- ADR `docs/decisions/002-eraser-data-and-truncation.md`: code-only ERASER download;
  512-truncation contract; mask-replacement erasure (deviation from literal removal);
  shap-optional decision.
- CHANGELOG `[Unreleased]` entry. Repo stays v0.x, unpromoted per §15.

## 10. Limitations (for REPORT)

- 512-truncation drops evidence in long reviews; mitigated by the coverage-gated headline
  but not eliminated.
- Mask-replacement erasure is an approximation of ERASER removal.
- `token_iou` is a custom metric, not ERASER's span-level IoU.
- Monte-Carlo explainer (LIME/SHAP) intra-example variance is not propagated.
- Attention-as-explanation is contested; we sidestep it by using Gradient×Input rather
  than rollout, and make no attention-faithfulness claim.
- Single model + single dataset; no generalization claim.

## 11. Review-traceability (what each change fixes)

- Truncation contract + coverage gating ← all 4 reviewers (BLOCKER).
- ERASER-exact suff/comp (predicted class j, `k_d` budget), AOPC as separate aggregate,
  AUPRC added, token-IoU relabelled ← methodology + Codex (BLOCKER/MAJOR).
- Mask-replacement erasure ← Gemini (MAJOR).
- Random baseline ← methodology + Codex + Gemini (MAJOR).
- Pairwise paired significance ← methodology + Codex + Gemini.
- Subword→word aggregation rule ← all 3 (MAJOR).
- Drop rollout → Gradient×Input on DeBERTa ← all 3 (Gemini BLOCKER); user decision.
- KernelSHAP → PartitionExplainer, shap optional ← Codex + Gemini + engineering; user
  decision.
- Enriched protocol + ModelAdapter ← Codex (MAJOR).
- Classifier accuracy/F1 + ECE + declared eval subset + stratified sampling ← Codex +
  methodology.
- Adapters in `src/awake/eval/explainers/` for coverage; stub-model smoke; cache schema
  with checkpoint-hash invalidation; missing `10_train.py`; bootstrap params ←
  engineering agent (BLOCKER/MAJOR).
- Discarded false positive: engineering agent claimed `awake.utils.seeding` is missing;
  it already exists at `src/awake/utils/seeding.py`.
