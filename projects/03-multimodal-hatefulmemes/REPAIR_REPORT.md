# P3 Repair Report — statistical estimand and Shapley-semantics corrections

**Date:** 2026-09-22
**Branch:** `fix/p3-hatefulmemes-estimand-stats`
**Scope:** `projects/03-multimodal-hatefulmemes` (+ its ADR 003 amendment) only.
**Original run status:** the full CLIP-L/14 run happened (its aggregates are committed in
`metrics.json`), but no per-example artifacts (embeddings, boosters, dev scores) were
cached on any available machine. Everything below was therefore repaired from committed
aggregates; per-example quantities are implemented and tested and populate on re-run.

## 1. Audit findings

| # | Concern | Verdict | Evidence |
|---|---|---|---|
| A | `auroc_diffs` are not AUROC differences | **CONFIRMED** | `scripts/11_eval.py:64-67` fed per-example **probability arrays** into `awake.eval.bootstrap.paired_diff_test` (a mean-difference test) and stored the result under `auroc_diffs`. The committed `fused_vs_image.mean_diff` (−0.0304) has the wrong sign and magnitude for an AUROC difference (0.711 − 0.692 = +0.019). |
| A2 | README/REPORT inferred significance from overlapping marginal CIs | **CONFIRMED** | README "fusion only marginally edges image-only (overlapping 95% CIs)"; REPORT §5.1 "the honest comparison is the overlapping AUROC CIs". CI overlap is not a paired test. |
| B | Signed share misinterpreted as dominance | **CONFIRMED** | REPORT §4.2 "positive means image-dominant, negative means text-dominant"; notebook §2/§5 and the worked-example print repeated it. The sign of `s = φ_img/(\|φ_img\|+\|φ_txt\|)` is a direction, not dominance. |
| C | Complementarity/interaction claims exceed the estimand | **CONFIRMED** | REPORT §6: mean `s` near 0 "suggests the fused head exploits complementary signals"; notebook §5: a positive AUROC gap "indicates complementary signal". Two Shapley values + a predictive gap establish neither. |
| D1 | "Train-only 5-fold CV" claim | **CONFIRMED absent** | `scripts/10_train.py` has no CV; hyperparameters come from `configs/train.yaml`. Claimed in README Method and REPORT §3. |
| D2 | Mean-baseline / balanced-background ablations "reported in metrics.json" | **CONFIRMED absent** | No ablation script exists; `metrics.json` has neither ablation. Claimed in README Limitations, REPORT §4.2/§7, notebook, and ADR 003 Decision 3. |
| D3 | Per-example φ "written to metrics.json" | **CONFIRMED absent** | `scripts/20_attribute.py` wrote only aggregates. Claimed in REPORT §4.2 and ADR 003 Decision 2. |
| D4 | Correctness- and label-stratified attribution breakdowns | **CONFIRMED absent** | Single histogram figure; no per-label/per-correctness outputs anywhere. Claimed in REPORT §4.2 and notebook §2. |
| — | Background seed missing from `metrics.json` | **PARTIALLY CONFIRMED** | Minor provenance gap; seed (1337) is in `configs/train.yaml`. |
| — | Shared `paired_diff_test` / `modality_shapley` implementations | **NOT CONFIRMED (no fault)** | Both are correct generic code; P3 misapplied the former. Shared `src/awake/` left untouched (P2 collision rule). |

## 2. Root causes

1. **Misapplied statistic:** `11_eval.py` called a paired *mean-difference* helper on
   probability arrays but named the key `auroc_diffs`; the docs then read it as an AUROC
   comparison. The executed notebook even printed `+nan` for these entries (schema
   mismatch `mean/lo/hi` vs `mean_diff/ci_low/ci_high`).
2. **Sign/dominance conflation:** one scalar `s` was asked to carry both direction and
   dominance. `share_mean = −0.034` (mean *signed* direction) sat next to "image-dominant"
   language, while the magnitude evidence (mean|φ| 0.84 vs 0.68) was conflated with it.
3. **Docs written ahead of implementation:** the CV, ablation, per-example persistence
   and stratified-breakdown claims described plans (ADR 003) that were never implemented;
   REPORT/README/notebook then cited them as results.

## 3. Corrected statistical estimands

**Paired AUROC bootstrap** (`scripts/_stats.py::paired_auroc_bootstrap`, used by
`11_eval.py`): resample example indices as paired units → recompute fused AUROC and
unimodal AUROC inside each resample → record the difference → percentile 95% CI of the
paired-difference distribution. Degenerate resamples (single-class label vectors) are
counted in `n_valid` and skipped, never coerced (this sklearn returns 0.5 for
single-class AUROC instead of raising, so the guard is explicit). Significance is read
from the paired CI; marginal CI overlap is not used as a test.

**Point estimates computable today** (exact, from committed AUROC means):

| Comparison | Point AUROC diff | Paired 95% CI |
|---|---|---|
| fused − image | **+0.0188** | unavailable (requires re-run) |
| fused − text | **+0.1356** | unavailable (requires re-run) |

The previous `auroc_diffs` entries (probability-array mean diffs: fused_vs_image −0.0304,
fused_vs_text +0.0411) are preserved under
`metrics.json → provenance → removed_prob_array_diffs` and used in no claim.

## 4. Corrected Shapley interpretation

- **Signed contribution** φ_image, φ_text: direction of each modality's push on the raw
  margin. **Signed share** `s` = mean direction summary (∈ [−1, 1]; 0 when total ≈ 0).
- **Magnitude share** `m = |φ_image| / (|φ_image| + |φ_text|)` (∈ [0, 1]; 0.5 when both
  ≈ 0): the dominance-appropriate quantity. Counterexample φ_image = −10, φ_text = +1
  gives m = 10/11 (image magnitude-dominant) with s < 0 — covered by a unit test.
- **Reported today:** mean|φ| image 0.842 vs text 0.675 (committed, real); aggregate
  magnitude share of the mean absolutes **image 0.555 / text 0.445** (derived from the
  committed mean absolutes); mean signed share −0.034 (committed, relabelled as
  direction). Mean signed φ, mean per-example magnitude share: `null` until re-run.
- "Balanced reliance" / "complementary signals" readings of a near-zero mean `s` are
  withdrawn.

## 5. Interaction contrast

**Added (bounded).** `I = v({img,txt}) − v({img}) − v({txt}) + v(∅)` is computed
per-example in `scripts/20_attribute.py` from the same four coalition values, with
aggregates (mean, mean|I|, bootstrap interval over examples) written to `metrics.json`.
Interpretation is restricted to *non-additivity of the margin under the interventional
game* — not semantic synergy, not causal interaction. **Not computed on real data:**
the coalition values of the original run were not cached; values are `null` until
re-run. Synthetic-game unit tests cover additive (I = 0) and non-additive cases.

## 6. Background sensitivity

**Not feasible from cached artifacts** (no train embeddings remain). The two ablations
(mean-embedding baseline; class-balanced background) were never implemented. All claims
that they "quantify" prior-dependence or off-manifold effects are withdrawn; ADR 003
Amendment marks them **future work**. No new GPU/CPU work was started for them.

## 7. Files changed

- `scripts/_stats.py` — **new**: paired metric/AUROC bootstrap (shared paired indices,
  degenerate handling, `n_valid`, determinism), signed/magnitude shares, interaction.
- `scripts/11_eval.py` — `auroc_diffs` now the corrected paired AUROC bootstrap.
- `scripts/20_attribute.py` — per-example rows (signed share, magnitude share,
  interaction) + corrected aggregates + background seed; two-panel hero figure on re-run;
  per-example rows to `outputs/attribution/<variant>/dev_attribution.json` (local).
- `scripts/25_repair_metrics.py` — **new**: regenerates the committed `metrics.json`
  from the original aggregates (point diffs, nulls, magnitude share, provenance,
  removed prob-array diffs, background seed).
- `metrics.json` — regenerated (see §3/§4).
- `README.md`, `REPORT.md` — corrected claims, definitions, results, limitations.
- `docs/decisions/003-hateful-memes-licence-and-modality-shapley.md` — amendment
  section (paired bootstrap contract, share semantics, per-example persistence,
  ablations not run, interaction, CV withdrawal).
- `notebooks/01-modality-attribution.{py,ipynb,html}` — corrected semantics + schema;
  re-executed offline (worked example prints its graceful skip; CLIP weights not cached).
- `tests/test_stats_unit.py` — **new**; `tests/test_attribute_smoke.py` — extended.

No shared `src/awake/` files changed. No P2 files changed.

## 8. Tests added

`tests/test_stats_unit.py` (all `unit`, deterministic):
paired AUROC bootstrap — known delta sign; exact point delta; **same resampled indices
for both models** (injected fixed-index rng); degenerate resamples counted and skipped,
never coerced; all-degenerate raises; shape mismatch raises; seed determinism.
Shapley semantics — efficiency of `shapley_2player`; the φ_img=−10/φ_txt=+1
sign-vs-magnitude counterexample; zero/zero defined (m = 0.5, s = 0); shares
complementary. Interaction — additive game → 0; super-/sub-additive games → expected
sign/magnitude. Committed `metrics.json` schema contract pinned. Plus two smoke tests
on `attribute_split` rows (efficiency, share ranges, finite interaction).

## 9. Old vs corrected headline metrics

| Quantity | Old (committed) | Corrected (rerun-authoritative) | Basis |
|---|---|---|---|
| `auroc_diffs.fused_vs_image` | −0.0304 [−0.045, −0.016] (prob-array mean diff, mislabeled) | **+0.019 [−0.013, +0.054]** paired AUROC bootstrap CI (2 000 resamples, all valid, seed 0) | 2026-09-22 rerun |
| `auroc_diffs.fused_vs_text` | +0.0411 (same mislabel) | **+0.136 [+0.095, +0.177]** paired AUROC bootstrap CI | 2026-09-22 rerun |
| Attribution dominance claim | "decision leans on the image" via signed-share framing | magnitude-based: mean|φ| image 0.842 vs text 0.675; mean per-example magnitude share 0.549; magnitude share of mean absolutes 0.555 / 0.445; mean signed φ −0.114 / −0.197; mean signed share −0.034 as direction only | 2026-09-22 rerun |
| Interaction | not reported | mean I −0.093 (mean|I| 0.358), 95% CI [−0.134, −0.053] — mildly sub-additive | 2026-09-22 rerun |
| Per-head AUROC/AUPRC/acc | 0.711 / 0.692 / 0.575 etc. | **bit-identical** (0.710608 / 0.691808 / 0.574960…) — rerun reproduced the original aggregates exactly | deterministic pipeline |

## 10. Claims withdrawn or narrowed

- "All hyperparameters are fixed by train-only 5-fold cross-validation" → withdrawn;
  fixed config, no CV.
- "mean-baseline ablation … quantifies the practical effect" / "balanced-background
  ablation quantifies the shift" → withdrawn (never run; future work).
- "All per-example φ values are written to metrics.json" → narrowed: per-example rows go
  to a local `outputs/attribution/...` artifact on re-run; aggregates only in
  `metrics.json`.
- "correctness-stratified and gold-label-stratified breakdowns are reported" → withdrawn
  (not computed; can be added on re-run).
- "fusion only marginally edges image-only (overlapping 95% CIs)" → replaced by the point
  difference + explicit statement that marginal CI overlap is not a test and the paired
  CI is pending re-run.
- "positive s = image-dominant, negative = text-dominant" → withdrawn; sign = direction,
  magnitude share = dominance.
- "s near 0 suggests complementary signals" → withdrawn (see §4).

## 11. Remaining limitations

- All per-example quantities were regenerated by the 2026-09-22 rerun and are retained
  under `outputs/` (hashes in `outputs/manifest.json`).
- Mean-baseline and balanced-background sensitivity analyses remain unimplemented
  future work (they would require additional pipeline runs; the current artifacts make
  them cheap to add later).
- Interventional/factorised off-manifold caveat and prior-dependence of raw φ remain
  inherent to the method (documented).
- The paired fused−image CI includes 0: no claim that fusion beats the image head at
  n = 500.

## 12. Reproduction

```bash
cd projects/03-multimodal-hatefulmemes
just data && just encode && just train && just eval && just attribute   # full re-run
just test                                                                # unit + smoke tests
uv run python scripts/25_repair_metrics.py                              # idempotent; only
                                                                        # meaningful without artifacts
```

## 13. Branch / PR

Branch `fix/p3-hatefulmemes-estimand-stats`, based on `main` (e02ba47). Draft PR:
https://github.com/Desmond-Mariita/interpretability-foundations/pull/2 (commit 9740f40).

## 14. Reproduction rerun (2026-09-22)

**Why.** The repair (§1–§12) fixed the code, tests and prose, but the corrected
per-example statistics (paired AUROC CIs, mean signed φ, mean magnitude share,
interaction) could not be computed because the original run's artifacts had not been
cached. The rerun regenerated them under the repaired protocol.

**Reproducibility freeze (from `outputs/manifest.json`, run id `20260922-025212Z`).**

- Dataset: Kaggle mirror `marafey/hateful-memes-dataset` (identity-verified: 8,500 /
  500 / 1,000 rows; labels 5,450/3,050 and 250/250; `LICENSE.txt` "Hateful Memes
  Dataset License Agreement"; jsonl sha256 in the manifest).
- CLIP backbone: `openai/clip-vit-large-patch14` @ HF revision
  `32bd64288804d66eefd0ccbe215aa642df71cc41`; 1024-d embeddings; frozen.
- Classifiers: LightGBM heads (fused 2048-d, image 1024-d, text 1024-d),
  `n_estimators=400, learning_rate=0.05, num_leaves=31`, `scale_pos_weight` from train
  ratio, `random_state=1337`. **No cross-validation**; dev used for evaluation only.
- Background: empirical train, N = 200, `background_seed=1337`.
- Seeds: eval marginal CIs seed 1337; paired bootstrap seed 0 (2 000 resamples, all
  valid); interaction bootstrap seed 0.
- Machine/device: `manyee`, RTX 3090 (encode), CPU (train/eval/attribute).
- Environment: Python 3.11.16, torch 2.12.0 (CUDA available), lightgbm/transformers
  versions recorded in the manifest; git `ec271c0` (this branch).

**Execution.** `scripts/00_data.py` (verify) → `01_encode.py --config clip_l14` →
`10_train.py` → `15_background.py` → `11_eval.py` (paired AUROC bootstrap) →
`20_attribute.py` (Shapley, shares, interaction, both figures) →
`26_run_manifest.py` (provenance + hashes). Total wall time ≈ 6 minutes (encode ≈ 4 min
on the 3090; the rest CPU-seconds).

**Validation of the original run.** The rerun reproduced every committed per-head
aggregate **bit-identically** (AUROC 0.710608 / 0.691808 / 0.5749600000000001; AUPRC
and accuracy likewise; mean|φ| and signed-share mean identical). The original headline
numbers were therefore correct as committed; only the *difference test* and the
share interpretation were wrong, as described in §1–§9.

**Authoritative corrected numbers** (also in `metrics.json`): see §9. In brief —
fused − image = +0.019 [−0.013, +0.054] (includes 0); fused − text = +0.136
[+0.095, +0.177] (excludes 0); mean|φ| image 0.842 vs text 0.675; mean per-example
magnitude share 0.549; mean signed φ image −0.114 / text −0.197; interaction mean
−0.093 [−0.134, −0.053] (mildly sub-additive; narrow interpretation only).

**Artifacts retained.** `outputs/embeddings/l14/{train,dev,test}.parquet`,
`outputs/models/l14/{fused,image,text}.txt + background.npz + head_meta.json`,
`outputs/attribution/l14/dev_attribution.json` (500 per-example rows), both figures in
`assets/`, all with sha256 hashes and full provenance in `outputs/manifest.json`
(gitignored, local; summarised here). The gated dataset itself is at
`~/.cache/hateful_memes` on `manyee` and is never committed.
