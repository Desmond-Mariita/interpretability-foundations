# 02 — Which text-classification explainer is actually faithful?

**Question.** When LIME, Integrated Gradients, Gradient×Input, and SHAP PartitionExplainer
disagree about which tokens drove a fine-tuned sentiment classifier's decision, which
explanations are *faithful* (reflect what the model actually used) and which are merely
*plausible* (agree with human rationales)?  Do the two properties coincide?

**Answer.** **They don't coincide.** On a RoBERTa-base classifier (test accuracy 0.925),
**Integrated Gradients is the only faithful explainer** — comprehensiveness 0.52 and AOPC
0.34, vs. ~0.02–0.06 for Gradient×Input, LIME, and even the random baseline (IG's lead is
significant, paired bootstrap p < 0.001). Yet on **plausibility** all four methods barely
clear the random floor (AUPRC 0.30–0.33), and IG is *not* the most plausible. The
"faithful **and** plausible" quadrant stays empty — confident saliency maps (Gradient×Input,
LIME) can be no more faithful than random. See [`REPORT.md`](REPORT.md) and
[`metrics.json`](metrics.json).

**Why it matters.** Confident-looking token-attribution explanations are easy to produce
and easy to trust by default.  The ERASER benchmark supplies human-annotated rationales
so we can check explainers against *both* an internal standard (faithfulness: does erasing
these tokens actually change the model's prediction?) and an external one (plausibility:
do humans agree these are the right tokens?).  An explanation that scores high on
plausibility but low on faithfulness offers an appealing narrative that does not accurately
represent the model's computation.

![hero](assets/faithfulness_plausibility.png)

## Method

Fine-tune `roberta-base` (binary sentiment head) on the ERASER Movies dataset. (The spec
named `microsoft/deberta-v3-base`, but DeBERTa-v3 diverges to NaN under this environment's
transformers/torch/CUDA stack — a library bug; RoBERTa-base trains cleanly and nothing in
the method is DeBERTa-specific. See ADR 002 / `REPORT.md §3`.) For each prediction across
the test split (199 examples), run four explainers plus a random baseline:

| Explainer | Role |
|---|---|
| LIME | Surrogate-model attribution (whitespace-level) |
| Integrated Gradients | Gradient-based; axiomatic attribution |
| Gradient×Input | Gradient × input embedding per token (replaces attention rollout) |
| SHAP PartitionExplainer | Shapley-value attribution (optional extra `[explain-shap]`) |
| Random baseline | Uniform random scores — the floor reference |

**Faithfulness metrics:** ERASER-exact comprehensiveness and sufficiency at the dataset
rationale budget `k_d`, plus AOPC (mean probability drop over progressive masking at
bins 0–50%).

**Plausibility metrics:** token F1 and AUPRC against human rationale masks, with
subword-to-word aggregation (max |score| over subwords of each whitespace word).

Bootstrap 95% CIs (paired, 2 000 resamples) + Bonferroni-corrected pairwise tests over
the real-explainer pairs.

**512-subword truncation contract.** Reviews are tokenized once and frozen; all erasure,
plausibility, and attribution operate on this frozen visible sequence.  Gold rationale
masks are clipped to the visible window.  Per-example `truncation_coverage` is recorded
(mean ~0.54 on this run; reviews average ~795 words vs. the 512-subword window) and
carried as a diagnostic and a stated limitation.

See [`REPORT.md`](REPORT.md) for the full methodology and limitations.

## Reproduce

Requires a CUDA GPU (CPU-only is slow but functional; set `model.yaml: fp16: false`).

```
# From the repo root:
just setup

# Download and prepare the ERASER Movies data (nothing committed):
just data && just prepare

# Fine-tune the classifier (roberta-base):
just train

# Run explainers (add [explain-shap] extra first if you want PartitionSHAP):
just explain

# Compute metrics, generate figures, write metrics.json:
just eval

# Render the notebook (requires executed outputs):
just notebook
```

To include the optional SHAP PartitionExplainer:

```
uv sync --extra explain-shap
just explain          # now picks up shap_partition.py as well
```

The `shap` extra pins `numba>=0.59` and `llvmlite>=0.42`; these are required for
Python 3.11 wheels.  See ADR 002 for the rationale.

## Limitations

- 512-subword truncation drops evidence in long reviews; see `REPORT.md §8`.
- Mask-replacement erasure is an approximation of ERASER's literal token-removal.
- `token_iou` is a custom metric, not ERASER's span-level IoU.
- Single model + single dataset; no generalisation claim.
