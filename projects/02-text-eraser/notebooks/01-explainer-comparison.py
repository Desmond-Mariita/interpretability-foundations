# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 -- Explainer comparison: faithfulness vs. plausibility
#
# **Project 2 — `02-text-eraser`**
#
# This notebook summarises the quantitative results from the full evaluation run and
# provides a qualitative look at individual token attributions.
#
# **Before running** this notebook, the evaluation pipeline must have completed:
#
# ```
# just data && just prepare && just train && just explain && just eval
# ```
#
# The notebook will detect missing outputs early and print a clear error rather than
# crashing mid-cell.  All cells below are idempotent once the outputs are present.
#
# ---
#
# ### What we are measuring
#
# We fine-tune `microsoft/deberta-v3-base` on the ERASER Movies sentiment dataset and
# run four explainers — LIME, Integrated Gradients, Gradient×Input, and (optionally)
# SHAP PartitionExplainer — plus a **random baseline** that assigns uniform random
# attribution scores.  For each explainer we compute:
#
# | Dimension | Metrics |
# |---|---|
# | **Faithfulness** | Comprehensiveness, Sufficiency (at budget `k_d`), AOPC |
# | **Plausibility** | Token P/R/F1 at `k_d`, AUPRC |
#
# Bootstrap 95% CIs (paired, 2 000 resamples) and pairwise Bonferroni-corrected
# significance tests are reported alongside every number.
#
# See [`REPORT.md`](../REPORT.md) for the full methodology and limitations.

# %%
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
"""Explainer comparison notebook: faithfulness vs. plausibility."""
from __future__ import annotations

import contextlib
import json
import pathlib
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Matplotlib style — matches the portfolio palette from P1
with contextlib.suppress(ImportError):
    import awake.viz.style  # noqa: F401  (registers the style on import)

# %%
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NOTEBOOK_DIR = pathlib.Path(__file__).parent
PROJECT_ROOT = NOTEBOOK_DIR.parent
METRICS_PATH = PROJECT_ROOT / "metrics.json"
HERO_PATH = PROJECT_ROOT / "assets" / "faithfulness_plausibility.png"
ATTRIBUTION_CACHE_DIR = PROJECT_ROOT / "outputs" / "attributions"
PREPARED_TEST = PROJECT_ROOT / "outputs" / "prepared" / "test.parquet"

# %%
# ---------------------------------------------------------------------------
# Guard: check that the pipeline has run
# ---------------------------------------------------------------------------
_missing = []
if not METRICS_PATH.exists():
    _missing.append(str(METRICS_PATH))
if not HERO_PATH.exists():
    _missing.append(str(HERO_PATH))

if _missing:
    print(
        "ERROR: The following outputs were not found:\n"
        + "\n".join(f"  {p}" for p in _missing)
        + "\n\nPlease run the evaluation pipeline first:\n"
        "  just eval\n"
        "from the project root (or `just pipeline && just eval` for a full run).",
        file=sys.stderr,
    )
    raise FileNotFoundError(
        "Pipeline outputs missing — run `just eval` first.  "
        "See the cell above for details."
    )

# %%
# ---------------------------------------------------------------------------
# Load metrics.json
# ---------------------------------------------------------------------------
with METRICS_PATH.open() as fh:
    metrics_raw: dict = json.load(fh)

# Expected top-level keys (see scripts/30_eval.py):
#   "explainers":  {explainer_name: {metric: {mean, ci_lo, ci_hi}, ...}}
#   "pairwise":    {"{a}_vs_{b}": {metric: {diff, p_bonferroni}}}
#   "classifier":  {accuracy, macro_f1, ece}

explainers_data: dict = metrics_raw.get("explainers", {})
classifier_data: dict = metrics_raw.get("classifier", {})

if not explainers_data:
    warnings.warn(
        "metrics.json loaded but 'explainers' key is empty — "
        "did `just eval` complete successfully?",
        stacklevel=1,
    )

# %%
# ---------------------------------------------------------------------------
# Classifier diagnostics
# ---------------------------------------------------------------------------
print("=== Classifier diagnostics ===")
for k, v in classifier_data.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

# %% [markdown]
# ## 1. Per-explainer metric table
#
# The table below shows faithfulness and plausibility metrics for every explainer,
# including the **random baseline** floor.  Bootstrap 95% confidence intervals are shown
# as `[lo, hi]`.  Numbers are populated from `metrics.json`; the run is described in
# `REPORT.md §6`.

# %%
# ---------------------------------------------------------------------------
# Build summary table
# ---------------------------------------------------------------------------
_METRIC_LABELS = {
    "comprehensiveness": "Comp.",
    "sufficiency": "Suff.",
    "aopc": "AOPC",
    "token_f1": "Token F1",
    "auprc": "AUPRC",
}

rows = []
for explainer_name, mdict in explainers_data.items():
    row: dict[str, object] = {"Explainer": explainer_name}
    for metric_key, col_label in _METRIC_LABELS.items():
        entry = mdict.get(metric_key, {})
        mean = entry.get("mean", float("nan"))
        ci_lo = entry.get("ci_lo", float("nan"))
        ci_hi = entry.get("ci_hi", float("nan"))
        row[col_label] = f"{mean:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]"
    rows.append(row)

df_metrics = pd.DataFrame(rows).set_index("Explainer")

print("\n=== Per-explainer faithfulness + plausibility ===")
print(df_metrics.to_string())
df_metrics  # noqa: B018  (intentional notebook display expression)

# %% [markdown]
# ## 2. Hero scatter — faithfulness vs. plausibility
#
# The scatter below places each explainer (and the random baseline) in a 2D space
# whose axes are a composite faithfulness score (mean of comprehensiveness and AOPC)
# and a composite plausibility score (mean of token F1 and AUPRC).  Error bars are
# paired bootstrap 95% CIs.
#
# The random baseline occupies the bottom-left corner, setting the floor: any
# explainer that does not meaningfully exceed it on *both* axes is providing neither
# faithful nor plausible attributions.

# %%
# ---------------------------------------------------------------------------
# Display hero figure (pre-rendered by scripts/30_eval.py)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
try:
    img = plt.imread(str(HERO_PATH))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Faithfulness vs. plausibility per explainer")
except Exception as exc:
    ax.text(
        0.5,
        0.5,
        f"Could not load hero figure:\n{exc}\n\nRun `just eval` to generate it.",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        color="firebrick",
    )
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Token heatmaps — qualitative inspection
#
# For 1–2 test examples we render a per-token colour map: each token is coloured by
# its **normalised attribution score** (range 0–1, warm = high, cool = low).  All
# explainers are shown side by side so divergences are immediately visible.  The random
# baseline column makes the floor concrete.
#
# The examples are picked as the first two from the evaluation subsample that satisfy
# `truncation_coverage >= 0.8` so that the gold rationale mask is largely intact in the
# visible window.

# %%
# ---------------------------------------------------------------------------
# Load a small slice of the attribution cache for the heatmap
# ---------------------------------------------------------------------------

HEATMAP_N = 2  # number of examples to visualise

_EXPLAINER_ORDER = [
    "lime",
    "integrated_gradients",
    "gradient_x_input",
    "shap_partition",
    "random",
]

_cache_available = ATTRIBUTION_CACHE_DIR.exists() and any(
    ATTRIBUTION_CACHE_DIR.glob("*.parquet")
)
_prepared_available = PREPARED_TEST.exists()

if not _cache_available or not _prepared_available:
    print(
        "Attribution cache or prepared test parquet not found.\n"
        "Run `just explain` (and `just prepare` if needed) to generate them.\n"
        "Skipping token heatmap cells.",
        file=sys.stderr,
    )
    _skip_heatmaps = True
else:
    _skip_heatmaps = False

# %%
# ---------------------------------------------------------------------------
# Helper: render one example as an HTML token heatmap (displayed via IPython)
# ---------------------------------------------------------------------------
from IPython.display import HTML, display  # noqa: E402


def _score_to_hex(score: float, cmap_name: str = "RdYlGn") -> str:
    """Map a normalised score in [0, 1] to an RGB hex colour."""
    cmap = plt.get_cmap(cmap_name)
    r, g, b, _ = cmap(float(score))
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def render_token_heatmap(
    tokens: list[str],
    scores_per_explainer: dict[str, np.ndarray],
) -> HTML:
    """Return an HTML table with one column per explainer and coloured tokens."""
    explainer_names = list(scores_per_explainer.keys())
    header = "".join(f"<th>{n}</th>" for n in explainer_names)
    cells = ""
    for i, tok in enumerate(tokens):
        cells += "<tr>"
        for name in explainer_names:
            scores = scores_per_explainer[name]
            val = float(scores[i]) if i < len(scores) else 0.0
            bg = _score_to_hex(val)
            cells += (
                f'<td style="background:{bg};padding:2px 4px;'
                f'font-family:monospace;font-size:11px;">{tok}</td>'
            )
        cells += "</tr>"
    html = (
        "<table style='border-collapse:collapse'>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{cells}</tbody>"
        "</table>"
    )
    return HTML(html)


# %%
# ---------------------------------------------------------------------------
# Load attribution caches and prepared test data; render heatmaps
# ---------------------------------------------------------------------------
if not _skip_heatmaps:
    try:
        df_prepared = pd.read_parquet(PREPARED_TEST)
        # Select high-coverage examples
        _cov_col = "truncation_coverage"
        if _cov_col in df_prepared.columns:
            _examples = df_prepared[df_prepared[_cov_col] >= 0.8].head(HEATMAP_N)
        else:
            _examples = df_prepared.head(HEATMAP_N)

        _attr_caches: dict[str, pd.DataFrame] = {}
        for parquet_file in sorted(ATTRIBUTION_CACHE_DIR.glob("*.parquet")):
            _name = parquet_file.stem  # e.g. "lime", "integrated_gradients"
            _attr_caches[_name] = pd.read_parquet(parquet_file)

        for _row_idx, (_idx, example_row) in enumerate(_examples.iterrows()):
            example_id = example_row.get("example_id", _idx)
            tokens = example_row.get("tokens_visible", example_row.get("tokens", []))
            if isinstance(tokens, str):
                tokens = tokens.split()

            print(f"\n--- Example {_row_idx + 1} (id={example_id}) ---")
            print(f"Label: {example_row.get('label', '?')}  |  "
                  f"Coverage: {example_row.get(_cov_col, '?'):.2f}")
            print("Tokens (first 30):", tokens[:30])

            scores_map: dict[str, np.ndarray] = {}
            for expl_name in _EXPLAINER_ORDER:
                if expl_name not in _attr_caches:
                    continue
                df_cache = _attr_caches[expl_name]
                ex_rows = df_cache[df_cache["example_id"] == example_id]
                if ex_rows.empty:
                    continue
                ex_rows = ex_rows.sort_values("token_idx")
                raw_scores = ex_rows["score"].to_numpy(dtype=float)
                # Normalise to [0, 1]
                s_min, s_max = raw_scores.min(), raw_scores.max()
                if s_max > s_min:
                    norm = (raw_scores - s_min) / (s_max - s_min)
                else:
                    norm = np.full_like(raw_scores, 0.5)
                scores_map[expl_name] = norm

            if scores_map:
                display(render_token_heatmap(tokens, scores_map))
            else:
                print("  (no cached attributions found for this example_id)")

    except Exception as exc:
        print(f"Heatmap rendering failed: {exc}\nRun `just explain` to regenerate "
              "the attribution cache.", file=sys.stderr)

# %% [markdown]
# ## 4. Pairwise significance summary
#
# The table below shows, for each pair of real explainers, whether the difference in
# comprehensiveness and AUPRC is statistically significant after Bonferroni correction
# over the 3 real-explainer pairs.  A p-value < 0.05 (post-correction) is bolded.

# %%
pairwise_data: dict = metrics_raw.get("pairwise", {})

if pairwise_data:
    _pw_rows = []
    for pair_key, pair_metrics in pairwise_data.items():
        _pw_row: dict[str, object] = {"Pair": pair_key}
        for metric_key in ("comprehensiveness", "auprc"):
            entry = pair_metrics.get(metric_key, {})
            diff = entry.get("diff", float("nan"))
            p_val = entry.get("p_bonferroni", float("nan"))
            sig = "*" if (not np.isnan(p_val) and p_val < 0.05) else ""
            _pw_row[metric_key] = f"{diff:+.3f}  p={p_val:.3f}{sig}"
        _pw_rows.append(_pw_row)
    df_pairwise = pd.DataFrame(_pw_rows).set_index("Pair")
    print("\n=== Pairwise significance (* = Bonferroni p < 0.05) ===")
    print(df_pairwise.to_string())
    display(df_pairwise)
else:
    print("No pairwise data found in metrics.json.")

# %% [markdown]
# ## 5. Faithfulness vs. plausibility: discussion
#
# _(This section will be completed once the reproduced run has produced `metrics.json`.
# The framing below sets out what the numbers should speak to.)_
#
# ### Anticipated structure of the scatter
#
# The 2D faithfulness–plausibility space can be partitioned into four quadrants:
#
# | | Low plausibility | High plausibility |
# |---|---|---|
# | **High faithfulness** | Model-faithful, human-opaque | Both faithful and plausible |
# | **Low faithfulness** | Neither | Human-plausible but not model-faithful |
#
# The key empirical question is whether the faithful quadrant and the plausible quadrant
# overlap — that is, whether the explanations that best reflect the model's internal
# computation are also the ones that align with human reading of the review.
#
# ### Interpretation template
#
# - If **Gradient×Input** lands in the top-right quadrant: gradient-based attribution is
#   doing what the ERASER paper hoped — it is both faithful (erasure tests degrade the
#   model's confidence when the right tokens are removed) and plausible (it agrees with
#   human annotators about what counts as a rationale).
# - If **LIME** lands high on plausibility but low on faithfulness: the surrogate model
#   captures surface-level lexical cues that humans would pick, but does not accurately
#   track the fine-tuned model's non-linear computation.
# - If the **random baseline** is non-trivially close to any real explainer on either
#   axis: that axis is not reliably measured on this dataset and model combination, and
#   the explainer should not be given credit for that dimension.
#
# ### Relationship to the limitations
#
# All numbers here are generated under the **512-subword truncation contract**: reviews
# longer than 512 subwords are truncated, and only the visible prefix is evaluated.  The
# `truncation_coverage` stratification in `metrics.json` (full-sample vs.
# coverage ≥ 0.8) provides a robustness check: if plausibility numbers shift
# substantially between strata, the headline should cite the high-coverage stratum.
#
# Mask-replacement erasure (replacing with `[MASK]` rather than deleting tokens) is a
# documented approximation.  It preserves sequence length and positional embeddings but
# introduces a `[MASK]` token that the model was never fine-tuned to ignore; this
# systematically biases comprehensiveness upward and sufficiency downward relative to a
# true-deletion protocol.  See `REPORT.md §7` (Limitations).
