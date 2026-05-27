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
# We fine-tune `roberta-base` on the ERASER Movies sentiment dataset (the spec named
# `microsoft/deberta-v3-base`, but DeBERTa-v3 NaNs under this env's stack — see ADR 002) and
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
import os
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
# Locate the project root robustly. ``__file__`` is undefined when nbconvert executes the
# notebook and the kernel cwd may be $HOME, so prefer an explicit env var (set by
# ``just notebook``), then fall back to several candidate paths, picking the first that
# actually contains the project (``configs/model.yaml`` or a produced ``metrics.json``).
_candidates = []
if os.environ.get("P2_PROJECT_ROOT"):
    _candidates.append(pathlib.Path(os.environ["P2_PROJECT_ROOT"]))
with contextlib.suppress(NameError):
    _candidates.append(pathlib.Path(__file__).resolve().parent.parent)
_cwd = pathlib.Path.cwd()
_candidates += [_cwd, _cwd.parent, _cwd / "projects" / "02-text-eraser"]
PROJECT_ROOT = next(
    (
        d
        for d in _candidates
        if (d / "configs" / "model.yaml").exists() or (d / "metrics.json").exists()
    ),
    _cwd.parent,
)
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

# Top-level keys (see scripts/30_eval.py):
#   "metrics":                    {explainer_name: {metric: {ci_low, mean, ci_high}, ...}}
#   "pairwise_comprehensiveness": {"{a}_vs_{b}": {mean_diff, p_value, significant, ...}}
#   "diagnostics":                {accuracy, macro_f1, ece, class_balance, n}

explainers_data: dict = metrics_raw.get("metrics", {})
classifier_data: dict = metrics_raw.get("diagnostics", {})
pairwise_data: dict = metrics_raw.get("pairwise_comprehensiveness", {})

if not explainers_data:
    warnings.warn(
        "metrics.json loaded but 'metrics' key is empty — "
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
        ci_lo = entry.get("ci_low", float("nan"))
        ci_hi = entry.get("ci_high", float("nan"))
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
# For 1–2 test examples we first show the **reconstructed model input as readable text**
# (detokenised from the RoBERTa byte-level tokens the explainers scored — no raw review
# file is read), with the headline explainer's top-15% tokens highlighted so the selected
# rationale reads as actual review phrases.  We then render a per-token colour map: each
# token is coloured by its **normalised attribution score** (range 0–1, warm = high, cool
# = low).  All explainers are shown side by side so divergences are immediately visible.
# The random baseline column makes the floor concrete.
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
    "grad_x_input",
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


_SPECIAL_TOKENS = {"<s>", "</s>", "<pad>", "<unk>", "<mask>"}


def reconstruct_text(tokens: list[str]) -> str:
    """Detokenise RoBERTa byte-level BPE tokens back to readable text.

    'G-with-dot' marks a leading space and 'C-with-dot' a newline in RoBERTa's
    byte-level vocabulary; special tokens are dropped. The result is the actual model
    input, made legible (no raw review file is read -- this is the tokenised input the
    explainers scored, rendered back to text).
    """
    parts = []
    for tok in tokens:
        if tok in _SPECIAL_TOKENS:
            continue
        lead = " " if tok.startswith("Ġ") else ""
        parts.append(lead + tok.replace("Ġ", "").replace("Ċ", " "))
    return "".join(parts).strip()


def render_readable_highlight(
    tokens: list[str],
    norm_scores: np.ndarray,
    top_frac: float = 0.15,
) -> HTML:
    """Render the reconstructed review with the top-scoring tokens highlighted inline.

    Maps each token's (normalised) attribution onto the readable text, so the rationale
    an explainer selected is visible as highlighted words rather than as an abstract
    subword grid.
    """
    scores = np.asarray(norm_scores, dtype=float)
    thresh = float(np.quantile(scores, 1.0 - top_frac)) if scores.size else 1.1
    spans = []
    for tok, sc in zip(tokens, scores, strict=False):
        if tok in _SPECIAL_TOKENS:
            continue
        lead = " " if tok.startswith("Ġ") else ""
        word = tok.replace("Ġ", "").replace("Ċ", " ")
        if sc >= thresh:
            spans.append(f'{lead}<mark style="background:#ffd54a;padding:0 1px">{word}</mark>')
        else:
            spans.append(lead + word)
    body = "".join(spans).strip()
    return HTML(
        '<div style="font-family:Georgia,serif;font-size:13px;line-height:1.7;'
        f'max-width:760px">{body}</div>'
    )


# %%
# ---------------------------------------------------------------------------
# Load attribution caches and prepared test data; render heatmaps
# ---------------------------------------------------------------------------
if not _skip_heatmaps:
    try:
        # The subsample drives the cache's example_id (0-based positional id), and each
        # cache carries its own token_str/score grid — render each explainer on its own
        # tokens so a heatmap reflects exactly what that explainer scored.
        subsample = pd.read_parquet(ATTRIBUTION_CACHE_DIR / "subsample.parquet").reset_index(
            drop=True
        )
        _attr_caches: dict[str, pd.DataFrame] = {}
        for parquet_file in sorted(ATTRIBUTION_CACHE_DIR.glob("*.parquet")):
            if parquet_file.stem == "subsample":
                continue
            _attr_caches[parquet_file.stem] = pd.read_parquet(parquet_file)

        # Prefer high-coverage examples for a fairer qualitative look.
        _cov = subsample["truncation_coverage"]
        _ids = subsample.index[_cov >= 0.8].tolist()[:HEATMAP_N]
        if not _ids:
            _ids = subsample.index.tolist()[:HEATMAP_N]

        # Headline explainer for the readable-text highlight (fall back to first present).
        _highlight_expl = next(
            (e for e in ("integrated_gradients", *_EXPLAINER_ORDER) if e in _attr_caches),
            None,
        )

        for _n, example_id in enumerate(_ids):
            row = subsample.loc[example_id]
            _label_txt = "hateful/positive (1)" if int(row["label"]) == 1 else "benign/negative (0)"
            print(f"\n--- Example {_n + 1} (id={example_id}) ---")
            print(f"Label: {row['label']} ({_label_txt})  |  coverage: {row['truncation_coverage']:.2f}")

            # Readable reconstruction with the headline explainer's top tokens highlighted,
            # so the rationale is visible as actual review phrases (not just a subword grid).
            if _highlight_expl is not None:
                _hl_rows = (
                    _attr_caches[_highlight_expl]
                    .query("example_id == @example_id")
                    .sort_values("token_idx")
                )
                if not _hl_rows.empty:
                    _hl_toks = _hl_rows["token_str"].tolist()
                    _hl_raw = _hl_rows["score"].to_numpy(dtype=float)
                    _lo, _hi = _hl_raw.min(), _hl_raw.max()
                    _hl_norm = (
                        (_hl_raw - _lo) / (_hi - _lo)
                        if _hi > _lo
                        else np.full_like(_hl_raw, 0.5)
                    )
                    print(f"  Reconstructed input (top-15% tokens by {_highlight_expl} highlighted):")
                    display(render_readable_highlight(_hl_toks, _hl_norm))

            print("  Per-explainer subword attribution (first 60 tokens):")
            for expl_name in _EXPLAINER_ORDER:
                df_cache = _attr_caches.get(expl_name)
                if df_cache is None:
                    continue
                ex_rows = df_cache[df_cache["example_id"] == example_id].sort_values("token_idx")
                if ex_rows.empty:
                    continue
                toks = ex_rows["token_str"].tolist()
                raw_scores = ex_rows["score"].to_numpy(dtype=float)
                s_min, s_max = raw_scores.min(), raw_scores.max()
                norm = (
                    (raw_scores - s_min) / (s_max - s_min)
                    if s_max > s_min
                    else np.full_like(raw_scores, 0.5)
                )
                print(f"  {expl_name}:")
                display(render_token_heatmap(toks[:60], {expl_name: norm[:60]}))

    except Exception as exc:
        print(f"Heatmap rendering failed: {exc}\nRun `just explain` to regenerate "
              "the attribution cache.", file=sys.stderr)

# %% [markdown]
# ## 4. Pairwise significance summary
#
# The table below shows, for each pair of real explainers, whether the difference in
# comprehensiveness is statistically significant after Bonferroni correction over the
# 3 real-explainer pairs (`*` marks significant pairs).

# %%
_bonf_alpha = metrics_raw.get("bonferroni_alpha", 0.05)

if pairwise_data:
    _pw_rows = []
    for pair_key, entry in pairwise_data.items():
        diff = entry.get("mean_diff", float("nan"))
        p_val = entry.get("p_value", float("nan"))
        sig = "*" if entry.get("significant") else ""
        _pw_rows.append(
            {
                "Pair": pair_key,
                "comprehensiveness diff": f"{diff:+.3f}  p={p_val:.3f}{sig}",
            }
        )
    df_pairwise = pd.DataFrame(_pw_rows).set_index("Pair")
    print(f"\n=== Pairwise comprehensiveness (* = Bonferroni p < {_bonf_alpha:.4f}) ===")
    print(df_pairwise.to_string())
    display(df_pairwise)
else:
    print("No pairwise data found in metrics.json.")

# %% [markdown]
# ## 5. Faithfulness vs. plausibility: discussion
#
# **Faithfulness and plausibility do not coincide here.** On this RoBERTa-base classifier
# (test accuracy 0.925), **Integrated Gradients is the only faithful explainer**:
# comprehensiveness 0.52 and AOPC 0.34, versus ~0.02–0.06 for Gradient×Input, LIME, and
# even the random baseline. IG's advantage is significant under a paired bootstrap with
# Bonferroni correction (p < 0.001 vs. both). Gradient×Input and LIME are statistically
# indistinguishable from random attribution on faithfulness.
#
# On **plausibility**, all four methods cluster in a narrow band just above the random
# floor (AUPRC 0.30–0.33), and the most *faithful* method (IG) is not the most *plausible*
# (Gradient×Input edges plausibility). The "faithful **and** plausible" quadrant stays
# empty — reproducing DeYoung et al. (2020) that the two are distinct axes.
#
# | | Low plausibility | High plausibility |
# |---|---|---|
# | **High faithfulness** | ← Integrated Gradients sits here | _(empty)_ |
# | **Low faithfulness** | random baseline | Gradient×Input, LIME (barely above floor) |
#
# The plausibility numbers are tempered by truncation (mean coverage ~0.54; see §8 of
# `REPORT.md`); faithfulness is computed entirely on the model-visible sequence and is not
# affected in the same way.
#
# ### Relationship to the limitations
#
# All numbers here are generated under the **512-subword truncation contract**: reviews
# longer than 512 subwords are truncated, and only the visible prefix is evaluated.  Mean
# `truncation_coverage` on this run is ~0.54, so a large fraction of human rationale falls
# outside the window — a stated limitation that bears on the (uniformly low) plausibility
# numbers. Reporting a coverage-gated headline is left as a v1.1 follow-up.
#
# Mask-replacement erasure (replacing with `[MASK]` rather than deleting tokens) is a
# documented approximation.  It preserves sequence length and positional embeddings but
# introduces a `[MASK]` token that the model was never fine-tuned to ignore; this
# systematically biases comprehensiveness upward and sufficiency downward relative to a
# true-deletion protocol.  See `REPORT.md §7` (Limitations).
