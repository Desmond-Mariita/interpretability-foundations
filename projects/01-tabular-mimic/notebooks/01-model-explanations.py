# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # P1 — Per-model explanations
#
# Companion to `REPORT.md`. For each of the four fitted models, this notebook
# shows the explanation form natural to that family (spec §6.1):
#
# 1. **L2 Logistic.** Standardised coefficient × per-feature standard deviation —
#    the model is the explanation; we read it directly.
# 2. **Decision Tree.** A `plot_tree` of the depth-5 fitted tree, plus the
#    decision path for one held-out example.
# 3. **EBM.** Shape functions for the top features — the additive
#    log-odds contribution as a function of the feature value.
# 4. **LightGBM.** Native gain importance for the global view + a TreeSHAP
#    waterfall for one held-out example.
#
# All explanations operate on the *same fitted models* that produced the test
# metrics in `metrics.json`; nothing is re-trained.

# %%
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree

# Make the project's scripts/ importable so we can reuse _data and _models.
PROJECT_ROOT = Path.cwd().resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from _data import load_modeling_frame, split_columns  # noqa: E402
from _models import predict_proba  # noqa: E402

from awake.viz import PALETTE, apply_style  # noqa: E402

apply_style()
plt.rcParams["figure.dpi"] = 120

OUTPUTS = PROJECT_ROOT / "outputs"
MODELS = OUTPUTS / "models"

# %%
X, y, meta = load_modeling_frame(
    OUTPUTS / "cohort.parquet",
    OUTPUTS / "features.parquet",
    OUTPUTS / "splits.parquet",
)
_, cat_cols = split_columns(X)
test_mask = meta["is_test"]
X_test, y_test = X[test_mask], y[test_mask]


def _load(name: str):
    """Load a pickled fitted model from outputs/models/."""
    with (MODELS / f"{name}.pkl").open("rb") as fh:
        return pickle.load(fh)


fitted = {name: _load(name) for name in ("logistic", "decision_tree", "ebm", "lightgbm")}
print(f"Test rows: {len(y_test):,}   positive rate: {float(y_test.mean()):.4f}")
print(f"Loaded models: {list(fitted)}")

# %% [markdown]
# ## 1. L2 Logistic — coefficients
#
# Each predictor is standardised before fitting, so a coefficient with a
# larger magnitude *is* a feature the model is using more. The sign of the
# coefficient indicates whether higher values of that feature increase
# (positive) or decrease (negative) the predicted mortality risk.

# %%
lr = fitted["logistic"]
pre = lr.named_steps["pre"]
clf = lr.named_steps["clf"]
coefs = clf.coef_.ravel()
feat_names = list(pre.get_feature_names_out())

# Multiply by the test-set std of each post-encoding column so one-hot
# dummies (which are not standardised) end up on the same scale as the
# centred-and-scaled numeric columns.
Xt = np.asarray(pre.transform(X_test))
stds = Xt.std(axis=0)
imp = pd.Series(coefs * stds, index=feat_names).sort_values(key=np.abs, ascending=False)

top = imp.head(20)[::-1]  # reverse for top-down barh ordering
fig, ax = plt.subplots(figsize=(7, 6))
colors = [PALETTE[1] if v > 0 else PALETTE[0] for v in top.values]
ax.barh(range(len(top)), top.values, color=colors, edgecolor="black", linewidth=0.4)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top.index, fontsize=9)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Standardised coefficient (positive = higher mortality risk)")
ax.set_title("L2 Logistic — top 20 features by |coefficient × std|")
plt.tight_layout()
plt.show()

# %% [markdown]
# The biggest single coefficient is **first BUN value** — kidney function on
# arrival is heavily upweighted. Categorical levels of `first_careunit` and
# `admission_type` also dominate, which is the linear model's only way to
# express what other models pick up as interactions.

# %% [markdown]
# ## 2. Decision Tree — decision path
#
# A depth-5 tree is the boundary of what's still "look at it and read it."
# The full plot is below; we then take one held-out positive example and
# walk through which splits decided its predicted risk.

# %%
dt = fitted["decision_tree"]
dt_pre = dt.named_steps["pre"]
dt_clf = dt.named_steps["clf"]
dt_feature_names = list(dt_pre.get_feature_names_out())

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    dt_clf,
    feature_names=dt_feature_names,
    class_names=["alive", "died"],
    filled=True,
    rounded=True,
    impurity=False,
    ax=ax,
    fontsize=7,
)
ax.set_title("Decision Tree (depth 5, class_weight='balanced')")
plt.tight_layout()
plt.show()

# %%
# Trace the decision path for one held-out positive example.
positive_idx = X_test.index[y_test.to_numpy().astype(bool)][0]
sample = X_test.loc[[positive_idx]]
Xt_sample = dt_pre.transform(sample)
node_indicator = dt_clf.decision_path(Xt_sample)
leaf_id = dt_clf.apply(Xt_sample)[0]
feature_index = dt_clf.tree_.feature
threshold = dt_clf.tree_.threshold

print(f"Sample stay_id={positive_idx}   true label=died")
print(f"Predicted P(died)={dt_clf.predict_proba(Xt_sample)[0, 1]:.3f}")
print()
print("Decision path:")
node_index = node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]
for node_id in node_index:
    if leaf_id == node_id:
        print(f"  leaf {node_id}: stop")
        break
    feat_idx = feature_index[node_id]
    feat_name = dt_feature_names[feat_idx]
    value = Xt_sample[0, feat_idx]
    op = "<=" if value <= threshold[node_id] else ">"
    print(f"  node {node_id}: {feat_name} = {value:.2f}  {op} {threshold[node_id]:.2f}")

# %% [markdown]
# ## 3. EBM — shape functions
#
# An EBM is an additive model where each feature contributes a *function*
# of its value (rather than a single coefficient). Plotting that function
# is the explanation: it shows the model's view of how risk changes with
# the feature, monotonically or otherwise.

# %%
ebm = fitted["ebm"]
ebm_global = ebm.explain_global()

# Pick the top 6 single-feature terms by avg-weight importance and plot each.
single_term_indices = [i for i, feats in enumerate(ebm.term_features_) if len(feats) == 1]
term_imps = ebm.term_importances("avg_weight")
ranked_singles = sorted(single_term_indices, key=lambda i: -term_imps[i])[:6]


def _shape_xy(ebm_model, term_idx):
    """Extract (x, y) points for an EBM single-feature shape function.

    Continuous features: x is the bin midpoint; y is the per-bin log-odds
    contribution (term_scores_, with the boundary classes dropped).
    Nominal features: x is the level name; y is the per-level contribution.
    """
    feat_idx = ebm_model.term_features_[term_idx][0]
    scores = np.asarray(ebm_model.term_scores_[term_idx])
    feat_type = ebm_model.feature_types_in_[feat_idx]
    bins = ebm_model.bins_[feat_idx]
    if feat_type in {"continuous"}:
        edges = np.asarray(bins[0])
        # term_scores_ has length len(edges)+1 (left/right NaN bins on the ends);
        # strip those and align with bin centres.
        mids = 0.5 * (edges[:-1] + edges[1:])
        core = scores[1 : 1 + len(mids)]
        return mids, core, feat_type
    # Nominal — bins[0] is a dict mapping level -> index in scores.
    mapping = bins[0]
    levels = list(mapping.keys())
    idxs = [mapping[k] for k in levels]
    return levels, scores[idxs], feat_type


fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, term_idx in zip(axes.ravel(), ranked_singles, strict=True):
    feat_idx = ebm.term_features_[term_idx][0]
    feat_name = ebm.feature_names_in_[feat_idx]
    x, ycontrib, feat_type = _shape_xy(ebm, term_idx)
    if feat_type == "continuous":
        ax.plot(x, ycontrib, color=PALETTE[2], linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.5)
    else:
        positions = np.arange(len(x))
        colors = [PALETTE[1] if v > 0 else PALETTE[0] for v in ycontrib]
        ax.bar(positions, ycontrib, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(positions)
        ax.set_xticklabels(x, rotation=30, ha="right", fontsize=8)
    ax.set_title(f"{feat_name}  ({feat_type})", fontsize=10)
    ax.set_ylabel("Δ log-odds")
fig.suptitle("EBM shape functions — top 6 features", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. LightGBM — gain importance + TreeSHAP waterfall
#
# LightGBM is the post-hoc tier: the model is opaque, but the
# `predict(pred_contrib=True)` machinery gives us TreeSHAP-style per-feature
# contributions for any single prediction.

# %%
lgbm = fitted["lightgbm"]
booster = lgbm.booster_
gains = booster.feature_importance(importance_type="gain")
lgbm_features = booster.feature_name()
lgbm_imp = pd.Series(gains, index=lgbm_features).sort_values(ascending=False).head(20)[::-1]

fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(range(len(lgbm_imp)), lgbm_imp.values, color=PALETTE[3], edgecolor="black", linewidth=0.4)
ax.set_yticks(range(len(lgbm_imp)))
ax.set_yticklabels(lgbm_imp.index, fontsize=9)
ax.set_xlabel("Total gain (sum across splits)")
ax.set_title("LightGBM — top 20 features by gain")
plt.tight_layout()
plt.show()

# %%
# Single-prediction TreeSHAP for the same positive example used above.
X_lgb = X_test.copy()
for c in cat_cols:
    X_lgb[c] = X_lgb[c].astype("category")
sample_lgb = X_lgb.loc[[positive_idx]]
contribs = lgbm.predict_proba(sample_lgb, raw_score=False, pred_contrib=True)[0]
# predict_proba(pred_contrib=True) on LGBMClassifier returns shape (n_features + 1,)
# in *log-odds* units; the last entry is the base value (expected log-odds).
base_value = contribs[-1]
contrib_series = pd.Series(contribs[:-1], index=lgbm_features).sort_values(key=np.abs)
top_contribs = pd.concat([contrib_series.head(5), contrib_series.tail(10)])

fig, ax = plt.subplots(figsize=(7, 6))
colors = [PALETTE[1] if v > 0 else PALETTE[0] for v in top_contribs.values]
ax.barh(range(len(top_contribs)), top_contribs.values, color=colors, edgecolor="black",
        linewidth=0.4)
ax.set_yticks(range(len(top_contribs)))
ax.set_yticklabels(top_contribs.index, fontsize=9)
ax.axvline(0, color="black", linewidth=0.8)
score = float(sum(contribs))
prob = 1.0 / (1.0 + np.exp(-score))
ax.set_xlabel("Δ log-odds (negative = lower mortality risk)")
ax.set_title(
    f"LightGBM TreeSHAP — stay {positive_idx}\n"
    f"base log-odds {base_value:.3f}  →  predicted log-odds {score:.3f}  →  "
    f"P(died)={prob:.3f}"
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Cross-model agreement
#
# `feature_agreement.csv` is generated by `scripts/11_eval.py` and shows
# each feature's rank in each model's top-10. Lower rank = more important;
# blanks mean the feature is outside that model's top-10.

# %%
ranks = pd.read_csv(PROJECT_ROOT / "feature_agreement.csv", index_col=0)
print(f"Features in the union of every model's top-10: {len(ranks)}")
ranks.head(15)

# %% [markdown]
# **Summary.** All four families agree at the top: `first_careunit`, `age`,
# the BUN family (`bun_first` / `bun_max` / `bun_min`), `admission_type`, and
# `bicarbonate` levels. Where they diverge is *how* they use those features —
# the EBM shape functions and the LightGBM TreeSHAP plot above show the
# additive model picking up smooth dose-response curves while the boosted
# trees model sharper thresholds. The L2 logistic baseline rounds those
# curves off to a single slope per feature, which is exactly why it loses
# ~3 AUROC points to the EBM.
