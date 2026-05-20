"""Test-set evaluation and headline plots for the P1 models.

Loads the pickled models written by ``10_train.py``, re-scores the held-out
test split, and writes three figures into ``assets/``:

- ``calibration.png``  — reliability diagrams for all four models on one axis.
- ``roc_curves.png``   — ROC curves for all four models on one axis.
- ``frontier.png``     — the spec's headline accuracy-vs-interpretability
  frontier plot. Interpretability is a coarse ordinal placement
  (LR ≈ DT > EBM > LightGBM) rather than a measured score; the figure
  shows the trade-off shape, not a continuous quantity.

Run via ``just eval``.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data import load_modeling_frame, split_columns
from _models import predict_proba
from _paths import CONFIGS_DIR, OUTPUTS_DIR, PROJECT_ROOT

from awake.viz import PALETTE, apply_style

LOG = logging.getLogger("p1.eval")

# Coarse interpretability ordering used for the frontier plot (higher = more
# interpretable). Justified by spec §6.1: DT and L2 Logistic are intrinsically
# read-off-the-coefficients/path; EBM is a glassbox additive model with
# inspectable shape functions; LightGBM requires post-hoc SHAP.
INTERPRETABILITY_RANK = {
    "logistic": 4,
    "decision_tree": 4,
    "ebm": 3,
    "lightgbm": 1,
}

PRETTY_NAMES = {
    "logistic": "L2 Logistic",
    "decision_tree": "Decision Tree",
    "ebm": "EBM",
    "lightgbm": "LightGBM",
}


def _load_models(models_dir: Path) -> dict[str, object]:
    """Load every ``*.pkl`` from ``models_dir`` keyed by stem.

    Args:
        models_dir: Directory containing one pickle per family.

    Returns:
        Mapping family-name -> fitted estimator.
    """
    out: dict[str, object] = {}
    for pkl in sorted(models_dir.glob("*.pkl")):
        with pkl.open("rb") as fh:
            out[pkl.stem] = pickle.load(fh)
    return out


def _score_test_set(
    fitted: dict[str, object],
    X_test: pd.DataFrame,
    cat_cols: list[str],
) -> pd.DataFrame:
    """Score the held-out test set with every fitted model.

    Args:
        fitted: Mapping family-name -> fitted estimator.
        X_test: Test feature frame.
        cat_cols: Categorical column names (LightGBM needs them).

    Returns:
        A DataFrame indexed by ``stay_id`` with one column per model holding
        positive-class probabilities.
    """
    cols = {}
    for name, est in fitted.items():
        LOG.info("scoring %s on %d rows", name, len(X_test))
        cols[name] = predict_proba(name, est, X_test, cat_cols)
    return pd.DataFrame(cols, index=X_test.index)


def _plot_calibration(scores: pd.DataFrame, y_true: pd.Series, out_path: Path) -> None:
    """Save a multi-model reliability diagram.

    Args:
        scores: One column per model with positive-class probabilities.
        y_true: Test labels aligned to ``scores``.
        out_path: Destination PNG path.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    for i, name in enumerate(scores.columns):
        prob_true, prob_pred = calibration_curve(y_true, scores[name], n_bins=10)
        ax.plot(
            prob_pred, prob_true, marker="o", color=PALETTE[i], label=PRETTY_NAMES.get(name, name)
        )
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title("Reliability diagram (test set, 10 bins)")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)
    LOG.info("wrote %s", out_path)


def _plot_roc(scores: pd.DataFrame, y_true: pd.Series, out_path: Path) -> None:
    """Save a multi-model ROC overlay.

    Args:
        scores: One column per model with positive-class probabilities.
        y_true: Test labels aligned to ``scores``.
        out_path: Destination PNG path.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    for i, name in enumerate(scores.columns):
        fpr, tpr, _ = roc_curve(y_true, scores[name])
        ax.plot(fpr, tpr, color=PALETTE[i], label=PRETTY_NAMES.get(name, name))
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)
    LOG.info("wrote %s", out_path)


def _model_feature_importance(name: str, est: object, X_test: pd.DataFrame) -> pd.Series:
    """Return one row per (post-encoding) feature with that model's importance.

    The notion of "importance" differs across families, so each branch maps
    its native quantity to a comparable scale (the absolute magnitudes are
    rescaled to sum to 1 within a model for the agreement table):

    - **Logistic.** Absolute coefficient times the feature's training standard
      deviation (handles unstandardised raw columns vs. one-hot dummies).
    - **Decision Tree.** Native ``feature_importances_`` (Gini-based).
    - **EBM.** ``term_importances('avg_weight')`` over individual terms only;
      pairwise interactions are excluded so the table stays per-feature.
    - **LightGBM.** Native ``booster_.feature_importance('gain')``.

    Args:
        name: Model family key.
        est: Fitted estimator.
        X_test: A reference frame used only for column names / std deviations
            on the LR path.

    Returns:
        A normalised ``Series`` indexed by feature name; the absolute values
        sum to one within a model so cross-model rank-ordering is meaningful.
    """
    if name == "logistic":
        pre = est.named_steps["pre"]
        clf = est.named_steps["clf"]
        coefs = clf.coef_.ravel()
        names = list(pre.get_feature_names_out())
        # Use the post-preprocessor column stds to scale coefficients onto a
        # comparable axis. Fit-time transforms include the standardiser so
        # numeric features already live on a unit scale; one-hot dummies don't.
        # Multiplying by std catches the latter case.
        Xt = pre.transform(X_test)
        stds = np.asarray(Xt).std(axis=0)
        imp = pd.Series(np.abs(coefs * stds), index=names)
    elif name == "decision_tree":
        pre = est.named_steps["pre"]
        clf = est.named_steps["clf"]
        names = list(pre.get_feature_names_out())
        imp = pd.Series(clf.feature_importances_, index=names)
    elif name == "ebm":
        names = list(est.feature_names_in_)
        importances = est.term_importances("avg_weight")
        # term_features_ contains tuples of feature indices; keep only the
        # singleton terms so we have one row per feature.
        rows = []
        for term_feats, val in zip(est.term_features_, importances, strict=True):
            if len(term_feats) == 1:
                rows.append((names[term_feats[0]], float(val)))
        imp = pd.Series(dict(rows))
    elif name == "lightgbm":
        booster = est.booster_
        gains = booster.feature_importance(importance_type="gain")
        names = booster.feature_name()
        imp = pd.Series(gains, index=names, dtype=float)
    else:
        raise KeyError(f"no importance implementation for {name}")

    if imp.sum() > 0:
        imp = imp / imp.sum()
    return imp.sort_values(ascending=False)


def _canonical_feature(name: str, categorical_bases: set[str]) -> str:
    """Map a model-internal feature name back to its canonical column.

    Strips the ``num__`` / ``cat__`` prefixes added by sklearn's
    ColumnTransformer and collapses ``cat__X_level`` one-hot dummies back to
    the underlying categorical column ``X``.

    Args:
        name: Raw feature name as reported by a fitted estimator.
        categorical_bases: The set of original categorical column names.

    Returns:
        The canonical name shared across models.
    """
    if name.startswith("num__"):
        return name[len("num__") :]
    if name.startswith("cat__"):
        stem = name[len("cat__") :]
        # Collapse one-hot dummies back to the base column.
        for base in categorical_bases:
            if stem == base or stem.startswith(f"{base}_"):
                return base
        return stem
    return name


def _build_feature_agreement_table(
    fitted: dict[str, object],
    X_test: pd.DataFrame,
    categorical_bases: set[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """Build the spec's "how the four models agree on top features" table.

    Feature names are canonicalised first (one-hot dummies collapsed,
    ``num__``/``cat__`` prefixes stripped) so the table compares like with
    like across the four families.

    Args:
        fitted: Mapping family-name -> fitted estimator.
        X_test: Reference frame for column ordering / std dev.
        categorical_bases: Set of original categorical column names.
        top_n: How many top features to keep per model.

    Returns:
        A ranks DataFrame indexed by canonical feature names. Each cell holds
        the model's rank for that feature (1 = most important) or ``NaN`` if
        it was outside that model's top-N. The frame is ordered by mean rank.
    """
    per_model = {}
    for name, est in fitted.items():
        raw = _model_feature_importance(name, est, X_test)
        # Canonicalise + sum across one-hot dummies that map to the same
        # original column (matters for the LR one-hot path).
        canonical = raw.groupby(
            [_canonical_feature(idx, categorical_bases) for idx in raw.index]
        ).sum()
        per_model[name] = canonical.sort_values(ascending=False)

    top_features: set[str] = set()
    for ser in per_model.values():
        top_features.update(ser.head(top_n).index.tolist())

    ranks = pd.DataFrame(index=sorted(top_features), columns=list(per_model.keys()), dtype=float)
    for name, ser in per_model.items():
        ranked = ser.rank(ascending=False, method="min")
        ranks[name] = [
            ranked[f] if f in ranked.index and ranked[f] <= top_n else np.nan for f in ranks.index
        ]
    ranks["mean_rank"] = ranks.mean(axis=1, skipna=True)
    ranks = ranks.sort_values("mean_rank")
    return ranks


def _plot_frontier(metrics: dict, out_path: Path) -> None:
    """Save the accuracy-vs-interpretability frontier figure.

    Args:
        metrics: Parsed ``metrics.json`` contents.
        out_path: Destination PNG path.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, (name, payload) in enumerate(metrics["models"].items()):
        x = INTERPRETABILITY_RANK[name] + np.random.default_rng(i).uniform(-0.05, 0.05)
        y = payload["test"]["auroc"]
        ax.scatter([x], [y], s=150, color=PALETTE[i], zorder=3, edgecolor="black")
        ax.annotate(
            PRETTY_NAMES.get(name, name),
            xy=(x, y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=11,
        )
    ax.set_xlim(0.3, 4.7)
    ax.set_xticks([1, 3, 4])
    ax.set_xticklabels(["Post-hoc only", "Glassbox", "Intrinsic"])
    ax.set_xlabel(
        "Interpretability tier  (post-hoc = needs SHAP; intrinsic = coef / decision path)"
    )
    ax.set_ylabel("Test AUROC")
    ax.set_title("Accuracy vs. interpretability on MIMIC-IV mortality")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_path)
    plt.close(fig)
    LOG.info("wrote %s", out_path)


def evaluate(
    cohort_path: Path,
    features_path: Path,
    splits_path: Path,
    metrics_path: Path,
    models_dir: Path,
    assets_dir: Path,
) -> dict:
    """Re-score the test set and produce the three headline figures.

    Args:
        cohort_path: ``cohort.parquet``.
        features_path: ``features.parquet``.
        splits_path: ``splits.parquet``.
        metrics_path: ``metrics.json`` written by train.py.
        models_dir: Directory of pickled per-family models.
        assets_dir: Where the PNGs go (typically committed).

    Returns:
        Mapping family-name -> test metrics, for sanity checks.
    """
    assets_dir.mkdir(parents=True, exist_ok=True)
    X, y, meta = load_modeling_frame(cohort_path, features_path, splits_path)
    _, cat_cols = split_columns(X)
    test_mask = meta["is_test"]
    X_test, y_test = X[test_mask], y[test_mask]
    LOG.info("test rows: %d  (positive rate %.4f)", len(y_test), float(y_test.mean()))

    fitted = _load_models(models_dir)
    if not fitted:
        raise FileNotFoundError(f"no pickled models in {models_dir}")
    scores = _score_test_set(fitted, X_test, cat_cols)

    metrics = yaml.safe_load(metrics_path.read_text()) if metrics_path.suffix == ".yaml" else None
    if metrics is None:
        import json

        metrics = json.loads(metrics_path.read_text())

    _plot_calibration(scores, y_test, assets_dir / "calibration.png")
    _plot_roc(scores, y_test, assets_dir / "roc_curves.png")
    _plot_frontier(metrics, assets_dir / "frontier.png")

    _, cat_cols_full = split_columns(X)
    ranks = _build_feature_agreement_table(
        fitted, X_test, categorical_bases=set(cat_cols_full), top_n=10
    )
    ranks_path = PROJECT_ROOT / "feature_agreement.csv"
    ranks.to_csv(ranks_path, float_format="%.0f")
    LOG.info("wrote %s", ranks_path)

    return {name: metrics["models"][name]["test"] for name in scores.columns}


def main() -> int:
    """Entry point; parses CLI flags and calls :func:`evaluate`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", type=Path, default=OUTPUTS_DIR / "cohort.parquet")
    p.add_argument("--features", type=Path, default=OUTPUTS_DIR / "features.parquet")
    p.add_argument("--splits", type=Path, default=OUTPUTS_DIR / "splits.parquet")
    p.add_argument("--metrics", type=Path, default=PROJECT_ROOT / "metrics.json")
    p.add_argument("--models-dir", type=Path, default=OUTPUTS_DIR / "models")
    p.add_argument("--assets-dir", type=Path, default=PROJECT_ROOT / "assets")
    p.add_argument("--config", type=Path, default=CONFIGS_DIR / "models.yaml")
    args = p.parse_args()
    _ = args.config  # reserved for future use
    evaluate(
        args.cohort,
        args.features,
        args.splits,
        args.metrics,
        args.models_dir,
        args.assets_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
