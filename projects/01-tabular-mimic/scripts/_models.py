"""Per-family model builders for P1.

Each builder takes the YAML config block for its model family plus the
``(num_cols, cat_cols)`` partition of the feature columns and returns a
fitted-ready estimator. Tree models and EBM accept the raw frame directly;
the L2-logistic builder wraps the estimator in a ColumnTransformer that
median-imputes + standardises numeric columns and one-hot-encodes
categoricals.

The returned object exposes ``fit(X, y)`` and ``predict_proba(X)``; downstream
training code does not need to know the family.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_logistic(cfg: dict, num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    """Build the L2-logistic pipeline with imputation + scaling + one-hot.

    Args:
        cfg: Model config block from ``models.yaml``.
        num_cols: Numeric feature columns.
        cat_cols: Categorical feature columns (string dtype).

    Returns:
        A scikit-learn ``Pipeline`` ready for ``fit``/``predict_proba``.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy=cfg.get("impute_strategy", "median"))),
            ("scale", StandardScaler() if cfg.get("standardise", True) else "passthrough"),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    clf = LogisticRegression(
        penalty="l2",
        C=float(cfg.get("C", 1.0)),
        max_iter=int(cfg.get("max_iter", 1000)),
        class_weight=cfg.get("class_weight"),
        solver="lbfgs",
    )
    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def build_decision_tree(cfg: dict, num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    """Build a shallow Decision Tree with ordinal-encoded categoricals.

    Args:
        cfg: Model config block from ``models.yaml``.
        num_cols: Numeric feature columns.
        cat_cols: Categorical feature columns.

    Returns:
        A scikit-learn ``Pipeline`` ready for ``fit``/``predict_proba``.
    """
    from sklearn.preprocessing import OrdinalEncoder

    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    clf = DecisionTreeClassifier(
        max_depth=int(cfg.get("max_depth", 5)),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 100)),
        class_weight=cfg.get("class_weight"),
        random_state=int(cfg.get("random_state", 1337)),
    )
    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def build_ebm(cfg: dict, num_cols: list[str], cat_cols: list[str]) -> Any:
    """Build an ExplainableBoostingClassifier with named feature types.

    EBM accepts the raw frame including NaN and categorical strings — we
    pass column types through so the shape functions are labelled correctly.

    Args:
        cfg: Model config block.
        num_cols: Numeric feature columns.
        cat_cols: Categorical feature columns.

    Returns:
        A fitted-ready ``ExplainableBoostingClassifier`` instance.
    """
    from interpret.glassbox import ExplainableBoostingClassifier

    feature_names = num_cols + cat_cols
    feature_types = ["continuous"] * len(num_cols) + ["nominal"] * len(cat_cols)
    return ExplainableBoostingClassifier(
        feature_names=feature_names,
        feature_types=feature_types,
        interactions=int(cfg.get("interactions", 10)),
        max_bins=int(cfg.get("max_bins", 256)),
        outer_bags=int(cfg.get("outer_bags", 8)),
        inner_bags=int(cfg.get("inner_bags", 0)),
        learning_rate=float(cfg.get("learning_rate", 0.01)),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 5)),
        random_state=int(cfg.get("random_state", 1337)),
    )


def build_lightgbm(cfg: dict, num_cols: list[str], cat_cols: list[str]) -> Any:
    """Build a LightGBM classifier with native NaN + categorical handling.

    Args:
        cfg: Model config block from ``models.yaml``.
        num_cols: Numeric feature columns (unused — LightGBM handles them
            implicitly, but the signature is consistent with the other builders).
        cat_cols: Categorical feature columns; passed via ``categorical_feature``.

    Returns:
        A ``LGBMClassifier`` instance.
    """
    from lightgbm import LGBMClassifier

    _ = num_cols  # signature parity
    return LGBMClassifier(
        n_estimators=int(cfg.get("n_estimators", 600)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        num_leaves=int(cfg.get("num_leaves", 63)),
        min_child_samples=int(cfg.get("min_child_samples", 200)),
        subsample=float(cfg.get("subsample", 0.9)),
        colsample_bytree=float(cfg.get("colsample_bytree", 0.9)),
        reg_lambda=float(cfg.get("reg_lambda", 1.0)),
        n_jobs=int(cfg.get("n_jobs", -1)),
        deterministic=bool(cfg.get("deterministic", True)),
        random_state=int(cfg.get("random_state", 1337)),
        verbose=-1,
    )


BUILDERS = {
    "l2_logistic": build_logistic,
    "decision_tree": build_decision_tree,
    "ebm": build_ebm,
    "lightgbm": build_lightgbm,
}


def fit_model(
    name: str,
    cfg: dict,
    X: pd.DataFrame,
    y: pd.Series,
    num_cols: list[str],
    cat_cols: list[str],
) -> Any:
    """Build and fit a model by its family name.

    LightGBM is fit with ``categorical_feature`` so it can split on string
    columns directly; for the other families, the build step already wires
    the necessary preprocessing.

    Args:
        name: Family key from ``BUILDERS``.
        cfg: Model config block for that family.
        X: Training feature frame.
        y: Training labels.
        num_cols: Numeric feature columns.
        cat_cols: Categorical feature columns.

    Returns:
        The fitted estimator.
    """
    _ = name  # only used for logging in callers; dispatch is by cfg["type"]
    model_type = cfg["type"]
    if model_type not in BUILDERS:
        raise KeyError(f"unknown model type: {model_type}")
    estimator = BUILDERS[model_type](cfg, num_cols, cat_cols)

    if model_type == "lightgbm":
        X_lgb = X.copy()
        for c in cat_cols:
            X_lgb[c] = X_lgb[c].astype("category")
        pos = float(y.sum())
        scale_pos_weight = (len(y) - pos) / max(pos, 1.0)
        estimator.set_params(scale_pos_weight=scale_pos_weight)
        estimator.fit(X_lgb, y, categorical_feature=cat_cols)
    elif model_type == "ebm":
        # EBM expects the columns in the order it was constructed with.
        estimator.fit(X[num_cols + cat_cols], y)
    else:
        estimator.fit(X, y)
    return estimator


def predict_proba(name: str, estimator: Any, X: pd.DataFrame, cat_cols: list[str]) -> np.ndarray:
    """Return the positive-class probability for a fitted model.

    Args:
        name: Family key (controls LightGBM categorical encoding).
        estimator: Output of :func:`fit_model`.
        X: Feature frame to score.
        cat_cols: Categorical column names; only used for LightGBM.

    Returns:
        A 1-D array of positive-class probabilities of length ``len(X)``.
    """
    if name == "lightgbm":
        X_lgb = X.copy()
        for c in cat_cols:
            X_lgb[c] = X_lgb[c].astype("category")
        return estimator.predict_proba(X_lgb)[:, 1]
    if name == "ebm":
        # EBM was fit with explicit feature_names; honour that column order.
        return estimator.predict_proba(X[list(estimator.feature_names_in_)])[:, 1]
    return estimator.predict_proba(X)[:, 1]
