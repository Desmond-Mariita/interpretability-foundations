"""Shared data-loading utilities for the P1 modeling pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Columns from cohort.parquet that are bookkeeping or future-leaking — never
# used as model features.
COHORT_DROP_COLS = (
    "subject_id",
    "hadm_id",
    "intime",
    "outtime",
    "admittime",
    "dischtime",
    "deathtime",
    "los_icu_days",
    "target",
)

DEMOGRAPHIC_FEATURES = ("age", "gender", "admission_type", "first_careunit")


def load_modeling_frame(
    cohort_path: Path,
    features_path: Path,
    splits_path: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Inner-join the three pipeline outputs into a single modeling frame.

    Args:
        cohort_path: Path to ``cohort.parquet``.
        features_path: Path to ``features.parquet``.
        splits_path: Path to ``splits.parquet``.

    Returns:
        ``(X, y, meta)``:

        - ``X`` is a feature DataFrame indexed by ``stay_id`` (numeric + the
          four demographic columns, with ``gender`` / ``admission_type`` /
          ``first_careunit`` left as ``object`` dtype for downstream encoding).
        - ``y`` is the binary target series aligned to ``X``.
        - ``meta`` is a small DataFrame indexed by ``stay_id`` carrying
          ``subject_id``, ``fold``, and ``is_test`` — needed for split-aware
          training and patient-leakage checks.
    """
    cohort = pd.read_parquet(cohort_path)
    features = pd.read_parquet(features_path)
    splits = pd.read_parquet(splits_path)

    frame = cohort.merge(features, on="stay_id", how="inner").merge(
        splits, on="stay_id", how="inner"
    )

    y = frame["target"].astype(np.int8)
    meta = frame[["stay_id", "subject_id", "fold", "is_test"]].set_index("stay_id")

    feature_cols = [
        c
        for c in frame.columns
        if c not in COHORT_DROP_COLS and c not in {"fold", "is_test", "stay_id"}
    ]
    X = frame[["stay_id", *feature_cols]].set_index("stay_id")
    y.index = X.index
    return X, y, meta


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return ``(numeric_cols, categorical_cols)`` for the modeling frame.

    Args:
        X: Feature DataFrame returned by :func:`load_modeling_frame`.

    Returns:
        Two lists of column names; their concatenation equals ``X.columns``.
    """
    cat_cols = [c for c in DEMOGRAPHIC_FEATURES if c in X.columns and X[c].dtype == object]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols
