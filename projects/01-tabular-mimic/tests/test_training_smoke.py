"""Smoke test for the P1 model wrappers on tiny synthetic data.

Builds a 200-row binary-classification frame with the same column layout as
the real modeling frame (numeric vitals + the four demographic categoricals)
and trains each of the four model families end-to-end. Asserts that every
model produces well-shaped probabilities and that the AUROC is non-trivial
(>= 0.55) on a signal-injected toy problem.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import roc_auc_score


@pytest.fixture
def synthetic_frame() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Return a synthetic ``(X, y, num_cols, cat_cols)`` quad with real signal.

    The target is driven by ``heart_rate_mean`` and ``age`` so all four model
    families should clear an AUROC of 0.55 even with their default configs.
    A handful of NaN values are sprinkled in to exercise the imputation path.
    """
    rng = np.random.default_rng(42)
    n = 200
    age = rng.integers(20, 90, size=n).astype(float)
    hr_mean = rng.normal(85, 15, size=n)
    sbp_mean = rng.normal(120, 20, size=n)
    creat_max = rng.normal(1.2, 0.5, size=n)
    gender = rng.choice(["M", "F"], size=n)
    admission_type = rng.choice(["EMERGENCY", "ELECTIVE", "URGENT"], size=n)
    first_careunit = rng.choice(["MICU", "SICU", "CCU"], size=n)

    # Construct a balanced toy target with real signal: take the linear
    # combination and threshold at its median, so we get ~50/50 labels and a
    # learnable AUROC well above 0.55 even on 200 samples.
    score = 0.04 * (age - 60) + 0.02 * (hr_mean - 85)
    y = (score > np.nanmedian(score)).astype(np.int8)

    # Inject a few NaN values in numeric columns.
    hr_mean[rng.choice(n, size=10, replace=False)] = np.nan
    sbp_mean[rng.choice(n, size=10, replace=False)] = np.nan

    X = pd.DataFrame(
        {
            "age": age,
            "heart_rate_mean": hr_mean,
            "sbp_mean": sbp_mean,
            "creatinine_max": creat_max,
            "gender": gender,
            "admission_type": admission_type,
            "first_careunit": first_careunit,
        }
    )
    num_cols = ["age", "heart_rate_mean", "sbp_mean", "creatinine_max"]
    cat_cols = ["gender", "admission_type", "first_careunit"]
    return X, pd.Series(y, name="target"), num_cols, cat_cols


def _load_models_cfg() -> dict:
    """Load the project's ``models.yaml`` config."""
    project_root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((project_root / "configs" / "models.yaml").read_text())


@pytest.mark.smoke
@pytest.mark.parametrize("family", ["logistic", "decision_tree", "ebm", "lightgbm"])
def test_model_family_fits_and_scores(
    family: str, synthetic_frame: tuple[pd.DataFrame, pd.Series, list[str], list[str]]
) -> None:
    """Each family fits, predicts, and clears a non-trivial AUROC.

    Args:
        family: Model family key (parametrized).
        synthetic_frame: ``(X, y, num_cols, cat_cols)`` fixture.
    """
    X, y, num_cols, cat_cols = synthetic_frame
    cfg = dict(_load_models_cfg()["models"][family])
    # Production configs use large leaf-size floors that are too restrictive
    # for the 200-row smoke fixture. Scale them down so the trees can split.
    if "min_samples_leaf" in cfg:
        cfg["min_samples_leaf"] = 5
    if "min_child_samples" in cfg:
        cfg["min_child_samples"] = 5
    if "n_estimators" in cfg:
        cfg["n_estimators"] = 50

    models = importlib.import_module("_models")
    fitted = models.fit_model(family, cfg, X, y, num_cols, cat_cols)
    scores = models.predict_proba(family, fitted, X, cat_cols)

    assert scores.shape == (len(X),)
    assert np.all((scores >= 0) & (scores <= 1))
    assert roc_auc_score(y, scores) >= 0.55
