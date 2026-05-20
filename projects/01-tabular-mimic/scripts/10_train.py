"""Train the four P1 models with subject-grouped 5-fold CV + final refit.

For each model family, runs CV on the train pool using the fold column from
``splits.parquet``, then refits on the full train pool and scores the held-out
test split. All metrics are logged to MLflow (local file backend) and to a
``metrics.json`` manifest committed at the project root.

The fitted-on-full-train-pool models are pickled to ``outputs/models/``
(gitignored) so the explanation script can load them without re-training.

Run via ``just train``.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import yaml
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data import load_modeling_frame, split_columns
from _models import fit_model, predict_proba
from _paths import CONFIGS_DIR, OUTPUTS_DIR, PROJECT_ROOT, ensure_outputs_dir

from awake.utils import seed_everything

LOG = logging.getLogger("p1.train")


def _metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Return the standard binary-classification metric dict.

    Args:
        y_true: 0/1 labels.
        y_score: Positive-class probabilities.

    Returns:
        Mapping of metric name -> float (auroc, auprc, brier, log_loss).
    """
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "log_loss": float(log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7))),
    }


def _cv_scores(
    name: str,
    cfg: dict,
    X_train_pool,
    y_train_pool,
    fold_train_pool,
    num_cols,
    cat_cols,
) -> list[dict[str, float]]:
    """Run group-aware K-fold CV using the precomputed fold column.

    Args:
        name: Model family key.
        cfg: Model config block.
        X_train_pool: Feature frame restricted to ``is_test == False``.
        y_train_pool: Labels for that pool.
        fold_train_pool: Integer fold-index series for the same pool.
        num_cols: Numeric feature columns.
        cat_cols: Categorical feature columns.

    Returns:
        One metric dict per fold.
    """
    per_fold: list[dict[str, float]] = []
    folds = sorted(int(f) for f in fold_train_pool.unique() if int(f) >= 0)
    for f in folds:
        val_mask = fold_train_pool == f
        tr_mask = ~val_mask
        est = fit_model(name, cfg, X_train_pool[tr_mask], y_train_pool[tr_mask], num_cols, cat_cols)
        scores = predict_proba(name, est, X_train_pool[val_mask], cat_cols)
        m = _metrics(y_train_pool[val_mask].to_numpy(), scores)
        m["fold"] = int(f)
        per_fold.append(m)
        LOG.info(
            "[%s] fold %d AUROC=%.4f AUPRC=%.4f Brier=%.4f",
            name,
            f,
            m["auroc"],
            m["auprc"],
            m["brier"],
        )
    return per_fold


def _aggregate_cv(per_fold: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate per-fold metrics to mean +/- std.

    Args:
        per_fold: Output of :func:`_cv_scores`.

    Returns:
        ``{"auroc": {"mean": ..., "std": ...}, ...}`` for each metric.
    """
    out: dict[str, dict[str, float]] = {}
    for k in ("auroc", "auprc", "brier", "log_loss"):
        vals = np.array([m[k] for m in per_fold])
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
    return out


def _train_one_family(
    name: str,
    cfg: dict,
    X,
    y,
    meta,
    num_cols,
    cat_cols,
    models_dir: Path,
) -> dict:
    """CV-evaluate, refit on full train pool, score test, persist artifact.

    Args:
        name: Family key (also used as the model nickname).
        cfg: Model config block from ``models.yaml``.
        X: Full feature frame.
        y: Full label series.
        meta: Metadata frame with ``fold`` and ``is_test`` columns.
        num_cols: Numeric feature columns.
        cat_cols: Categorical feature columns.
        models_dir: Directory to pickle the refit model into.

    Returns:
        A dict with ``cv`` aggregates and ``test`` metrics, ready to drop into
        the top-level ``metrics.json``.
    """
    train_mask = ~meta["is_test"]
    test_mask = meta["is_test"]
    fold_train = meta.loc[train_mask, "fold"]
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    LOG.info("[%s] CV …", name)
    t0 = time.time()
    per_fold = _cv_scores(name, cfg, X_train, y_train, fold_train, num_cols, cat_cols)
    cv_agg = _aggregate_cv(per_fold)
    LOG.info(
        "[%s] CV AUROC %.4f +/- %.4f (took %.1fs)",
        name,
        cv_agg["auroc"]["mean"],
        cv_agg["auroc"]["std"],
        time.time() - t0,
    )

    LOG.info("[%s] refit on full train pool …", name)
    t1 = time.time()
    final = fit_model(name, cfg, X_train, y_train, num_cols, cat_cols)
    refit_s = time.time() - t1

    scores_test = predict_proba(name, final, X_test, cat_cols)
    test_m = _metrics(y_test.to_numpy(), scores_test)
    LOG.info(
        "[%s] test AUROC=%.4f AUPRC=%.4f Brier=%.4f (refit %.1fs)",
        name,
        test_m["auroc"],
        test_m["auprc"],
        test_m["brier"],
        refit_s,
    )

    # Persist artefacts.
    models_dir.mkdir(parents=True, exist_ok=True)
    with (models_dir / f"{name}.pkl").open("wb") as fh:
        pickle.dump(final, fh)

    mlflow.log_metrics(
        {
            f"{name}_cv_auroc_mean": cv_agg["auroc"]["mean"],
            f"{name}_cv_auroc_std": cv_agg["auroc"]["std"],
            f"{name}_cv_auprc_mean": cv_agg["auprc"]["mean"],
            f"{name}_cv_brier_mean": cv_agg["brier"]["mean"],
            f"{name}_test_auroc": test_m["auroc"],
            f"{name}_test_auprc": test_m["auprc"],
            f"{name}_test_brier": test_m["brier"],
            f"{name}_test_log_loss": test_m["log_loss"],
        }
    )

    return {
        "type": cfg["type"],
        "cv": cv_agg,
        "cv_per_fold": per_fold,
        "test": test_m,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


def train(
    cohort_path: Path,
    features_path: Path,
    splits_path: Path,
    config_path: Path,
    *,
    models_dir: Path | None = None,
    metrics_path: Path | None = None,
    mlruns_dir: Path | None = None,
    families: list[str] | None = None,
    seed: int = 1337,
) -> dict:
    """Train and evaluate all configured model families.

    Args:
        cohort_path: Path to ``cohort.parquet``.
        features_path: Path to ``features.parquet``.
        splits_path: Path to ``splits.parquet``.
        config_path: Path to ``models.yaml``.
        models_dir: Override for the persisted-models directory.
        metrics_path: Override for the metrics JSON output path.
        mlruns_dir: Override for the MLflow tracking directory.
        families: Optional whitelist of family keys to train.
        seed: Master RNG seed.

    Returns:
        The full metrics dict that was written to ``metrics_path``.
    """
    seed_everything(seed)
    cfg = yaml.safe_load(config_path.read_text())
    models_dir = models_dir or OUTPUTS_DIR / "models"
    mlruns_dir = mlruns_dir or OUTPUTS_DIR / "mlruns"
    metrics_path = metrics_path or PROJECT_ROOT / cfg["outputs"]["metrics"]
    ensure_outputs_dir()

    mlflow.set_tracking_uri(f"file:{mlruns_dir}")
    mlflow.set_experiment("p1-tabular-mimic")

    X, y, meta = load_modeling_frame(cohort_path, features_path, splits_path)
    num_cols, cat_cols = split_columns(X)
    LOG.info(
        "modeling frame: %d rows x %d cols (%d numeric, %d categorical)",
        len(X),
        X.shape[1],
        len(num_cols),
        len(cat_cols),
    )

    families = families or list(cfg["models"].keys())
    results: dict[str, dict] = {}
    with mlflow.start_run(run_name=f"all-{int(time.time())}"):
        mlflow.log_params({"n_features": X.shape[1], "n_train_test": len(X), "seed": seed})
        for fam in families:
            with mlflow.start_run(run_name=fam, nested=True):
                mlflow.log_params(
                    {k: v for k, v in cfg["models"][fam].items() if not isinstance(v, list | dict)}
                )
                results[fam] = _train_one_family(
                    fam,
                    cfg["models"][fam],
                    X,
                    y,
                    meta,
                    num_cols,
                    cat_cols,
                    models_dir,
                )

    payload = {
        "cohort_size": len(X),
        "n_features_pre_encoding": X.shape[1],
        "feature_columns": {"numeric": num_cols, "categorical": cat_cols},
        "models": results,
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    LOG.info("wrote %s", metrics_path)
    return payload


def main() -> int:
    """Entry point. Parses CLI flags and calls :func:`train`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", type=Path, default=OUTPUTS_DIR / "cohort.parquet")
    p.add_argument("--features", type=Path, default=OUTPUTS_DIR / "features.parquet")
    p.add_argument("--splits", type=Path, default=OUTPUTS_DIR / "splits.parquet")
    p.add_argument("--config", type=Path, default=CONFIGS_DIR / "models.yaml")
    p.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="Optional whitelist of family keys (default: all).",
    )
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    train(
        args.cohort,
        args.features,
        args.splits,
        args.config,
        families=args.families,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
