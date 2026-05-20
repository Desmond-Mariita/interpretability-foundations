"""Build subject-grouped CV splits for the P1 cohort.

Holds out 15% of *subjects* as the test set, then runs 5-fold ``GroupKFold``
on the remaining 85% — also on ``subject_id``. Writes one row per stay
indicating fold index (``-1`` if held out for test) and the ``is_test``
boolean.

This is the load-bearing leakage prevention specified in §6.1 of the design
spec. Deterministic via ``awake.utils.seed_everything``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import OUTPUTS_DIR, ensure_outputs_dir

from awake.utils import seed_everything

LOG = logging.getLogger("p1.splits")


def make_splits(
    cohort_path: Path,
    *,
    test_subject_frac: float = 0.15,
    n_folds: int = 5,
    seed: int = 1337,
    out_dir: Path | None = None,
) -> Path:
    """Persist a stay-level splits table to ``splits.parquet``.

    Args:
        cohort_path: Path to ``cohort.parquet``.
        test_subject_frac: Fraction of subjects held out for the test set.
        n_folds: Number of GroupKFold splits inside the training pool.
        seed: Master RNG seed.
        out_dir: Override the output directory (used by tests). Defaults to
            the project's ``outputs/``.

    Returns:
        Absolute path to the written ``splits.parquet``.
    """
    seed_everything(seed)
    cohort = pd.read_parquet(cohort_path, columns=["stay_id", "subject_id"])
    subjects = cohort["subject_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_test = round(len(subjects) * test_subject_frac)
    test_subjects = set(subjects[:n_test])
    LOG.info(
        "holding out %d/%d subjects (%.1f%%) for test",
        n_test,
        len(subjects),
        100 * n_test / max(1, len(subjects)),
    )

    is_test = cohort["subject_id"].isin(test_subjects)
    fold = np.full(len(cohort), -1, dtype=np.int8)

    train_mask = ~is_test
    train_subjects = cohort.loc[train_mask, "subject_id"].to_numpy()
    if train_mask.sum() < n_folds:
        raise ValueError("not enough training rows for the requested fold count")
    gkf = GroupKFold(n_splits=n_folds)
    for fold_idx, (_, val_idx) in enumerate(
        gkf.split(X=np.zeros(train_mask.sum()), groups=train_subjects)
    ):
        positions = np.flatnonzero(train_mask.to_numpy())[val_idx]
        fold[positions] = fold_idx

    out = pd.DataFrame(
        {"stay_id": cohort["stay_id"].to_numpy(), "fold": fold, "is_test": is_test.to_numpy()}
    )
    outputs = out_dir if out_dir is not None else ensure_outputs_dir()
    outputs.mkdir(parents=True, exist_ok=True)
    parquet_path = outputs / "splits.parquet"
    out.to_parquet(parquet_path, index=False)
    LOG.info(
        "wrote %s — train rows: %d, test rows: %d",
        parquet_path,
        (~out["is_test"]).sum(),
        out["is_test"].sum(),
    )
    return parquet_path


def main() -> int:
    """Entry point; parses CLI args and calls ``make_splits``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", type=Path, default=OUTPUTS_DIR / "cohort.parquet")
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    make_splits(
        args.cohort,
        test_subject_frac=args.test_frac,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
