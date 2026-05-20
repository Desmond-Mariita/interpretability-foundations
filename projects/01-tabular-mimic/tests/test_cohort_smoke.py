"""End-to-end smoke test for the P1 pipeline on 8-row synthetic data.

The fixture writes a minimal MIMIC-IV-shaped directory tree (gzipped CSVs)
for one ICU + one hosp module, runs the cohort SQL, feature extraction,
and split generation, and asserts the basic shape and invariants.

No real MIMIC data is involved.
"""

from __future__ import annotations

import gzip
import importlib
from pathlib import Path

import pandas as pd
import pytest
import yaml


def _write_gz_csv(path: Path, df: pd.DataFrame) -> None:
    """Write ``df`` to ``path`` as a gzip-compressed CSV.

    Args:
        path: Output ``.csv.gz`` path; parent must exist.
        df: Dataframe to serialise.
    """
    with gzip.open(path, "wt", newline="") as fh:
        df.to_csv(fh, index=False)


@pytest.fixture
def synthetic_mimic(tmp_path: Path) -> Path:
    """Materialise a tiny MIMIC-IV-shaped directory tree.

    The cohort has four stays in total. Two are eligible (adult, first ICU
    stay, ≥24h LOS) — one a survivor, one a death. The other two stays must
    be excluded: one is a paediatric patient, the other is a second ICU stay
    on a different admission.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        The MIMIC root path; the caller passes it as ``--mimic-path``.
    """
    root = tmp_path / "mimic"
    (root / "hosp").mkdir(parents=True)
    (root / "icu").mkdir(parents=True)

    # Six base subjects:
    #   10001 — adult survivor (eligible)
    #   10002 — adult death (eligible, target=1)
    #   10003 — paediatric (excluded)
    #   10004 — adult with LOS<24h, no death (excluded)
    #   10005 — adult survivor (eligible)
    #   10006 — adult survivor (eligible)
    #   10007 — adult survivor (eligible)
    # Plus a second ICU stay on 10001 (stay 30099) that must be excluded by the
    # first-ICU-stay-per-admission rule.
    patients = pd.DataFrame(
        {
            "subject_id": [10001, 10002, 10003, 10004, 10005, 10006, 10007],
            "gender": ["M", "F", "M", "F", "M", "F", "M"],
            "anchor_age": [65, 40, 10, 78, 55, 70, 62],
            "anchor_year": [2150] * 7,
            "anchor_year_group": ["2008 - 2010"] * 7,
            "dod": [None, "2150-01-05 10:00:00", None, None, None, None, None],
        }
    )
    admissions = pd.DataFrame(
        {
            "subject_id": [10001, 10002, 10003, 10004, 10005, 10006, 10007],
            "hadm_id": [20001, 20002, 20003, 20004, 20005, 20006, 20007],
            "admittime": [
                "2150-01-01 10:00:00",
                "2150-01-02 09:00:00",
                "2150-01-03 09:00:00",
                "2150-01-04 12:00:00",
                "2150-01-05 08:00:00",
                "2150-01-06 09:00:00",
                "2150-01-07 09:00:00",
            ],
            "dischtime": [
                "2150-01-06 10:00:00",
                "2150-01-05 11:00:00",
                "2150-01-06 09:00:00",
                "2150-01-06 12:00:00",
                "2150-01-10 10:00:00",
                "2150-01-11 09:00:00",
                "2150-01-12 09:00:00",
            ],
            "deathtime": [None, "2150-01-05 10:00:00", None, None, None, None, None],
            "admission_type": ["EMERGENCY"] * 7,
            "hospital_expire_flag": [0, 1, 0, 0, 0, 0, 0],
        }
    )
    icustays = pd.DataFrame(
        {
            "subject_id": [10001, 10002, 10003, 10004, 10005, 10006, 10007, 10001],
            "hadm_id": [20001, 20002, 20003, 20004, 20005, 20006, 20007, 20001],
            "stay_id": [30001, 30002, 30003, 30004, 30005, 30006, 30007, 30099],
            "first_careunit": ["MICU", "SICU", "PICU", "CCU", "MICU", "SICU", "TSICU", "MICU"],
            "last_careunit": ["MICU", "SICU", "PICU", "CCU", "MICU", "SICU", "TSICU", "MICU"],
            "intime": [
                "2150-01-01 12:00:00",
                "2150-01-02 10:00:00",
                "2150-01-03 10:00:00",
                "2150-01-04 13:00:00",
                "2150-01-05 09:00:00",
                "2150-01-06 10:00:00",
                "2150-01-07 10:00:00",
                "2150-01-02 10:00:00",  # 10001 second ICU stay → must be excluded
            ],
            "outtime": [
                "2150-01-05 12:00:00",
                "2150-01-05 10:00:00",
                "2150-01-06 09:00:00",
                "2150-01-06 11:00:00",
                "2150-01-08 09:00:00",
                "2150-01-09 10:00:00",
                "2150-01-09 10:00:00",
                "2150-01-03 09:00:00",
            ],
            "los": [4.0, 3.0, 3.0, 0.5, 3.0, 3.0, 2.0, 0.96],
        }
    )

    chartevents = pd.DataFrame(
        {
            "subject_id": [10001, 10001, 10002, 10004],
            "hadm_id": [20001, 20001, 20002, 20004],
            "stay_id": [30001, 30001, 30002, 30004],
            "charttime": [
                "2150-01-01 13:00:00",
                "2150-01-01 18:00:00",
                "2150-01-02 11:00:00",
                "2150-01-04 14:00:00",
            ],
            "itemid": [220045, 220045, 220045, 220045],
            "valuenum": [85.0, 92.0, 110.0, 70.0],
            "valueuom": ["bpm"] * 4,
        }
    )
    labevents = pd.DataFrame(
        {
            "subject_id": [10001, 10002],
            "hadm_id": [20001, 20002],
            "charttime": ["2150-01-01 14:00:00", "2150-01-02 12:00:00"],
            "itemid": [50912, 50912],
            "valuenum": [1.1, 2.4],
            "valueuom": ["mg/dL", "mg/dL"],
        }
    )

    _write_gz_csv(root / "hosp" / "patients.csv.gz", patients)
    _write_gz_csv(root / "hosp" / "admissions.csv.gz", admissions)
    _write_gz_csv(root / "icu" / "icustays.csv.gz", icustays)
    _write_gz_csv(root / "icu" / "chartevents.csv.gz", chartevents)
    _write_gz_csv(root / "hosp" / "labevents.csv.gz", labevents)
    return root


def _load_configs(project_root: Path) -> tuple[dict, dict]:
    """Load both config YAMLs from the project's configs directory."""
    cohort = yaml.safe_load((project_root / "configs" / "cohort.yaml").read_text())
    features = yaml.safe_load((project_root / "configs" / "features.yaml").read_text())
    return cohort, features


@pytest.mark.smoke
def test_pipeline_end_to_end(synthetic_mimic: Path, tmp_path: Path) -> None:
    """Cohort -> features -> splits runs on synthetic data and respects all filters.

    Args:
        synthetic_mimic: Path to the synthetic MIMIC root (fixture).
        tmp_path: Per-test temp dir for outputs (fixture).
    """
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = tmp_path / "outputs"

    cohort_cfg, features_cfg = _load_configs(project_root)

    build_cohort = importlib.import_module("00_build_cohort").build_cohort
    extract_features = importlib.import_module("01_extract_features").extract_features
    make_splits = importlib.import_module("02_make_splits").make_splits

    _, stats = build_cohort(synthetic_mimic, cohort_cfg, out_dir=outputs_dir)
    cohort_path = outputs_dir / "cohort.parquet"
    assert cohort_path.exists()
    cohort = pd.read_parquet(cohort_path)

    # Eligible: 10001 (survivor), 10002 (death), 10005, 10006, 10007 — five stays.
    # Excluded: 10003 (paediatric), 10004 (LOS<24h, no death), plus 10001's 2nd
    # ICU stay (30099) by the first-ICU-stay-per-admission rule.
    assert set(cohort["subject_id"].tolist()) == {10001, 10002, 10005, 10006, 10007}
    assert set(cohort["stay_id"].tolist()) == {30001, 30002, 30005, 30006, 30007}
    assert stats["n_stays"] == 5
    assert stats["target_positive"] == 1
    assert stats["target_base_rate"] == pytest.approx(0.2)

    features_path = extract_features(synthetic_mimic, features_cfg, out_dir=outputs_dir)
    feats = pd.read_parquet(features_path)
    assert 30001 in feats["stay_id"].tolist()
    # Subject 10001 has two heart-rate readings (85, 92); aggregates must reflect that.
    row_10001 = feats[feats["stay_id"] == 30001].iloc[0]
    assert row_10001["heart_rate_first"] == pytest.approx(85.0)
    assert row_10001["heart_rate_min"] == pytest.approx(85.0)
    assert row_10001["heart_rate_max"] == pytest.approx(92.0)
    assert row_10001["heart_rate_mean"] == pytest.approx(88.5)

    splits_path = make_splits(
        cohort_path, test_subject_frac=0.2, n_folds=2, seed=7, out_dir=outputs_dir
    )
    splits = pd.read_parquet(splits_path)
    assert splits["is_test"].sum() == 1
    assert set(splits["fold"].unique()) <= {-1, 0, 1}
    # Patient-level leakage check: no subject appears in both test and train.
    cohort_subjects = cohort.set_index("stay_id")["subject_id"]
    test_subjects = cohort_subjects.loc[splits.loc[splits["is_test"], "stay_id"]].unique()
    train_subjects = cohort_subjects.loc[splits.loc[~splits["is_test"], "stay_id"]].unique()
    assert not set(test_subjects) & set(train_subjects)
