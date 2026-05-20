"""Build the P1 cohort from MIMIC-IV CSVs via DuckDB.

Filters to adults on their first ICU stay per hospital admission with either
≥24h of ICU LOS or an in-hospital death inside the first 24h. Writes a
single Parquet table indexed by ``stay_id`` plus a small JSON manifest with
cohort-level descriptive statistics.

Run via ``just cohort`` (or ``uv run python scripts/00_build_cohort.py``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import duckdb
import yaml

# Make the project's scripts/ importable when invoked from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import CONFIGS_DIR, PROJECT_ROOT, ensure_outputs_dir

LOG = logging.getLogger("p1.cohort")

COHORT_SQL = """
WITH icu_ranked AS (
    SELECT
        subject_id, hadm_id, stay_id,
        first_careunit, intime, outtime, los,
        ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) AS icu_seq
    FROM read_csv_auto(?, compression = 'gzip')
),
adm AS (
    SELECT
        subject_id, hadm_id, admittime, dischtime, deathtime,
        admission_type, hospital_expire_flag
    FROM read_csv_auto(?, compression = 'gzip')
),
pat AS (
    SELECT subject_id, gender, anchor_age, anchor_year
    FROM read_csv_auto(?, compression = 'gzip')
)
SELECT
    i.subject_id,
    i.hadm_id,
    i.stay_id,
    i.first_careunit,
    i.intime,
    i.outtime,
    i.los AS los_icu_days,
    a.admittime,
    a.dischtime,
    a.deathtime,
    a.admission_type,
    p.gender,
    CAST(p.anchor_age + (date_part('year', a.admittime) - p.anchor_year) AS INTEGER) AS age,
    a.hospital_expire_flag AS target
FROM icu_ranked i
JOIN adm a USING (subject_id, hadm_id)
JOIN pat p USING (subject_id)
WHERE i.icu_seq = 1
  AND (p.anchor_age + (date_part('year', a.admittime) - p.anchor_year)) >= ?
  AND (
      i.los * 24 >= ?
      OR (a.deathtime IS NOT NULL AND a.deathtime <= i.intime + INTERVAL '24 hours')
  )
"""


def build_cohort(
    mimic_path: Path, config: dict, *, out_dir: Path | None = None
) -> tuple[Path, dict]:
    """Run the cohort SQL and persist its Parquet + stats sidecar.

    Args:
        mimic_path: Root of the MIMIC-IV download (must contain ``hosp/`` and
            ``icu/``).
        config: Parsed ``cohort.yaml`` contents.
        out_dir: Override the output directory (used by tests). Defaults to
            the project's ``outputs/`` directory.

    Returns:
        A pair of ``(parquet_path, stats_dict)``. ``stats_dict`` is also
        written to ``cohort_stats.json`` in the output directory.
    """
    icu_csv = mimic_path / "icu" / "icustays.csv.gz"
    adm_csv = mimic_path / "hosp" / "admissions.csv.gz"
    pat_csv = mimic_path / "hosp" / "patients.csv.gz"
    for p in (icu_csv, adm_csv, pat_csv):
        if not p.exists():
            raise FileNotFoundError(f"required MIMIC table missing: {p}")

    outputs = out_dir if out_dir is not None else ensure_outputs_dir()
    outputs.mkdir(parents=True, exist_ok=True)
    parquet_path = outputs / Path(config["outputs"]["cohort"]).name
    # Stats manifest is small and committed; resolve it relative to the
    # project root when running normally, or alongside the parquet in tests.
    stats_cfg = Path(config["outputs"]["stats"])
    if out_dir is None:
        stats_path = (PROJECT_ROOT / stats_cfg).resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        stats_path = outputs / stats_cfg.name

    min_age = int(config["min_age_years"])
    min_los_hours = int(config["min_los_icu_hours"])

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    LOG.info("running cohort SQL (min_age=%d, min_los_h=%d)", min_age, min_los_hours)
    df = con.execute(
        COHORT_SQL,
        [str(icu_csv), str(adm_csv), str(pat_csv), min_age, min_los_hours],
    ).fetch_df()
    LOG.info("cohort rows: %d (unique subjects: %d)", len(df), df["subject_id"].nunique())

    df.to_parquet(parquet_path, index=False)
    LOG.info("wrote %s", parquet_path)

    stats = _summarise(df, mimic_version=str(config["mimic_version"]))
    stats_path.write_text(json.dumps(stats, indent=2, default=str) + "\n")
    LOG.info("wrote %s", stats_path)

    return parquet_path, stats


def _summarise(df, mimic_version: str) -> dict:
    """Return a JSON-serialisable summary dict of cohort-level descriptive stats.

    Args:
        df: Cohort dataframe returned by the SQL.
        mimic_version: MIMIC-IV version string for the manifest.

    Returns:
        Dict with cohort size, unique subjects, base rate, age and gender
        breakdowns, careunit distribution, and the source MIMIC version.
    """
    n = len(df)
    n_pos = int(df["target"].sum())
    return {
        "mimic_version": mimic_version,
        "n_stays": n,
        "n_subjects": int(df["subject_id"].nunique()),
        "n_admissions": int(df["hadm_id"].nunique()),
        "target_positive": n_pos,
        "target_base_rate": (n_pos / n) if n else 0.0,
        "age": {
            "median": float(df["age"].median()),
            "p25": float(df["age"].quantile(0.25)),
            "p75": float(df["age"].quantile(0.75)),
        },
        "gender_pct": (df["gender"].value_counts(normalize=True) * 100).round(2).to_dict(),
        "first_careunit_pct": (df["first_careunit"].value_counts(normalize=True) * 100)
        .round(2)
        .to_dict(),
        "los_icu_days": {
            "median": float(df["los_icu_days"].median()),
            "p25": float(df["los_icu_days"].quantile(0.25)),
            "p75": float(df["los_icu_days"].quantile(0.75)),
        },
    }


def main() -> int:
    """Entry point. Parses CLI args, loads the cohort YAML, runs the SQL.

    Returns:
        ``0`` on success, non-zero on failure.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--mimic-path", required=True, type=Path)
    p.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "cohort.yaml",
        help="cohort YAML config (default: configs/cohort.yaml)",
    )
    args = p.parse_args()

    config = yaml.safe_load(args.config.read_text())
    build_cohort(args.mimic_path, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
