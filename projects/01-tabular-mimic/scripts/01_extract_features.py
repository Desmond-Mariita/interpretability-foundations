"""Extract first-24h vitals and labs for the P1 cohort.

Reads ``outputs/cohort.parquet``, scans ``chartevents.csv.gz`` and
``labevents.csv.gz`` once each via DuckDB filtered to the relevant itemids,
clips values to physiologic ranges, aggregates ``first``/``min``/``max``/
``mean`` per (stay_id, feature), pivots wide, and writes a single
``outputs/features.parquet`` indexed by ``stay_id``.

Run via ``just features``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import CONFIGS_DIR, ensure_outputs_dir

LOG = logging.getLogger("p1.features")


def _itemid_case_sql(spec: dict[str, dict]) -> tuple[str, list[int]]:
    """Build a ``CASE`` expression mapping each itemid to its feature name.

    Args:
        spec: Mapping of feature-name -> ``{"itemids": [...], ...}``.

    Returns:
        A pair of ``(sql_expression, sorted_itemid_list)``. The expression
        evaluates to the feature name and is ``NULL`` for unmapped itemids
        (those are filtered out upstream by the ``IN`` clause).
    """
    branches: list[str] = []
    all_ids: list[int] = []
    for feature, conf in spec.items():
        ids = list(conf["itemids"])
        all_ids.extend(ids)
        in_list = ",".join(str(i) for i in ids)
        branches.append(f"WHEN itemid IN ({in_list}) THEN '{feature}'")
    case_sql = "CASE " + " ".join(branches) + " ELSE NULL END"
    return case_sql, sorted(set(all_ids))


def _temperature_normalisation_sql(spec: dict[str, dict]) -> tuple[str, list[int]]:
    """Build the SQL fragment that converts Fahrenheit temperatures to Celsius.

    Args:
        spec: ``vitals`` config block (so we can inspect ``temperature_c``).

    Returns:
        A ``(case_expression, fahrenheit_itemids)`` pair. The case expression
        operates on the bound parameter ``itemid`` and ``valuenum``; itemids
        outside ``fahrenheit_itemids`` pass through unchanged.
    """
    f_ids = list(spec.get("temperature_c", {}).get("coerce_from_f_itemids", []))
    if not f_ids:
        return "valuenum", []
    in_list = ",".join(str(i) for i in f_ids)
    return (
        f"CASE WHEN itemid IN ({in_list}) THEN (valuenum - 32) * 5.0 / 9.0 ELSE valuenum END",
        f_ids,
    )


def _plausibility_filter_sql(plausibility: dict[str, list[float]]) -> str:
    """Build the post-pivot ``WHERE`` clause that drops out-of-range values.

    Args:
        plausibility: Mapping of feature -> ``[low, high]`` inclusive bounds.

    Returns:
        A SQL fragment safe to splice after ``WHERE`` (joins clauses with
        ``AND``); empty mapping yields ``TRUE``.
    """
    if not plausibility:
        return "TRUE"
    clauses = [
        f"(feature <> '{feat}' OR (valuenum BETWEEN {lo} AND {hi}))"
        for feat, (lo, hi) in plausibility.items()
    ]
    return "(" + " AND ".join(clauses) + ")"


def _aggregate_sql(aggregates: list[str]) -> str:
    """Build the per-feature aggregation SELECT list."""
    parts: list[str] = []
    for agg in aggregates:
        if agg == "first":
            parts.append("FIRST(valuenum ORDER BY charttime) AS agg_first")
        elif agg == "min":
            parts.append("MIN(valuenum) AS agg_min")
        elif agg == "max":
            parts.append("MAX(valuenum) AS agg_max")
        elif agg == "mean":
            parts.append("AVG(valuenum) AS agg_mean")
        else:
            raise ValueError(f"unknown aggregate: {agg}")
    return ", ".join(parts)


def extract_features(mimic_path: Path, config: dict, *, out_dir: Path | None = None) -> Path:
    """Run the feature extraction pipeline and write Parquet.

    Args:
        mimic_path: Root of the MIMIC-IV download.
        config: Parsed ``features.yaml`` contents.
        out_dir: Override the output directory (used by tests). Reads
            ``cohort.parquet`` from this directory too. Defaults to the
            project's ``outputs/``.

    Returns:
        Absolute path to the written ``features.parquet``.
    """
    chart_csv = mimic_path / "icu" / "chartevents.csv.gz"
    lab_csv = mimic_path / "hosp" / "labevents.csv.gz"
    outputs = out_dir if out_dir is not None else ensure_outputs_dir()
    outputs.mkdir(parents=True, exist_ok=True)
    cohort_path = outputs / "cohort.parquet"
    for p in (chart_csv, lab_csv, cohort_path):
        if not p.exists():
            raise FileNotFoundError(f"required input missing: {p}")

    parquet_path = outputs / Path(config["outputs"]["features"]).name
    window_h = int(config["window_hours"])
    aggregates: list[str] = list(config["aggregates"])

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    # CREATE VIEW does not accept prepared parameters in DuckDB. The path is
    # internally constructed (not user input), so inlining is safe; we still
    # quote-escape defensively.
    cohort_uri = str(cohort_path).replace("'", "''")
    con.execute(f"CREATE OR REPLACE VIEW cohort AS SELECT * FROM read_parquet('{cohort_uri}')")

    # ── vitals from chartevents ────────────────────────────────────────────
    vitals_case, vital_ids = _itemid_case_sql(config["vitals"])
    temp_case, _ = _temperature_normalisation_sql(config["vitals"])
    vital_id_list = ",".join(str(i) for i in vital_ids)
    LOG.info("scanning chartevents for %d vital itemids", len(vital_ids))
    vitals_long = con.execute(
        f"""
        SELECT
            c.stay_id,
            {vitals_case} AS feature,
            {temp_case}   AS valuenum,
            ce.charttime
        FROM cohort c
        JOIN read_csv_auto(?, compression = 'gzip') ce
          ON ce.stay_id = c.stay_id
        WHERE ce.itemid IN ({vital_id_list})
          AND ce.valuenum IS NOT NULL
          AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '{window_h} hours'
        """,
        [str(chart_csv)],
    ).fetch_df()
    LOG.info("vitals_long rows: %d", len(vitals_long))

    # ── labs from labevents ────────────────────────────────────────────────
    labs_case, lab_ids = _itemid_case_sql(config["labs"])
    lab_id_list = ",".join(str(i) for i in lab_ids)
    LOG.info("scanning labevents for %d lab itemids", len(lab_ids))
    labs_long = con.execute(
        f"""
        SELECT
            c.stay_id,
            {labs_case} AS feature,
            le.valuenum AS valuenum,
            le.charttime
        FROM cohort c
        JOIN read_csv_auto(?, compression = 'gzip') le
          ON le.subject_id = c.subject_id
         AND le.hadm_id    = c.hadm_id
        WHERE le.itemid IN ({lab_id_list})
          AND le.valuenum IS NOT NULL
          AND le.charttime BETWEEN c.intime AND c.intime + INTERVAL '{window_h} hours'
        """,
        [str(lab_csv)],
    ).fetch_df()
    LOG.info("labs_long rows: %d", len(labs_long))

    long_df = pd.concat([vitals_long, labs_long], ignore_index=True)
    long_df = long_df[long_df["feature"].notnull()].copy()
    con.register("long_df", long_df)
    plaus_clause = _plausibility_filter_sql(config.get("plausibility", {}))
    agg_select = _aggregate_sql(aggregates)

    wide = con.execute(
        f"""
        WITH clipped AS (
            SELECT * FROM long_df WHERE {plaus_clause}
        ),
        agg AS (
            SELECT stay_id, feature, {agg_select}
            FROM clipped
            GROUP BY stay_id, feature
        )
        PIVOT agg
        ON feature
        USING FIRST(agg_first) AS first, FIRST(agg_min) AS min,
              FIRST(agg_max)   AS max,   FIRST(agg_mean) AS mean
        """
    ).fetch_df()
    LOG.info("wide feature table: %d rows x %d cols", len(wide), len(wide.columns))

    wide.to_parquet(parquet_path, index=False)
    LOG.info("wrote %s", parquet_path)
    return parquet_path


def main() -> int:
    """Entry point; parses CLI args and runs ``extract_features``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--mimic-path", required=True, type=Path)
    p.add_argument("--config", type=Path, default=CONFIGS_DIR / "features.yaml")
    args = p.parse_args()
    config = yaml.safe_load(args.config.read_text())
    extract_features(args.mimic_path, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
