"""Consistency guards between P1's public docs and its committed metric artifact.

Every headline number in ``README.md`` and ``REPORT.md`` must trace to
``metrics.json`` (the machine-readable artifact of the final run), and the
probability/uncertainty language must stay within what the evidence
supports. These tests protect the scientific-integrity repair against
regression and require no MIMIC data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]

# Row order of the headline tables in both README.md and REPORT.md.
MODEL_ORDER = ["lightgbm", "ebm", "logistic", "decision_tree"]


def _read(name: str) -> str:
    return (PROJECT_ROOT / name).read_text()


def _load_metrics() -> dict:
    return json.loads(_read("metrics.json"))


@pytest.fixture(scope="module")
def metrics() -> dict:
    """Return the parsed ``metrics.json`` artifact, cached per module."""
    return _load_metrics()


def _table_rows(doc: str, header_contains: str) -> list[list[str]]:
    """Return the cell lists of the first markdown table whose header matches.

    Args:
        doc: Raw markdown document.
        header_contains: Substring that identifies the header row.

    Returns:
        One list of stripped cells per data row (bold markers removed).
    """
    lines = doc.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("|") and header_contains in line:
            rows: list[list[str]] = []
            for ln in lines[i + 2 :]:  # skip the |---| separator row
                if not ln.lstrip().startswith("|"):
                    break
                cells = [c.strip().strip("*") for c in ln.strip().strip("|").split("|")]
                rows.append(cells)
            if rows:
                return rows
    raise AssertionError(f"no markdown table found containing {header_contains!r}")


def _cv_pair(cell: str) -> tuple[str, str]:
    """Split a ``mean ± std`` table cell into its two rounded strings."""
    m = re.fullmatch(r"([0-9.]+)\s*±\s*([0-9.]+)", cell.strip())
    if not m:
        raise AssertionError(f"CV cell {cell!r} does not match 'mean ± std'")
    return m.group(1), m.group(2)


def _assert_rounds(value: float, doc_cell: str) -> None:
    """Assert that ``value`` rounds to the 3-decimal string in the doc."""
    assert f"{round(value, 3):.3f}" == doc_cell, (
        f"{value:.6f} does not round to the documented {doc_cell!r}"
    )


@pytest.mark.unit
def test_readme_headline_table_matches_metrics(metrics: dict) -> None:
    """README headline numbers equal metrics.json values rounded to 3 dp."""
    rows = _table_rows(_read("README.md"), "CV AUROC")
    assert len(rows) == 4
    for row, key in zip(rows, MODEL_ORDER, strict=True):
        m = metrics["models"][key]
        mean_doc, std_doc = _cv_pair(row[1])
        _assert_rounds(m["cv"]["auroc"]["mean"], mean_doc)
        _assert_rounds(m["cv"]["auroc"]["std"], std_doc)
        _assert_rounds(m["test"]["auroc"], row[2])
        _assert_rounds(m["test"]["auprc"], row[3])
        _assert_rounds(m["test"]["brier"], row[4])


@pytest.mark.unit
def test_report_headline_table_matches_metrics(metrics: dict) -> None:
    """REPORT headline numbers (incl. log-loss) equal metrics.json values."""
    rows = _table_rows(_read("REPORT.md"), "Log-loss")
    assert len(rows) == 4
    for row, key in zip(rows, MODEL_ORDER, strict=True):
        m = metrics["models"][key]
        mean_doc, std_doc = _cv_pair(row[1])
        _assert_rounds(m["cv"]["auroc"]["mean"], mean_doc)
        _assert_rounds(m["cv"]["auroc"]["std"], std_doc)
        _assert_rounds(m["test"]["auroc"], row[2])
        _assert_rounds(m["test"]["auprc"], row[3])
        _assert_rounds(m["test"]["brier"], row[4])
        _assert_rounds(m["test"]["log_loss"], row[5])


@pytest.mark.unit
def test_metrics_schema_invariants(metrics: dict) -> None:
    """metrics.json carries complete per-fold and test blocks for each family."""
    n_total = metrics["cohort_size"]
    assert isinstance(n_total, int) and n_total > 0
    for key in MODEL_ORDER:
        m = metrics["models"][key]
        assert set(m["cv"]) == {"auroc", "auprc", "brier", "log_loss"}
        for stat in m["cv"].values():
            assert set(stat) == {"mean", "std"}
            assert stat["std"] >= 0
        assert [f["fold"] for f in m["cv_per_fold"]] == [0, 1, 2, 3, 4]
        assert set(m["test"]) == {"auroc", "auprc", "brier", "log_loss"}
        for value in m["test"].values():
            assert 0.0 <= value <= 1.0
        assert m["n_train"] + m["n_test"] == n_total


@pytest.mark.unit
def test_doc_counts_agree_with_artifacts() -> None:
    """The stay counts quoted in the docs match cohort_stats.json/metrics.json."""
    stats = json.loads(_read("cohort_stats.json"))
    metrics = _load_metrics()
    readme = _read("README.md")
    report = _read("REPORT.md")
    # Cohort size and test size are quoted in both documents.
    assert f"{stats['n_stays']:,}" in readme
    assert f"{stats['n_stays']:,}" in report
    test_n = metrics["models"]["lightgbm"]["n_test"]
    assert f"{test_n:,}" in readme
    assert f"{test_n:,}" in report
    # The modeling frame drops exactly one cohort stay (no in-window
    # measurements), which is what metrics.json's cohort_size records.
    assert metrics["cohort_size"] == stats["n_stays"] - 1


@pytest.mark.unit
def test_docs_avoid_unsupported_claim_language() -> None:
    """Claim language stays inside the evidence: no CI/Brier-calibration wording.

    Guards the repaired defects: Brier must not be equated with calibration
    ("best-calibrated ... by Brier"), the reliability diagram must not be
    misdescribed ("under-confident" for a below-diagonal curve), and fold SD
    must not be dressed up as inferential statistics.
    """
    docs = {
        "projects/01-tabular-mimic/README.md": _read("README.md"),
        "projects/01-tabular-mimic/REPORT.md": _read("REPORT.md"),
        "README.md (repo root)": (REPO_ROOT / "README.md").read_text(),
    }
    # Normalise whitespace so phrase checks survive benign line re-wrapping.
    flat = {name: re.sub(r"\s+", " ", doc) for name, doc in docs.items()}
    for name, doc in flat.items():
        for banned in ("best-calibrated", "under-confident"):
            assert banned not in doc, f"{name} still contains {banned!r}"
    # The repaired wording must actually be present in the P1 docs.
    for name in ("projects/01-tabular-mimic/README.md", "projects/01-tabular-mimic/REPORT.md"):
        doc = flat[name]
        for required in (
            "not a confidence interval",
            "fold-to-fold",
            "retrospective",
            "no post-hoc calibration was applied",
        ):
            assert required in doc, f"{name} is missing the required wording {required!r}"


@pytest.mark.unit
def test_readme_has_no_duplicated_sections() -> None:
    """README carries a single Reproduce and a single Limitations section."""
    readme = _read("README.md")
    assert readme.count("## Reproduce") == 1
    assert readme.count("## Limitations") == 1
