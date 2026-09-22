"""Guards for the corrected P4 semantics and provenance.

These tests pin the publication artifacts to the authoritative run record:

- REPORT tables must match the committed metrics snapshot (copied from the executed
  notebook's recorded stdout).
- Counts must be internally consistent (n = leak-flagged + leakage-free).
- Pipeline labels must map to the configured model ids.
- The corrected estimand language must be present, and no chance-baseline claim may
  creep back in ("consistency" is not accuracy; the no-explanation rate is empirical).
- The leakage-free subset construction must be deterministic, dataset-side only, and
  must not be described as removing all leakage.
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
REPORT = (PROJECT / "REPORT.md").read_text()
README = (PROJECT / "README.md").read_text()
SNAPSHOT = json.loads((PROJECT / "assets" / "metrics_snapshot.json").read_text())

A_LABEL = r"A \(BLIP-2 \+ Qwen-7B\)"
B_LABEL = r"B \(Qwen-VL-3B\)"
B7_LABEL = r"B7 \(Qwen-VL-7B\)"


def _section(text: str, heading: str) -> str:
    """Return the text between ``heading`` and the next same-level heading."""
    match = re.search(rf"^{re.escape(heading)}.*$", text, re.MULTILINE)
    assert match, f"heading {heading!r} not found"
    rest = text[match.end() :]
    next_heading = re.search(r"^#{2,5} ", rest, re.MULTILINE)
    return rest[: next_heading.start()] if next_heading else rest


@pytest.mark.unit
def test_report_headline_tables_match_snapshot():
    """5.1/5.2/5.4 tables reproduce the notebook-verified snapshot values."""
    tol = 0.0015

    s51 = _section(REPORT, "### 5.1")
    s52 = _section(REPORT, "### 5.2")
    s54 = _section(REPORT, "### 5.4")
    rows_51 = {
        "A": re.search(
            rf"\|\s*{A_LABEL}\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", s51
        ),
        "B": re.search(
            rf"\|\s*{B_LABEL}\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", s51
        ),
        "B7": re.search(
            rf"\|\s*{B7_LABEL}\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", s51
        ),
    }
    rows_52 = {
        "A": re.search(
            rf"\|\s*{A_LABEL}\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*\**([\d.]+)\**\s*"
            r"\[\s*([\d.]+),\s*([\d.]+)\s*\]\s*\|",
            s52,
        ),
        "B": re.search(
            rf"\|\s*{B_LABEL}\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*\**([\d.]+)\**\s*"
            r"\[\s*([\d.]+),\s*([\d.]+)\s*\]\s*\|",
            s52,
        ),
        "B7": re.search(
            rf"\|\s*{B7_LABEL}\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*\**([\d.]+)\**\s*"
            r"\[\s*([\d.]+),\s*([\d.]+)\s*\]\s*\|",
            s52,
        ),
    }
    rows_54 = {
        p: re.search(
            rf"\|\s*{p}\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\[\s*([\d.]+),\s*([\d.]+)\s*\]\s*\|", s54
        )
        for p in ("A", "B", "B7")
    }

    for p, row in rows_51.items():
        assert row, f"5.1 row for {p} not found"
        snap = SNAPSHOT["unfiltered"]["pipelines"][p]
        assert float(row.group(1)) == pytest.approx(snap["accuracy"], abs=tol)
        assert float(row.group(2)) == pytest.approx(snap["parse_rate_answer"], abs=tol)
        assert float(row.group(3)) == pytest.approx(snap["expl_leak_rate"], abs=tol)
    for p, row in rows_52.items():
        assert row, f"5.2 row for {p} not found"
        snap = SNAPSHOT["unfiltered"]["pipelines"][p]["consistency"]
        assert float(row.group(1)) == pytest.approx(snap["with_expl"], abs=tol)
        assert float(row.group(2)) == pytest.approx(snap["no_expl"], abs=tol)
        assert float(row.group(3)) == pytest.approx(snap["delta"], abs=tol)
        assert float(row.group(4)) == pytest.approx(snap["delta_ci"][0], abs=tol)
        assert float(row.group(5)) == pytest.approx(snap["delta_ci"][1], abs=tol)
    for p, row in rows_54.items():
        assert row, f"5.4 row for {p} not found"
        snap = SNAPSHOT["filtered"]["pipelines"][p]["consistency"]
        assert float(row.group(2)) == pytest.approx(snap["delta"], abs=tol)
        assert float(row.group(3)) == pytest.approx(snap["delta_ci"][0], abs=tol)
        assert float(row.group(4)) == pytest.approx(snap["delta_ci"][1], abs=tol)


@pytest.mark.unit
def test_counts_internally_consistent():
    """N = leak-flagged + leakage-free, in the REPORT header and the snapshot."""
    match = re.search(r"n=(\d+);\s*(\d+) leak-flagged,\s*(\d+) leakage-free", REPORT)
    assert match
    n, flagged, clean = (int(g) for g in match.groups())
    assert n == flagged + clean
    assert SNAPSHOT["run"]["n"] == n == 1145
    assert SNAPSHOT["run"]["n_leak_flagged"] == flagged
    assert SNAPSHOT["run"]["n_filtered"] == clean == 340


@pytest.mark.unit
def test_model_label_mapping_matches_config():
    """A/B/B7 labels map to the configured model ids; revisions are 40-hex shas."""
    import yaml

    cfg = yaml.safe_load((PROJECT / "configs" / "pipelines.yaml").read_text())["models"]
    assert cfg == {
        "blip2": "Salesforce/blip2-opt-2.7b",
        "qwen_lm": "Qwen/Qwen2.5-7B-Instruct",
        "qwen_vl_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen_vl_7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    }
    revisions = SNAPSHOT["run"]["model_revisions"]
    assert set(revisions) == set(cfg)
    for sha in revisions.values():
        assert re.fullmatch(r"[0-9a-f]{40}", sha)


@pytest.mark.unit
def test_snapshot_schema_has_expected_keys():
    """The snapshot carries the four separated constructs, not one merged number."""
    for p in ("A", "B", "B7"):
        pipe = SNAPSHOT["unfiltered"]["pipelines"][p]
        assert set(pipe) == {"accuracy", "parse_rate_answer", "expl_leak_rate", "consistency"}
        assert set(pipe["consistency"]) == {"with_expl", "no_expl", "delta", "delta_ci"}
        assert 0.0 <= pipe["expl_leak_rate"] <= 1.0
        # Snapshot values are recorded to 4 dp, so delta == with - no only up to rounding.
        assert pipe["consistency"]["delta"] == pytest.approx(
            pipe["consistency"]["with_expl"] - pipe["consistency"]["no_expl"], abs=0.0002
        )


@pytest.mark.unit
def test_estimand_language_present_and_no_chance_baseline():
    """The corrected estimand is stated; no chance-level claim remains."""
    estimand = (
        "incremental answer recoverability from the supplied explanation under null visual input"
    )
    assert estimand in REPORT
    assert estimand in README
    # The no-explanation rate is an empirical baseline; the 0.25 uniform-guess number may
    # appear only to be explicitly disclaimed as the null.
    assert "chance-like" not in REPORT
    assert "no chance null is claimed" in REPORT


@pytest.mark.unit
def test_leakage_subset_deterministic_dataset_side_and_incomplete():
    """prepare_rows flags gold-text leakage only, deterministically, never model output."""
    mod = importlib.import_module("00_data")
    raw = [
        {
            "id": "q1",
            "question": "q?",
            "choices": ["owl", "cat", "dog", "fox"],
            "correct_choice_idx": 0,
            "rationales": ["it is an owl at night"],
        },
        {
            "id": "q2",
            "question": "q?",
            "choices": ["owl", "cat", "dog", "fox"],
            "correct_choice_idx": 0,
            "rationales": ["the answer is option A"],
        },
        {
            "id": "q3",
            "question": "q?",
            "choices": ["owl", "cat", "dog", "fox"],
            "correct_choice_idx": 0,
            "rationales": ["it is a nocturnal bird of prey"],  # paraphrase: NOT flagged
        },
    ]
    first = mod.prepare_rows(raw)
    second = mod.prepare_rows(raw)  # deterministic
    assert [r["leakage_flag"] for r in first] == [True, False, False]
    assert [r["leakage_flag"] for r in first] == [r["leakage_flag"] for r in second]
    # The flag is a pure function of the dataset row (gold text vs. rationale), so the
    # bare-letter rationale above is not flagged: letters would false-positive.
    assert first[1]["leakage_flag"] is False
    # Paraphrase leakage is NOT caught: the filter removes only verbatim gold-choice text.
    assert first[2]["leakage_flag"] is False


@pytest.mark.unit
def test_no_causal_attribution_of_divergence_or_leakage():
    """Divergence attribution and filtered-subset leakage claims stay bounded.

    The design does not isolate the modality-stack factor (B7 bounds only the
    parameter-count confound), and the verbatim-gold-text filter removes only one
    leakage channel -- so neither may be stated as a causal conclusion.
    """
    banned = (
        "driven by the modality stack",
        "attributable to the modality stack",
        "not an artifact of dataset-side answer leakage",
    )
    notebook_py = (PROJECT / "notebooks" / "01-vqa-consistency.py").read_text()
    for doc in (REPORT, README, notebook_py):
        for phrase in banned:
            assert phrase not in doc
