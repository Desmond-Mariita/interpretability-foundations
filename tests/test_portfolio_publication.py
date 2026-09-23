"""Guard the public portfolio summary against superseded headline claims."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]


def test_root_readme_uses_repaired_p2_metrics() -> None:
    """Keep corrected P2 numbers and remove superseded v1 headline values."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    metrics = json.loads((ROOT / "projects/02-text-eraser/metrics.json").read_text())
    ig = metrics["metrics"]["integrated_gradients"]

    assert f"{ig['comprehensiveness']['mean']:.3f}" in readme
    assert f"{ig['aopc']['mean']:.3f}" in readme
    assert f"{ig['auprc']['mean']:.3f}" in readme
    assert "comprehensiveness 0.52" not in readme
    assert "AOPC 0.34" not in readme
    assert "only faithful explainer" not in readme.lower()


def test_root_readme_preserves_p3_paired_uncertainty() -> None:
    """Require the fused-image paired interval instead of a categorical winner claim."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8").replace("\u2212", "-")
    metrics = json.loads((ROOT / "projects/03-multimodal-hatefulmemes/metrics.json").read_text())
    diff = metrics["auroc_diffs"]["fused_vs_image"]

    assert f"{diff['point_diff']:+.3f}" in readme
    assert f"{diff['ci_low']:+.3f}" in readme
    assert f"{diff['ci_high']:+.3f}" in readme
    assert "**The image.**" not in readme


def test_root_readme_states_p4_estimand_boundary() -> None:
    """Keep P4 framed as recoverability under null visual input."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Incremental answer recoverability under null visual input" in readme
    assert "not proof of original-answer faithfulness or grounding" in readme


def test_portfolio_summary_exists_and_links_all_projects() -> None:
    """Keep one authoritative cross-project summary covering P1 through P5."""
    summary = (ROOT / "docs/PORTFOLIO_SUMMARY.md").read_text(encoding="utf-8")
    for label in ("P1", "P2", "P3", "P4", "P5"):
        assert label in summary
    assert "Tests validate software and analysis contracts" in summary
