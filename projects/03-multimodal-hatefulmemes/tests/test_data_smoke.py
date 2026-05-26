"""Smoke tests for the P3 data-loading script (00_data)."""

import importlib
import json

import pytest

data_mod = importlib.import_module("00_data")


@pytest.mark.unit
def test_load_split_parses_records(tmp_path):
    """Verify load_split reads both records from a JSONL file, including one without a trailing newline."""
    f = tmp_path / "dev.jsonl"
    f.write_text(
        json.dumps({"id": 1, "img": "img/1.png", "label": 0, "text": "a"})
        + "\n"
        + json.dumps({"id": 2, "img": "img/2.png", "label": 1, "text": "b"})  # no trailing nl
    )
    rows = data_mod.load_split(f)
    assert len(rows) == 2  # both records, despite missing trailing newline
    assert rows[1]["label"] == 1
