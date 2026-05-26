"""Smoke and unit tests for the P2 fetch (00_fetch_data) and prepare (01_prepare) scripts."""

import hashlib
import importlib

import pytest

fetch = importlib.import_module("00_fetch_data")
prepare = importlib.import_module("01_prepare")


@pytest.mark.unit
def test_verify_sha256_accepts_match(tmp_path):
    """verify_sha256 returns True when the file's digest matches the expected value."""
    f = tmp_path / "blob"
    f.write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    assert fetch.verify_sha256(f, digest) is True


@pytest.mark.unit
def test_verify_sha256_rejects_mismatch(tmp_path):
    """verify_sha256 returns False when the file's digest does not match the expected value."""
    f = tmp_path / "blob"
    f.write_bytes(b"hello")
    assert fetch.verify_sha256(f, "deadbeef") is False


@pytest.mark.smoke
def test_prepare_builds_word_mask_and_coverage(tmp_path):
    """build_record produces correct label, word list, gold mask, and word count for a sample ERD example."""
    doc = "the movie was absolutely terrible and boring throughout"
    words = doc.split()
    # evidence covers words 3..5 ("absolutely terrible and")
    example = {
        "annotation_id": "neg_0",
        "classification": "NEG",
        "evidences": [
            [
                {
                    "docid": "neg_0",
                    "start_token": 3,
                    "end_token": 6,
                    "text": "absolutely terrible and",
                }
            ]
        ],
    }
    record = prepare.build_record(example, doc_text=doc)
    assert record["label"] == 0
    assert record["words"] == words
    assert record["gold_mask"] == [0, 0, 0, 1, 1, 1, 0, 0]
    assert record["n_words"] == len(words)


@pytest.mark.unit
def test_prepare_drops_comparison_multidoc_evidence():
    """is_comparison returns True when an evidence group spans multiple distinct document IDs."""
    example = {
        "annotation_id": "x_0",
        "classification": "POS",
        "evidences": [
            [
                {"docid": "x_0", "start_token": 0, "end_token": 1, "text": "a"},
                {"docid": "OTHER_1", "start_token": 0, "end_token": 1, "text": "b"},
            ]
        ],
    }
    assert prepare.is_comparison(example, docid="x_0") is True
