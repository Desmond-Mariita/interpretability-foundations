import hashlib
import importlib

import pytest

fetch = importlib.import_module("00_fetch_data")


@pytest.mark.unit
def test_verify_sha256_accepts_match(tmp_path):
    f = tmp_path / "blob"
    f.write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    assert fetch.verify_sha256(f, digest) is True


@pytest.mark.unit
def test_verify_sha256_rejects_mismatch(tmp_path):
    f = tmp_path / "blob"
    f.write_bytes(b"hello")
    assert fetch.verify_sha256(f, "deadbeef") is False
