"""Fetch + verify + parse UD English-EWT into prepared parquet (CC BY-SA 4.0)."""

from __future__ import annotations

import hashlib

import pyarrow as pa
from _paths import CONLLU, PREPARED, ensure_dirs, load_config
from _udparse import parse_conllu

_SCHEMA = pa.schema(
    [
        ("sent_id", pa.string()),
        ("text", pa.string()),
        ("words", pa.list_(pa.string())),
        ("upos", pa.list_(pa.string())),
        ("number", pa.list_(pa.string())),
        ("space_after", pa.list_(pa.bool_())),
    ]
)


def rows_to_table(sents: list[dict]) -> pa.Table:
    """Build a pyarrow Table with an explicit schema (so empty strings never coerce to null)."""
    cols = {f.name: [s[f.name] for s in sents] for f in _SCHEMA}
    return pa.table(cols, schema=_SCHEMA)


def _fetch(url: str, dest, sha: str) -> str:  # pragma: no cover - network/slow
    import urllib.request

    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    got = hashlib.sha256(dest.read_bytes()).hexdigest()
    if sha and got != sha:
        raise ValueError(f"SHA-256 mismatch for {dest.name}: expected {sha}, got {got}")
    return got


def main() -> None:  # pragma: no cover - slow path
    """Download the three .conllu files (verify SHA), parse, write prepared parquet."""
    import pyarrow.parquet as pq

    cfg = load_config("data")
    ensure_dirs(CONLLU, PREPARED)
    for split, fname in cfg["files"].items():
        url = f"{cfg['base_url']}/{cfg['ud_tag']}/{fname}"
        dest = CONLLU / fname
        got = _fetch(url, dest, cfg["sha256"].get(split, ""))
        print(f"{split}: sha256={got}")
        sents = parse_conllu(dest.read_text(encoding="utf-8"))
        pq.write_table(rows_to_table(sents), PREPARED / f"{split}.parquet")
        print(f"  parsed {len(sents)} sentences")


if __name__ == "__main__":  # pragma: no cover
    main()
