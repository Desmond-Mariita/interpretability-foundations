"""Verify (and if needed extract) the gated Hateful Memes data. Never downloads/commits."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from _paths import DATA_PATH


def load_split(path: Path) -> list[dict]:
    """Parse a Hateful Memes ``.jsonl`` split into a list of records."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _verify(root: Path) -> bool:
    need = [root / "img", root / "train.jsonl", root / "dev.jsonl", root / "test.jsonl"]
    return all(p.exists() for p in need)


def main() -> None:
    """Verify the dataset at ``--data-path``; extract from a local archive if present."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", type=Path, default=DATA_PATH)
    ap.add_argument("--archive", type=Path, default=Path.home() / "Downloads/archive.zip")
    args = ap.parse_args()
    root = args.data_path
    if not _verify(root) and args.archive.exists():
        print(f"Extracting {args.archive} -> {root}")
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.archive) as z:
            z.extractall(root)
        if (root / "data" / "dev.jsonl").exists():
            root = root / "data"
    if not _verify(root):
        raise SystemExit(
            "FATAL: Hateful Memes not found.\n"
            "This dataset is gated by Meta's licence (accept terms; do not redistribute).\n"
            f"Place the licensed archive at {args.archive} or the extracted files at {root}\n"
            "(expected: img/, train.jsonl, dev.jsonl, test.jsonl)."
        )
    counts = {s: len(load_split(root / f"{s}.jsonl")) for s in ("train", "dev", "test")}
    print(f"OK: {root}  counts={counts}")


if __name__ == "__main__":
    main()
