"""Download + verify + extract the ERASER Movies tarball. Commits no data."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

from _paths import DATA_PATH, load_config


def verify_sha256(path: Path, expected: str) -> bool:
    """Return True iff the file's SHA-256 hex digest equals ``expected``."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def main() -> None:
    """CLI: download the tarball to ``--data-path`` and extract it."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", type=Path, default=DATA_PATH)
    args = ap.parse_args()
    cfg = load_config("data")
    args.data_path.mkdir(parents=True, exist_ok=True)
    tar_path = args.data_path / "movies.tar.gz"

    if not tar_path.exists():
        print(f"Downloading {cfg['url']} -> {tar_path}")
        urllib.request.urlretrieve(cfg["url"], tar_path)

    expected = cfg.get("sha256", "")
    if expected and expected != "PLACEHOLDER_FILL_FROM_DOWNLOAD":
        if not verify_sha256(tar_path, expected):
            raise SystemExit(f"FATAL: checksum mismatch on {tar_path}")
    else:
        actual = hashlib.sha256(tar_path.read_bytes()).hexdigest()
        print(f"NOTE: set configs/data.yaml sha256 to: {actual}")

    with tarfile.open(tar_path) as t:
        t.extractall(args.data_path, filter="data")
    print(f"Extracted to {args.data_path}")


if __name__ == "__main__":
    main()
