"""Filesystem paths and config loading for project 04."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(os.environ.get("P4_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
OUTPUTS = PROJECT_ROOT / "outputs"
ASSETS = PROJECT_ROOT / "assets"
CONFIGS = PROJECT_ROOT / "configs"
IMAGES = OUTPUTS / "images"
PREPARED = OUTPUTS / "prepared"
GEN = OUTPUTS / "gen"


def load_config(name: str) -> dict:
    """Load ``configs/<name>.yaml`` as a dict."""
    with open(CONFIGS / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def ensure_dirs(*paths: Path) -> None:
    """Create each path (and parents)."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
