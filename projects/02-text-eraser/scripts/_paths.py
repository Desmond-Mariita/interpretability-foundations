"""Filesystem paths and config loading for project 02."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
PREPARED = OUTPUTS / "prepared"
MODEL_DIR = OUTPUTS / "model"
CACHE_DIR = OUTPUTS / "attributions"
ASSETS = PROJECT_ROOT / "assets"
CONFIGS = PROJECT_ROOT / "configs"

DATA_PATH = Path(os.environ.get("DATA_PATH", Path.home() / ".cache/eraser/movies"))


def load_config(name: str) -> dict:
    """Load ``configs/<name>.yaml`` as a dict."""
    with open(CONFIGS / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    """Create all output directories."""
    for d in (OUTPUTS, PREPARED, MODEL_DIR, CACHE_DIR, ASSETS):
        d.mkdir(parents=True, exist_ok=True)
