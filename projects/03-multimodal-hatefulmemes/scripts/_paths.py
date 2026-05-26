"""Filesystem paths and config loading for project 03."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
ASSETS = PROJECT_ROOT / "assets"
CONFIGS = PROJECT_ROOT / "configs"
DATA_PATH = Path(os.environ.get("DATA_PATH", Path.home() / ".cache/hateful_memes"))


def load_config(name: str) -> dict:
    """Load ``configs/<name>.yaml`` as a dict."""
    with open(CONFIGS / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def embeddings_dir(variant_cfg: dict) -> Path:
    """Return the embedding cache dir for a CLIP variant config."""
    return OUTPUTS / variant_cfg["embedding_subdir"]


def models_dir(variant_cfg: dict) -> Path:
    """Return the models dir for a CLIP variant config."""
    return OUTPUTS / "models" / variant_cfg["embedding_subdir"].split("/")[-1]


def ensure_dirs(*paths: Path) -> None:
    """Create each path (and parents)."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
