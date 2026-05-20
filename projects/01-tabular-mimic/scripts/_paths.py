"""Shared filesystem helpers for the P1 pipeline scripts."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def ensure_outputs_dir() -> Path:
    """Create ``outputs/`` if it doesn't exist and return its path.

    Returns:
        The absolute path to the project's outputs directory.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR
