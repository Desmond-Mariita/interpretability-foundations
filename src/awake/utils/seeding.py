"""Deterministic seeding for Python and NumPy RNGs."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> int:
    """Seed Python's ``random``, NumPy's global RNG, and ``PYTHONHASHSEED``.

    Torch is intentionally not seeded here: importing it would make
    ``awake.utils`` a heavyweight dependency, and projects that use torch
    seed it themselves immediately after calling this function.

    Args:
        seed: Integer seed shared across all RNGs.

    Returns:
        The seed value, returned unchanged so callers can log or chain it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed
