from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy, and PYTHONHASHSEED. Returns the seed for logging.

    Torch is seeded by the caller when present; we don't import it here to keep
    `awake.utils` lightweight (the shared library is a CPU-tier dependency).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed
