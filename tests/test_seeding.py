from __future__ import annotations

import random

import numpy as np
import pytest

from awake.utils import seed_everything


@pytest.mark.unit
def test_seed_everything_is_deterministic() -> None:
    seed_everything(1337)
    py_first = random.random()
    np_first = np.random.rand(4).tolist()

    seed_everything(1337)
    assert random.random() == py_first
    assert np.random.rand(4).tolist() == np_first


@pytest.mark.unit
def test_seed_everything_returns_seed() -> None:
    assert seed_everything(7) == 7
