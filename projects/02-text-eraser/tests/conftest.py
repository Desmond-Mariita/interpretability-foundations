"""Make the project's scripts/ importable in tests.

Sibling projects (e.g. ``projects/01-tabular-mimic``) also expose bare-named
script modules such as ``_paths`` and ``10_train``. In a combined pytest
session those names collide in ``sys.modules``. The autouse fixture below
isolates this project's scripts on every test: it puts this ``scripts/`` dir
first on ``sys.path`` and evicts this project's owned module names from the
import cache (before and after each test) so they always re-import from here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
_OWNED = {p.stem for p in SCRIPTS_DIR.glob("*.py") if not p.stem.startswith("__")}

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture(autouse=True)
def _isolate_project_scripts():
    """Give this project's scripts import priority and a clean module cache."""
    path = str(SCRIPTS_DIR)
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
    for name in _OWNED:
        sys.modules.pop(name, None)
    yield
    for name in _OWNED:
        sys.modules.pop(name, None)
