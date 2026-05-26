"""Make this project's scripts/ importable in tests; isolate from sibling projects."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
_OWNED = {p.stem for p in SCRIPTS_DIR.glob("*.py") if not p.stem.startswith("__")}

# Evict stale sibling modules at conftest import time so module-level importlib
# calls in test files pick up THIS project's versions in a combined session.
path = str(SCRIPTS_DIR)
while path in sys.path:
    sys.path.remove(path)
sys.path.insert(0, path)
for _name in _OWNED:
    sys.modules.pop(_name, None)


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
