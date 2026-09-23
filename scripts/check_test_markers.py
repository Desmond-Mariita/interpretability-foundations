"""Enforce explicit execution-class markers on repository tests.

Every test must be classified as ``unit``, ``smoke``, or ``slow`` either at module,
class, or function level. This prevents an unmarked test from being silently omitted by
the fast PR-CI selector.
"""

from __future__ import annotations

import ast
from pathlib import Path

APPROVED = {"unit", "smoke", "slow"}
ROOT = Path(__file__).resolve().parents[1]
TEST_ROOTS = (ROOT / "tests", ROOT / "projects", ROOT / "apps")


def _marker_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        is_marker = (
            isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Attribute)
            and isinstance(child.value.value, ast.Name)
        )
        if not is_marker:
            continue
        if child.value.value.id != "pytest" or child.value.attr != "mark":
            continue
        if child.attr in APPROVED:
            names.add(child.attr)
    return names


def _module_markers(tree: ast.Module) -> set[str]:
    markers: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign | ast.AnnAssign):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        is_pytestmark = any(
            isinstance(target, ast.Name) and target.id == "pytestmark" for target in targets
        )
        if is_pytestmark:
            markers |= _marker_names(node)
    return markers


def _test_files() -> list[Path]:
    files: set[Path] = set()
    for root in TEST_ROOTS:
        if root.exists():
            files.update(root.rglob("test_*.py"))
    kept = [path for path in files if "legacy" not in path.parts and "notebooks" not in path.parts]
    return sorted(kept)


def _is_test_function(node: ast.AST) -> bool:
    is_function = isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    return is_function and node.name.startswith("test_")


def _decorator_markers(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> set[str]:
    markers: set[str] = set()
    for decorator in node.decorator_list:
        markers |= _marker_names(decorator)
    return markers


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module_marks = _module_markers(tree)
    violations: list[str] = []

    for node in tree.body:
        if _is_test_function(node):
            marks = module_marks | _decorator_markers(node)
            if not marks:
                violations.append(f"{path.relative_to(ROOT)}::{node.name}")
            continue

        if not isinstance(node, ast.ClassDef) or not node.name.startswith("Test"):
            continue
        class_marks = module_marks | _decorator_markers(node)
        for method in node.body:
            if not _is_test_function(method):
                continue
            marks = class_marks | _decorator_markers(method)
            if not marks:
                violations.append(f"{path.relative_to(ROOT)}::{node.name}::{method.name}")
    return violations


def main() -> int:
    """Return nonzero when any test lacks an approved execution marker."""
    violations: list[str] = []
    for path in _test_files():
        violations.extend(_violations(path))

    if violations:
        print("Tests missing an execution marker (unit/smoke/slow):")
        for item in violations:
            print(f"  - {item}")
        return 1

    print("All repository tests are explicitly classified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
