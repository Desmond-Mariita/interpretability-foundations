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
        if (
            isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Attribute)
            and isinstance(child.value.value, ast.Name)
            and child.value.value.id == "pytest"
            and child.value.attr == "mark"
            and child.attr in APPROVED
        ):
            names.add(child.attr)
    return names


def _module_markers(tree: ast.Module) -> set[str]:
    markers: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in targets):
                markers |= _marker_names(node)
    return markers


def _test_files() -> list[Path]:
    files: set[Path] = set()
    for root in TEST_ROOTS:
        if root.exists():
            files.update(root.rglob("test_*.py"))
    return sorted(
        p for p in files if "legacy" not in p.parts and "notebooks" not in p.parts
    )


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module_marks = _module_markers(tree)
    violations: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(
            "test_"
        ):
            marks = module_marks | set().union(
                *(_marker_names(d) for d in node.decorator_list)
            )
            if not marks:
                violations.append(f"{path.relative_to(ROOT)}::{node.name}")
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            class_marks = module_marks | set().union(
                *(_marker_names(d) for d in node.decorator_list)
            )
            for method in node.body:
                if isinstance(
                    method, ast.FunctionDef | ast.AsyncFunctionDef
                ) and method.name.startswith("test_"):
                    marks = class_marks | set().union(
                        *(_marker_names(d) for d in method.decorator_list)
                    )
                    if not marks:
                        violations.append(
                            f"{path.relative_to(ROOT)}::{node.name}::{method.name}"
                        )
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
