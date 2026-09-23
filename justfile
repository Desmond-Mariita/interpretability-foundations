set shell := ["bash", "-uc"]

# Default: list available recipes.
default:
    @just --list

# Reproduce the committed environment and install local pre-commit hooks.
setup:
    uv sync --frozen --all-extras
    uv run pre-commit install

# Run the repository hygiene suite exactly as CI does.
precommit:
    uv run pre-commit run --all-files

# Enforce unit/smoke/slow classification for every collected test.
markers:
    uv run python scripts/check_test_markers.py

# Lint + format check.
lint:
    uv run ruff check .
    uv run ruff format --check .

# Auto-fix lint + format.
fix:
    uv run ruff check --fix .
    uv run ruff format .

# Type check the shared library. Future production targets are added incrementally.
typecheck:
    uv run mypy src/awake/

# Run unit + smoke tests (the PR CI budget).
test:
    uv run pytest -m "unit or smoke"

# Run every non-slow test. Used by the scheduled quality workflow.
test-nonslow:
    uv run pytest -m "not slow"

# Everything required in PR CI.
ci: precommit markers test

# Run the HF Space app locally.
space:
    cd apps/hatefulmemes-space && uv run python app.py

# Per-project recipes are defined in projects/NN-*/justfile and invoked from there.
