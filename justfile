set shell := ["bash", "-uc"]

# Default: list available recipes.
default:
    @just --list

# Create venv + install dev deps + pre-commit hooks.
setup:
    uv sync --all-extras
    uv run pre-commit install

# Lint + format check.
lint:
    uv run ruff check .
    uv run ruff format --check .

# Auto-fix lint + format.
fix:
    uv run ruff check --fix .
    uv run ruff format .

# Type check the shared library.
typecheck:
    uv run mypy src/awake/

# Run unit + smoke tests (the CI budget).
test:
    uv run pytest -m "unit or smoke"

# Everything CI runs.
ci: lint typecheck test

# Run the HF Space app locally.
space:
    cd apps/hatefulmemes-space && uv run python app.py

# Per-project recipes are defined in projects/NN-*/justfile and invoked from there.
