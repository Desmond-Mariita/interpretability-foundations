# Contributing

This repository is a research portfolio. Changes must preserve both software quality and scientific traceability.

## Environment

- Python: **3.11.x** only.
- Environment/dependency manager: **uv**.
- Reproduce the committed environment with `uv sync --frozen --all-extras`.
- `uv.lock` is authoritative. CI must not silently resolve a different dependency set.

## Repository layout

- `src/awake/` — reusable scientific/evaluation utilities shared across projects.
- `projects/` — bounded research experiments.
- `apps/` — interactive/public applications built from frozen project logic.
- `legacy/` — preserved historical material; do not modernize it opportunistically.
- `docs/` — ADRs, standards, templates, and research-engineering documentation.

Reusable deterministic logic should move to `src/awake/` when it is genuinely shared. Project-specific orchestration remains inside the relevant project.

## Formatting and linting

Run `uv run pre-commit run --all-files`.

This is the repository hygiene source of truth and includes whitespace/EOF checks, YAML/TOML validation, merge-conflict and large-file checks, Ruff lint/formatting, mypy for the current shared-library target, and Gitleaks.

Ruff's configured line length (100) is a formatter target, **not** a hard maximum. `E501` intentionally remains disabled; Ruff formatter is authoritative.

## Docstrings

Google-style docstrings are required for public production functions, classes, and methods.

- Public production API: required.
- Private helpers: recommended when behaviour/contracts are non-obvious.
- Nested closures: normally unnecessary.
- Tests: prefer descriptive names and focused comments over mandatory docstrings.
- Notebooks and `legacy/`: outside the production docstring contract.

Ruff's `D` rules enforce the production policy; do not add a second overlapping docstring linter without a demonstrated gap.

## Typing

The shared library (`src/awake/`) is the current required mypy target.

Typing expands incrementally: shared reusable logic, deterministic project modules, production application code, then orchestration code where dynamic third-party APIs make checking useful rather than noisy.

Every new project must state which production modules are type-checked. A project may defer a dynamic orchestration file, but the reason must be documented in its README or ADR.

## Tests

Every test must carry one execution-class marker, directly or through its module/class:

- `unit` — fast pure logic, no external resources;
- `smoke` — tiny wiring/end-to-end checks, no heavyweight model/data dependency;
- `slow` — model-, GPU-, network-, or data-heavy integration work.

CI enforces this with `scripts/check_test_markers.py`.

PR CI runs `uv run pytest -m "unit or smoke"`. A scheduled workflow runs `uv run pytest -m "not slow"`.

Slow/GPU tests must remain reproducible and documented. They are run manually on capable hardware until a suitable runner is available; do not force GPU/model code into hosted CPU CI merely to increase coverage.

## Coverage

Coverage is interpreted by code role, not as one repository-wide vanity number.

- `src/awake/`: hard coverage gate (currently 90%).
- Deterministic project logic: must have focused project tests.
- GPU/model/data orchestration: validate with smoke/integration assertions and scientific controls; line coverage is not a substitute for pipeline validity.

Future projects should report shared-library coverage separately from their deterministic project logic.

## Scientific artifact contract

A test passing is evidence for a **code contract**, not proof that a scientific claim is true.

Every finished project must identify:

1. research question and estimand;
2. data/model identity and revision where applicable;
3. frozen configuration;
4. authoritative machine-readable metrics artifact;
5. run/provenance record sufficient to trace published numbers;
6. controls and uncertainty procedure;
7. claim boundary and known limitations;
8. tests that protect estimand/aggregation semantics.

Do not quietly replace missing provenance with reconstructed assumptions.

## Adding a project

Start from `docs/project-template/`.

A new project should contain, as applicable:

```text
NN-project-name/
├── README.md
├── REPORT.md
├── configs/
├── scripts/
├── tests/
├── assets/
└── outputs/        # gitignored runtime artifacts
```

The README must cover data/model provenance, reproduction, test commands, authoritative artifacts, claim boundaries, and limitations.

Before opening a PR, run:

```bash
just ci
git diff --check
```

Model/GPU experiments should also run their documented project-specific slow validation before results are published.

## Security and dependencies

- Never commit credentials or API keys.
- Gitleaks runs through pre-commit in CI.
- CI uses the committed lockfile with `--frozen`.
- Dependency auditing runs on the scheduled workflow initially as a visible, non-blocking signal; ML-stack advisories must be triaged rather than ignored or blindly waived.
