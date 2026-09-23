# Research project template

Copy this structure when adding a new portfolio project.

```text
NN-project-name/
├── README.md
├── REPORT.md
├── configs/
├── scripts/
├── tests/
├── assets/
└── outputs/        # runtime artifacts; normally gitignored
```

## README contract

A project README must include:

1. **Research question** — the exact question being tested.
2. **Estimand / target quantity** — what the headline number means.
3. **Data and model identity** — source, split, revisions, inclusion/exclusion rules.
4. **Method** — enough detail to trace data → model → intervention/metric.
5. **Controls** — what could falsify the preferred interpretation.
6. **Results** — machine-readable artifact is authoritative.
7. **Uncertainty** — bootstrap/test unit and any multiplicity caveats.
8. **Claim boundary** — what the result does and does not establish.
9. **Limitations** — known scope and validity limits.
10. **Reproduction** — commands, environment, seeds, and hardware-sensitive steps.
11. **Tests** — unit/smoke/slow commands.
12. **Provenance** — hashes/manifests/revisions for publication-critical artifacts.

## Quality contract

- New tests must be explicitly marked `unit`, `smoke`, or `slow`.
- Public production APIs use Google-style docstrings.
- Reusable deterministic logic should live in `src/awake/` where justified.
- New deterministic production modules should be added to the typing plan.
- Do not commit large/raw datasets, credentials, model caches, or transient outputs.
- Tests protect code/estimand contracts; they do not certify scientific truth.
