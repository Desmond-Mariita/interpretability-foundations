# New project checklist

- [ ] Research question and estimand are explicit.
- [ ] Dataset/model identity and revisions are recorded.
- [ ] Train/dev/test or pilot/confirmatory boundaries are frozen where relevant.
- [ ] Configs and seeds are committed.
- [ ] Authoritative metrics artifact is machine-readable.
- [ ] Publication-critical local outputs have provenance/hashes.
- [ ] Controls are defined before interpreting the headline result.
- [ ] Claim boundary and limitations are documented.
- [ ] Every test is marked unit/smoke/slow.
- [ ] Deterministic logic has focused unit tests.
- [ ] Pipeline wiring has a smoke test.
- [ ] Slow/model/GPU validation is documented.
- [ ] Public production APIs satisfy the docstring policy.
- [ ] Type-check coverage is stated for production modules.
- [ ] `uv run pre-commit run --all-files` passes.
- [ ] `uv run python scripts/check_test_markers.py` passes.
- [ ] `uv run pytest -m "unit or smoke"` passes.
- [ ] No credentials, caches, raw restricted data, or large transient artifacts are tracked.
