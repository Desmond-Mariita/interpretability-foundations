# Layerwise Probing of Pythia-160M (Project 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `projects/05-mechanistic-pythia`: per-layer linear probes on Pythia-160M's residual stream for three binary properties (is_noun, is_verb, noun_number) from UD English-EWT, each paired with a Hewitt & Liang control task; headline = selectivity (balanced-acc probe − balanced-acc control) by depth.

**Architecture:** A pure, I/O-free, 90%-covered core in `src/awake/eval/probing.py` (control-label assignment, balanced accuracy, selectivity, type overlap, emergence point) plus cluster-bootstrap additions to `src/awake/eval/bootstrap.py`, driven entirely by unit tests. Heavy work is config-driven numbered scripts in `projects/05-mechanistic-pythia/` (`00_data` fetch/parse UD → `10_extract` hook activations → `20_probe` standardise+LR+control → `30_eval` metrics+figures); every script takes an **injectable callable** (probe-fitter / model) so smoke tests drive the whole flow on tiny synthetic arrays with no model download. Real model code is lazy-imported inside functions and `slow`-marked. No HF Space.

**Tech Stack:** Python 3.11, uv, pytest (unit/smoke/slow), transformers (`GPTNeoXForCausalLM`, forward hooks), torch, scikit-learn (LogisticRegression, StandardScaler), numpy, pandas/pyarrow, matplotlib, stdlib urllib.

**Reference spec:** `docs/superpowers/specs/2026-05-27-mechanistic-pythia-design.md` (v2.1) — read it first. **Governance:** UD English-EWT is **CC BY-SA 4.0** (attribute in REPORT + ADR); never commit the `.conllu`/parquet/`outputs/`; the notebook worked example uses a **self-authored** sentence, never UD text.

---

## Conventions for every task

- Run from the **repo root**. `uv` is at `$HOME/.local/bin/uv` (NOT on PATH): prefix every command with `export PATH="$HOME/.local/bin:$PATH"`. `just`/`gh` not installed — run `uv run …`.
- **Unit tests for `src/awake/`** live in repo-root `tests/`; **project smoke tests** in `projects/05-mechanistic-pythia/tests/`.
- Markers: `@pytest.mark.unit` (fast pure), `@pytest.mark.smoke` (tiny stub, CPU, no model/no dataset), `@pytest.mark.slow` (real Pythia/UD, excluded from CI).
- Single-file run: `uv run pytest <path> -v --no-cov`. Coverage gate (≥90% on `awake`): `uv run pytest tests -m unit`. CI: `uv run pytest tests projects apps -m "unit or smoke"`.
- ruff: Google docstrings on `src/awake/` **and project test functions** (project `tests/**` are NOT in the `tests/**` D-ignore — give each test fn a one-line docstring, as P3/P4 do), line ≤100, double quotes, **ASCII only** (no Unicode `×`/`–`/Δ; write `x`/`-`/`Delta`). Commit after each task.
- mypy clean on `src/awake/`. Reuse `seed_everything` from `awake.utils.seeding` if needed.
- Per-project conftest isolates `scripts/` (sys.modules eviction) — copied from P4 in Task 1.

## File structure (created by this plan)

```
src/awake/eval/probing.py        # NEW pure core
src/awake/eval/bootstrap.py      # + cluster_bootstrap_ci, paired_cluster_bootstrap (MODIFY)
src/awake/eval/__init__.py       # + re-export new names (NOT a second `accuracy`) (MODIFY)
tests/test_eval_probing.py
tests/test_eval_cluster_bootstrap.py
projects/05-mechanistic-pythia/
  configs/{data,probe}.yaml
  scripts/__init__.py _paths.py _models.py _stub.py 00_data.py 10_extract.py 20_probe.py 30_eval.py
  tests/{conftest.py, __init__.py, test_data_smoke.py, test_probe_smoke.py}
  notebooks/01-layerwise-probing.py
  assets/{probe_is_noun,probe_is_verb,probe_noun_number,hero}.png   # from real run
  REPORT.md   (README.md exists — updated)
docs/decisions/005-probing-pythia-and-control-tasks.md
.gitignore                       # + projects/05-mechanistic-pythia/outputs/ (MODIFY)
CHANGELOG.md                     # + [Unreleased] P5 entry (MODIFY)
```

---

## Task 1: Scaffold, gitignore, conftest, configs

**Files:** Create `projects/05-mechanistic-pythia/scripts/__init__.py` (empty), `scripts/_paths.py`, `tests/__init__.py` (empty), `tests/conftest.py`, `configs/data.yaml`, `configs/probe.yaml`. Modify `.gitignore`.

- [ ] **Step 1: gitignore** — append after the P4 line:

```
projects/05-mechanistic-pythia/outputs/
```

- [ ] **Step 2: `scripts/_paths.py`**

```python
"""Filesystem paths and config loading for project 05."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(os.environ.get("P5_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
OUTPUTS = PROJECT_ROOT / "outputs"
ASSETS = PROJECT_ROOT / "assets"
CONFIGS = PROJECT_ROOT / "configs"
PREPARED = OUTPUTS / "prepared"
ACTS = OUTPUTS / "acts"
CONLLU = OUTPUTS / "conllu"


def load_config(name: str) -> dict:
    """Load ``configs/<name>.yaml`` as a dict."""
    with open(CONFIGS / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def ensure_dirs(*paths: Path) -> None:
    """Create each path (and parents)."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 3: `tests/conftest.py`** — copy P4's isolation fixture verbatim, with P5's owned module set:

```python
"""Make this project's scripts/ importable in tests; isolate from sibling projects."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
_OWNED = {p.stem for p in SCRIPTS_DIR.glob("*.py") if not p.stem.startswith("__")}

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
```

- [ ] **Step 4: `configs/data.yaml`**

```yaml
# UD English-EWT (CC BY-SA 4.0). Fetched as .conllu from a pinned release tag.
ud_repo: "UniversalDependencies/UD_English-EWT"
ud_tag: "r2.14"
base_url: "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT"
files:
  train: "en_ewt-ud-train.conllu"
  dev: "en_ewt-ud-dev.conllu"
  test: "en_ewt-ud-test.conllu"
# SHA-256 of each file at ud_tag; fill in at first run (00_data prints them) then pin here.
sha256:
  train: ""
  dev: ""
  test: ""
```

- [ ] **Step 5: `configs/probe.yaml`**

```yaml
model_id: "EleutherAI/pythia-160m"
model_revision: "main"          # pin to a commit SHA after first run
n_blocks: 12
properties: ["is_noun", "is_verb", "noun_number"]
alignment: "last"               # last overlapping subword
probe:
  C_grid: [0.01, 0.1, 1.0]      # chosen once per property on dev
  max_iter: 2000
  random_state: 0
control:
  seeds: [0, 1, 2, 3, 4]        # K=5
train_token_cap: 60000          # class-stratified seeded cap for probe fits
train_cap_seed: 0
bootstrap:
  n_resamples: 2000
  seed: 0
noun_number_min_train: 3000     # below this -> flag underpowered
```

- [ ] **Step 6: verify + commit**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest projects/05-mechanistic-pythia -q --no-cov`
Expected: exit 0 or 5 (no tests). Then:

```bash
git add .gitignore projects/05-mechanistic-pythia/
git commit -m "feat(p5): scaffold project, gitignore outputs, configs, conftest"
```

---

## Task 2: Pure core — `balanced_accuracy`, `base_rate`, `majority_class`, `selectivity`

**Files:** Create `src/awake/eval/probing.py`; Test `tests/test_eval_probing.py`.

- [ ] **Step 1: failing tests**

```python
"""Unit tests for the pure probing core."""

import pytest

from awake.eval.probing import balanced_accuracy, base_rate, majority_class, selectivity


@pytest.mark.unit
def test_balanced_accuracy_is_mean_per_class_recall():
    # class 0: 3 gold, 2 correct -> recall 2/3; class 1: 1 gold, 1 correct -> recall 1
    y_true = [0, 0, 0, 1]
    y_pred = [0, 0, 1, 1]
    assert balanced_accuracy(y_true, y_pred) == pytest.approx((2 / 3 + 1.0) / 2)


@pytest.mark.unit
def test_balanced_accuracy_chance_is_half_under_imbalance():
    y_true = [0] * 9 + [1]
    y_pred = [0] * 10  # predict all-negative
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.5)  # recall0=1, recall1=0


@pytest.mark.unit
def test_balanced_accuracy_empty_and_mismatch():
    assert balanced_accuracy([], []) == 0.0
    with pytest.raises(ValueError):
        balanced_accuracy([0, 1], [0])


@pytest.mark.unit
def test_base_rate_majority_class_selectivity():
    assert base_rate([0, 0, 1, 1]) == 0.5
    assert base_rate([]) == 0.0
    assert majority_class([0, 0, 0, 1]) == 0
    assert majority_class([1, 1, 0]) == 1
    assert selectivity(0.9, 0.6) == pytest.approx(0.3)
```

- [ ] **Step 2: run → fail** (`ModuleNotFoundError`). `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests/test_eval_probing.py -v --no-cov`

- [ ] **Step 3: implement**

```python
"""Pure, I/O-free core for the layerwise-probing project (P5).

No models, no sklearn, no file I/O: every function takes Python lists/dicts and returns a number
or dict, so the methodology is locked down by fast unit tests. See
docs/superpowers/specs/2026-05-27-mechanistic-pythia-design.md (v2.1). All metric callables use the
sklearn argument order (y_true, y_pred). P5 deliberately does NOT define `accuracy` here — it reuses
the existing `awake.eval.accuracy`.
"""

from __future__ import annotations

import random


def balanced_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Mean of per-class recall (chance = 0.5 regardless of prevalence).

    Args:
        y_true: Gold binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Mean over classes present in ``y_true`` of (correct / gold) for that class;
        ``0.0`` for empty input.

    Raises:
        ValueError: If the inputs differ in length.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not y_true:
        return 0.0
    recalls = []
    for cls in {*y_true}:
        gold = [p for t, p in zip(y_true, y_pred, strict=True) if t == cls]
        recalls.append(sum(p == cls for p in gold) / len(gold))
    return sum(recalls) / len(recalls)


def base_rate(labels: list[int]) -> float:
    """Share of positive labels (``0.0`` for empty)."""
    return sum(labels) / len(labels) if labels else 0.0


def majority_class(train_labels: list[int]) -> int:
    """Majority class (1 if more than half positive, else 0)."""
    return 1 if base_rate(train_labels) > 0.5 else 0


def selectivity(probe_metric: float, control_metric: float) -> float:
    """probe_metric - control_metric."""
    return probe_metric - control_metric
```

- [ ] **Step 4: run → pass.** **Step 5: commit** `feat(eval): add balanced_accuracy/base_rate/majority_class/selectivity for P5`.

---

## Task 3: Pure core — `assign_control_labels` + `control_vector`

**Files:** Modify `src/awake/eval/probing.py`; Test `tests/test_eval_probing.py`.

- [ ] **Step 1: failing tests**

```python
from awake.eval.probing import assign_control_labels, control_vector


@pytest.mark.unit
def test_assign_control_labels_token_rate_matched_and_deterministic():
    # "the" is very frequent; matching in TOKEN space must account for that.
    counts = {"the": 100, "dog": 10, "cat": 10, "run": 10, "sit": 10}
    all_types = set(counts)
    m1 = assign_control_labels(all_types, counts, base_rate=0.3, seed=0)
    m2 = assign_control_labels(all_types, counts, base_rate=0.3, seed=0)
    assert m1 == m2  # deterministic
    assert set(m1) == all_types and set(m1.values()) <= {0, 1}
    # realised TOKEN-level positive share within tolerance of 0.3
    tot = sum(counts.values())
    pos = sum(counts[t] for t, lbl in m1.items() if lbl == 1)
    assert abs(pos / tot - 0.3) < 0.15
    assert assign_control_labels(all_types, counts, 0.3, seed=1) != m1  # seed changes map


@pytest.mark.unit
def test_control_vector_maps_and_raises_on_unseen():
    m = {"a": 0, "b": 1}
    assert control_vector(["a", "b", "a"], m) == [0, 1, 0]
    with pytest.raises(KeyError):
        control_vector(["a", "z"], m)
```

- [ ] **Step 2: run → fail.**

- [ ] **Step 3: implement** (append to `probing.py`)

```python
def assign_control_labels(
    all_types: set[str],
    train_counts: dict[str, int],
    base_rate: float,
    seed: int,
) -> dict[str, int]:
    """Hewitt-Liang control: a deterministic random binary label for every type.

    Labels every type in ``all_types`` (the train+dev+test union, so no scored token is ever
    unseen). The positive share is matched in TOKEN space: types are visited in a seeded-random
    order and assigned 1 until their cumulative TRAIN token mass (from ``train_counts``, 0 for
    types absent from train) reaches ``base_rate`` of the total train mass; the rest get 0.

    Args:
        all_types: Every word type that will be scored (train union dev union test).
        train_counts: Train-token frequency per type (missing -> 0).
        base_rate: Target positive share in token space.
        seed: RNG seed (determinism).

    Returns:
        ``{type: 0|1}`` for every type in ``all_types``.
    """
    types = sorted(all_types)
    random.Random(seed).shuffle(types)
    total = sum(train_counts.values())
    target = base_rate * total
    labels: dict[str, int] = {}
    acc = 0.0
    for t in types:
        if acc < target:
            labels[t] = 1
            acc += train_counts.get(t, 0)
        else:
            labels[t] = 0
    return labels


def control_vector(words: list[str], type_to_label: dict[str, int]) -> list[int]:
    """Map each token's exact-surface-form type to its control label.

    Args:
        words: Per-token surface forms.
        type_to_label: Map built over the train+dev+test union.

    Returns:
        Per-token control labels.

    Raises:
        KeyError: If a word's type is absent from ``type_to_label`` (a programming error,
            since the map must be built over the full union).
    """
    return [type_to_label[w] for w in words]
```

- [ ] **Step 4: run → pass.** **Step 5: commit** `feat(eval): add token-rate-matched control-label assignment for P5`.

---

## Task 4: Pure core — `type_overlap` + `emergence_point`

**Files:** Modify `src/awake/eval/probing.py`; Test `tests/test_eval_probing.py`.

- [ ] **Step 1: failing tests**

```python
from awake.eval.probing import emergence_point, type_overlap


@pytest.mark.unit
def test_type_overlap_seen_and_oov_token_rates():
    train = ["a", "a", "b"]
    test = ["a", "a", "c", "c"]  # "a" seen (2 tokens), "c" oov (2 tokens)
    out = type_overlap(train, test)
    assert out["seen_type_token_rate"] == pytest.approx(0.5)
    assert out["oov_type_token_rate"] == pytest.approx(0.5)


@pytest.mark.unit
def test_emergence_point_peak_and_earliest_within_ci():
    sel = {"embedding": 0.0, "block_0": 0.1, "block_1": 0.45, "block_2": 0.5, "ln_f": 0.9}
    ci = {
        "embedding": (-0.05, 0.05),
        "block_0": (0.0, 0.2),
        "block_1": (0.38, 0.52),   # overlaps peak's [0.42, 0.58]
        "block_2": (0.42, 0.58),   # peak
        "ln_f": (0.8, 1.0),        # excluded from emergence (not a depth point)
    }
    out = emergence_point(sel, ci)
    assert out["peak"] == "block_2"               # ln_f ignored despite higher selectivity
    assert out["earliest_within_peak_ci"] == "block_1"
```

- [ ] **Step 2: run → fail.**

- [ ] **Step 3: implement** (append)

```python
_DEPTH_PREFIXES = ("embedding", "block_")


def type_overlap(train_words: list[str], test_words: list[str]) -> dict:
    """Token-weighted train/test type overlap.

    Returns:
        ``{"seen_type_token_rate": frac of test tokens whose type appears in train,
        "oov_type_token_rate": 1 - that}``; both ``0.0`` for empty test.
    """
    if not test_words:
        return {"seen_type_token_rate": 0.0, "oov_type_token_rate": 0.0}
    train_types = set(train_words)
    seen = sum(w in train_types for w in test_words) / len(test_words)
    return {"seen_type_token_rate": seen, "oov_type_token_rate": 1.0 - seen}


def _is_depth_point(name: str) -> bool:
    return name == "embedding" or name.startswith("block_")


def emergence_point(
    sel_by_point: dict[str, float],
    sel_ci_by_point: dict[str, tuple[float, float]],
) -> dict:
    """Peak and earliest-emergence depth point (the 'ln_f' extra point is excluded).

    Args:
        sel_by_point: Selectivity per point (may include non-depth points like 'ln_f').
        sel_ci_by_point: ``(lo, hi)`` CI per point.

    Returns:
        ``{"peak": <point with max selectivity among depth points>,
        "earliest_within_peak_ci": <earliest depth point whose CI overlaps the peak's CI>}``.
        Overlap: ``lo_j <= hi_peak and lo_peak <= hi_j``. Lowest-index tie-break (depth points
        are ordered embedding, block_0, block_1, …).
    """
    def order(name: str) -> int:
        return 0 if name == "embedding" else 1 + int(name.split("_")[1])

    depth = sorted((p for p in sel_by_point if _is_depth_point(p)), key=order)
    peak = max(depth, key=lambda p: (sel_by_point[p], -order(p)))
    lo_pk, hi_pk = sel_ci_by_point[peak]
    for p in depth:
        lo, hi = sel_ci_by_point[p]
        if lo <= hi_pk and lo_pk <= hi:
            return {"peak": peak, "earliest_within_peak_ci": p}
    return {"peak": peak, "earliest_within_peak_ci": peak}
```

- [ ] **Step 4: run → pass.** **Step 5: commit** `feat(eval): add type_overlap and emergence_point for P5`.

---

## Task 5: `bootstrap.py` — cluster + paired cluster bootstrap

**Files:** Modify `src/awake/eval/bootstrap.py`; Test `tests/test_eval_cluster_bootstrap.py`.

- [ ] **Step 1: failing tests**

```python
"""Unit tests for cluster (sentence-grouped) bootstrap helpers."""

import numpy as np
import pytest

from awake.eval.bootstrap import cluster_bootstrap_ci, paired_cluster_bootstrap


def _acc(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


@pytest.mark.unit
def test_cluster_bootstrap_ci_is_deterministic_and_brackets_mean():
    y_true = [1, 1, 0, 0, 1, 1]
    y_pred = [1, 1, 0, 0, 1, 0]      # 5/6 correct
    groups = [0, 0, 1, 1, 2, 2]
    lo, mean, hi = cluster_bootstrap_ci(y_true, y_pred, groups, _acc, n_resamples=200, seed=0)
    assert lo <= mean <= hi
    assert (lo, mean, hi) == cluster_bootstrap_ci(y_true, y_pred, groups, _acc, n_resamples=200, seed=0)


@pytest.mark.unit
def test_paired_cluster_bootstrap_detects_positive_gap():
    # model a perfect, model b always-wrong -> paired diff strongly positive, CI excludes 0
    y_true = [1, 0, 1, 0, 1, 0, 1, 0]
    pred_a = list(y_true)
    pred_b = [1 - y for y in y_true]
    groups = [0, 0, 1, 1, 2, 2, 3, 3]
    lo, mean, hi = paired_cluster_bootstrap(y_true, pred_a, pred_b, groups, _acc,
                                            n_resamples=300, seed=0)
    assert mean == pytest.approx(1.0)
    assert lo > 0.0
```

- [ ] **Step 2: run → fail.**

- [ ] **Step 3: implement** (append to `bootstrap.py`)

```python
def _resample_group_indices(groups, rng):
    """Return row indices for one cluster-bootstrap resample (sample groups with replacement)."""
    import numpy as np

    groups = np.asarray(groups)
    uniq = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    drawn = uniq[rng.integers(0, uniq.size, uniq.size)]
    return np.concatenate([by_group[g] for g in drawn])


def cluster_bootstrap_ci(y_true, y_pred, groups, metric_fn, n_resamples=2000, alpha=0.05, seed=0):
    """Cluster (group-resampled) bootstrap CI for ``metric_fn(y_true, y_pred)``.

    Resamples GROUPS (e.g. sentences) with replacement — the correct unit of independence when
    rows within a group are correlated. ``metric_fn`` is called ``metric_fn(y_true', y_pred')``.

    Returns:
        ``(lo, mean, hi)`` at the ``1 - alpha`` level (``mean`` = metric on the full sample).
    """
    import numpy as np

    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = _resample_group_indices(groups, rng)
        stats[i] = metric_fn(yt[idx], yp[idx])
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, float(metric_fn(yt, yp)), hi


def paired_cluster_bootstrap(y_true, pred_a, pred_b, groups, metric_fn,
                             n_resamples=2000, alpha=0.05, seed=0):
    """Paired cluster bootstrap CI for ``metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b)``.

    Both metrics are recomputed on the SAME resampled groups each replicate (paired), so the CI
    correctly reflects their correlation on a shared test set.

    Returns:
        ``(lo, mean_diff, hi)`` at the ``1 - alpha`` level.
    """
    import numpy as np

    yt, pa, pb = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = _resample_group_indices(groups, rng)
        diffs[i] = metric_fn(yt[idx], pa[idx]) - metric_fn(yt[idx], pb[idx])
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return lo, float(metric_fn(yt, pa) - metric_fn(yt, pb)), hi
```

- [ ] **Step 4: run → pass.** **Step 5: commit** `feat(eval): add cluster + paired-cluster bootstrap for sentence-grouped CIs`.

---

## Task 6: Export new names from `awake.eval`

**Files:** Modify `src/awake/eval/__init__.py`; Test `tests/test_eval_probing.py`.

- [ ] **Step 1: failing test**

```python
@pytest.mark.unit
def test_public_api_reexports_probing_without_shadowing_accuracy():
    import awake.eval as e

    for name in ("assign_control_labels", "control_vector", "balanced_accuracy", "base_rate",
                 "majority_class", "selectivity", "type_overlap", "emergence_point",
                 "cluster_bootstrap_ci", "paired_cluster_bootstrap"):
        assert hasattr(e, name), name
    # existing P4 accuracy must remain (probing must NOT shadow it with a different contract)
    assert e.accuracy([0, None, 2], [0, 1, 2]) == pytest.approx(1 / 3)
```

- [ ] **Step 2: run → fail.**

- [ ] **Step 3: implement** — add to `src/awake/eval/__init__.py` (keep groups/`__all__` sorted; do NOT import/export `accuracy` from probing):

```python
from awake.eval.bootstrap import (  # noqa: F401  (existing import line extended)
    bootstrap_ci,
    cluster_bootstrap_ci,
    paired_cluster_bootstrap,
    paired_diff_test,
)
from awake.eval.probing import (
    assign_control_labels,
    balanced_accuracy,
    base_rate,
    control_vector,
    emergence_point,
    majority_class,
    selectivity,
    type_overlap,
)
```

Add each new name to `__all__` (sorted). Leave the existing `accuracy` export from `vqa_consistency` untouched.

- [ ] **Step 4: run full unit suite + coverage** `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests -m unit` → PASS, awake coverage ≥90% (new modules ~100% from Tasks 2–5).

- [ ] **Step 5: commit** `feat(eval): re-export P5 probing + cluster-bootstrap core from awake.eval`.

---

## Task 7: `_stub.py` + UD parse helper + smoke

**Files:** Create `scripts/_stub.py`, `scripts/_udparse.py`; Test `tests/test_data_smoke.py`.

`_udparse.py` holds the **pure** CoNLL-U parsing (testable without network); `_stub.py` holds tiny synthetic fixtures for probe smoke tests.

- [ ] **Step 1: failing test**

```python
"""Smoke tests for UD parsing + stub fixtures (no network, no model)."""

import pytest

from _stub import tiny_acts
from _udparse import parse_conllu


_SAMPLE = """\
# sent_id = 1
# text = The dogs run.
1\tThe\tthe\tDET\tDT\t_\t2\tdet\t_\t_
2\tdogs\tdog\tNOUN\tNNS\tNumber=Plur\t3\tnsubj\t_\t_
3\trun\trun\tVERB\tVBP\t_\t0\troot\t_\tSpaceAfter=No
4\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_
"""


@pytest.mark.smoke
def test_parse_conllu_extracts_fields_and_spaceafter():
    sents = parse_conllu(_SAMPLE)
    assert len(sents) == 1
    s = sents[0]
    assert s["words"] == ["The", "dogs", "run", "."]
    assert s["upos"] == ["DET", "NOUN", "VERB", "PUNCT"]
    assert s["number"] == ["", "Plur", "", ""]
    assert s["space_after"] == [True, True, False, True]  # "run" has SpaceAfter=No
    assert s["text"] == "The dogs run."


@pytest.mark.smoke
def test_tiny_acts_shapes():
    acts, meta = tiny_acts()
    assert set(acts) >= {"embedding", "block_0", "block_1"}
    assert acts["embedding"].shape[0] == len(meta["upos"])
```

- [ ] **Step 2: run → fail.**

- [ ] **Step 3: implement `_udparse.py`**

```python
"""Pure CoNLL-U parsing for UD English-EWT (no network, no I/O)."""

from __future__ import annotations


def parse_conllu(text: str) -> list[dict]:
    """Parse CoNLL-U text into per-sentence dicts.

    Skips comments (except ``# text =``) and multiword/empty rows (ID containing ``-``/``.``).

    Returns:
        List of ``{sent_id, text, words, upos, number, space_after}``. ``number`` is the
        ``Number=`` FEAT value (``""`` if absent); ``space_after`` is ``False`` iff MISC has
        ``SpaceAfter=No``.
    """
    sents: list[dict] = []
    cur = _blank()
    for line in text.splitlines():
        if not line.strip():
            if cur["words"]:
                sents.append(cur)
            cur = _blank()
            continue
        if line.startswith("#"):
            if line.startswith("# text ="):
                cur["text"] = line.split("=", 1)[1].strip()
            elif line.startswith("# sent_id ="):
                cur["sent_id"] = line.split("=", 1)[1].strip()
            continue
        cols = line.split("\t")
        if "-" in cols[0] or "." in cols[0]:
            continue
        feats, misc = cols[5], cols[9]
        number = ""
        for f in feats.split("|"):
            if f.startswith("Number="):
                number = f.split("=", 1)[1]
        cur["words"].append(cols[1])
        cur["upos"].append(cols[3])
        cur["number"].append(number)
        cur["space_after"].append("SpaceAfter=No" not in misc.split("|"))
    if cur["words"]:
        sents.append(cur)
    return sents


def _blank() -> dict:
    return {"sent_id": "", "text": "", "words": [], "upos": [], "number": [], "space_after": []}
```

- [ ] **Step 4: implement `_stub.py`**

```python
"""Tiny synthetic fixtures for P5 smoke tests (no model, no dataset)."""

from __future__ import annotations

import numpy as np


def tiny_acts(n: int = 60, d: int = 16, seed: int = 0):
    """Return (acts, meta): per-point residual matrices where the is_noun label is linearly
    encoded ONLY at deep points, so selectivity should rise with depth.

    acts: {point: (n, d) float32}; meta: {"upos": [...], "number": [...], "words": [...],
    "sent_id": [...]}.
    """
    rng = np.random.default_rng(seed)
    upos = ["NOUN" if i % 3 == 0 else ("VERB" if i % 3 == 1 else "DET") for i in range(n)]
    is_noun = np.array([u == "NOUN" for u in upos], dtype=float)
    points = {}
    for li, name in enumerate(["embedding", "block_0", "block_1"]):
        noise = rng.normal(size=(n, d))
        # signal strength grows with depth; deep points encode is_noun in dim 0
        signal = (li / 2.0) * is_noun[:, None] * np.concatenate([[3.0], np.zeros(d - 1)])[None, :]
        points[name] = (noise + signal).astype(np.float32)
    sent_id = [str(i // 5) for i in range(n)]  # 5 tokens per fake sentence
    words = [f"w{i}" for i in range(n)]
    number = ["Plur" if (u == "NOUN" and i % 2 == 0) else ("Sing" if u == "NOUN" else "")
              for i, u in enumerate(upos)]
    return points, {"upos": upos, "number": number, "words": words, "sent_id": sent_id}
```

- [ ] **Step 5: run → pass.** **Step 6: commit** `feat(p5): add pure CoNLL-U parser and smoke fixtures`.

---

## Task 8: `00_data.py` — fetch/verify/parse UD → parquet

**Files:** Create `scripts/00_data.py`. (Pure parse already covered; this is the slow fetch wrapper + a pure `rows_to_table` smoke.)

- [ ] **Step 1: failing test** (append to `tests/test_data_smoke.py`)

```python
@pytest.mark.smoke
def test_rows_to_table_roundtrip_preserves_empty_strings():
    import importlib

    mod = importlib.import_module("00_data")
    sents = [{"sent_id": "1", "text": "a b", "words": ["a", "b"], "upos": ["DET", "NOUN"],
              "number": ["", "Sing"], "space_after": [True, True]}]
    tbl = mod.rows_to_table(sents)
    df = tbl.to_pandas()
    assert list(df.loc[0, "number"]) == ["", "Sing"]      # empty string survived (not null)
    assert list(df.loc[0, "upos"]) == ["DET", "NOUN"]
```

- [ ] **Step 2: run → fail.**

- [ ] **Step 3: implement**

```python
"""Fetch + verify + parse UD English-EWT into prepared parquet (CC BY-SA 4.0)."""

from __future__ import annotations

import hashlib

import pyarrow as pa

from _paths import CONLLU, PREPARED, ensure_dirs, load_config
from _udparse import parse_conllu

_SCHEMA = pa.schema([
    ("sent_id", pa.string()), ("text", pa.string()),
    ("words", pa.list_(pa.string())), ("upos", pa.list_(pa.string())),
    ("number", pa.list_(pa.string())), ("space_after", pa.list_(pa.bool_())),
])


def rows_to_table(sents: list[dict]) -> pa.Table:
    """Build a pyarrow Table with an explicit schema (so empty strings never coerce to null)."""
    cols = {f.name: [s[f.name] for s in sents] for f in _SCHEMA}
    return pa.table(cols, schema=_SCHEMA)


def _fetch(url: str, dest, sha: str) -> str:  # pragma: no cover - network/slow
    import urllib.request

    if not dest.exists():
        urllib.request.urlretrieve(url, dest)  # noqa: S310 - pinned raw.githubusercontent URL
    got = hashlib.sha256(dest.read_bytes()).hexdigest()
    if sha and got != sha:
        raise ValueError(f"SHA-256 mismatch for {dest.name}: expected {sha}, got {got}")
    return got


def main() -> None:  # pragma: no cover - slow path
    """Download the three .conllu files (verify SHA), parse, write prepared parquet."""
    import pyarrow.parquet as pq

    cfg = load_config("data")
    ensure_dirs(CONLLU, PREPARED)
    for split, fname in cfg["files"].items():
        url = f"{cfg['base_url']}/{cfg['ud_tag']}/{fname}"
        dest = CONLLU / fname
        got = _fetch(url, dest, cfg["sha256"].get(split, ""))
        print(f"{split}: sha256={got}")
        sents = parse_conllu(dest.read_text(encoding="utf-8"))
        pq.write_table(rows_to_table(sents), PREPARED / f"{split}.parquet")
        print(f"  parsed {len(sents)} sentences")


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 4: run → pass.** **Step 5: commit** `feat(p5): add 00_data UD fetch/verify/parse to parquet`.

---

## Task 9: `_models.py` — lazy Pythia load + hook extraction + alignment

**Files:** Create `scripts/_models.py`. (Heavy/slow; verified by import-without-download + slow run. The pure alignment helper is smoke-tested.)

- [ ] **Step 1: failing test** (append to `tests/test_data_smoke.py`)

```python
@pytest.mark.smoke
def test_align_last_subword_uses_overlap_not_containment():
    from _models import align_words_to_tokens

    # offsets mimic byte-level BPE: leading space attached to the token (start one before word)
    words = ["hello", "world"]
    word_spans = [(0, 5), (6, 11)]              # "hello world", space at idx 5
    token_offsets = [(0, 5), (5, 11)]            # token 1 = "hello", token 2 = " world"
    # last overlapping token for each word: word0 -> tok0; word1 -> tok1 (overlap, not containment)
    assert align_words_to_tokens(word_spans, token_offsets) == [0, 1]
```

- [ ] **Step 2: run → fail.**

- [ ] **Step 3: implement**

```python
"""Lazy Pythia-160M loader + residual-stream hook extraction + word alignment (P5).

Heavy imports (torch/transformers) live inside functions so importing this module downloads
nothing. The pure ``align_words_to_tokens`` is smoke-tested; extraction is slow-only.
"""

from __future__ import annotations


def align_words_to_tokens(
    word_spans: list[tuple[int, int]],
    token_offsets: list[tuple[int, int]],
) -> list[int | None]:
    """For each word char-span, the index of its LAST overlapping subword token.

    Overlap (not containment): ``tok_end > ws and tok_start < we`` — byte-level BPE attaches the
    leading space to the token, so a containment test would drop every non-first word.

    Returns:
        Per-word last-overlapping token index, or ``None`` if no token overlaps (dropped).
    """
    out: list[int | None] = []
    for ws, we in word_spans:
        last = None
        for ti, (ts, te) in enumerate(token_offsets):
            if te > ws and ts < we:
                last = ti
        out.append(last)
    return out


def load_pythia(model_id: str, revision: str, device: str = "cpu"):  # pragma: no cover - slow
    """Load a frozen Pythia (GPT-NeoX) model + Fast tokenizer (eval, no grad)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.set_grad_enabled(False)
    return model, tok


def extract_points(model, tok, sentence_words, space_after, n_blocks, device="cpu"):  # pragma: no cover - slow
    """Return {point: (n_words, d_model) np.float16} for one sentence via forward hooks.

    Points: 'embedding' (embed_in output) + 'block_0..N-1' (each GPTNeoXLayer output, resid_post)
    + 'ln_f' (final_layer_norm output). Words are aligned to their LAST overlapping subword.
    """
    import numpy as np
    import torch

    # Reconstruct surface string + per-word char spans from FORM + SpaceAfter.
    text, spans, pos = "", [], 0
    for w, sa in zip(sentence_words, space_after, strict=True):
        spans.append((pos, pos + len(w)))
        text += w + (" " if sa else "")
        pos += len(w) + (1 if sa else 0)
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    last_tok = align_words_to_tokens(spans, offsets)

    captured: dict[str, torch.Tensor] = {}
    handles = []
    base = model.gpt_neox

    def mk(name):
        def hook(_m, _i, out):
            captured[name] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    handles.append(base.embed_in.register_forward_hook(mk("embedding")))
    for i in range(n_blocks):
        handles.append(base.layers[i].register_forward_hook(mk(f"block_{i}")))
    handles.append(base.final_layer_norm.register_forward_hook(mk("ln_f")))
    try:
        model(**{k: v.to(device) for k, v in enc.items()})
    finally:
        for h in handles:
            h.remove()

    out = {}
    for name, t in captured.items():
        seq = t[0].to(torch.float64).cpu().numpy()  # (seq, d)
        rows = [seq[ti] for ti in last_tok if ti is not None]
        out[name] = np.asarray(rows, dtype=np.float16)
    keep = [ti is not None for ti in last_tok]
    return out, keep
```

- [ ] **Step 4: verify import is download-free** `export PATH="$HOME/.local/bin:$PATH" && uv run python -c "import sys; sys.path.insert(0,'projects/05-mechanistic-pythia/scripts'); import _models; print('ok')"` → `ok`.
- [ ] **Step 5: run smoke → pass.** **Step 6: commit** `feat(p5): add lazy Pythia loader + hook extraction + overlap alignment`.

---

## Task 10: `10_extract.py` — drive extraction → per-point npy

**Files:** Create `scripts/10_extract.py`. Slow `main`; no new pure logic (alignment tested in Task 9). A thin smoke confirms it imports without torch.

- [ ] **Step 1: implement** (no failing-test step beyond an import smoke in Task 12's file)

```python
"""Extract per-point residual-stream activations for each split -> outputs/acts/<split>/."""

from __future__ import annotations

from _paths import ACTS, PREPARED, ensure_dirs, load_config


def main() -> None:  # pragma: no cover - slow path
    """Run Pythia over each split, hook residual points, write per-point float16 .npy + meta."""
    import numpy as np
    import pandas as pd

    from _models import extract_points, load_pythia

    cfg = load_config("probe")
    model, tok = load_pythia(cfg["model_id"], cfg["model_revision"])
    points = ["embedding", *[f"block_{i}" for i in range(cfg["n_blocks"])], "ln_f"]
    for split in ("train", "dev", "test"):
        df = pd.read_parquet(PREPARED / f"{split}.parquet")
        out_dir = ACTS / split
        ensure_dirs(out_dir)
        acc = {p: [] for p in points}
        meta = {"word": [], "upos": [], "number": [], "sent_id": []}
        for row in df.itertuples():
            pts, keep = extract_points(model, tok, list(row.words), list(row.space_after),
                                       cfg["n_blocks"])
            if not pts:
                continue
            for p in points:
                acc[p].append(pts[p])
            for w, u, nu, k in zip(row.words, row.upos, row.number, keep, strict=True):
                if k:
                    meta["word"].append(w); meta["upos"].append(u); meta["number"].append(nu)
                    meta["sent_id"].append(row.sent_id)
        for p in points:
            np.save(out_dir / f"{p}.npy", np.concatenate(acc[p], axis=0))
        pd.DataFrame(meta).to_parquet(out_dir / "meta.parquet")
        print(f"{split}: {len(meta['word'])} aligned words")


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 2: import-smoke** (add to `tests/test_probe_smoke.py` in Task 11): `importlib.import_module("10_extract")` must not import torch. **Step 3: commit** `feat(p5): add 10_extract activation hooks driver`.

---

## Task 11: `20_probe.py` — standardise + LR + control (injectable fitter) + smoke

**Files:** Create `scripts/20_probe.py`; Test `tests/test_probe_smoke.py`.

- [ ] **Step 1: failing test**

```python
"""Smoke tests for the probing driver on tiny synthetic activations (no model, no sklearn-heavy)."""

import importlib

import numpy as np
import pytest

from _stub import tiny_acts


def _linear_fit(x_train, y_train):
    """Trivial injectable fitter: threshold on dim-0 sign (recovers the planted signal)."""
    def predict(x):
        return (x[:, 0] > x_train[:, 0].mean()).astype(int)
    return predict


@pytest.mark.smoke
def test_probe_property_selectivity_rises_with_depth():
    mod = importlib.import_module("20_probe")
    acts, meta = tiny_acts()
    res = mod.probe_property(
        acts, meta, label_fn=lambda m: [int(u == "NOUN") for u in m["upos"]],
        subset_fn=lambda m: [True] * len(m["upos"]),
        fit_predict=_linear_fit, control_seeds=[0, 1], base_rate=1 / 3,
    )
    pts = {r["point"]: r for r in res}
    assert pts["block_1"]["selectivity"] >= pts["embedding"]["selectivity"] + 0.2


@pytest.mark.smoke
def test_extract_module_imports_without_torch():
    import sys
    importlib.import_module("10_extract")
    assert "torch" not in sys.modules
```

- [ ] **Step 2: run → fail.**

- [ ] **Step 3: implement**

```python
"""Per-point linear probing + Hewitt-Liang control on extracted activations.

``probe_property`` is pure-ish (numpy in, dict out) given an injectable ``fit_predict`` callable
(default = standardise + sklearn LR; stub in tests). ``main`` wires real data + sklearn.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from awake.eval.probing import assign_control_labels, balanced_accuracy, base_rate, control_vector


def _train_test_split_idx(meta_train, meta_test, subset_fn):
    return np.array(subset_fn(meta_train), bool), np.array(subset_fn(meta_test), bool)


def probe_property(
    acts_by_split: dict,
    meta_by_split: dict,
    label_fn: Callable[[dict], list[int]],
    subset_fn: Callable[[dict], list[bool]],
    fit_predict: Callable,
    control_seeds: list[int],
    base_rate: float,
) -> list[dict]:
    """Return per-point {point, balanced_acc, control_balanced_acc, selectivity} for one property.

    ``acts_by_split``/``meta_by_split`` map 'train'/'test' -> {point: (n,d)} / meta dict. For the
    smoke stub a single split is reused for train and test.
    """
    train_m = meta_by_split.get("train", meta_by_split.get("all"))
    test_m = meta_by_split.get("test", meta_by_split.get("all"))
    tr_acts = acts_by_split.get("train", acts_by_split.get("all"))
    te_acts = acts_by_split.get("test", acts_by_split.get("all"))

    tr_sub, te_sub = np.array(subset_fn(train_m), bool), np.array(subset_fn(test_m), bool)
    y_tr = np.array(label_fn(train_m))[tr_sub]
    y_te = np.array(label_fn(test_m))[te_sub]
    words_tr = [w for w, k in zip(train_m["word"], tr_sub, strict=True) if k]
    words_te = [w for w, k in zip(test_m["word"], te_sub, strict=True) if k]

    # control label map over the union, token-rate-matched on train frequencies
    counts: dict[str, int] = {}
    for w in words_tr:
        counts[w] = counts.get(w, 0) + 1
    all_types = set(words_tr) | set(words_te)

    results = []
    for point in tr_acts:
        x_tr = tr_acts[point][tr_sub].astype(np.float64)
        x_te = te_acts[point][te_sub].astype(np.float64)
        predict = fit_predict(x_tr, y_tr)
        probe_ba = balanced_accuracy(list(y_te), list(predict(x_te)))
        ctrl_bas = []
        for seed in control_seeds:
            cmap = assign_control_labels(all_types, counts, base_rate, seed)
            c_tr = np.array(control_vector(words_tr, cmap))
            c_te = np.array(control_vector(words_te, cmap))
            cpred = fit_predict(x_tr, c_tr)
            ctrl_bas.append(balanced_accuracy(list(c_te), list(cpred(x_te))))
        ctrl_ba = float(np.mean(ctrl_bas))
        results.append({"point": point, "balanced_acc": probe_ba,
                        "control_balanced_acc": ctrl_ba, "selectivity": probe_ba - ctrl_ba})
    return results


def sklearn_fitter(C: float, max_iter: int, random_state: int):  # pragma: no cover - slow
    """Default fit_predict: StandardScaler(train) + balanced LogisticRegression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def fit_predict(x_train, y_train):
        scaler = StandardScaler().fit(x_train)
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter,
                                 random_state=random_state).fit(scaler.transform(x_train), y_train)

        def predict(x):
            return clf.predict(scaler.transform(x))
        return predict
    return fit_predict


def main() -> None:  # pragma: no cover - slow path
    """Load per-point acts, choose C on dev per property, probe+control on test, store raw preds."""
    # Full wiring: load ACTS/<split>/<point>.npy + meta; for each property build label_fn/subset_fn;
    # grid-search C on dev (balanced acc on dev probe); run probe_property on test with the chosen C
    # and config control seeds; persist per-token (gold, probe_pred, control_preds, sent_id) to
    # outputs/probe/<property>.npz for 30_eval's cluster bootstrap. (Mechanical; see spec section 5/7.)
    raise NotImplementedError  # implemented during the real-run task, exercised by `slow`/real run


if __name__ == "__main__":  # pragma: no cover
    main()
```

(Implementer note: `main` is a `slow`/real-run path; the audited unit is `probe_property` + `sklearn_fitter`. Flesh out `main`'s persistence during Task 14 so `30_eval` has per-token preds + `sent_id` for the paired cluster bootstrap.)

- [ ] **Step 4: run smoke → pass.** **Step 5: commit** `feat(p5): add 20_probe per-point probing + control (injectable fitter)`.

---

## Task 12: `30_eval.py` — metrics.json + figures + smoke

**Files:** Create `scripts/30_eval.py`; Test `tests/test_probe_smoke.py`.

- [ ] **Step 1: failing test**

```python
@pytest.mark.smoke
def test_assemble_metrics_shape_and_emergence():
    mod = importlib.import_module("30_eval")
    # synthetic per-token preds for 2 points, one property, 2 sentences
    per_token = {
        "gold": [1, 0, 1, 0],
        "sent_id": ["0", "0", "1", "1"],
        "points": {
            "embedding": {"probe": [1, 0, 0, 1], "control": [[1, 0, 0, 1]]},
            "block_0":   {"probe": [1, 0, 1, 0], "control": [[0, 1, 1, 0]]},
        },
    }
    out = mod.assemble_property_metrics(per_token, n_resamples=100, seed=0)
    assert {"points", "emergence"} <= set(out)
    pts = {p["point"]: p for p in out["points"]}
    assert "selectivity_ci" in pts["block_0"]
    assert out["emergence"]["peak"] in ("embedding", "block_0")
```

- [ ] **Step 2: run → fail.**

- [ ] **Step 3: implement**

```python
"""Assemble metrics.json (per property x per point, cluster-bootstrap CIs) + figures."""

from __future__ import annotations

import numpy as np

from awake.eval.bootstrap import cluster_bootstrap_ci, paired_cluster_bootstrap
from awake.eval.probing import balanced_accuracy, emergence_point


def _bal(y_true, y_pred):
    return balanced_accuracy(list(y_true), list(y_pred))


def assemble_property_metrics(per_token: dict, n_resamples: int, seed: int) -> dict:
    """Per-point balanced-acc + control + selectivity with cluster-bootstrap CIs + emergence."""
    gold = per_token["gold"]
    groups = per_token["sent_id"]
    points_out, sel, sel_ci = [], {}, {}
    for point, preds in per_token["points"].items():
        plo, pmean, phi = cluster_bootstrap_ci(gold, preds["probe"], groups, _bal,
                                               n_resamples=n_resamples, seed=seed)
        # control: average balanced acc over seeds; CI from the first seed's paired difference
        ctrl_means = [_bal(gold, c) for c in preds["control"]]
        cmean = float(np.mean(ctrl_means))
        slo, smean, shi = paired_cluster_bootstrap(gold, preds["probe"], preds["control"][0],
                                                   groups, _bal, n_resamples=n_resamples, seed=seed)
        points_out.append({"point": point, "balanced_acc": pmean, "balanced_acc_ci": [plo, phi],
                           "control_balanced_acc": cmean, "selectivity": pmean - cmean,
                           "selectivity_ci": [slo, shi]})
        sel[point] = pmean - cmean
        sel_ci[point] = (slo, shi)
    return {"points": points_out, "emergence": emergence_point(sel, sel_ci)}


def main() -> None:  # pragma: no cover - slow/real-run path
    """Load outputs/probe/<property>.npz, assemble metrics.json, render per-property + hero figs."""
    raise NotImplementedError  # implemented during the real run (mechanical I/O + matplotlib)


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 4: run smoke → pass.** **Step 5: full project + unit suite** `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests projects/05-mechanistic-pythia -m "unit or smoke" --no-cov` → PASS. **Step 6: commit** `feat(p5): add 30_eval metrics assembly + cluster-bootstrap CIs`.

---

## Task 13: Docs — ADR, REPORT skeleton, README, CHANGELOG, notebook scaffold

**Files:** Create `docs/decisions/005-probing-pythia-and-control-tasks.md`, `projects/05-mechanistic-pythia/REPORT.md`, `notebooks/01-layerwise-probing.py`; Modify `README.md`, `CHANGELOG.md`.

- [ ] **Step 1: ADR** mirroring `004-*.md` style — decisions: hook-based resid_post extraction (+ why not the `hidden_states` tuple) + separate `ln_f`; offset-overlap word alignment; balanced-accuracy + token-rate-matched multi-seed control + StandardScaler + paired sentence-cluster bootstrap; "necessary not sufficient" selectivity framing; UD-EWT **CC BY-SA 4.0** (attributed) over PTB; scope = probing only. Status Accepted, 2026-05-27.
- [ ] **Step 2: REPORT.md** with placeholder cells **clearly labelled "(filled by the real run)"** (no fabricated numbers), §sections: question, data (with CC BY-SA attribution), method, results (per-property tables + figures `assets/probe_*.png`, hero), emergence summary, limitations (copy spec §12).
- [ ] **Step 3: notebook** `notebooks/01-layerwise-probing.py` (jupytext percent): resolve root via `os.environ["P5_PROJECT_ROOT"]`, load `outputs/metrics.json`, render the per-property selectivity-by-depth panels + print the emergence summary, and a **self-authored-sentence worked example** (never UD text) — run a short hand-written sentence through Pythia + the trained is_noun probe, print per-token noun-probability at the peak point and one token's trajectory across depth. Guard so it degrades gracefully if `outputs/` is absent.
- [ ] **Step 4: README** — replace stub with question, method (Pythia-160M, per-layer probes, control tasks), how-to-run (`00_data`→`30_eval`), pointer to REPORT + ADR; keep the v1.0 scope/limitations.
- [ ] **Step 5: CHANGELOG** `[Unreleased]` Added: "Project 5 (`05-mechanistic-pythia`): per-layer linear probes on Pythia-160M with Hewitt & Liang control tasks (selectivity by depth) on UD English-EWT; pure `awake.eval.probing` core + cluster-bootstrap helpers."
- [ ] **Step 6: commit** `docs(p5): add ADR-005, REPORT skeleton, README, CHANGELOG, notebook`.

---

## Task 14: Green-CI gate (lint + types + unit + smoke)

- [ ] **Step 1:** `export PATH="$HOME/.local/bin:$PATH" && uv run ruff check . && uv run ruff format --check . && uv run mypy src/awake/` → fix any issues (ASCII-only; project test docstrings).
- [ ] **Step 2:** `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests projects apps -m "unit or smoke"` → PASS, awake coverage ≥90%.
- [ ] **Step 3:** commit any fixups `chore(p5): satisfy lint/type/coverage gates`.

---

## Task 15: Real run (slow) — produce metrics, figures, executed notebook, REPORT

> PLAYBOOK §1 real run. Flesh out the `20_probe.main` and `30_eval.main` slow paths (mechanical I/O + the dev-C grid search + matplotlib figures) here, run end-to-end, fill REPORT with real numbers. No fabrication.

- [ ] **Step 1:** `00_data.py` → fetch+verify UD; **record the printed SHA-256s into `configs/data.yaml`** and pin `model_revision` to the resolved commit SHA in `configs/probe.yaml`; commit the config pin.
- [ ] **Step 2:** `10_extract.py` → per-point `.npy` + meta (watch RAM; per-point load). **Step 3:** `20_probe.py` (C chosen on dev per property; control over K seeds; train-token cap). **Step 4:** `30_eval.py` → `metrics.json` + `assets/probe_{is_noun,is_verb,noun_number}.png` + `hero.png`. Sanity: selectivity curves sensible; is_verb balanced-acc not pinned at 0.5; noun_number train_n ≥ min (else flagged).
- [ ] **Step 5:** execute notebook with outputs (`P5_PROJECT_ROOT=… uv run --extra dev jupytext --to notebook --execute …`; then nbconvert to html). Confirm 0 error cells and **no UD text** in outputs.
- [ ] **Step 6:** fill REPORT.md with real numbers + the emergence story; verify every number matches `metrics.json`.
- [ ] **Step 7:** commit `feat(p5): real run — layerwise probing metrics, figures, executed notebook, report` (assets + notebook + REPORT + config pins; `outputs/` stays gitignored).

---

## Task 16: Final review + finish branch

- [ ] **Step 1:** dispatch a final whole-branch reviewer (spec-compliance + quality + governance: no UD text committed, numbers match metrics.json). Fix blockers/majors.
- [ ] **Step 2:** re-run the full gate green.
- [ ] **Step 3:** superpowers:finishing-a-development-branch → merge `p5-mechanistic-pythia` to `main` (no-ff), re-run gate on merge, push. No HF Space.

---

## Self-review notes (author)

- **Spec coverage:** §2 data → Task 8 (+ pure parse Task 7); §3 extraction/alignment → Task 9–10 (overlap predicate in `align_words_to_tokens`); §4 properties → label_fn/subset_fn in Task 11 + real run; §5 probe/control/metric/CI → Tasks 2–5 (pure) + 11–12; §6 pure-core signatures → Tasks 2–6 (no second `accuracy`; `metric_fn(y_true,y_pred)`; `balanced_accuracy(y_true,y_pred)`; union control map; `emergence_point` excludes `ln_f`); §7 metrics/figures → Task 12 + 15; §8 tests → throughout; §9 governance/ADR → Task 13; §10 real-run → Task 15; §11 layout → all; §12 limitations → Task 13 REPORT.
- **Type consistency:** all metric callables `(y_true, y_pred)`; `assign_control_labels(all_types, train_counts, base_rate, seed)`; `control_vector` raises KeyError only on a union-violation; cluster bootstraps call `metric_fn(y_true, y_pred)`.
- **No placeholders:** complete code in every code step except the two `main()` slow paths (Tasks 11–12) explicitly deferred to Task 15 and `# pragma: no cover`; REPORT numbers come from the real run (deliberately not fabricated).
- **Governance:** UD CC BY-SA attributed (ADR + REPORT); `.conllu`/parquet/`outputs/` gitignored; notebook worked example uses a self-authored sentence, never UD text.
