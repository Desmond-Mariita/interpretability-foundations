# VQA A-OKVQA Vision-Ablation Probe (Project 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `projects/04-vqa-aokvqa`: two zero-shot multiple-choice VQA pipelines on A-OKVQA — caption-then-LLM (Pipeline A: BLIP-2 → Qwen2.5-7B) and direct VLM (Pipeline B: Qwen2.5-VL-3B, plus a size-matched B7 = Qwen2.5-VL-7B) — measured by accuracy, inter-pipeline divergence, and a **paired-baseline vision-ablation probe** whose headline is `Δ = consistency(with-explanation) − consistency(no-explanation)`.

**Architecture:** All metric semantics live in a pure, I/O-free, 90%-covered core `src/awake/eval/vqa_consistency.py` (parse answers, leakage flags, accuracy/consistency/divergence) driven entirely by unit tests on hand-built lists. Heavy generation is config-driven numbered scripts in `projects/04-vqa-aokvqa/` (`00_data` → `10_run_pipelines` → `20_probe` → `30_eval`); every script takes an **injectable `generate` callable** so smoke tests drive the full data→metrics flow on tiny synthetic items with a stub generator (no model downloads, no dataset). Real model code is imported lazily inside functions and marked `slow` (excluded from CI). No HF Space.

**Tech Stack:** Python 3.11, uv, pytest (markers unit/smoke/slow), transformers (`Blip2ForConditionalGeneration`, `Qwen2_5_VLForConditionalGeneration`, `AutoModelForCausalLM`), torch, datasets, numpy, pandas/pyarrow, matplotlib, pillow, qwen-vl-utils.

**Reference spec:** `docs/superpowers/specs/2026-05-26-vqa-aokvqa-design.md` (v2.1) — read it before starting. **Governance:** A-OKVQA is open, but treat per PLAYBOOK §4 — never commit the dataset, decoded images, or `outputs/`. Commit only code, configs, `assets/hero.png`, the notebook (with outputs, no raw dataset dumps), and `REPORT.md`.

---

## Conventions for every task

- Run from the **repo root** unless stated otherwise. `uv` is at `$HOME/.local/bin/uv` (NOT on PATH): start every command with `export PATH="$HOME/.local/bin:$PATH"` or call the absolute path. `just`/`gh` are not installed — run the underlying `uv run …`.
- **Unit tests for `src/awake/`** live in repo-root `tests/` (so `--cov=awake` counts them). **Project plumbing/smoke tests** live in `projects/04-vqa-aokvqa/tests/`.
- Markers: `@pytest.mark.unit` (fast pure), `@pytest.mark.smoke` (tiny stub, CPU, no models/no dataset), `@pytest.mark.slow` (real BLIP-2/Qwen/dataset, excluded from CI).
- Single-file runs: `uv run pytest <path> -v --no-cov`. Coverage gate (must stay ≥90% on `awake`): `uv run pytest tests -m unit`.
- CI command (must pass): `uv run pytest tests projects apps -m "unit or smoke"`.
- ruff: Google docstrings on `src/awake/`, line ≤100, double quotes, **ASCII only in code** (no `×`/`–`/`Δ` in code or docstrings — RUF002/003; write `Delta`/`x`/`-`). Commit after each task.
- `seed_everything` exists: `from awake.utils.seeding import seed_everything`.
- Reuse `awake.eval.bootstrap` (`bootstrap_ci(values, n_resamples=2000, alpha=0.05, seed=0) -> (lo, mean, hi)`; `paired_diff_test(a, b, n_resamples=2000, seed=0) -> dict` with `mean_diff/ci_low/ci_high/p_value`) for all CIs and paired deltas.
- Per-project conftest isolates `scripts/` imports (sys.modules eviction) — copied verbatim from P3 in Task 1.

## File structure (created by this plan)

```
src/awake/eval/
  vqa_consistency.py        # NEW pure core: normalize_text, extract_choice, leakage fns,
                            #   parse_rate, accuracy, consistency_rate, pipeline_divergence
  __init__.py               # + re-export the above (MODIFY)
tests/                      # repo-root awake unit tests
  test_eval_vqa_consistency.py
projects/04-vqa-aokvqa/
  configs/{data,pipelines}.yaml
  scripts/
    __init__.py  _paths.py  _models.py  _stub.py
    00_data.py 10_run_pipelines.py 20_probe.py 30_eval.py
  tests/{conftest.py, test_data_smoke.py, test_pipeline_smoke.py, test_eval_smoke.py}
  notebooks/01-vqa-consistency.py    # committed WITH outputs
  assets/hero.png                    # committed (multi-panel figure)
  REPORT.md                          # (README.md exists)
docs/decisions/004-vqa-pipelines-and-vision-ablation.md
.gitignore                           # + projects/04-vqa-aokvqa/outputs/ (MODIFY)
pyproject.toml                       # + [project.optional-dependencies] vqa (MODIFY)
CHANGELOG.md                         # + [Unreleased] P4 entry (MODIFY)
```

---

## Task 1: Project scaffold, gitignore, optional-deps, conftest

**Files:**
- Create: `projects/04-vqa-aokvqa/scripts/__init__.py` (empty), `projects/04-vqa-aokvqa/scripts/_paths.py`, `projects/04-vqa-aokvqa/tests/conftest.py`, `projects/04-vqa-aokvqa/tests/__init__.py` (empty)
- Modify: `.gitignore`, `pyproject.toml`

- [ ] **Step 1: Add P4 outputs to `.gitignore`**

Append after the existing `projects/03-multimodal-hatefulmemes/outputs/` line:

```
projects/04-vqa-aokvqa/outputs/
```

- [ ] **Step 2: Add the `vqa` optional-dependency group to `pyproject.toml`**

After the `explain-shap = [...]` line in `[project.optional-dependencies]`:

```toml
vqa = ["qwen-vl-utils>=0.0.10", "accelerate>=0.30"]
```

(Note: `transformers>=5.9` is the effective floor for Qwen2.5-VL and is already satisfied by the installed stack; the `vqa` group is only needed for the slow/real run, not for unit+smoke CI.)

- [ ] **Step 3: Create `scripts/_paths.py`** (mirrors P3)

```python
"""Filesystem paths and config loading for project 04."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(os.environ.get("P4_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
OUTPUTS = PROJECT_ROOT / "outputs"
ASSETS = PROJECT_ROOT / "assets"
CONFIGS = PROJECT_ROOT / "configs"
IMAGES = OUTPUTS / "images"
PREPARED = OUTPUTS / "prepared"
GEN = OUTPUTS / "gen"


def load_config(name: str) -> dict:
    """Load ``configs/<name>.yaml`` as a dict."""
    with open(CONFIGS / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def ensure_dirs(*paths: Path) -> None:
    """Create each path (and parents)."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Create `tests/conftest.py`** (copy P3's isolation fixture verbatim)

```python
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
```

- [ ] **Step 5: Verify nothing breaks and commit**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest projects/04-vqa-aokvqa -q --no-cov`
Expected: no tests collected (or all pass) with exit code 0 or 5 (no tests). Then:

```bash
git add .gitignore pyproject.toml projects/04-vqa-aokvqa/
git commit -m "feat(p4): scaffold project, gitignore outputs, vqa optional-deps, conftest"
```

---

## Task 2: Pure core — `normalize_text`

**Files:**
- Create: `src/awake/eval/vqa_consistency.py`
- Test: `tests/test_eval_vqa_consistency.py`

- [ ] **Step 1: Write the failing test**

```python
"""Unit tests for the pure VQA consistency core."""

import pytest

from awake.eval.vqa_consistency import normalize_text


@pytest.mark.unit
def test_normalize_text_lowercases_strips_collapses_and_drops_punctuation():
    assert normalize_text("  A Red, Car!! ") == "a red car"
    assert normalize_text("It's GREEN.") == "it s green"
    assert normalize_text("") == ""
    assert normalize_text("multiple   spaces\tand\nnewlines") == "multiple spaces and newlines"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests/test_eval_vqa_consistency.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'awake.eval.vqa_consistency'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Pure, I/O-free metric core for the A-OKVQA vision-ablation probe.

No model loading and no file I/O live here: every function takes parsed ints or
strings and returns a number or dict, so the metric definitions are locked down by
fast unit tests. See docs/superpowers/specs/2026-05-26-vqa-aokvqa-design.md (v2.1).
"""

from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Lowercase, drop punctuation, and collapse whitespace.

    Args:
        s: Raw text.

    Returns:
        Normalized string (lowercased, punctuation replaced by spaces, runs of
        whitespace collapsed to single spaces, stripped).
    """
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s)
    return s.strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests/test_eval_vqa_consistency.py -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/awake/eval/vqa_consistency.py tests/test_eval_vqa_consistency.py
git commit -m "feat(eval): add normalize_text for VQA answer matching"
```

---

## Task 3: Pure core — `extract_choice` (strict-then-text)

**Files:**
- Modify: `src/awake/eval/vqa_consistency.py`
- Test: `tests/test_eval_vqa_consistency.py`

- [ ] **Step 1: Write the failing tests**

```python
from awake.eval.vqa_consistency import extract_choice


@pytest.mark.unit
def test_extract_choice_strict_first_line_letter():
    choices = ["red car", "blue truck", "green bus", "yellow van"]
    assert extract_choice("Answer: B\nBecause it is blue.", choices) == (1, "strict")
    assert extract_choice("answer : C", choices) == (2, "strict")
    # strict wins even if other choice text also appears
    assert extract_choice("Answer: A\nthe blue truck is wrong", choices) == (0, "strict")


@pytest.mark.unit
def test_extract_choice_text_fallback_unique_substring():
    choices = ["red car", "blue truck", "green bus", "yellow van"]
    assert extract_choice("I think it is a blue truck.", choices) == (1, "text")


@pytest.mark.unit
def test_extract_choice_ambiguous_or_missing_returns_none():
    choices = ["cat", "black cat", "dog", "bird"]
    # output contains "black cat" -> both "cat" and "black cat" match -> ambiguous
    assert extract_choice("clearly a black cat", choices) == (None, "none")
    assert extract_choice("no opinion here", choices) == (None, "none")


@pytest.mark.unit
def test_extract_choice_strict_letter_out_of_range_falls_through():
    choices = ["red car", "blue truck"]  # only A, B valid
    # 'Answer: D' is out of range -> not strict; no text match -> none
    assert extract_choice("Answer: D", choices) == (None, "none")
```

- [ ] **Step 2: Run to verify it fails**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests/test_eval_vqa_consistency.py -k extract_choice -v --no-cov`
Expected: FAIL — `cannot import name 'extract_choice'`.

- [ ] **Step 3: Implement**

Add to `vqa_consistency.py`:

```python
_STRICT = re.compile(r"^\s*answer\s*[:\-]?\s*([a-d])\b", re.IGNORECASE | re.MULTILINE)


def extract_choice(model_output: str, choices: list[str]) -> tuple[int | None, str]:
    """Parse the chosen choice index from a model's free-text output (strict-then-text).

    Primary (strict): the first ``Answer: <A-D>`` match, mapped to a 0-based index;
    used only if that index is valid for ``choices``. Fallback (text): exactly one
    choice's normalized text is a substring of the normalized output; ambiguous or
    zero matches yield ``None``.

    Args:
        model_output: Raw generated text.
        choices: The multiple-choice option strings (index order defines A, B, C, ...).

    Returns:
        ``(index, parsed_by)`` where ``index`` is a 0-based int or ``None`` and
        ``parsed_by`` is one of ``"strict"``, ``"text"``, ``"none"``.
    """
    m = _STRICT.search(model_output)
    if m is not None:
        idx = "abcd".index(m.group(1).lower())
        if idx < len(choices):
            return idx, "strict"
    norm_out = normalize_text(model_output)
    hits = [i for i, c in enumerate(choices) if normalize_text(c) and normalize_text(c) in norm_out]
    if len(hits) == 1:
        return hits[0], "text"
    return None, "none"
```

- [ ] **Step 4: Run to verify it passes**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests/test_eval_vqa_consistency.py -k extract_choice -v --no-cov`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/awake/eval/vqa_consistency.py tests/test_eval_vqa_consistency.py
git commit -m "feat(eval): add extract_choice strict-then-text parser"
```

---

## Task 4: Pure core — leakage flags

**Files:**
- Modify: `src/awake/eval/vqa_consistency.py`
- Test: `tests/test_eval_vqa_consistency.py`

- [ ] **Step 1: Write the failing tests**

```python
from awake.eval.vqa_consistency import explanation_leaks_answer, rationale_leaks_answer


@pytest.mark.unit
def test_explanation_leaks_answer_substring_match_normalized():
    assert explanation_leaks_answer("Because it is a RED car.", "red car") is True
    assert explanation_leaks_answer("Because it is blue.", "red car") is False
    assert explanation_leaks_answer("anything", "") is False  # empty choice never leaks


@pytest.mark.unit
def test_rationale_leaks_answer_any_rationale_matches():
    assert rationale_leaks_answer(["it flies", "it is a red car"], "red car") is True
    assert rationale_leaks_answer(["it flies", "it is blue"], "red car") is False
    assert rationale_leaks_answer([], "red car") is False


@pytest.mark.unit
def test_leakage_does_not_match_bare_letter():
    # a lone letter token must NOT cause a false leak (the choice TEXT is what matters)
    assert explanation_leaks_answer("the answer is A", "red car") is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests/test_eval_vqa_consistency.py -k leak -v --no-cov`
Expected: FAIL — import error.

- [ ] **Step 3: Implement**

```python
def explanation_leaks_answer(explanation: str, chosen_choice_text: str) -> bool:
    """True iff the chosen choice's text appears verbatim (normalized) in the explanation.

    Detects the direct leak path for the probe: a self-explanation that simply restates
    the chosen answer. Matches the choice TEXT, never the bare letter.

    Args:
        explanation: The model's free-text explanation.
        chosen_choice_text: The text of the choice the model selected.

    Returns:
        ``True`` if ``chosen_choice_text`` is non-empty and its normalized form is a
        substring of the normalized explanation.
    """
    needle = normalize_text(chosen_choice_text)
    if not needle:
        return False
    return needle in normalize_text(explanation)


def rationale_leaks_answer(rationales: list[str], gold_choice_text: str) -> bool:
    """True iff any human rationale contains the gold choice text (normalized).

    Used to build the leakage-sensitivity split on A-OKVQA's ``rationales`` field.

    Args:
        rationales: The human-written rationales for an item.
        gold_choice_text: The text of the gold (correct) choice.

    Returns:
        ``True`` if any rationale leaks the gold choice text.
    """
    return any(explanation_leaks_answer(r, gold_choice_text) for r in rationales)
```

- [ ] **Step 4: Run to verify it passes**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests/test_eval_vqa_consistency.py -k leak -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/awake/eval/vqa_consistency.py tests/test_eval_vqa_consistency.py
git commit -m "feat(eval): add explanation/rationale leakage flags (choice-text match)"
```

---

## Task 5: Pure core — `parse_rate` and `accuracy`

**Files:**
- Modify: `src/awake/eval/vqa_consistency.py`
- Test: `tests/test_eval_vqa_consistency.py`

- [ ] **Step 1: Write the failing tests**

```python
from awake.eval.vqa_consistency import accuracy, parse_rate


@pytest.mark.unit
def test_parse_rate_counts_non_none_fraction():
    assert parse_rate([0, 1, None, 2]) == 0.75
    assert parse_rate([None, None]) == 0.0
    assert parse_rate([]) == 0.0


@pytest.mark.unit
def test_accuracy_none_counts_as_wrong():
    assert accuracy([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    assert accuracy([0, None, 2, 1], [0, 1, 2, 3]) == 0.5  # None and 1!=3 both wrong
    assert accuracy([], []) == 0.0


@pytest.mark.unit
def test_accuracy_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy([0, 1], [0])
```

- [ ] **Step 2: Run to verify it fails**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests/test_eval_vqa_consistency.py -k "parse_rate or accuracy" -v --no-cov`
Expected: FAIL — import error.

- [ ] **Step 3: Implement**

```python
def parse_rate(parsed: list[int | None]) -> float:
    """Fraction of items that parsed to a choice index (not ``None``).

    Args:
        parsed: Per-item parsed indices (``None`` = unparseable).

    Returns:
        Share in [0, 1]; ``0.0`` for an empty list.
    """
    if not parsed:
        return 0.0
    return sum(p is not None for p in parsed) / len(parsed)


def accuracy(pred: list[int | None], gold: list[int]) -> float:
    """Top-1 accuracy where unparseable predictions count as wrong.

    Args:
        pred: Per-item predicted indices (``None`` = unparseable = wrong).
        gold: Per-item gold indices.

    Returns:
        Share correct in [0, 1] (denominator = all items); ``0.0`` for empty.

    Raises:
        ValueError: If ``pred`` and ``gold`` differ in length.
    """
    if len(pred) != len(gold):
        raise ValueError("pred and gold must have equal length")
    if not pred:
        return 0.0
    return sum(p is not None and p == g for p, g in zip(pred, gold)) / len(pred)
```

- [ ] **Step 4: Run to verify it passes**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests/test_eval_vqa_consistency.py -k "parse_rate or accuracy" -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/awake/eval/vqa_consistency.py tests/test_eval_vqa_consistency.py
git commit -m "feat(eval): add parse_rate and accuracy (None=wrong policy)"
```

---

## Task 6: Pure core — `consistency_rate`

**Files:**
- Modify: `src/awake/eval/vqa_consistency.py`
- Test: `tests/test_eval_vqa_consistency.py`

- [ ] **Step 1: Write the failing tests**

```python
from awake.eval.vqa_consistency import consistency_rate


@pytest.mark.unit
def test_consistency_rate_primary_none_either_side_is_inconsistent():
    orig = [0, 1, 2, 3]
    abl = [0, 1, None, 0]  # item0,1 consistent; item2 None; item3 differs
    assert consistency_rate(orig, abl) == 0.5  # 2 of 4


@pytest.mark.unit
def test_consistency_rate_paired_only_drops_unparseable_pairs():
    orig = [0, 1, None, 3]
    abl = [0, 1, 2, None]  # paired-parsed pairs: (0,0)(1,1) only -> both consistent
    assert consistency_rate(orig, abl, paired_only=True) == 1.0
    # primary: 2 consistent of 4
    assert consistency_rate(orig, abl) == 0.5


@pytest.mark.unit
def test_consistency_rate_empty_and_mismatch():
    assert consistency_rate([], []) == 0.0
    assert consistency_rate([0], [0], paired_only=True) == 1.0
    with pytest.raises(ValueError):
        consistency_rate([0, 1], [0])
```

- [ ] **Step 2: Run to verify it fails**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests/test_eval_vqa_consistency.py -k consistency -v --no-cov`
Expected: FAIL — import error.

- [ ] **Step 3: Implement**

```python
def consistency_rate(
    original: list[int | None],
    ablated: list[int | None],
    paired_only: bool = False,
) -> float:
    """Rate at which the ablated answer matches the original answer.

    Args:
        original: Per-item original parsed indices.
        ablated: Per-item ablated parsed indices.
        paired_only: If ``False`` (primary policy), a pair is consistent iff both
            parse AND are equal, and the denominator is all items (unparseable on
            either side counts as inconsistent). If ``True`` (documented sensitivity),
            pairs with ``None`` on either side are dropped and the denominator is the
            pairs where both sides parsed.

    Returns:
        Consistency share in [0, 1]; ``0.0`` when the denominator is empty.

    Raises:
        ValueError: If the two lists differ in length.
    """
    if len(original) != len(ablated):
        raise ValueError("original and ablated must have equal length")
    pairs = list(zip(original, ablated))
    if paired_only:
        pairs = [(o, a) for o, a in pairs if o is not None and a is not None]
    if not pairs:
        return 0.0
    return sum(o is not None and a is not None and o == a for o, a in pairs) / len(pairs)
```

- [ ] **Step 4: Run to verify it passes**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests/test_eval_vqa_consistency.py -k consistency -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/awake/eval/vqa_consistency.py tests/test_eval_vqa_consistency.py
git commit -m "feat(eval): add consistency_rate with paired_only sensitivity"
```

---

## Task 7: Pure core — `pipeline_divergence`

**Files:**
- Modify: `src/awake/eval/vqa_consistency.py`
- Test: `tests/test_eval_vqa_consistency.py`

- [ ] **Step 1: Write the failing test**

```python
from awake.eval.vqa_consistency import pipeline_divergence


@pytest.mark.unit
def test_pipeline_divergence_overall_and_contingency():
    #            gold:  0    1    2    3
    a = [0, 1, 2, 0]  # correct, correct, correct, wrong
    b = [0, 2, 2, 1]  # correct, wrong,   correct, wrong
    gold = [0, 1, 2, 3]
    out = pipeline_divergence(a, b, gold)
    # disagreements: item1 (1 vs 2), item3 (0 vs 1) -> 2/4
    assert out["overall"] == 0.5
    c = out["contingency"]
    # item0: both correct + agree
    assert c["both_correct"] == {"agree": 1, "disagree": 0}
    # item1: a correct, b wrong, disagree
    assert c["a_correct_b_wrong"] == {"agree": 0, "disagree": 1}
    # item2: both correct + agree
    # (folded into both_correct above => both_correct agree == 2)
    assert c["both_correct"]["agree"] == 2
    # item3: both wrong, disagree
    assert c["both_wrong"] == {"agree": 0, "disagree": 1}
    assert c["a_wrong_b_correct"] == {"agree": 0, "disagree": 0}


@pytest.mark.unit
def test_pipeline_divergence_none_counts_as_disagree():
    out = pipeline_divergence([None, 0], [0, 0], [0, 0])
    assert out["overall"] == 0.5  # item0 None vs 0 disagrees; item1 agrees
```

(Note: the first test's assertions about `both_correct` reflect that items 0 and 2 are both-correct-and-agree, so that cell accumulates `agree: 2`. Adjust the inline `assert c["both_correct"] == {"agree": 1, ...}` line — keep only the `both_correct"]["agree"] == 2` assertion. The implementer should write the test so it passes against the spec semantics below; the canonical expected contingency is: `both_correct={agree:2,disagree:0}`, `a_correct_b_wrong={agree:0,disagree:1}`, `a_wrong_b_correct={agree:0,disagree:0}`, `both_wrong={agree:0,disagree:1}`.)

- [ ] **Step 2: Run to verify it fails**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests/test_eval_vqa_consistency.py -k divergence -v --no-cov`
Expected: FAIL — import error.

- [ ] **Step 3: Implement**

```python
def pipeline_divergence(
    a: list[int | None],
    b: list[int | None],
    gold: list[int],
) -> dict:
    """Inter-pipeline divergence plus a correctness-conditioned 2x2 contingency.

    Two pipelines "agree" on an item iff both parse and predict the same index;
    ``None`` on either side counts as a disagreement. Each item is also bucketed by
    (a-correct, b-correct) into one of four cells, and within each cell counted as
    agree/disagree.

    Args:
        a: Pipeline A per-item parsed indices.
        b: Pipeline B per-item parsed indices.
        gold: Per-item gold indices.

    Returns:
        ``{"overall": disagree_rate, "contingency": {cell: {"agree": int,
        "disagree": int}}}`` with cells ``both_correct``, ``a_correct_b_wrong``,
        ``a_wrong_b_correct``, ``both_wrong``.

    Raises:
        ValueError: If the three lists differ in length.
    """
    if not (len(a) == len(b) == len(gold)):
        raise ValueError("a, b, gold must have equal length")
    cells = {
        "both_correct": {"agree": 0, "disagree": 0},
        "a_correct_b_wrong": {"agree": 0, "disagree": 0},
        "a_wrong_b_correct": {"agree": 0, "disagree": 0},
        "both_wrong": {"agree": 0, "disagree": 0},
    }
    disagree = 0
    for ai, bi, gi in zip(a, b, gold):
        agree = ai is not None and bi is not None and ai == bi
        if not agree:
            disagree += 1
        a_ok = ai is not None and ai == gi
        b_ok = bi is not None and bi == gi
        if a_ok and b_ok:
            cell = "both_correct"
        elif a_ok and not b_ok:
            cell = "a_correct_b_wrong"
        elif b_ok and not a_ok:
            cell = "a_wrong_b_correct"
        else:
            cell = "both_wrong"
        cells[cell]["agree" if agree else "disagree"] += 1
    overall = disagree / len(a) if a else 0.0
    return {"overall": overall, "contingency": cells}
```

- [ ] **Step 4: Run to verify it passes**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests/test_eval_vqa_consistency.py -k divergence -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/awake/eval/vqa_consistency.py tests/test_eval_vqa_consistency.py
git commit -m "feat(eval): add pipeline_divergence with correctness contingency"
```

---

## Task 8: Export pure core from `awake.eval` + coverage gate

**Files:**
- Modify: `src/awake/eval/__init__.py`
- Test: `tests/test_eval_vqa_consistency.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.unit
def test_public_api_reexports_vqa_consistency():
    import awake.eval as e

    for name in (
        "normalize_text",
        "extract_choice",
        "explanation_leaks_answer",
        "rationale_leaks_answer",
        "parse_rate",
        "accuracy",
        "consistency_rate",
        "pipeline_divergence",
    ):
        assert hasattr(e, name), name
```

- [ ] **Step 2: Run to verify it fails**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests/test_eval_vqa_consistency.py -k reexports -v --no-cov`
Expected: FAIL — attributes missing.

- [ ] **Step 3: Implement** — add the import block and `__all__` entries to `src/awake/eval/__init__.py`

Add this import (keep alphabetical grouping with the others):

```python
from awake.eval.vqa_consistency import (
    accuracy,
    consistency_rate,
    explanation_leaks_answer,
    extract_choice,
    normalize_text,
    parse_rate,
    pipeline_divergence,
    rationale_leaks_answer,
)
```

And insert these names into the `__all__` list (keep it sorted):
`"accuracy"`, `"consistency_rate"`, `"explanation_leaks_answer"`, `"extract_choice"`, `"normalize_text"`, `"parse_rate"`, `"pipeline_divergence"`, `"rationale_leaks_answer"`.

- [ ] **Step 4: Run the full unit suite + coverage gate**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests -m unit`
Expected: PASS, total coverage on `awake` ≥90% (the new module should be ~100% covered by Tasks 2-7).

- [ ] **Step 5: Commit**

```bash
git add src/awake/eval/__init__.py tests/test_eval_vqa_consistency.py
git commit -m "feat(eval): re-export VQA consistency core from awake.eval"
```

---

## Task 9: Configs — `data.yaml` and `pipelines.yaml`

**Files:**
- Create: `projects/04-vqa-aokvqa/configs/data.yaml`, `projects/04-vqa-aokvqa/configs/pipelines.yaml`

- [ ] **Step 1: Write `configs/data.yaml`**

```yaml
# A-OKVQA data preparation config.
dataset: "HuggingFaceM4/A-OKVQA"   # bundles the COCO image as a PIL object per item
split: "validation"                 # labelled headline split (~1.1k)
# Subset is a PILOT/SMOKE convenience ONLY -- never the reported headline.
# Leave subset_n null for the real headline run; set + record seed for pilots.
subset_n: null
seed: 0
image_format: "jpg"
```

- [ ] **Step 2: Write `configs/pipelines.yaml`**

```yaml
# Models, generation settings, and prompts for the two pipelines.
generation:
  do_sample: false
  max_new_tokens: 256
  torch_dtype: "float16"
  device_map: "cuda:0"

models:
  blip2: "Salesforce/blip2-opt-2.7b"
  qwen_lm: "Qwen/Qwen2.5-7B-Instruct"
  qwen_vl_3b: "Qwen/Qwen2.5-VL-3B-Instruct"
  qwen_vl_7b: "Qwen/Qwen2.5-VL-7B-Instruct"   # B7: required for the headline

caption:
  prompt: "Question: describe the image in detail. Answer:"

null_caption: "(no visual information available)"   # Pipeline A vision ablation

# Two prompt variants. 'main' is the headline; 'alt' is a wording-only sensitivity arm.
# Placeholders: {question} {choices_block}; optional {caption}; optional {explanation}.
prompts:
  main:
    answer: |
      Answer the multiple-choice question. Respond with a first line exactly
      'Answer: <letter>' (one of A-D), then one sentence explaining why.
      Question: {question}
      Choices:
      {choices_block}
    answer_with_caption: |
      Answer the multiple-choice question using the image description.
      Respond with a first line exactly 'Answer: <letter>' (one of A-D), then one
      sentence explaining why.
      Image description: {caption}
      Question: {question}
      Choices:
      {choices_block}
    ablate_with_expl: |
      You previously answered this question and explained: "{explanation}".
      Re-answer it now. Respond with a first line exactly 'Answer: <letter>' (one of
      A-D), then one sentence explaining why.
      Question: {question}
      Choices:
      {choices_block}
    ablate_no_expl: |
      Answer the multiple-choice question. Respond with a first line exactly
      'Answer: <letter>' (one of A-D), then one sentence explaining why.
      Question: {question}
      Choices:
      {choices_block}
  alt:
    answer: |
      Choose the single best option. Begin your reply with 'Answer: <letter>'
      (A, B, C, or D) and add one sentence of justification.
      Q: {question}
      Options:
      {choices_block}
    answer_with_caption: |
      Using only this description of a photo, choose the single best option. Begin
      with 'Answer: <letter>' (A-D) and add one sentence of justification.
      Description: {caption}
      Q: {question}
      Options:
      {choices_block}
    ablate_with_expl: |
      Earlier you said: "{explanation}". Now choose the single best option. Begin
      with 'Answer: <letter>' (A-D) and add one sentence of justification.
      Q: {question}
      Options:
      {choices_block}
    ablate_no_expl: |
      Choose the single best option. Begin with 'Answer: <letter>' (A-D) and add one
      sentence of justification.
      Q: {question}
      Options:
      {choices_block}
```

- [ ] **Step 3: Commit**

```bash
git add projects/04-vqa-aokvqa/configs/
git commit -m "feat(p4): add data and pipelines configs (prompts, models, generation)"
```

---

## Task 10: `_stub.py` and prompt helpers + unit tests

**Files:**
- Create: `projects/04-vqa-aokvqa/scripts/_stub.py`
- Create: `projects/04-vqa-aokvqa/scripts/_prompts.py`
- Test: `projects/04-vqa-aokvqa/tests/test_pipeline_smoke.py` (created here; extended in later tasks)

This task builds the deterministic stub generator and the pure prompt-formatting helpers that the orchestration scripts (Tasks 12-13) depend on. Keeping them separate keeps heavy scripts importable without models.

- [ ] **Step 1: Write the failing test**

```python
"""Smoke tests for prompt formatting and the stub generator (no models, no dataset)."""

import pytest

from _prompts import format_choices, render
from _stub import stub_generate


@pytest.mark.smoke
def test_format_choices_letters_options():
    assert format_choices(["red", "blue"]) == "A. red\nB. blue"


@pytest.mark.smoke
def test_render_fills_placeholders():
    tmpl = "Q: {question}\n{choices_block}"
    out = render(tmpl, question="why?", choices_block="A. x")
    assert out == "Q: why?\nA. x"


@pytest.mark.smoke
def test_stub_generate_returns_answer_line_and_is_deterministic():
    item = {"id": "q1", "choices": ["red", "blue", "green", "black"], "correct_choice_idx": 2}
    out = stub_generate("any prompt", item)
    assert out.splitlines()[0].startswith("Answer: ")
    assert out == stub_generate("any prompt", item)  # deterministic
```

- [ ] **Step 2: Run to verify it fails**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest projects/04-vqa-aokvqa/tests/test_pipeline_smoke.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named '_prompts'`.

- [ ] **Step 3: Implement `_prompts.py`**

```python
"""Pure prompt-formatting helpers (no models, no I/O)."""

from __future__ import annotations

_LETTERS = "ABCD"


def format_choices(choices: list[str]) -> str:
    """Render choices as a lettered block: ``A. <c0>\\nB. <c1>`` ..."""
    return "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))


def render(template: str, **kwargs: str) -> str:
    """Fill a prompt template's ``{name}`` placeholders, stripping trailing whitespace."""
    return template.format(**kwargs).strip()
```

- [ ] **Step 4: Implement `_stub.py`**

```python
"""Deterministic stand-in generator for smoke tests (no real models)."""

from __future__ import annotations

_LETTERS = "ABCD"


def stub_generate(prompt: str, item: dict) -> str:
    """Return a canned ``Answer: <letter>`` + explanation string for an item.

    Picks the item's gold choice so smoke runs produce a sensible, deterministic
    signal; the explanation restates the chosen choice text (exercises the leakage
    flag path).

    Args:
        prompt: The (ignored) prompt text; present to match the real generator's
            ``(prompt, item) -> str`` interface.
        item: Dict with ``choices`` and ``correct_choice_idx``.

    Returns:
        A two-line string ``"Answer: <L>\\nBecause it is <choice text>."``.
    """
    idx = int(item["correct_choice_idx"])
    choice_text = item["choices"][idx]
    return f"Answer: {_LETTERS[idx]}\nBecause it is {choice_text}."
```

- [ ] **Step 5: Run to verify it passes**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest projects/04-vqa-aokvqa/tests/test_pipeline_smoke.py -v --no-cov`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add projects/04-vqa-aokvqa/scripts/_prompts.py projects/04-vqa-aokvqa/scripts/_stub.py projects/04-vqa-aokvqa/tests/test_pipeline_smoke.py
git commit -m "feat(p4): add prompt helpers and deterministic stub generator"
```

---

## Task 11: `_models.py` — lazy real-model loaders + generators

**Files:**
- Create: `projects/04-vqa-aokvqa/scripts/_models.py`

This module holds all heavy model code. It is imported by the scripts but every transformers/torch import is **inside a function**, so importing the scripts (and running smoke tests with the stub) downloads nothing. There are no unit tests here (covered by `slow` only); the smoke path never touches it.

- [ ] **Step 1: Implement `_models.py`**

```python
"""Lazy BLIP-2 / Qwen LM / Qwen-VL loaders and (prompt[, image]) -> text generators.

All heavy imports are inside functions so importing this module triggers no model
download. Exercised only by slow tests and the real run.
"""

from __future__ import annotations

from collections.abc import Callable


def _dtype(name: str):
    import torch

    return getattr(torch, name)


def load_blip2_captioner(model_id: str, gen_cfg: dict) -> Callable[[str], str]:
    """Return ``caption_fn(image_path) -> caption`` backed by BLIP-2."""
    import torch
    from PIL import Image
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    proc = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=_dtype(gen_cfg["torch_dtype"]), device_map=gen_cfg["device_map"]
    ).eval()

    @torch.no_grad()
    def caption_fn(image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = proc(images=image, return_tensors="pt").to(model.device, _dtype(gen_cfg["torch_dtype"]))
        out = model.generate(**inputs, do_sample=False, max_new_tokens=gen_cfg["max_new_tokens"])
        return proc.batch_decode(out, skip_special_tokens=True)[0].strip()

    return caption_fn


def load_qwen_lm(model_id: str, gen_cfg: dict) -> Callable[[str], str]:
    """Return ``generate(prompt) -> text`` backed by a Qwen2.5 instruct LM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=_dtype(gen_cfg["torch_dtype"]), device_map=gen_cfg["device_map"]
    ).eval()

    @torch.no_grad()
    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, do_sample=False, max_new_tokens=gen_cfg["max_new_tokens"])
        gen = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(gen, skip_special_tokens=True).strip()

    return generate


def load_qwen_vl(model_id: str, gen_cfg: dict) -> Callable[[str, str | None], str]:
    """Return ``generate(prompt, image_path|None) -> text`` backed by Qwen2.5-VL.

    When ``image_path`` is ``None`` the model is given a black tile (the vision
    ablation): same prompt structure, no real visual evidence.
    """
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=_dtype(gen_cfg["torch_dtype"]), device_map=gen_cfg["device_map"]
    ).eval()

    @torch.no_grad()
    def generate(prompt: str, image_path: str | None) -> str:
        image = Image.open(image_path).convert("RGB") if image_path else Image.new("RGB", (224, 224))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = proc(text=[text], images=image_inputs, videos=video_inputs,
                      padding=True, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, do_sample=False, max_new_tokens=gen_cfg["max_new_tokens"])
        gen = out[0][inputs["input_ids"].shape[1]:]
        return proc.decode(gen, skip_special_tokens=True).strip()

    return generate


def model_revisions(model_ids: dict[str, str]) -> dict[str, str]:
    """Resolve each model id to its current HF commit hash (best-effort)."""
    revisions = {}
    for key, mid in model_ids.items():
        try:
            from huggingface_hub import HfApi

            revisions[key] = HfApi().model_info(mid).sha
        except Exception:  # noqa: BLE001 -- revision logging must never break a run
            revisions[key] = "unknown"
    return revisions
```

- [ ] **Step 2: Verify the module imports without downloading anything**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run python -c "import sys; sys.path.insert(0, 'projects/04-vqa-aokvqa/scripts'); import _models; print('ok')"`
Expected: prints `ok` with no model download.

- [ ] **Step 3: Commit**

```bash
git add projects/04-vqa-aokvqa/scripts/_models.py
git commit -m "feat(p4): add lazy BLIP-2 / Qwen-LM / Qwen-VL loaders and generators"
```

---

## Task 12: `00_data.py` — prepare A-OKVQA (decode, leakage flag) + smoke test

**Files:**
- Create: `projects/04-vqa-aokvqa/scripts/00_data.py`
- Test: `projects/04-vqa-aokvqa/tests/test_data_smoke.py`

- [ ] **Step 1: Write the failing smoke test** (drives the pure prepare logic on synthetic rows; no `datasets`, no images)

```python
"""Smoke test for data preparation logic (no HF download, no real images)."""

import importlib

import pytest


@pytest.mark.smoke
def test_prepare_rows_adds_leakage_flag_and_keeps_fields():
    mod = importlib.import_module("00_data")
    raw = [
        {"id": "q1", "question": "color?", "choices": ["red", "blue", "green", "black"],
         "correct_choice_idx": 0, "rationales": ["it is clearly red", "warm tone"]},
        {"id": "q2", "question": "animal?", "choices": ["cat", "dog", "fox", "owl"],
         "correct_choice_idx": 1, "rationales": ["it barks", "a pet"]},
    ]
    rows = mod.prepare_rows(raw)
    assert rows[0]["leakage_flag"] is True   # "it is clearly red" leaks gold "red"
    assert rows[1]["leakage_flag"] is False  # no rationale contains "dog"
    assert rows[0]["image_path"].endswith("q1.jpg")
    assert set(rows[0]) >= {"id", "question", "choices", "correct_choice_idx",
                            "rationales", "leakage_flag", "image_path"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest projects/04-vqa-aokvqa/tests/test_data_smoke.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named '00_data'`.

- [ ] **Step 3: Implement `00_data.py`** (pure `prepare_rows` + a `slow`-only `main` that downloads/decodes)

```python
"""Prepare A-OKVQA validation: decode images, build leakage flag, write parquet.

Pure ``prepare_rows`` is smoke-tested; ``main`` (download + image decode) is slow.
"""

from __future__ import annotations

from awake.eval.vqa_consistency import rationale_leaks_answer

from _paths import IMAGES, PREPARED, ensure_dirs, load_config


def prepare_rows(raw: list[dict]) -> list[dict]:
    """Build prepared rows with a leakage flag and image_path (no I/O).

    Args:
        raw: Items with ``id, question, choices, correct_choice_idx, rationales``.

    Returns:
        Rows augmented with ``leakage_flag`` (any rationale leaks gold choice text)
        and ``image_path`` (``<IMAGES>/<id>.jpg``).
    """
    rows = []
    for r in raw:
        gold_text = r["choices"][int(r["correct_choice_idx"])]
        rows.append({
            "id": r["id"],
            "question": r["question"],
            "choices": list(r["choices"]),
            "correct_choice_idx": int(r["correct_choice_idx"]),
            "rationales": list(r.get("rationales", [])),
            "leakage_flag": rationale_leaks_answer(list(r.get("rationales", [])), gold_text),
            "image_path": str(IMAGES / f"{r['id']}.jpg"),
        })
    return rows


def main() -> None:  # pragma: no cover - slow path, exercised only in the real run
    """Download A-OKVQA, decode images idempotently, write prepared parquet."""
    import pandas as pd
    from datasets import load_dataset

    cfg = load_config("data")
    ensure_dirs(IMAGES, PREPARED)
    ds = load_dataset(cfg["dataset"], split=cfg["split"])
    if cfg.get("subset_n"):
        ds = ds.shuffle(seed=cfg["seed"]).select(range(cfg["subset_n"]))

    raw = []
    for ex in ds:
        ex_id = str(ex.get("question_id", ex.get("id")))
        img_path = IMAGES / f"{ex_id}.jpg"
        if not img_path.exists():  # idempotent: skip re-decode
            ex["image"].convert("RGB").save(img_path, format="JPEG")
        raw.append({
            "id": ex_id,
            "question": ex["question"],
            "choices": ex["choices"],
            "correct_choice_idx": ex["correct_choice_idx"],
            "rationales": ex.get("rationales", []),
        })

    rows = prepare_rows(raw)
    pd.DataFrame(rows).to_parquet(PREPARED / "val.parquet", index=False)
    print(f"prepared {len(rows)} items; leak={sum(r['leakage_flag'] for r in rows)}")


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest projects/04-vqa-aokvqa/tests/test_data_smoke.py -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add projects/04-vqa-aokvqa/scripts/00_data.py projects/04-vqa-aokvqa/tests/test_data_smoke.py
git commit -m "feat(p4): add 00_data prepare (leakage flag, idempotent decode)"
```

---

## Task 13: `10_run_pipelines.py` — generate answers + smoke test

**Files:**
- Create: `projects/04-vqa-aokvqa/scripts/10_run_pipelines.py`
- Modify: `projects/04-vqa-aokvqa/tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write the failing test** (drive both pipelines with stub callables; assert parquet-ready row shape)

```python
from _prompts import format_choices
from _stub import stub_generate


def _items():
    return [
        {"id": "q1", "question": "color?", "choices": ["red", "blue", "green", "black"],
         "correct_choice_idx": 0},
        {"id": "q2", "question": "animal?", "choices": ["cat", "dog", "fox", "owl"],
         "correct_choice_idx": 1},
    ]


@pytest.mark.smoke
def test_run_pipeline_b_rows_have_expected_fields():
    import importlib

    mod = importlib.import_module("10_run_pipelines")
    prompts = {"answer": "Q: {question}\n{choices_block}"}
    rows = mod.run_pipeline_b(_items(), vlm_generate=lambda p, img: stub_generate(p, _items()[0]),
                              prompts=prompts)
    r = rows[0]
    assert set(r) >= {"id", "answer_idx", "explanation", "raw_output", "parsed_by", "expl_leaks"}
    assert r["parsed_by"] == "strict"
    assert r["answer_idx"] == 0


@pytest.mark.smoke
def test_run_pipeline_a_includes_caption():
    import importlib

    mod = importlib.import_module("10_run_pipelines")
    items = _items()
    prompts = {"answer_with_caption": "{caption}\nQ: {question}\n{choices_block}"}
    rows = mod.run_pipeline_a(
        items,
        caption_fn=lambda path: "a description",
        llm_generate=lambda p: stub_generate(p, items[0]),
        prompts=prompts,
    )
    assert rows[0]["caption"] == "a description"
    assert format_choices(items[0]["choices"]).startswith("A. red")
```

- [ ] **Step 2: Run to verify it fails**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest projects/04-vqa-aokvqa/tests/test_pipeline_smoke.py -k run_pipeline -v --no-cov`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `10_run_pipelines.py`** (pure orchestration over injectable callables; `slow` `main` wires real models)

```python
"""Run both VQA pipelines and write per-item answer + explanation parquet.

The ``run_pipeline_*`` functions take injectable generate callables (real models in
``main``; stubs in smoke tests). The first line of output is treated as the answer
line and any following lines as the explanation.
"""

from __future__ import annotations

from collections.abc import Callable

from awake.eval.vqa_consistency import explanation_leaks_answer, extract_choice

from _paths import GEN, PREPARED, ensure_dirs, load_config
from _prompts import format_choices, render


def _split_answer_explanation(raw: str) -> tuple[str, str]:
    """Return (answer_line, explanation) from a raw generation."""
    lines = raw.splitlines()
    answer_line = lines[0] if lines else ""
    explanation = "\n".join(lines[1:]).strip()
    return answer_line, explanation


def _row(item: dict, raw: str, caption: str | None = None) -> dict:
    idx, parsed_by = extract_choice(raw, item["choices"])
    _, explanation = _split_answer_explanation(raw)
    chosen_text = item["choices"][idx] if idx is not None else ""
    row = {
        "id": item["id"],
        "answer_idx": idx,
        "explanation": explanation,
        "raw_output": raw,
        "parsed_by": parsed_by,
        "expl_leaks": explanation_leaks_answer(explanation, chosen_text),
    }
    if caption is not None:
        row["caption"] = caption
    return row


def run_pipeline_a(items: list[dict], caption_fn: Callable[[str], str],
                   llm_generate: Callable[[str], str], prompts: dict) -> list[dict]:
    """Caption-then-LLM: caption each image, then answer from (question, caption, choices)."""
    rows = []
    for item in items:
        caption = caption_fn(item.get("image_path", ""))
        prompt = render(prompts["answer_with_caption"], question=item["question"],
                        caption=caption, choices_block=format_choices(item["choices"]))
        rows.append(_row(item, llm_generate(prompt), caption=caption))
    return rows


def run_pipeline_b(items: list[dict], vlm_generate: Callable[[str, str | None], str],
                   prompts: dict) -> list[dict]:
    """Direct VLM: answer from (question, image, choices)."""
    rows = []
    for item in items:
        prompt = render(prompts["answer"], question=item["question"],
                        choices_block=format_choices(item["choices"]))
        rows.append(_row(item, vlm_generate(prompt, item.get("image_path"))))
    return rows


def main() -> None:  # pragma: no cover - slow path
    """Load each model in turn and write outputs/gen/{A,B,B7}.parquet."""
    import gc

    import pandas as pd
    import torch

    from _models import (
        load_blip2_captioner,
        load_qwen_lm,
        load_qwen_vl,
        model_revisions,
    )

    cfg = load_config("pipelines")
    gen_cfg, prompts = cfg["generation"], cfg["prompts"]["main"]
    ensure_dirs(GEN)
    items = pd.read_parquet(PREPARED / "val.parquet").to_dict("records")

    def _free():
        gc.collect()
        torch.cuda.empty_cache()

    # Pipeline A: BLIP-2 caption -> Qwen LM
    caption_fn = load_blip2_captioner(cfg["models"]["blip2"], gen_cfg)
    captions = {it["id"]: caption_fn(it["image_path"]) for it in items}
    del caption_fn
    _free()
    llm = load_qwen_lm(cfg["models"]["qwen_lm"], gen_cfg)
    rows_a = []  # captions already computed above; answer with cached captions
    for it in items:
        prompt = render(prompts["answer_with_caption"], question=it["question"],
                        caption=captions[it["id"]], choices_block=format_choices(it["choices"]))
        rows_a.append(_row(it, llm(prompt), caption=captions[it["id"]]))
    pd.DataFrame(rows_a).to_parquet(GEN / "A.parquet", index=False)
    del llm
    _free()

    # Pipeline B (3B) and B7 (7B)
    for key, mid in (("B", cfg["models"]["qwen_vl_3b"]), ("B7", cfg["models"]["qwen_vl_7b"])):
        vlm = load_qwen_vl(mid, gen_cfg)
        rows = run_pipeline_b(items, vlm_generate=vlm, prompts=prompts)
        pd.DataFrame(rows).to_parquet(GEN / f"{key}.parquet", index=False)
        del vlm
        _free()

    revs = model_revisions(cfg["models"])
    (GEN / "model_revisions.json").write_text(__import__("json").dumps(revs, indent=2))
    print("wrote A/B/B7 parquet")


if __name__ == "__main__":  # pragma: no cover
    main()
```

(Implementer note: the `if False else None` then re-loop in `main` is awkward — replace it with the clean cached-caption loop shown directly below it. The smoke-tested `run_pipeline_a/b` are the audited paths; `main` is `slow`-only and may be tidied as long as it loads one model at a time and writes the three parquet files.)

- [ ] **Step 4: Run to verify it passes**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest projects/04-vqa-aokvqa/tests/test_pipeline_smoke.py -k run_pipeline -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add projects/04-vqa-aokvqa/scripts/10_run_pipelines.py projects/04-vqa-aokvqa/tests/test_pipeline_smoke.py
git commit -m "feat(p4): add 10_run_pipelines (injectable generators, A/B/B7)"
```

---

## Task 14: `20_probe.py` — two ablation arms + smoke test

**Files:**
- Create: `projects/04-vqa-aokvqa/scripts/20_probe.py`
- Modify: `projects/04-vqa-aokvqa/tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.smoke
def test_ablate_arms_differ_only_by_explanation():
    import importlib

    mod = importlib.import_module("20_probe")
    items = _items()
    gen_rows = [{"id": "q1", "answer_idx": 0, "explanation": "because red"},
                {"id": "q2", "answer_idx": 1, "explanation": "because dog"}]
    prompts = {
        "ablate_with_expl": "prev: {explanation}\nQ: {question}\n{choices_block}",
        "ablate_no_expl": "Q: {question}\n{choices_block}",
    }
    seen = {}

    def vlm_generate(prompt, image_path):
        seen["with_expl"] = "prev:" in prompt
        return stub_generate(prompt, items[0])

    rows_with = mod.ablate_pipeline_b(items, gen_rows, vlm_generate, prompts, with_expl=True)
    assert seen["with_expl"] is True
    rows_without = mod.ablate_pipeline_b(items, gen_rows, vlm_generate, prompts, with_expl=False)
    assert seen["with_expl"] is False
    assert {"id", "ablated_idx", "parsed_by"} <= set(rows_with[0])
```

- [ ] **Step 2: Run to verify it fails**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest projects/04-vqa-aokvqa/tests/test_pipeline_smoke.py -k ablate -v --no-cov`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `20_probe.py`**

```python
"""Vision-ablation probe: re-answer with the image/caption removed, two arms.

with_expl arm includes the model's own prior explanation; the no_expl baseline arm
is identical except it omits that explanation. Pipeline A ablates by replacing the
caption with the null-caption string; Pipeline B ablates by passing image_path=None
(the loader substitutes a black tile).
"""

from __future__ import annotations

from collections.abc import Callable

from awake.eval.vqa_consistency import extract_choice

from _paths import GEN, PREPARED, ensure_dirs, load_config
from _prompts import format_choices, render


def _expl_by_id(gen_rows: list[dict]) -> dict[str, str]:
    return {r["id"]: r.get("explanation", "") for r in gen_rows}


def _ablated_row(item: dict, raw: str) -> dict:
    idx, parsed_by = extract_choice(raw, item["choices"])
    return {"id": item["id"], "ablated_idx": idx, "parsed_by": parsed_by, "raw_output": raw}


def _prompt_for(item: dict, expl: str, prompts: dict, with_expl: bool, caption: str | None) -> str:
    key = "ablate_with_expl" if with_expl else "ablate_no_expl"
    kwargs = {"question": item["question"], "choices_block": format_choices(item["choices"])}
    if with_expl:
        kwargs["explanation"] = expl
    if caption is not None:
        kwargs["caption"] = caption
    return render(prompts[key], **kwargs)


def ablate_pipeline_b(items: list[dict], gen_rows: list[dict],
                      vlm_generate: Callable[[str, str | None], str], prompts: dict,
                      with_expl: bool) -> list[dict]:
    """Re-answer Pipeline B with a black tile (image_path=None)."""
    expl = _expl_by_id(gen_rows)
    rows = []
    for item in items:
        prompt = _prompt_for(item, expl.get(item["id"], ""), prompts, with_expl, caption=None)
        rows.append(_ablated_row(item, vlm_generate(prompt, None)))
    return rows


def ablate_pipeline_a(items: list[dict], gen_rows: list[dict],
                      llm_generate: Callable[[str], str], prompts: dict, with_expl: bool,
                      null_caption: str) -> list[dict]:
    """Re-answer Pipeline A with the caption replaced by the null-caption string."""
    expl = _expl_by_id(gen_rows)
    rows = []
    for item in items:
        prompt = _prompt_for(item, expl.get(item["id"], ""), prompts, with_expl,
                             caption=null_caption)
        rows.append(_ablated_row(item, llm_generate(prompt)))
    return rows


def main() -> None:  # pragma: no cover - slow path
    """Run both ablation arms for A/B/B7 and write *_ablated_{expl,noexpl}.parquet."""
    import gc

    import pandas as pd
    import torch

    from _models import load_qwen_lm, load_qwen_vl

    cfg = load_config("pipelines")
    gen_cfg, prompts = cfg["generation"], cfg["prompts"]["main"]
    null_caption = cfg["null_caption"]
    ensure_dirs(GEN)
    items = pd.read_parquet(PREPARED / "val.parquet").to_dict("records")

    def _free():
        gc.collect()
        torch.cuda.empty_cache()

    # Pipeline A ablation (LLM only; no image needed)
    gen_a = pd.read_parquet(GEN / "A.parquet").to_dict("records")
    llm = load_qwen_lm(cfg["models"]["qwen_lm"], gen_cfg)
    for with_expl, tag in ((True, "expl"), (False, "noexpl")):
        rows = ablate_pipeline_a(items, gen_a, llm, prompts, with_expl, null_caption)
        pd.DataFrame(rows).to_parquet(GEN / f"A_ablated_{tag}.parquet", index=False)
    del llm
    _free()

    for key, mid in (("B", cfg["models"]["qwen_vl_3b"]), ("B7", cfg["models"]["qwen_vl_7b"])):
        gen = pd.read_parquet(GEN / f"{key}.parquet").to_dict("records")
        vlm = load_qwen_vl(mid, gen_cfg)
        for with_expl, tag in ((True, "expl"), (False, "noexpl")):
            rows = ablate_pipeline_b(items, gen, vlm, prompts, with_expl)
            pd.DataFrame(rows).to_parquet(GEN / f"{key}_ablated_{tag}.parquet", index=False)
        del vlm
        _free()
    print("wrote ablation arms")


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest projects/04-vqa-aokvqa/tests/test_pipeline_smoke.py -k ablate -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add projects/04-vqa-aokvqa/scripts/20_probe.py projects/04-vqa-aokvqa/tests/test_pipeline_smoke.py
git commit -m "feat(p4): add 20_probe (paired with/without-explanation ablation arms)"
```

---

## Task 15: `30_eval.py` — metrics.json + hero figure + smoke test

**Files:**
- Create: `projects/04-vqa-aokvqa/scripts/30_eval.py`
- Test: `projects/04-vqa-aokvqa/tests/test_eval_smoke.py`

- [ ] **Step 1: Write the failing test** (drives the pure `compute_subset_metrics` aggregator on hand-built rows)

```python
"""Smoke test for the eval aggregator (pure; no plotting, no files)."""

import importlib

import pytest


@pytest.mark.smoke
def test_compute_subset_metrics_shapes_and_delta():
    mod = importlib.import_module("30_eval")
    gold = [0, 1, 2, 3]
    gen = {
        "A": {"answer_idx": [0, 1, 2, 0], "expl_leaks": [True, False, False, False],
              "parsed_by": ["strict"] * 4},
        "B": {"answer_idx": [0, 1, 2, 3], "expl_leaks": [False] * 4,
              "parsed_by": ["strict"] * 4},
        "B7": {"answer_idx": [0, 1, 2, 3], "expl_leaks": [False] * 4,
               "parsed_by": ["strict"] * 4},
    }
    abl = {  # ablated answers per pipeline per arm
        "A": {"expl": [0, 1, 2, 0], "noexpl": [0, 0, 0, 0]},
        "B": {"expl": [0, 1, 2, 3], "noexpl": [0, 1, 2, 3]},
        "B7": {"expl": [0, 1, 2, 3], "noexpl": [0, 1, 2, 3]},
    }
    m = mod.compute_subset_metrics(gen, abl, gold, n_resamples=200, seed=0)
    assert set(m["pipelines"]) == {"A", "B", "B7"}
    a = m["pipelines"]["A"]
    assert 0.0 <= a["accuracy"] <= 1.0
    assert set(a["consistency"]) >= {"with_expl", "no_expl", "delta", "delta_ci"}
    assert set(a["parse_rate"]) == {"answer", "abl_expl", "abl_noexpl"}
    assert set(m["divergence"]) == {"A_vs_B", "A_vs_B7", "B_vs_B7"}
    assert "contingency" in m["divergence"]["A_vs_B"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest projects/04-vqa-aokvqa/tests/test_eval_smoke.py -v --no-cov`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `30_eval.py`**

```python
"""Aggregate generation + ablation outputs into metrics.json and the hero figure.

``compute_subset_metrics`` is pure (lists in, dict out) and smoke-tested; ``main``
reads parquet, splits filtered/unfiltered, writes metrics.json, and renders the
multi-panel hero figure.
"""

from __future__ import annotations

import numpy as np

from awake.eval.bootstrap import bootstrap_ci, paired_diff_test
from awake.eval.vqa_consistency import (
    accuracy,
    consistency_rate,
    parse_rate,
    pipeline_divergence,
)

PIPELINES = ("A", "B", "B7")
PAIRS = (("A", "B"), ("A", "B7"), ("B", "B7"))


def _consistency_vector(original: list, ablated: list) -> np.ndarray:
    """Per-item 1.0/0.0 consistency (primary policy) for bootstrapping the delta."""
    return np.array(
        [float(o is not None and a is not None and o == a) for o, a in zip(original, ablated)]
    )


def compute_subset_metrics(gen: dict, abl: dict, gold: list[int],
                           n_resamples: int = 2000, seed: int = 0) -> dict:
    """Compute per-pipeline accuracy/parse/consistency-delta + pairwise divergence.

    Args:
        gen: ``{pipeline: {"answer_idx": [...], "expl_leaks": [...], "parsed_by": [...]}}``.
        abl: ``{pipeline: {"expl": [...], "noexpl": [...]}}`` ablated answer indices.
        gold: Per-item gold indices.
        n_resamples: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        ``{"pipelines": {...}, "divergence": {pair: {...}}}``.
    """
    pipelines = {}
    for p in PIPELINES:
        ans = gen[p]["answer_idx"]
        with_e = consistency_rate(ans, abl[p]["expl"])
        no_e = consistency_rate(ans, abl[p]["noexpl"])
        delta_test = paired_diff_test(
            _consistency_vector(ans, abl[p]["expl"]),
            _consistency_vector(ans, abl[p]["noexpl"]),
            n_resamples=n_resamples, seed=seed,
        )
        pipelines[p] = {
            "accuracy": accuracy(ans, gold),
            "expl_leak_rate": float(np.mean(gen[p]["expl_leaks"])) if gen[p]["expl_leaks"] else 0.0,
            "parse_rate": {
                "answer": parse_rate(ans),
                "abl_expl": parse_rate(abl[p]["expl"]),
                "abl_noexpl": parse_rate(abl[p]["noexpl"]),
            },
            "consistency": {
                "with_expl": with_e,
                "no_expl": no_e,
                "delta": with_e - no_e,
                "delta_ci": [delta_test["ci_low"], delta_test["ci_high"]],
            },
        }

    divergence = {}
    for x, y in PAIRS:
        div = pipeline_divergence(gen[x]["answer_idx"], gen[y]["answer_idx"], gold)
        agree_vec = np.array([
            float(ax is not None and ay is not None and ax == ay)
            for ax, ay in zip(gen[x]["answer_idx"], gen[y]["answer_idx"])
        ])
        lo, _, hi = bootstrap_ci(1.0 - agree_vec, n_resamples=n_resamples, seed=seed)
        divergence[f"{x}_vs_{y}"] = {
            "overall": div["overall"],
            "overall_ci": [lo, hi],
            "contingency": div["contingency"],
        }
    return {"pipelines": pipelines, "divergence": divergence}


def _gen_dict(df) -> dict:
    return {"answer_idx": _none_list(df["answer_idx"]),
            "expl_leaks": list(df["expl_leaks"]),
            "parsed_by": list(df["parsed_by"])}


def _none_list(series) -> list:
    """pandas reads missing ints as NaN; map back to None."""
    import math

    return [None if (isinstance(v, float) and math.isnan(v)) else int(v) for v in series]


def main() -> None:  # pragma: no cover - slow/real-run path
    """Read parquet, compute filtered/unfiltered metrics, write metrics.json + hero.png."""
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from _models import model_revisions
    from _paths import ASSETS, GEN, OUTPUTS, PREPARED, ensure_dirs, load_config

    ensure_dirs(ASSETS, OUTPUTS)
    prep = pd.read_parquet(PREPARED / "val.parquet")
    cfg = load_config("pipelines")

    def _subset(mask) -> dict:
        gold = list(prep.loc[mask, "correct_choice_idx"].astype(int))
        ids = set(prep.loc[mask, "id"])
        gen, abl = {}, {}
        for p in PIPELINES:
            g = pd.read_parquet(GEN / f"{p}.parquet")
            g = g[g["id"].isin(ids)]
            gen[p] = _gen_dict(g)
            abl[p] = {}
            for arm, tag in (("expl", "expl"), ("noexpl", "noexpl")):
                a = pd.read_parquet(GEN / f"{p}_ablated_{tag}.parquet")
                a = a[a["id"].isin(ids)]
                abl[p][arm] = _none_list(a["ablated_idx"])
        return compute_subset_metrics(gen, abl, gold)

    metrics = {
        "split": load_config("data")["split"],
        "n": int(len(prep)),
        "n_filtered": int((~prep["leakage_flag"]).sum()),
        "prompt_variant": "main",
        "model_revisions": model_revisions(cfg["models"]),
        "b7_completed": (GEN / "B7.parquet").exists(),
        "subsets": {
            "unfiltered": _subset(prep["id"].notna()),
            "filtered": _subset(~prep["leakage_flag"]),
        },
    }
    (OUTPUTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Hero figure: 3 panels (delta, accuracy, parse_rate) over unfiltered pipelines.
    u = metrics["subsets"]["unfiltered"]["pipelines"]
    labels = list(PIPELINES)
    deltas = [u[p]["consistency"]["delta"] for p in labels]
    accs = [u[p]["accuracy"] for p in labels]
    parses = [u[p]["parse_rate"]["answer"] for p in labels]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(labels, deltas); axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title("self-rationale recoverability gain (Delta)")
    axes[1].bar(labels, accs); axes[1].set_title("accuracy"); axes[1].set_ylim(0, 1)
    axes[2].bar(labels, parses); axes[2].set_title("parse rate"); axes[2].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(ASSETS / "hero.png", dpi=150)
    print("wrote metrics.json + hero.png")


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest projects/04-vqa-aokvqa/tests/test_eval_smoke.py -v --no-cov`
Expected: PASS.

- [ ] **Step 5: Run the full project smoke + unit suite**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests projects/04-vqa-aokvqa -m "unit or smoke" --no-cov`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add projects/04-vqa-aokvqa/scripts/30_eval.py projects/04-vqa-aokvqa/tests/test_eval_smoke.py
git commit -m "feat(p4): add 30_eval (metrics.json pairs + multi-panel hero figure)"
```

---

## Task 16: Docs — ADR, REPORT.md, README, CHANGELOG, notebook scaffold

**Files:**
- Create: `docs/decisions/004-vqa-pipelines-and-vision-ablation.md`, `projects/04-vqa-aokvqa/REPORT.md`, `projects/04-vqa-aokvqa/notebooks/01-vqa-consistency.py`
- Modify: `projects/04-vqa-aokvqa/README.md`, `CHANGELOG.md`

- [ ] **Step 1: Write the ADR** `docs/decisions/004-vqa-pipelines-and-vision-ablation.md`

Mirror the style of `003-hateful-memes-licence-and-modality-shapley.md`. Cover: the paired-baseline probe and why raw consistency is mislabelled without the no-explanation arm (the `Delta` headline); the Pipeline-A null-caption ablation (black-tile captioning is secondary only); the narrowed A-vs-B claim with the required size-matched B7 arm bounding the parameter-count confound; strict-then-text parsing + the None-as-wrong / paired-only denominator policy; determinism (`do_sample=False`, logged model revisions) and one-model-resident memory discipline. Status: Accepted, 2026-05-26.

- [ ] **Step 2: Write `REPORT.md`** with a results section whose numbers are placeholders **clearly labelled "filled by the real run"** (no fabricated values), and a Limitations section copied from spec §9. The hero figure is referenced as `assets/hero.png`.

- [ ] **Step 3: Write the notebook** `notebooks/01-vqa-consistency.py` as a jupytext percent-format script that: resolves the project root via `os.environ["P4_PROJECT_ROOT"]`, loads `outputs/metrics.json`, renders the three-panel summary inline, and prints the headline `Delta` per pipeline with CIs. It must NOT dump raw dataset images/text. (It will be executed and committed WITH outputs in Task 18.)

- [ ] **Step 4: Update `README.md`** — replace the stub with: one-paragraph question, the two-pipeline + B7 design, how to run (`uv run python projects/04-vqa-aokvqa/scripts/00_data.py` ... `30_eval.py`), and a pointer to REPORT.md + the ADR.

- [ ] **Step 5: Update `CHANGELOG.md`** — add under `[Unreleased]`:

```
### Added
- Project 4 (`04-vqa-aokvqa`): caption-then-LLM vs direct-VLM A-OKVQA pipelines with a
  paired-baseline vision-ablation probe (consistency gain Delta), inter-pipeline divergence,
  and the pure `awake.eval.vqa_consistency` metric core.
```

- [ ] **Step 6: Verify markdown + commit**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run python -c "import pathlib; print('docs ok')"`

```bash
git add docs/decisions/004-vqa-pipelines-and-vision-ablation.md projects/04-vqa-aokvqa/REPORT.md projects/04-vqa-aokvqa/notebooks/ projects/04-vqa-aokvqa/README.md CHANGELOG.md
git commit -m "docs(p4): add ADR-004, REPORT, README, CHANGELOG, notebook scaffold"
```

---

## Task 17: Full green-CI gate (lint + types + unit + smoke)

**Files:** none (verification + fixups only)

- [ ] **Step 1: Lint**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run ruff check src/awake/eval/vqa_consistency.py projects/04-vqa-aokvqa/ && uv run ruff format --check src/awake/eval/vqa_consistency.py projects/04-vqa-aokvqa/scripts/`
Expected: PASS. Fix any RUF002/003 (non-ASCII) or docstring issues, then re-run.

- [ ] **Step 2: Full CI command**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests projects apps -m "unit or smoke"`
Expected: PASS with `awake` coverage ≥90%.

- [ ] **Step 3: Commit any fixups**

```bash
git add -A && git commit -m "chore(p4): satisfy lint + coverage gates" || echo "nothing to fix"
```

---

## Task 18: Real run (slow; GPU) — produce real metrics, figure, notebook

**Files:** writes `outputs/` (gitignored) + commits `assets/hero.png`, notebook with outputs, REPORT.md numbers.

> This is the PLAYBOOK §1 real run. Headline = **full `validation`, all three arms {A, B, B7}** (spec §10). No fabricated numbers. If B7 cannot finish, record `b7_completed: false` and report the size confound as unmitigated (do not drop it silently).

- [ ] **Step 1: Install the vqa extra**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv sync --extra vqa` (or `uv pip install -e ".[vqa]"`).

- [ ] **Step 2: Prepare data**

Run: `export PATH="$HOME/.local/bin:$PATH" && P4_PROJECT_ROOT=$PWD/projects/04-vqa-aokvqa uv run python projects/04-vqa-aokvqa/scripts/00_data.py`
Expected: prints `prepared N items; leak=K`; `outputs/prepared/val.parquet` + `outputs/images/*.jpg` exist.

- [ ] **Step 3: Run pipelines** (one model resident at a time; expect a long wall-clock)

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run python projects/04-vqa-aokvqa/scripts/10_run_pipelines.py`
Expected: `outputs/gen/{A,B,B7}.parquet` + `model_revisions.json`. Watch `nvidia-smi` to confirm only one large model resident; if OOM, lower nothing in code — the unload discipline should suffice on 24 GB.

- [ ] **Step 4: Run the probe**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run python projects/04-vqa-aokvqa/scripts/20_probe.py`
Expected: `outputs/gen/{A,B,B7}_ablated_{expl,noexpl}.parquet`.

- [ ] **Step 5: Evaluate + figure**

Run: `export PATH="$HOME/.local/bin:$PATH" && uv run python projects/04-vqa-aokvqa/scripts/30_eval.py`
Expected: `outputs/metrics.json` + `assets/hero.png`. Sanity-check: parse_rate not pathologically low; `Delta` and CIs present for A/B/B7; divergence pairs present.

- [ ] **Step 6: Execute the notebook with outputs**

Run: `export PATH="$HOME/.local/bin:$PATH" && P4_PROJECT_ROOT=$PWD/projects/04-vqa-aokvqa uv run jupytext --to notebook --execute projects/04-vqa-aokvqa/notebooks/01-vqa-consistency.py`
Expected: `01-vqa-consistency.ipynb` produced WITH outputs (committed; no raw dataset dumps).

- [ ] **Step 7: Fill REPORT.md with the real numbers** (from `metrics.json`), interpret `Delta` (positive = self-rationale recoverability red flag; ≈0 = language priors; <0 = noise), and state whether B7 completed.

- [ ] **Step 8: Commit the real artifacts**

```bash
git add projects/04-vqa-aokvqa/assets/hero.png projects/04-vqa-aokvqa/notebooks/01-vqa-consistency.ipynb projects/04-vqa-aokvqa/REPORT.md
git commit -m "feat(p4): real A-OKVQA run -- metrics, hero figure, executed notebook, report"
```

---

## Task 19: Final review + finish branch

**Files:** none (review + merge)

- [ ] **Step 1: Dispatch a final code reviewer** over the whole P4 diff (per subagent-driven-development): spec compliance against the v2.1 spec + code quality. Fix any blocking findings.
- [ ] **Step 2: Confirm green CI** one last time: `export PATH="$HOME/.local/bin:$PATH" && uv run pytest tests projects apps -m "unit or smoke"`.
- [ ] **Step 3: Use superpowers:finishing-a-development-branch** to merge `p4-vqa-aokvqa` into `main` and push. (No HF Space to deploy for P4.)

---

## Self-review notes (author)

- **Spec coverage:** §1 deliverables → Tasks 2-8 (core) + 15 (metrics/figure); §2 data + leakage → Task 12; §3 pipelines + B7 + determinism + parser → Tasks 9,11,13; §4 two-arm probe + arm symmetry → Task 14; §5 pure-core signatures → Tasks 2-8 (note `extract_choice` returns `(idx, parsed_by)`, `consistency_rate(paired_only=...)`); §6 metrics schema + pair keys + per-arm parse_rate + leak rate + hero → Task 15; §7 tests (unit/smoke/slow) → throughout; §8 deps + ADR + CHANGELOG → Tasks 1,16; §9 limitations → Task 16 REPORT; §10 real-run headline policy → Task 18; §11 layout → all.
- **Type consistency:** `extract_choice -> (int|None, str)`; callers in `10_run_pipelines`/`20_probe` unpack `(idx, parsed_by)`. `consistency_rate(original, ablated, paired_only=False)`. `pipeline_divergence` returns `{"overall", "contingency"}` with the four named cells. `compute_subset_metrics(gen, abl, gold, ...)` consumes those exact shapes.
- **No placeholders:** every code step shows complete code except Task 16 docs (prose deliverables) and Task 18 (numbers come from the real run — deliberately not fabricated, per PLAYBOOK §1).
- **`main` paths are slow-only:** the `main()` orchestrators (Tasks 12-15) are `# pragma: no cover` and never run in CI; the audited, smoke-tested paths are the pure `run_pipeline_a/b`, `ablate_pipeline_a/b`, `prepare_rows`, and `compute_subset_metrics` functions.
