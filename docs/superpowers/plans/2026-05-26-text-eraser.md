# Text-Eraser (Project 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `projects/02-text-eraser`: fine-tune DeBERTa-v3-base on ERASER Movies, run four explainers + a random baseline, and score each on ERASER-exact faithfulness and plausibility, producing a faithfulness-vs-plausibility scatter, `metrics.json`, a notebook, and a REPORT.

**Architecture:** Pure, tokenizer-agnostic metric + erasure code lives in `src/awake/eval/` (carries the 90% coverage floor via fast unit tests on toy `predict_fn`s). Explainer adapters live in `src/awake/eval/explainers/` (so they too count toward coverage). Project-local numbered scripts in `projects/02-text-eraser/scripts/` orchestrate data → train → explain → eval; all heavy/GPU work is marked `slow` and excluded from CI, while CPU smoke tests exercise the plumbing on a tiny stub model.

**Tech Stack:** Python 3.11, uv, just, pytest (markers `unit`/`smoke`/`slow`), transformers, torch, captum, lime, shap (optional extra), numpy, pandas/pyarrow, matplotlib, scikit-learn.

**Reference spec:** `docs/superpowers/specs/2026-05-26-text-eraser-design.md`. Read it before starting.

---

## Conventions for every task

- Run commands from the **repo root** unless stated otherwise.
- Unit tests for `src/awake/` live in the **repo-root `tests/`** dir (so `--cov=awake` counts them); project plumbing tests live in `projects/02-text-eraser/tests/`.
- Test markers: `@pytest.mark.unit` (fast, pure), `@pytest.mark.smoke` (tiny stub model, CPU, < CI budget), `@pytest.mark.slow` (real DeBERTa/GPU, excluded from CI).
- Run the awake unit suite with coverage: `uv run pytest tests -m unit`.
- Run a single test: `uv run pytest tests/test_x.py::test_name -v --no-cov`.
- Commit after every task with the message shown.
- `seed_everything` already exists: `from awake.utils.seeding import seed_everything` (seeds random+numpy+PYTHONHASHSEED, returns the seed; seed torch separately).

---

## File structure (created by this plan)

```
src/awake/eval/
  __init__.py                 # re-exports public API
  attribution.py              # TokenAttribution dataclass, Explainer + ModelAdapter Protocols
  erasure.py                  # erase() — mask-replacement, length-preserving
  faithfulness.py             # comprehensiveness, sufficiency, aopc (predicted class, k_d budget)
  plausibility.py             # subword→word aggregation, P/R/F1, AUPRC, token_iou, mask clipping
  bootstrap.py                # paired bootstrap CIs + pairwise paired-diff tests (Bonferroni)
  explainers/
    __init__.py
    random_baseline.py        # RandomExplainer (floor)
    gradient_x_input.py       # Gradient×Input (replaces attention rollout)
    integrated_gradients.py   # captum LayerIntegratedGradients
    lime_text.py              # LIME (whitespace-level, identity alignment)
    shap_partition.py         # SHAP PartitionExplainer (optional extra)

tests/                        # repo-root: awake unit tests (counted by --cov=awake)
  test_eval_attribution.py
  test_eval_erasure.py
  test_eval_faithfulness.py
  test_eval_plausibility.py
  test_eval_bootstrap.py
  test_eval_explainers.py

projects/02-text-eraser/
  justfile
  configs/{data,model,explainers}.yaml
  scripts/
    __init__.py
    _paths.py                 # path + config helpers
    _model_adapter.py         # HFModelAdapter implementing ModelAdapter
    _explainers.py            # config → adapter → explainer wiring (thin driver)
    _stub_model.py            # tiny HF model for smoke tests
    00_fetch_data.py
    01_prepare.py
    10_train.py
    20_explain.py
    30_eval.py
  tests/
    conftest.py
    test_prepare_smoke.py
    test_train_smoke.py
    test_explain_eval_smoke.py
  notebooks/01-explainer-comparison.py
  REPORT.md
  README.md                   # already exists; updated in final task

docs/decisions/002-eraser-data-and-truncation.md
```

---

## Phase 0 — Scaffold and dependencies

### Task 1: Add dependencies and verify the uv resolve

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`

- [ ] **Step 1: Add runtime deps + optional shap extra to `pyproject.toml`.**

In `[project].dependencies`, append:
```toml
  "transformers>=4.44",
  "torch>=2.2",
  "datasets>=2.20",
  "captum>=0.7",
  "lime>=0.2.0.1",
  "scikit-learn>=1.4",
```
Add a new optional-dependencies group (shap drags numba/llvmlite — keep it out of the default resolve, mirroring P1's `interpret`→`interpret-core` lesson):
```toml
[project.optional-dependencies]
explain-shap = ["shap>=0.46"]
```

- [ ] **Step 2: Ignore project outputs.** Append to `.gitignore`:
```
projects/02-text-eraser/outputs/
```

- [ ] **Step 3: Verify the default resolve has NO numba/llvmlite.**

Run: `uv lock && uv sync --all-extras=false --extra dev`
Then: `uv pip list | grep -iE "numba|llvmlite" || echo "OK: no numba in default resolve"`
Expected: `OK: no numba in default resolve`. If numba appears, stop — a default dep pulled it; investigate before proceeding.

- [ ] **Step 4: Verify the shap extra resolves at all (may pull numba; that's why it's opt-in).**

Run: `uv sync --extra dev --extra explain-shap`
Expected: resolves successfully. Record the outcome for ADR 002 (Task 22).

- [ ] **Step 5: Commit.**
```bash
git add pyproject.toml uv.lock .gitignore
git commit -m "P2: add transformers/torch/captum/lime deps + optional shap extra"
```

---

## Phase 1 — Shared eval core (`src/awake/eval/`), strict TDD

### Task 2: `TokenAttribution` + `Explainer`/`ModelAdapter` protocols

**Files:**
- Create: `src/awake/eval/attribution.py`
- Test: `tests/test_eval_attribution.py`

- [ ] **Step 1: Write the failing test.**
```python
# tests/test_eval_attribution.py
import numpy as np
import pytest
from awake.eval.attribution import TokenAttribution


@pytest.mark.unit
def test_token_attribution_validates_lengths():
    attr = TokenAttribution(
        tokens=["a", "b"],
        offsets=[(0, 1), (2, 3)],
        scores=np.array([0.1, 0.9]),
        visible_mask=np.array([True, True]),
        predicted_class=1,
        class_scores=np.array([0.3, 0.7]),
    )
    assert attr.n_tokens == 2
    assert attr.predicted_class == 1


@pytest.mark.unit
def test_token_attribution_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        TokenAttribution(
            tokens=["a", "b"],
            offsets=[(0, 1)],
            scores=np.array([0.1, 0.9]),
            visible_mask=np.array([True, True]),
            predicted_class=0,
            class_scores=np.array([0.5, 0.5]),
        )
```

- [ ] **Step 2: Run to verify failure.** `uv run pytest tests/test_eval_attribution.py -v --no-cov` → FAIL (module not found).

- [ ] **Step 3: Implement.**
```python
# src/awake/eval/attribution.py
"""Core attribution datatype and explainer/model protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class TokenAttribution:
    """Per-token importance scores for one model-visible sequence.

    Attributes:
        tokens: Subword (or whitespace, for LIME) token strings.
        offsets: Character ``(start, end)`` spans into the original text.
        scores: Per-token importance toward the predicted class.
        visible_mask: True for real tokens, False for special/pad tokens.
        predicted_class: The class index the metrics are scored against.
        class_scores: Full predicted probability vector for the input.
    """

    tokens: list[str]
    offsets: list[tuple[int, int]]
    scores: np.ndarray
    visible_mask: np.ndarray
    predicted_class: int
    class_scores: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.tokens)
        if not (len(self.offsets) == n == len(self.scores) == len(self.visible_mask)):
            raise ValueError(
                f"length mismatch: tokens={n}, offsets={len(self.offsets)}, "
                f"scores={len(self.scores)}, visible_mask={len(self.visible_mask)}"
            )

    @property
    def n_tokens(self) -> int:
        """Number of tokens in this attribution."""
        return len(self.tokens)


@runtime_checkable
class ModelAdapter(Protocol):
    """Wraps a fine-tuned classifier over its frozen visible tokenization."""

    def predict_proba(self, token_ids_batch: np.ndarray) -> np.ndarray:
        """Return ``(batch, n_classes)`` probabilities for token-id rows."""
        ...


@runtime_checkable
class Explainer(Protocol):
    """Produces a :class:`TokenAttribution` for one prepared example."""

    name: str

    def attribute(self, example: dict) -> TokenAttribution:
        """Explain one example dict (keys: ``input_ids``, ``text``, ...)."""
        ...
```

- [ ] **Step 4: Run to verify pass.** `uv run pytest tests/test_eval_attribution.py -v --no-cov` → PASS.

- [ ] **Step 5: Commit.**
```bash
git add src/awake/eval/attribution.py tests/test_eval_attribution.py
git commit -m "P2: TokenAttribution dataclass + Explainer/ModelAdapter protocols"
```

### Task 3: `erase()` — mask-replacement, length-preserving

**Files:**
- Create: `src/awake/eval/erasure.py`
- Test: `tests/test_eval_erasure.py`

- [ ] **Step 1: Write the failing test.**
```python
# tests/test_eval_erasure.py
import numpy as np
import pytest
from awake.eval.erasure import erase, top_k_mask


@pytest.mark.unit
def test_erase_replaces_dropped_positions_and_preserves_length():
    ids = np.array([5, 6, 7, 8])
    keep = np.array([True, False, True, False])
    out = erase(ids, keep_mask=keep, mask_token_id=99)
    assert out.shape == ids.shape
    assert out.tolist() == [5, 99, 7, 99]


@pytest.mark.unit
def test_top_k_mask_selects_highest_scores_among_visible():
    scores = np.array([0.1, 0.9, 0.5, 0.3])
    visible = np.array([True, True, True, False])
    # k_fraction 0.5 of 3 visible -> ceil(1.5)=2 -> indices 1 and 2
    mask = top_k_mask(scores, visible_mask=visible, k_fraction=0.5)
    assert mask.tolist() == [False, True, True, False]
```

- [ ] **Step 2: Run to verify failure.** `uv run pytest tests/test_eval_erasure.py -v --no-cov` → FAIL.

- [ ] **Step 3: Implement.**
```python
# src/awake/eval/erasure.py
"""Length-preserving token erasure for faithfulness metrics.

We replace erased tokens with the tokenizer's mask id rather than deleting
them, so positional embeddings stay aligned (deletion shifts positions and
produces out-of-distribution inputs). This is a deliberate, documented
deviation from ERASER's literal token-removal (see ADR 002).
"""

from __future__ import annotations

import math

import numpy as np


def erase(token_ids: np.ndarray, keep_mask: np.ndarray, mask_token_id: int) -> np.ndarray:
    """Replace tokens where ``keep_mask`` is False with ``mask_token_id``.

    Args:
        token_ids: 1-D array of input token ids.
        keep_mask: Boolean array; True keeps the original token.
        mask_token_id: Id substituted into erased positions.

    Returns:
        A new array, same shape, with erased positions masked.
    """
    out = token_ids.copy()
    out[~keep_mask] = mask_token_id
    return out


def top_k_mask(scores: np.ndarray, visible_mask: np.ndarray, k_fraction: float) -> np.ndarray:
    """Boolean mask selecting the top ``k_fraction`` of visible tokens by score.

    Args:
        scores: Per-token importance.
        visible_mask: True for tokens eligible for selection.
        k_fraction: Fraction in ``[0, 1]`` of visible tokens to select.

    Returns:
        Boolean mask, True for selected (top-scoring visible) tokens.
    """
    mask = np.zeros_like(visible_mask, dtype=bool)
    visible_idx = np.flatnonzero(visible_mask)
    if visible_idx.size == 0 or k_fraction <= 0:
        return mask
    k = min(visible_idx.size, math.ceil(k_fraction * visible_idx.size))
    order = visible_idx[np.argsort(scores[visible_idx])[::-1]]
    mask[order[:k]] = True
    return mask
```

- [ ] **Step 4: Run to verify pass.** `uv run pytest tests/test_eval_erasure.py -v --no-cov` → PASS.

- [ ] **Step 5: Commit.**
```bash
git add src/awake/eval/erasure.py tests/test_eval_erasure.py
git commit -m "P2: mask-replacement erasure + top_k_mask helper"
```

### Task 4: `comprehensiveness` and `sufficiency` (predicted class, fixed budget)

**Files:**
- Create: `src/awake/eval/faithfulness.py`
- Test: `tests/test_eval_faithfulness.py`

- [ ] **Step 1: Write the failing test.** Use a toy `predict_fn` whose predicted-class
  probability is proportional to the count of a "signal" token id (`7`); erasing it must
  drop the probability.
```python
# tests/test_eval_faithfulness.py
import numpy as np
import pytest
from awake.eval.faithfulness import comprehensiveness, sufficiency


def _toy_predict_fn(batch: np.ndarray) -> np.ndarray:
    # p(class 1) = fraction of tokens equal to the signal id 7.
    p1 = (batch == 7).mean(axis=1)
    return np.stack([1 - p1, p1], axis=1)


@pytest.mark.unit
def test_comprehensiveness_positive_when_important_token_removed():
    ids = np.array([7, 7, 0, 0])              # p(class1)=0.5
    scores = np.array([1.0, 1.0, 0.0, 0.0])   # signal tokens rank top
    visible = np.array([True, True, True, True])
    comp = comprehensiveness(_toy_predict_fn, ids, scores, visible,
                             predicted_class=1, mask_token_id=0, k_fraction=0.5)
    assert comp > 0  # removing the two 7s drops p(class1) from 0.5 to 0


@pytest.mark.unit
def test_sufficiency_small_when_rationale_alone_preserves_prediction():
    ids = np.array([7, 7, 0, 0])
    scores = np.array([1.0, 1.0, 0.0, 0.0])
    visible = np.array([True, True, True, True])
    suff = sufficiency(_toy_predict_fn, ids, scores, visible,
                       predicted_class=1, mask_token_id=0, k_fraction=0.5)
    # keeping only the two 7s -> p(class1)=0.5 == original -> suff ~ 0
    assert abs(suff) < 1e-9
```

- [ ] **Step 2: Run to verify failure.** `uv run pytest tests/test_eval_faithfulness.py -v --no-cov` → FAIL.

- [ ] **Step 3: Implement.**
```python
# src/awake/eval/faithfulness.py
"""ERASER faithfulness metrics, scored on the original predicted class.

comprehensiveness = p_j(x) - p_j(x with top-k rationale erased)
sufficiency       = p_j(x) - p_j(x with only top-k rationale kept)
where j is the predicted class fixed per example, and the top-k budget is a
fixed dataset fraction (k_d), not the per-example gold length.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from awake.eval.erasure import erase, top_k_mask

PredictFn = Callable[[np.ndarray], np.ndarray]


def _prob_j(predict_fn: PredictFn, token_ids: np.ndarray, predicted_class: int) -> float:
    probs = predict_fn(token_ids[None, :])
    return float(probs[0, predicted_class])


def comprehensiveness(
    predict_fn: PredictFn,
    token_ids: np.ndarray,
    scores: np.ndarray,
    visible_mask: np.ndarray,
    predicted_class: int,
    mask_token_id: int,
    k_fraction: float,
) -> float:
    """Drop in predicted-class prob when the top-k rationale is erased."""
    base = _prob_j(predict_fn, token_ids, predicted_class)
    rationale = top_k_mask(scores, visible_mask, k_fraction)
    reduced = erase(token_ids, keep_mask=~rationale, mask_token_id=mask_token_id)
    return base - _prob_j(predict_fn, reduced, predicted_class)


def sufficiency(
    predict_fn: PredictFn,
    token_ids: np.ndarray,
    scores: np.ndarray,
    visible_mask: np.ndarray,
    predicted_class: int,
    mask_token_id: int,
    k_fraction: float,
) -> float:
    """Drop in predicted-class prob when only the top-k rationale is kept."""
    base = _prob_j(predict_fn, token_ids, predicted_class)
    rationale = top_k_mask(scores, visible_mask, k_fraction)
    # keep rationale + special tokens (non-visible) so structure is preserved
    keep = rationale | (~visible_mask)
    kept = erase(token_ids, keep_mask=keep, mask_token_id=mask_token_id)
    return base - _prob_j(predict_fn, kept, predicted_class)
```

- [ ] **Step 4: Run to verify pass.** `uv run pytest tests/test_eval_faithfulness.py -v --no-cov` → PASS.

- [ ] **Step 5: Commit.**
```bash
git add src/awake/eval/faithfulness.py tests/test_eval_faithfulness.py
git commit -m "P2: comprehensiveness + sufficiency on predicted class with fixed budget"
```

### Task 5: `aopc` over bins (separate aggregate)

**Files:**
- Modify: `src/awake/eval/faithfulness.py`
- Modify: `tests/test_eval_faithfulness.py`

- [ ] **Step 1: Add the failing test.**
```python
# append to tests/test_eval_faithfulness.py
from awake.eval.faithfulness import aopc_comprehensiveness


@pytest.mark.unit
def test_aopc_averages_per_bin_prob_drops():
    ids = np.array([7, 7, 7, 7, 0, 0, 0, 0, 0, 0])  # p(class1)=0.4
    scores = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    visible = np.ones(10, dtype=bool)
    val = aopc_comprehensiveness(
        _toy_predict_fn, ids, scores, visible,
        predicted_class=1, mask_token_id=0, bins=(0.0, 0.1, 0.2, 0.5),
    )
    # bin 0.0 -> drop 0; 0.1 -> remove 1 signal -> p=0.3, drop 0.1;
    # 0.2 -> remove 2 -> p=0.2, drop 0.2; 0.5 -> remove 5 (4 signal) -> p=0, drop 0.4
    # mean([0, 0.1, 0.2, 0.4]) = 0.175
    assert val == pytest.approx(0.175, abs=1e-9)
```

- [ ] **Step 2: Run to verify failure.** → FAIL (import error).

- [ ] **Step 3: Implement (append to `faithfulness.py`).**
```python
def aopc_comprehensiveness(
    predict_fn: PredictFn,
    token_ids: np.ndarray,
    scores: np.ndarray,
    visible_mask: np.ndarray,
    predicted_class: int,
    mask_token_id: int,
    bins: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10, 0.20, 0.50),
) -> float:
    """Mean over bins of the predicted-class prob drop after erasing top-k%.

    Includes the 0% bin (drop 0) per DeYoung et al. 2020.
    """
    base = _prob_j(predict_fn, token_ids, predicted_class)
    drops = []
    for frac in bins:
        rationale = top_k_mask(scores, visible_mask, frac)
        reduced = erase(token_ids, keep_mask=~rationale, mask_token_id=mask_token_id)
        drops.append(base - _prob_j(predict_fn, reduced, predicted_class))
    return float(np.mean(drops))
```

- [ ] **Step 4: Run to verify pass.** → PASS.

- [ ] **Step 5: Commit.**
```bash
git add src/awake/eval/faithfulness.py tests/test_eval_faithfulness.py
git commit -m "P2: AOPC comprehensiveness as separate binned aggregate"
```

### Task 6: subword→word aggregation (max of |score|)

**Files:**
- Create: `src/awake/eval/plausibility.py`
- Test: `tests/test_eval_plausibility.py`

- [ ] **Step 1: Write the failing test.**
```python
# tests/test_eval_plausibility.py
import numpy as np
import pytest
from awake.eval.plausibility import aggregate_subwords_to_words


@pytest.mark.unit
def test_aggregate_takes_max_abs_over_word_subwords():
    # word_ids maps each subword to a word index; None for special tokens.
    word_ids = [None, 0, 0, 1, None]
    scores = np.array([9.0, 0.2, -0.8, 0.5, 9.0])
    word_scores = aggregate_subwords_to_words(scores, word_ids, n_words=2)
    # word 0: max(|0.2|, |-0.8|) = 0.8 ; word 1: 0.5
    assert word_scores.tolist() == [0.8, 0.5]
```

- [ ] **Step 2: Run to verify failure.** → FAIL.

- [ ] **Step 3: Implement.**
```python
# src/awake/eval/plausibility.py
"""Plausibility metrics vs. human rationales (word level).

Subword attributions are aggregated to whitespace-word level by taking the
max absolute score among a word's subwords. LIME, already whitespace-level,
bypasses this via an identity word_ids mapping. All functions are
tokenizer-agnostic: callers pass ``word_ids`` (e.g. from a fast tokenizer's
``word_ids()``) and a clipped gold mask, never a live tokenizer.
"""

from __future__ import annotations

import numpy as np


def aggregate_subwords_to_words(
    scores: np.ndarray, word_ids: list[int | None], n_words: int
) -> np.ndarray:
    """Aggregate subword scores to word level by max absolute value.

    Args:
        scores: Per-subword scores.
        word_ids: Word index per subword; None for special tokens.
        n_words: Total number of whitespace words in the (clipped) document.

    Returns:
        Array of length ``n_words`` with the max |score| per word (0 if a word
        has no surviving subwords).
    """
    word_scores = np.zeros(n_words, dtype=float)
    for score, wid in zip(scores, word_ids):
        if wid is None:
            continue
        word_scores[wid] = max(word_scores[wid], abs(float(score)))
    return word_scores
```

- [ ] **Step 4: Run to verify pass.** → PASS.

- [ ] **Step 5: Commit.**
```bash
git add src/awake/eval/plausibility.py tests/test_eval_plausibility.py
git commit -m "P2: subword->word max-abs aggregation for plausibility"
```

### Task 7: plausibility metrics — P/R/F1 @ k_d, AUPRC, custom token_iou, mask clipping

**Files:**
- Modify: `src/awake/eval/plausibility.py`
- Modify: `tests/test_eval_plausibility.py`

- [ ] **Step 1: Add failing tests.**
```python
# append to tests/test_eval_plausibility.py
from awake.eval.plausibility import (
    clip_gold_mask_to_window,
    token_prf1_at_k,
    token_auprc,
    token_iou,
)


@pytest.mark.unit
def test_clip_gold_mask_to_window():
    gold = np.array([0, 1, 1, 0, 1])      # 5 words in full doc
    out = clip_gold_mask_to_window(gold, n_visible_words=3)
    assert out.tolist() == [0, 1, 1]


@pytest.mark.unit
def test_token_prf1_at_k_selects_top_scores():
    word_scores = np.array([0.1, 0.9, 0.8, 0.2])
    gold = np.array([0, 1, 1, 0])
    p, r, f1 = token_prf1_at_k(word_scores, gold, k=2)
    assert (p, r, f1) == pytest.approx((1.0, 1.0, 1.0))


@pytest.mark.unit
def test_token_auprc_perfect_ranking_is_one():
    word_scores = np.array([0.1, 0.9, 0.8, 0.2])
    gold = np.array([0, 1, 1, 0])
    assert token_auprc(word_scores, gold) == pytest.approx(1.0)


@pytest.mark.unit
def test_token_iou_custom_metric():
    pred = np.array([0, 1, 1, 0], dtype=bool)
    gold = np.array([0, 1, 0, 0], dtype=bool)
    # intersection=1, union=2 -> 0.5
    assert token_iou(pred, gold) == pytest.approx(0.5)
```

- [ ] **Step 2: Run to verify failure.** → FAIL.

- [ ] **Step 3: Implement (append to `plausibility.py`).**
```python
from sklearn.metrics import average_precision_score


def clip_gold_mask_to_window(gold_mask: np.ndarray, n_visible_words: int) -> np.ndarray:
    """Clip a full-document word-level gold mask to the visible window."""
    return gold_mask[:n_visible_words]


def token_prf1_at_k(
    word_scores: np.ndarray, gold_mask: np.ndarray, k: int
) -> tuple[float, float, float]:
    """Token precision/recall/F1 when the top-k scored words are predicted.

    Args:
        word_scores: Word-level attribution scores.
        gold_mask: Binary word-level human rationale (clipped to window).
        k: Number of words to mark as predicted rationale (the k_d budget).

    Returns:
        ``(precision, recall, f1)``.
    """
    k = max(0, min(k, word_scores.size))
    pred = np.zeros_like(gold_mask, dtype=bool)
    if k > 0:
        top = np.argsort(word_scores)[::-1][:k]
        pred[top] = True
    gold = gold_mask.astype(bool)
    tp = int(np.sum(pred & gold))
    p = tp / pred.sum() if pred.sum() else 0.0
    r = tp / gold.sum() if gold.sum() else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def token_auprc(word_scores: np.ndarray, gold_mask: np.ndarray) -> float:
    """Area under the precision-recall curve over continuous word scores."""
    gold = gold_mask.astype(int)
    if gold.sum() == 0 or gold.sum() == gold.size:
        return float("nan")
    return float(average_precision_score(gold, word_scores))


def token_iou(pred_mask: np.ndarray, gold_mask: np.ndarray) -> float:
    """Custom token-set IoU (NOT ERASER's span-level IoU). Diagnostic only."""
    pred = pred_mask.astype(bool)
    gold = gold_mask.astype(bool)
    union = int(np.sum(pred | gold))
    return int(np.sum(pred & gold)) / union if union else 0.0
```

- [ ] **Step 4: Run to verify pass.** → PASS.

- [ ] **Step 5: Commit.**
```bash
git add src/awake/eval/plausibility.py tests/test_eval_plausibility.py
git commit -m "P2: plausibility P/R/F1@k, AUPRC, custom token_iou, mask clipping"
```

### Task 8: paired bootstrap CIs + pairwise paired-difference tests

**Files:**
- Create: `src/awake/eval/bootstrap.py`
- Test: `tests/test_eval_bootstrap.py`

- [ ] **Step 1: Write the failing test.**
```python
# tests/test_eval_bootstrap.py
import numpy as np
import pytest
from awake.eval.bootstrap import bootstrap_ci, paired_diff_test


@pytest.mark.unit
def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    rng_vals = np.linspace(0.0, 1.0, 200)
    lo, mean, hi = bootstrap_ci(rng_vals, n_resamples=500, alpha=0.05, seed=0)
    assert lo < mean < hi
    lo2, mean2, hi2 = bootstrap_ci(rng_vals, n_resamples=500, alpha=0.05, seed=0)
    assert (lo, mean, hi) == (lo2, mean2, hi2)  # seeded determinism


@pytest.mark.unit
def test_paired_diff_test_detects_clear_difference():
    a = np.full(100, 0.8)
    b = np.full(100, 0.2)
    res = paired_diff_test(a, b, n_resamples=500, seed=0)
    assert res["mean_diff"] == pytest.approx(0.6)
    assert res["p_value"] < 0.05
```

- [ ] **Step 2: Run to verify failure.** → FAIL.

- [ ] **Step 3: Implement.**
```python
# src/awake/eval/bootstrap.py
"""Paired bootstrap CIs and pairwise difference tests over per-example metrics."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Returns:
        ``(low, mean, high)`` at the ``1 - alpha`` confidence level.
    """
    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = values[rng.integers(0, n, n)].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, float(values.mean()), hi


def paired_diff_test(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 2000, seed: int = 0
) -> dict:
    """Paired bootstrap test of ``mean(a) - mean(b)`` over shared examples.

    Returns:
        Dict with ``mean_diff``, ``ci_low``, ``ci_high`` and a two-sided
        bootstrap ``p_value`` (fraction of resampled diffs crossing zero).
    """
    if a.shape != b.shape:
        raise ValueError("paired arrays must have equal shape")
    diff = a - b
    lo, mean_diff, hi = bootstrap_ci(diff, n_resamples=n_resamples, seed=seed)
    rng = np.random.default_rng(seed + 1)
    n = diff.size
    centered = diff - diff.mean()
    resampled = np.array(
        [centered[rng.integers(0, n, n)].mean() for _ in range(n_resamples)]
    )
    p = float((np.abs(resampled) >= abs(diff.mean())).mean())
    return {"mean_diff": mean_diff, "ci_low": lo, "ci_high": hi, "p_value": p}
```

- [ ] **Step 4: Run to verify pass.** → PASS.

- [ ] **Step 5: Verify the coverage floor for the whole eval core.**
Run: `uv run pytest tests -m unit`
Expected: PASS with coverage on `awake` ≥ 90% (the addopts `--cov-fail-under=90` must not trip).

- [ ] **Step 6: Add `src/awake/eval/__init__.py` public re-exports + commit.**
```python
# src/awake/eval/__init__.py
"""Shared interpretability evaluation utilities."""

from awake.eval.attribution import Explainer, ModelAdapter, TokenAttribution
from awake.eval.bootstrap import bootstrap_ci, paired_diff_test
from awake.eval.erasure import erase, top_k_mask
from awake.eval.faithfulness import aopc_comprehensiveness, comprehensiveness, sufficiency
from awake.eval.plausibility import (
    aggregate_subwords_to_words,
    clip_gold_mask_to_window,
    token_auprc,
    token_iou,
    token_prf1_at_k,
)

__all__ = [
    "Explainer", "ModelAdapter", "TokenAttribution",
    "bootstrap_ci", "paired_diff_test", "erase", "top_k_mask",
    "aopc_comprehensiveness", "comprehensiveness", "sufficiency",
    "aggregate_subwords_to_words", "clip_gold_mask_to_window",
    "token_auprc", "token_iou", "token_prf1_at_k",
]
```
```bash
git add src/awake/eval/bootstrap.py src/awake/eval/__init__.py tests/test_eval_bootstrap.py
git commit -m "P2: paired bootstrap CIs + pairwise diff tests; eval public API"
```

---

## Phase 2 — Project scaffold, configs, data pipeline

### Task 9: project scaffold — `_paths.py`, configs, justfile, conftest

**Files:**
- Create: `projects/02-text-eraser/scripts/__init__.py` (empty)
- Create: `projects/02-text-eraser/scripts/_paths.py`
- Create: `projects/02-text-eraser/configs/{data,model,explainers}.yaml`
- Create: `projects/02-text-eraser/justfile`
- Create: `projects/02-text-eraser/tests/conftest.py`

- [ ] **Step 1: `_paths.py`.**
```python
# projects/02-text-eraser/scripts/_paths.py
"""Filesystem paths and config loading for project 02."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
PREPARED = OUTPUTS / "prepared"
MODEL_DIR = OUTPUTS / "model"
CACHE_DIR = OUTPUTS / "attributions"
ASSETS = PROJECT_ROOT / "assets"
CONFIGS = PROJECT_ROOT / "configs"

DATA_PATH = Path(os.environ.get("DATA_PATH", Path.home() / ".cache/eraser/movies"))


def load_config(name: str) -> dict:
    """Load ``configs/<name>.yaml`` as a dict."""
    with open(CONFIGS / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    """Create all output directories."""
    for d in (OUTPUTS, PREPARED, MODEL_DIR, CACHE_DIR, ASSETS):
        d.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: configs.**
```yaml
# configs/data.yaml
url: "https://www.eraserbenchmark.com/zipped/movies.tar.gz"
sha256: "PLACEHOLDER_FILL_FROM_DOWNLOAD"   # set in Task 10 after first verified download
max_seq_len: 512
coverage_threshold: 0.8
```
> Note: the `sha256` is the one value legitimately filled during Task 10 (you cannot
> know a download's checksum before downloading it once and verifying the contents).
```yaml
# configs/model.yaml
model_name: "microsoft/deberta-v3-base"
num_labels: 2
lr: 2.0e-5
epochs: 3
batch_size: 8
fp16: true
seed: 42
```
```yaml
# configs/explainers.yaml
subsample_n: 200
k_d: 0.20            # dataset rationale budget (fraction); recomputed in Task 11
aopc_bins: [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]
bootstrap:
  n_resamples: 2000
  alpha: 0.05
  seed: 0
lime:
  num_samples: 1000
shap:
  max_evals: 500
ig:
  n_steps: 50
```

- [ ] **Step 3: justfile (mirror P1).**
```just
set shell := ["bash", "-uc"]
set dotenv-load := true

project_root := justfile_directory()
repo_root := justfile_directory() + "/../.."
default_data_path := env_var_or_default("DATA_PATH", env_var("HOME") + "/.cache/eraser/movies")

default:
    @just --list

data:
    cd "{{project_root}}" && uv run python scripts/00_fetch_data.py --data-path "{{default_data_path}}"

prepare:
    cd "{{project_root}}" && uv run python scripts/01_prepare.py

train *FLAGS:
    cd "{{project_root}}" && uv run python scripts/10_train.py {{FLAGS}}

explain *FLAGS:
    cd "{{project_root}}" && uv run python scripts/20_explain.py {{FLAGS}}

eval *FLAGS:
    cd "{{project_root}}" && uv run python scripts/30_eval.py {{FLAGS}}

pipeline: data prepare

notebook NAME="01-explainer-comparison":
    cd "{{project_root}}/notebooks" && \
        uv run jupytext --to ipynb {{NAME}}.py && \
        uv run jupyter nbconvert --to notebook --execute --inplace {{NAME}}.ipynb && \
        uv run jupyter nbconvert --to html {{NAME}}.ipynb && \
        uv run nbstripout {{NAME}}.ipynb

test:
    cd "{{repo_root}}" && uv run pytest projects/02-text-eraser/tests -m "unit or smoke" --no-cov
```

- [ ] **Step 4: conftest (make scripts importable, like P1).**
```python
# projects/02-text-eraser/tests/conftest.py
"""Make the project's scripts/ importable in tests."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
```

- [ ] **Step 5: Commit.**
```bash
git add projects/02-text-eraser/scripts/__init__.py projects/02-text-eraser/scripts/_paths.py \
        projects/02-text-eraser/configs projects/02-text-eraser/justfile \
        projects/02-text-eraser/tests/conftest.py
git commit -m "P2: project scaffold — paths, configs, justfile, conftest"
```

### Task 10: `00_fetch_data.py` — download, verify, extract

**Files:**
- Create: `projects/02-text-eraser/scripts/00_fetch_data.py`
- Test: `projects/02-text-eraser/tests/test_prepare_smoke.py` (the fetch verifier portion)

- [ ] **Step 1: Write a failing unit test for the checksum verifier (pure, no network).**
```python
# projects/02-text-eraser/tests/test_prepare_smoke.py
import hashlib
import importlib

import pytest

fetch = importlib.import_module("00_fetch_data")


@pytest.mark.unit
def test_verify_sha256_accepts_match(tmp_path):
    f = tmp_path / "blob"
    f.write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    assert fetch.verify_sha256(f, digest) is True


@pytest.mark.unit
def test_verify_sha256_rejects_mismatch(tmp_path):
    f = tmp_path / "blob"
    f.write_bytes(b"hello")
    assert fetch.verify_sha256(f, "deadbeef") is False
```

- [ ] **Step 2: Run to verify failure.** `uv run pytest projects/02-text-eraser/tests/test_prepare_smoke.py -v --no-cov` → FAIL.

- [ ] **Step 3: Implement.**
```python
# projects/02-text-eraser/scripts/00_fetch_data.py
"""Download + verify + extract the ERASER Movies tarball. Commits no data."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

from _paths import DATA_PATH, load_config


def verify_sha256(path: Path, expected: str) -> bool:
    """Return True iff the file's SHA-256 hex digest equals ``expected``."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def main() -> None:
    """CLI: download the tarball to ``--data-path`` and extract it."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", type=Path, default=DATA_PATH)
    args = ap.parse_args()
    cfg = load_config("data")
    args.data_path.mkdir(parents=True, exist_ok=True)
    tar_path = args.data_path / "movies.tar.gz"

    if not tar_path.exists():
        print(f"Downloading {cfg['url']} -> {tar_path}")
        urllib.request.urlretrieve(cfg["url"], tar_path)  # noqa: S310

    expected = cfg.get("sha256", "")
    if expected and expected != "PLACEHOLDER_FILL_FROM_DOWNLOAD":
        if not verify_sha256(tar_path, expected):
            raise SystemExit(f"FATAL: checksum mismatch on {tar_path}")
    else:
        actual = hashlib.sha256(tar_path.read_bytes()).hexdigest()
        print(f"NOTE: set configs/data.yaml sha256 to: {actual}")

    with tarfile.open(tar_path) as t:
        t.extractall(args.data_path, filter="data")
    print(f"Extracted to {args.data_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass.** → PASS (network test deferred; verifier covered).

- [ ] **Step 5: One-time real fetch + record checksum.** Run `just data`, copy the printed
  `sha256` into `configs/data.yaml`. (Manual; this is the legitimate fill of that placeholder.)

- [ ] **Step 6: Commit.**
```bash
git add projects/02-text-eraser/scripts/00_fetch_data.py \
        projects/02-text-eraser/tests/test_prepare_smoke.py projects/02-text-eraser/configs/data.yaml
git commit -m "P2: ERASER tarball fetch + sha256 verify"
```

### Task 11: `01_prepare.py` — parse, masks, visible sequence, coverage, drop comparison docs

**Files:**
- Create: `projects/02-text-eraser/scripts/01_prepare.py`
- Modify: `projects/02-text-eraser/tests/test_prepare_smoke.py`

- [ ] **Step 1: Add a smoke test on a tiny ERASER-shaped fixture.** It builds a fake
  `docs/` file + one `val.jsonl` line, runs the pure helpers, and asserts the visible
  sequence is frozen, the gold mask is word-level, and coverage is computed.
```python
# append to projects/02-text-eraser/tests/test_prepare_smoke.py
import json
import numpy as np
prepare = importlib.import_module("01_prepare")


@pytest.mark.smoke
def test_prepare_builds_word_mask_and_coverage(tmp_path):
    doc = "the movie was absolutely terrible and boring throughout"
    words = doc.split()
    # evidence covers words 3..5 ("absolutely terrible and")
    example = {
        "annotation_id": "neg_0",
        "classification": "NEG",
        "evidences": [[{"docid": "neg_0", "start_token": 3, "end_token": 6,
                        "text": "absolutely terrible and"}]],
    }
    record = prepare.build_record(example, doc_text=doc)
    assert record["label"] == 0
    assert record["words"] == words
    assert record["gold_mask"] == [0, 0, 0, 1, 1, 1, 0, 0]
    assert record["n_words"] == len(words)


@pytest.mark.unit
def test_prepare_drops_comparison_multidoc_evidence():
    example = {
        "annotation_id": "x_0", "classification": "POS",
        "evidences": [[{"docid": "x_0", "start_token": 0, "end_token": 1, "text": "a"},
                       {"docid": "OTHER_1", "start_token": 0, "end_token": 1, "text": "b"}]],
    }
    assert prepare.is_comparison(example, docid="x_0") is True
```

- [ ] **Step 2: Run to verify failure.** → FAIL.

- [ ] **Step 3: Implement.**
```python
# projects/02-text-eraser/scripts/01_prepare.py
"""Parse ERASER Movies into prepared parquet with frozen visible sequences."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA_PATH, PREPARED, ensure_dirs, load_config

LABELS = {"NEG": 0, "POS": 1}


def is_comparison(example: dict, docid: str) -> bool:
    """True if any evidence span references a document other than ``docid``."""
    for group in example["evidences"]:
        for ev in group:
            if ev["docid"] != docid:
                return True
    return False


def build_record(example: dict, doc_text: str) -> dict:
    """Build a word-level record (text, label, words, gold_mask, n_words)."""
    words = doc_text.split()
    gold = np.zeros(len(words), dtype=int)
    for group in example["evidences"]:
        for ev in group:
            gold[ev["start_token"]:ev["end_token"]] = 1
    return {
        "annotation_id": example["annotation_id"],
        "label": LABELS[example["classification"]],
        "text": doc_text,
        "words": words,
        "gold_mask": gold.tolist(),
        "n_words": len(words),
    }


def freeze_visible(text: str, tokenizer, max_len: int) -> dict:
    """Tokenize once, truncate to max_len, capture offsets + word_ids."""
    enc = tokenizer(
        text, truncation=True, max_length=max_len,
        return_offsets_mapping=True, return_tensors=None,
    )
    word_ids = enc.word_ids()
    return {
        "input_ids": enc["input_ids"],
        "offsets": enc["offset_mapping"],
        "word_ids": word_ids,
    }


def truncation_coverage(word_ids: list[int | None], gold_mask: list[int]) -> float:
    """Fraction of gold-rationale words that survive truncation."""
    gold_words = {i for i, g in enumerate(gold_mask) if g}
    if not gold_words:
        return 1.0
    visible_words = {w for w in word_ids if w is not None}
    return len(gold_words & visible_words) / len(gold_words)


def main() -> None:
    """Parse each split, freeze visible sequences, write prepared parquet."""
    from transformers import AutoTokenizer

    ensure_dirs()
    cfg_d = load_config("data")
    cfg_m = load_config("model")
    tok = AutoTokenizer.from_pretrained(cfg_m["model_name"])
    stats = {"dropped_comparison": 0}

    for split in ("train", "val", "test"):
        rows = []
        with open(DATA_PATH / "movies" / f"{split}.jsonl") as f:
            for line in f:
                ex = json.loads(line)
                docid = ex["annotation_id"]
                if is_comparison(ex, docid):
                    stats["dropped_comparison"] += 1
                    continue
                doc_text = (DATA_PATH / "movies" / "docs" / docid).read_text()
                rec = build_record(ex, doc_text)
                vis = freeze_visible(rec["text"], tok, cfg_d["max_seq_len"])
                rec.update(vis)
                rec["truncation_coverage"] = truncation_coverage(
                    vis["word_ids"], rec["gold_mask"]
                )
                rows.append(rec)
        pd.DataFrame(rows).to_parquet(PREPARED / f"{split}.parquet")
        print(f"{split}: {len(rows)} examples")

    (PREPARED.parent / "prepare_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"stats: {stats}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass.** `uv run pytest projects/02-text-eraser/tests/test_prepare_smoke.py -v --no-cov` → PASS.

- [ ] **Step 5: Commit.**
```bash
git add projects/02-text-eraser/scripts/01_prepare.py projects/02-text-eraser/tests/test_prepare_smoke.py
git commit -m "P2: prepare ERASER Movies — masks, frozen visible seq, coverage, drop comparison"
```

---

## Phase 3 — Stub model, training, model adapter

### Task 12: `_stub_model.py` + `_model_adapter.py` (HFModelAdapter)

**Files:**
- Create: `projects/02-text-eraser/scripts/_stub_model.py`
- Create: `projects/02-text-eraser/scripts/_model_adapter.py`
- Create: `projects/02-text-eraser/tests/test_train_smoke.py`

- [ ] **Step 1: Write the smoke test (tiny model, CPU).**
```python
# projects/02-text-eraser/tests/test_train_smoke.py
import importlib
import numpy as np
import pytest

stub = importlib.import_module("_stub_model")
adapter_mod = importlib.import_module("_model_adapter")


@pytest.mark.smoke
def test_model_adapter_predict_proba_shape():
    model, tok = stub.build_stub_model_and_tokenizer()
    adapter = adapter_mod.HFModelAdapter(model, tok, device="cpu")
    ids = np.array([[tok.cls_token_id, 5, 6, tok.sep_token_id]])
    probs = adapter.predict_proba(ids)
    assert probs.shape == (1, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
```

- [ ] **Step 2: Run to verify failure.** → FAIL.

- [ ] **Step 3: Implement the stub model.**
```python
# projects/02-text-eraser/scripts/_stub_model.py
"""A tiny BERT-class model + tokenizer for CPU smoke tests (no downloads)."""

from __future__ import annotations

from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
)


def build_stub_model_and_tokenizer():
    """Return a randomly-initialised 2-layer BERT classifier + a tiny tokenizer."""
    config = BertConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=128, num_labels=2,
    )
    model = BertForSequenceClassification(config)
    # minimal fast tokenizer over a tiny vocab
    vocab = {f"[{t}]": i for i, t in enumerate(["PAD", "UNK", "CLS", "SEP", "MASK"])}
    for i in range(5, 64):
        vocab[f"w{i}"] = i
    import tempfile, os
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "vocab.txt"), "w") as f:
        for tokstr in vocab:
            f.write(tokstr + "\n")
    tok = BertTokenizerFast(
        vocab_file=os.path.join(d, "vocab.txt"),
        cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]",
        unk_token="[UNK]", mask_token="[MASK]",
    )
    return model, tok
```

- [ ] **Step 4: Implement the adapter.**
```python
# projects/02-text-eraser/scripts/_model_adapter.py
"""HFModelAdapter: ModelAdapter over a fine-tuned HF sequence classifier."""

from __future__ import annotations

import numpy as np
import torch


class HFModelAdapter:
    """Adapts an HF classifier to ``awake.eval.ModelAdapter.predict_proba``."""

    def __init__(self, model, tokenizer, device: str = "cpu") -> None:
        """Store the model/tokenizer and move the model to ``device``."""
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def predict_proba(self, token_ids_batch: np.ndarray) -> np.ndarray:
        """Softmax probabilities for a ``(batch, seq)`` array of token ids."""
        ids = torch.as_tensor(token_ids_batch, dtype=torch.long, device=self.device)
        attn = (ids != self.tokenizer.pad_token_id).long()
        logits = self.model(input_ids=ids, attention_mask=attn).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()
```

- [ ] **Step 5: Run to verify pass + commit.**
```bash
uv run pytest projects/02-text-eraser/tests/test_train_smoke.py -v --no-cov   # PASS
git add projects/02-text-eraser/scripts/_stub_model.py projects/02-text-eraser/scripts/_model_adapter.py \
        projects/02-text-eraser/tests/test_train_smoke.py
git commit -m "P2: stub model for smoke tests + HFModelAdapter"
```

### Task 13: `10_train.py` — fine-tune + checkpoint + SHA

**Files:**
- Create: `projects/02-text-eraser/scripts/10_train.py`
- Modify: `projects/02-text-eraser/tests/test_train_smoke.py`

- [ ] **Step 1: Add a smoke test of the training loop on the stub (1 step).**
```python
# append to test_train_smoke.py
train_mod = importlib.import_module("10_train")


@pytest.mark.smoke
def test_train_one_step_on_stub(tmp_path):
    model, tok = stub.build_stub_model_and_tokenizer()
    texts = ["w5 w6 w7", "w8 w9 w10"]
    labels = [0, 1]
    out = train_mod.train_loop(model, tok, texts, labels, out_dir=tmp_path,
                               epochs=1, batch_size=2, lr=1e-3, max_len=16, fp16=False)
    assert (tmp_path / "model_sha256.txt").exists()
    assert out["sha256"]
```

- [ ] **Step 2: Run to verify failure.** → FAIL.

- [ ] **Step 3: Implement.**
```python
# projects/02-text-eraser/scripts/10_train.py
"""Fine-tune a sequence classifier on prepared ERASER Movies; save + hash it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from awake.utils.seeding import seed_everything
from _paths import MODEL_DIR, PREPARED, ensure_dirs, load_config


class _TextDataset(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(list(texts), truncation=True, max_length=max_len,
                       padding="max_length", return_tensors="pt")
        self.labels = torch.tensor(list(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i],
                "labels": self.labels[i]}


def _hash_dir(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def train_loop(model, tok, texts, labels, out_dir: Path, epochs, batch_size, lr,
               max_len, fp16) -> dict:
    """Minimal training loop; saves model + checkpoint SHA. Returns metadata."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).train()
    ds = _TextDataset(texts, labels, tok, max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        for batch in dl:
            opt.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            opt.step()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    sha = _hash_dir(out_dir)
    (out_dir / "model_sha256.txt").write_text(sha)
    return {"sha256": sha}


def main() -> None:
    """Fine-tune DeBERTa on the prepared train split and save the checkpoint."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ensure_dirs()
    cfg = load_config("model")
    seed_everything(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    df = pd.read_parquet(PREPARED / "train.parquet")
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=cfg["num_labels"]
    )
    meta = train_loop(model, tok, df["text"], df["label"], MODEL_DIR,
                      cfg["epochs"], cfg["batch_size"], cfg["lr"], 512, cfg["fp16"])
    (MODEL_DIR / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved model, sha={meta['sha256'][:12]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass + commit.**
```bash
uv run pytest projects/02-text-eraser/tests/test_train_smoke.py -v --no-cov   # PASS
git add projects/02-text-eraser/scripts/10_train.py projects/02-text-eraser/tests/test_train_smoke.py
git commit -m "P2: fine-tune script with checkpoint hashing (smoke-tested on stub)"
```

---

## Phase 4 — Explainers (`src/awake/eval/explainers/`)

### Task 14: `RandomExplainer` (the baseline floor)

**Files:**
- Create: `src/awake/eval/explainers/__init__.py` (empty)
- Create: `src/awake/eval/explainers/random_baseline.py`
- Test: `tests/test_eval_explainers.py`

- [ ] **Step 1: Write the failing unit test.**
```python
# tests/test_eval_explainers.py
import numpy as np
import pytest
from awake.eval.explainers.random_baseline import RandomExplainer


@pytest.mark.unit
def test_random_explainer_is_seed_deterministic_and_right_length():
    ex = {"input_ids": [2, 5, 6, 3], "visible_mask": [False, True, True, False],
          "offsets": [(0, 0), (0, 1), (2, 3), (0, 0)], "tokens": ["[CLS]", "a", "b", "[SEP]"],
          "predicted_class": 1, "class_scores": [0.4, 0.6]}
    r1 = RandomExplainer(seed=0).attribute(ex)
    r2 = RandomExplainer(seed=0).attribute(ex)
    assert r1.n_tokens == 4
    assert np.allclose(r1.scores, r2.scores)
```

- [ ] **Step 2: Run to verify failure.** → FAIL.

- [ ] **Step 3: Implement.**
```python
# src/awake/eval/explainers/random_baseline.py
"""Uniform-random attribution baseline (the floor in every comparison)."""

from __future__ import annotations

import numpy as np

from awake.eval.attribution import TokenAttribution


class RandomExplainer:
    """Assigns uniform random scores; reference floor for all metrics."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        """Store the RNG seed."""
        self.seed = seed

    def attribute(self, example: dict) -> TokenAttribution:
        """Return random per-token scores for the example's visible tokens."""
        rng = np.random.default_rng(self.seed + hash(str(example["input_ids"])) % 10_000)
        n = len(example["input_ids"])
        return TokenAttribution(
            tokens=list(example["tokens"]),
            offsets=list(example["offsets"]),
            scores=rng.random(n),
            visible_mask=np.asarray(example["visible_mask"], dtype=bool),
            predicted_class=int(example["predicted_class"]),
            class_scores=np.asarray(example["class_scores"], dtype=float),
        )
```

- [ ] **Step 4: Run to verify pass + commit.**
```bash
uv run pytest tests/test_eval_explainers.py -v --no-cov   # PASS
git add src/awake/eval/explainers/__init__.py src/awake/eval/explainers/random_baseline.py \
        tests/test_eval_explainers.py
git commit -m "P2: RandomExplainer baseline floor"
```

### Task 15: `GradientXInputExplainer`

**Files:**
- Create: `src/awake/eval/explainers/gradient_x_input.py`
- Modify: `tests/test_eval_explainers.py`

- [ ] **Step 1: Add a smoke test (stub model, CPU).** It builds the stub, wraps it, and
  asserts the attribution has one score per token and is finite.
```python
# append to tests/test_eval_explainers.py
@pytest.mark.smoke
def test_gradient_x_input_runs_on_stub():
    import importlib, sys, pathlib
    sys.path.insert(0, str(pathlib.Path("projects/02-text-eraser/scripts").resolve()))
    stub = importlib.import_module("_stub_model")
    from awake.eval.explainers.gradient_x_input import GradientXInputExplainer
    model, tok = stub.build_stub_model_and_tokenizer()
    gxi = GradientXInputExplainer(model, tok, device="cpu")
    ex = {"text": "w5 w6 w7", "predicted_class": None}
    attr = gxi.attribute(ex)
    assert attr.n_tokens > 0
    assert np.all(np.isfinite(attr.scores))
```

- [ ] **Step 2: Run to verify failure.** → FAIL.

- [ ] **Step 3: Implement.**
```python
# src/awake/eval/explainers/gradient_x_input.py
"""Gradient×Input attribution (valid on DeBERTa's disentangled attention).

Replaces attention rollout, which is ill-defined for DeBERTa's disentangled
attention matrices.
"""

from __future__ import annotations

import numpy as np
import torch

from awake.eval.attribution import TokenAttribution


class GradientXInputExplainer:
    """Saliency = sum over embedding dims of (grad · input embedding)."""

    name = "grad_x_input"

    def __init__(self, model, tokenizer, device: str = "cpu") -> None:
        """Store model/tokenizer and the compute device."""
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    def attribute(self, example: dict) -> TokenAttribution:
        """Compute gradient×input saliency toward the predicted class."""
        enc = self.tokenizer(example["text"], truncation=True, max_length=512,
                             return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        word_ids = enc.word_ids()
        ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        emb_layer = self.model.get_input_embeddings()
        embeds = emb_layer(ids).clone().detach().requires_grad_(True)
        logits = self.model(inputs_embeds=embeds, attention_mask=attn).logits
        pred = int(logits.argmax(dim=-1)) if example.get("predicted_class") is None \
            else int(example["predicted_class"])
        logits[0, pred].backward()
        sal = (embeds.grad * embeds).sum(dim=-1)[0].detach().cpu().numpy()
        visible = np.array([w is not None for w in word_ids], dtype=bool)
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        return TokenAttribution(
            tokens=self.tokenizer.convert_ids_to_tokens(ids[0].tolist()),
            offsets=[tuple(o) for o in offsets],
            scores=sal,
            visible_mask=visible,
            predicted_class=pred,
            class_scores=probs,
        )
```

- [ ] **Step 4: Run to verify pass + commit.**
```bash
uv run pytest tests/test_eval_explainers.py::test_gradient_x_input_runs_on_stub -v --no-cov  # PASS
git add src/awake/eval/explainers/gradient_x_input.py tests/test_eval_explainers.py
git commit -m "P2: Gradient×Input explainer (replaces attention rollout)"
```

### Task 16: `IntegratedGradientsExplainer` (captum)

**Files:**
- Create: `src/awake/eval/explainers/integrated_gradients.py`
- Modify: `tests/test_eval_explainers.py`

- [ ] **Step 1: Add a smoke test on the stub (small `n_steps`).**
```python
# append to tests/test_eval_explainers.py
@pytest.mark.smoke
def test_integrated_gradients_runs_on_stub():
    import importlib, sys, pathlib
    sys.path.insert(0, str(pathlib.Path("projects/02-text-eraser/scripts").resolve()))
    stub = importlib.import_module("_stub_model")
    from awake.eval.explainers.integrated_gradients import IntegratedGradientsExplainer
    model, tok = stub.build_stub_model_and_tokenizer()
    ig = IntegratedGradientsExplainer(model, tok, device="cpu", n_steps=4)
    attr = ig.attribute({"text": "w5 w6 w7", "predicted_class": None})
    assert attr.n_tokens > 0
    assert np.all(np.isfinite(attr.scores))
```

- [ ] **Step 2: Run to verify failure.** → FAIL.

- [ ] **Step 3: Implement.**
```python
# src/awake/eval/explainers/integrated_gradients.py
"""Integrated Gradients on the embedding layer via captum."""

from __future__ import annotations

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients

from awake.eval.attribution import TokenAttribution


class IntegratedGradientsExplainer:
    """LayerIntegratedGradients over the model's input embeddings."""

    name = "integrated_gradients"

    def __init__(self, model, tokenizer, device: str = "cpu", n_steps: int = 50) -> None:
        """Store model/tokenizer, device, and the IG step count."""
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.n_steps = n_steps

    def _forward(self, input_ids, attention_mask):
        return torch.softmax(
            self.model(input_ids=input_ids, attention_mask=attention_mask).logits, dim=-1
        )

    def attribute(self, example: dict) -> TokenAttribution:
        """Attribute the predicted class with IG; pad baseline as reference."""
        enc = self.tokenizer(example["text"], truncation=True, max_length=512,
                             return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        word_ids = enc.word_ids()
        ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        probs = self._forward(ids, attn)[0].detach().cpu().numpy()
        pred = int(probs.argmax()) if example.get("predicted_class") is None \
            else int(example["predicted_class"])
        baseline = torch.full_like(ids, self.tokenizer.pad_token_id)
        lig = LayerIntegratedGradients(self._forward, self.model.get_input_embeddings())
        atts = lig.attribute(ids, baselines=baseline, target=pred,
                             additional_forward_args=(attn,), n_steps=self.n_steps)
        sal = atts.sum(dim=-1)[0].detach().cpu().numpy()
        visible = np.array([w is not None for w in word_ids], dtype=bool)
        return TokenAttribution(
            tokens=self.tokenizer.convert_ids_to_tokens(ids[0].tolist()),
            offsets=[tuple(o) for o in offsets],
            scores=sal, visible_mask=visible,
            predicted_class=pred, class_scores=probs,
        )
```

- [ ] **Step 4: Run to verify pass + commit.**
```bash
uv run pytest tests/test_eval_explainers.py::test_integrated_gradients_runs_on_stub -v --no-cov  # PASS
git add src/awake/eval/explainers/integrated_gradients.py tests/test_eval_explainers.py
git commit -m "P2: Integrated Gradients explainer via captum"
```

### Task 17: `LimeExplainer` (whitespace-level, identity alignment)

**Files:**
- Create: `src/awake/eval/explainers/lime_text.py`
- Modify: `tests/test_eval_explainers.py`

- [ ] **Step 1: Add a smoke test on the stub (small `num_samples`).**
```python
# append to tests/test_eval_explainers.py
@pytest.mark.smoke
def test_lime_runs_on_stub():
    import importlib, sys, pathlib
    sys.path.insert(0, str(pathlib.Path("projects/02-text-eraser/scripts").resolve()))
    stub = importlib.import_module("_stub_model")
    from awake.eval.explainers.lime_text import LimeExplainer
    model, tok = stub.build_stub_model_and_tokenizer()
    lime = LimeExplainer(model, tok, device="cpu", num_samples=20)
    attr = lime.attribute({"text": "w5 w6 w7", "predicted_class": None})
    assert attr.n_tokens == 3            # whitespace-level: one score per word
    assert attr.word_level is True
```

- [ ] **Step 2: Run to verify failure.** → FAIL (also needs a `word_level` flag on
  `TokenAttribution`).

- [ ] **Step 3: Add `word_level: bool = False` field to `TokenAttribution`** (default
  False so existing constructions are unaffected; LIME sets it True). Update
  `__post_init__` not required (it doesn't affect length checks). Then implement LIME:
```python
# src/awake/eval/explainers/lime_text.py
"""LIME text explainer; produces whitespace-word-level attributions.

LIME perturbs whitespace tokens, so its scores are already word-level and use
the identity alignment path in plausibility scoring (word_level=True).
"""

from __future__ import annotations

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer

from awake.eval.attribution import TokenAttribution


class LimeExplainer:
    """Wraps lime.lime_text over an HF classifier (word-level scores)."""

    name = "lime"

    def __init__(self, model, tokenizer, device: str = "cpu", num_samples: int = 1000) -> None:
        """Store model/tokenizer, device, and the LIME sample budget."""
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.num_samples = num_samples
        self.explainer = LimeTextExplainer(class_names=["neg", "pos"], bow=False)

    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(texts, truncation=True, max_length=512,
                             padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def attribute(self, example: dict) -> TokenAttribution:
        """Run LIME and map per-word weights back to word order."""
        text = example["text"]
        words = text.split()
        probs = self._predict_proba([text])[0]
        pred = int(probs.argmax()) if example.get("predicted_class") is None \
            else int(example["predicted_class"])
        exp = self.explainer.explain_instance(
            text, self._predict_proba, num_features=len(words),
            num_samples=self.num_samples, labels=(pred,),
        )
        scores = np.zeros(len(words))
        # LIME returns (token_index_in_split, weight) pairs when bow=False
        for idx, weight in exp.as_map()[pred]:
            if idx < len(words):
                scores[idx] = weight
        return TokenAttribution(
            tokens=words,
            offsets=[(0, 0)] * len(words),
            scores=scores,
            visible_mask=np.ones(len(words), dtype=bool),
            predicted_class=pred,
            class_scores=probs,
            word_level=True,
        )
```

- [ ] **Step 4: Run to verify pass.** Re-run the attribution unit test (Task 2) to confirm
  the new field didn't break it, then the LIME smoke test. Both PASS.

- [ ] **Step 5: Commit.**
```bash
git add src/awake/eval/attribution.py src/awake/eval/explainers/lime_text.py tests/test_eval_explainers.py
git commit -m "P2: LIME explainer (word-level) + word_level flag on TokenAttribution"
```

### Task 18: `ShapPartitionExplainer` (optional extra, slow)

**Files:**
- Create: `src/awake/eval/explainers/shap_partition.py`
- Modify: `tests/test_eval_explainers.py`

- [ ] **Step 1: Add a test that skips cleanly when shap is absent.**
```python
# append to tests/test_eval_explainers.py
@pytest.mark.slow
def test_shap_partition_runs_on_stub():
    pytest.importorskip("shap")
    import importlib, sys, pathlib
    sys.path.insert(0, str(pathlib.Path("projects/02-text-eraser/scripts").resolve()))
    stub = importlib.import_module("_stub_model")
    from awake.eval.explainers.shap_partition import ShapPartitionExplainer
    model, tok = stub.build_stub_model_and_tokenizer()
    sx = ShapPartitionExplainer(model, tok, device="cpu", max_evals=50)
    attr = sx.attribute({"text": "w5 w6 w7", "predicted_class": None})
    assert attr.n_tokens > 0
```

- [ ] **Step 2: Run to verify failure (or skip).** `uv run pytest -m slow -k shap --no-cov`
  → FAIL if shap installed, else skipped.

- [ ] **Step 3: Implement.**
```python
# src/awake/eval/explainers/shap_partition.py
"""SHAP PartitionExplainer for text (stable on long inputs; optional dep)."""

from __future__ import annotations

import numpy as np
import torch

from awake.eval.attribution import TokenAttribution


class ShapPartitionExplainer:
    """Hierarchical PartitionExplainer over the classifier's predicted class."""

    name = "shap_partition"

    def __init__(self, model, tokenizer, device: str = "cpu", max_evals: int = 500) -> None:
        """Store model/tokenizer, device, and the SHAP evaluation budget."""
        import shap

        self.shap = shap
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_evals = max_evals

    def _predict(self, texts):
        enc = self.tokenizer(list(texts), truncation=True, max_length=512,
                             padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def attribute(self, example: dict) -> TokenAttribution:
        """Compute SHAP values for the predicted class via a text masker."""
        text = example["text"]
        words = text.split()
        probs = self._predict([text])[0]
        pred = int(probs.argmax()) if example.get("predicted_class") is None \
            else int(example["predicted_class"])
        masker = self.shap.maskers.Text(self.tokenizer)
        explainer = self.shap.PartitionExplainer(self._predict, masker)
        sv = explainer([text], max_evals=self.max_evals, silent=True)
        scores = np.asarray(sv.values[0][:, pred]).ravel()[: len(words)]
        if scores.size < len(words):
            scores = np.pad(scores, (0, len(words) - scores.size))
        return TokenAttribution(
            tokens=words, offsets=[(0, 0)] * len(words), scores=scores,
            visible_mask=np.ones(len(words), dtype=bool),
            predicted_class=pred, class_scores=probs, word_level=True,
        )
```

- [ ] **Step 4: Run to verify pass (only if shap installed) + commit.**
```bash
git add src/awake/eval/explainers/shap_partition.py tests/test_eval_explainers.py
git commit -m "P2: SHAP PartitionExplainer (optional extra)"
```

---

## Phase 5 — Explain, evaluate, figures, report

### Task 19: `_explainers.py` wiring + `20_explain.py` with cache schema

**Files:**
- Create: `projects/02-text-eraser/scripts/_explainers.py`
- Create: `projects/02-text-eraser/scripts/20_explain.py`
- Create: `projects/02-text-eraser/tests/test_explain_eval_smoke.py`

- [ ] **Step 1: Smoke test — explain a tiny prepared frame on the stub, assert cache
  parquet + metadata.**
```python
# projects/02-text-eraser/tests/test_explain_eval_smoke.py
import importlib, sys, pathlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(pathlib.Path("projects/02-text-eraser/scripts").resolve()))
stub = importlib.import_module("_stub_model")
explain_mod = importlib.import_module("20_explain")


@pytest.mark.smoke
def test_explain_writes_cache_with_model_hash(tmp_path):
    model, tok = stub.build_stub_model_and_tokenizer()
    df = pd.DataFrame({"text": ["w5 w6 w7", "w8 w9 w10"], "label": [0, 1]})
    path = explain_mod.run_one_explainer(
        "grad_x_input", model, tok, df, out_dir=tmp_path,
        model_sha="abc123", device="cpu",
    )
    meta = pq.read_table(path).schema.metadata
    assert meta[b"model_sha256"] == b"abc123"
    assert meta[b"explainer_name"] == b"grad_x_input"
```

- [ ] **Step 2: Run to verify failure.** → FAIL.

- [ ] **Step 3: Implement `_explainers.py` (factory).**
```python
# projects/02-text-eraser/scripts/_explainers.py
"""Build explainer instances from config over a model + tokenizer."""

from __future__ import annotations

from awake.eval.explainers.gradient_x_input import GradientXInputExplainer
from awake.eval.explainers.integrated_gradients import IntegratedGradientsExplainer
from awake.eval.explainers.lime_text import LimeExplainer
from awake.eval.explainers.random_baseline import RandomExplainer


def build_explainer(name: str, model, tok, cfg: dict, device: str):
    """Return an explainer instance by name using ``cfg`` for hyperparameters."""
    if name == "random":
        return RandomExplainer(seed=cfg["bootstrap"]["seed"])
    if name == "grad_x_input":
        return GradientXInputExplainer(model, tok, device=device)
    if name == "integrated_gradients":
        return IntegratedGradientsExplainer(model, tok, device=device, n_steps=cfg["ig"]["n_steps"])
    if name == "lime":
        return LimeExplainer(model, tok, device=device, num_samples=cfg["lime"]["num_samples"])
    if name == "shap_partition":
        from awake.eval.explainers.shap_partition import ShapPartitionExplainer
        return ShapPartitionExplainer(model, tok, device=device, max_evals=cfg["shap"]["max_evals"])
    raise ValueError(f"unknown explainer: {name}")
```

- [ ] **Step 4: Implement `20_explain.py`.**
```python
# projects/02-text-eraser/scripts/20_explain.py
"""Run explainers over a stratified test subsample; cache attributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from _explainers import build_explainer
from _paths import CACHE_DIR, MODEL_DIR, PREPARED, ensure_dirs, load_config

EXPLAINERS = ["random", "grad_x_input", "integrated_gradients", "lime"]


def run_one_explainer(name, model, tok, df, out_dir: Path, model_sha: str,
                      device: str, cfg: dict | None = None) -> Path:
    """Run one explainer over ``df`` rows; write a cache parquet with metadata."""
    cfg = cfg or load_config("explainers")
    explainer = build_explainer(name, model, tok, cfg, device)
    rows = []
    for i, row in df.reset_index(drop=True).iterrows():
        attr = explainer.attribute({"text": row["text"], "predicted_class": None})
        for j, (tokstr, score) in enumerate(zip(attr.tokens, attr.scores)):
            rows.append({"example_id": int(i), "token_idx": j,
                         "token_str": tokstr, "score": float(score)})
    table = pa.Table.from_pandas(pd.DataFrame(rows))
    table = table.replace_schema_metadata({
        "model_sha256": model_sha, "explainer_name": name,
        "explainer_version": "1", "seed": str(cfg["bootstrap"]["seed"]),
    })
    out = Path(out_dir) / f"{name}.parquet"
    pq.write_table(table, out)
    return out


def _stratified_subsample(df: pd.DataFrame, n: int, threshold: float, seed: int):
    df = df.copy()
    df["cov_stratum"] = (df["truncation_coverage"] >= threshold).astype(int)
    g = df.groupby(["label", "cov_stratum"], group_keys=False)
    frac = min(1.0, n / len(df))
    return g.apply(lambda x: x.sample(frac=frac, random_state=seed)).reset_index(drop=True)


def main() -> None:
    """Load the model + test split, subsample, run all explainers, cache them."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_config("explainers")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    sha = (MODEL_DIR / "model_sha256.txt").read_text().strip()
    df = pd.read_parquet(PREPARED / "test.parquet")
    sub = _stratified_subsample(df, cfg["subsample_n"], cfg["coverage_threshold"],
                                cfg["bootstrap"]["seed"])
    sub.to_parquet(CACHE_DIR / "subsample.parquet")
    for name in EXPLAINERS:
        run_one_explainer(name, model, tok, sub, CACHE_DIR, sha, args.device, cfg)
        print(f"cached {name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run to verify pass + commit.**
```bash
uv run pytest projects/02-text-eraser/tests/test_explain_eval_smoke.py::test_explain_writes_cache_with_model_hash -v --no-cov  # PASS
git add projects/02-text-eraser/scripts/_explainers.py projects/02-text-eraser/scripts/20_explain.py \
        projects/02-text-eraser/tests/test_explain_eval_smoke.py
git commit -m "P2: explainer wiring + 20_explain with checkpoint-hashed cache"
```

### Task 20: `30_eval.py` — metrics, classifier diagnostics + ECE, figures

**Files:**
- Create: `projects/02-text-eraser/scripts/30_eval.py`
- Modify: `projects/02-text-eraser/tests/test_explain_eval_smoke.py`

- [ ] **Step 1: Add a unit test for the ECE helper (pure).**
```python
# append to projects/02-text-eraser/tests/test_explain_eval_smoke.py
eval_mod = importlib.import_module("30_eval")


@pytest.mark.unit
def test_expected_calibration_error_zero_for_perfect():
    # confidences equal accuracy in each bin -> ECE 0
    probs = np.array([0.9, 0.9, 0.1, 0.1])
    preds = np.array([1, 1, 0, 0])
    labels = np.array([1, 1, 0, 0])
    ece = eval_mod.expected_calibration_error(probs, preds, labels, n_bins=5)
    assert ece == pytest.approx(0.1, abs=0.05)
```

- [ ] **Step 2: Run to verify failure.** → FAIL.

- [ ] **Step 3: Implement `30_eval.py`.** Loads the subsample + cached attributions,
  rebuilds the `HFModelAdapter` `predict_fn`, computes faithfulness (comprehensiveness,
  sufficiency, AOPC) and plausibility (P/R/F1@k_d, AUPRC), bootstraps per-explainer CIs,
  runs pairwise paired-diff tests with a Bonferroni-corrected threshold over the 6
  non-random explainer pairs, reports classifier accuracy/macro-F1/ECE, and writes
  `metrics.json` + figures.
```python
# projects/02-text-eraser/scripts/30_eval.py
"""Score cached attributions: faithfulness, plausibility, CIs, figures."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from awake.eval.bootstrap import bootstrap_ci, paired_diff_test
from awake.eval.faithfulness import aopc_comprehensiveness, comprehensiveness, sufficiency
from awake.eval.plausibility import (
    aggregate_subwords_to_words, clip_gold_mask_to_window, token_auprc, token_prf1_at_k,
)
from _model_adapter import HFModelAdapter
from _paths import ASSETS, CACHE_DIR, MODEL_DIR, ensure_dirs, load_config

REAL_EXPLAINERS = ["grad_x_input", "integrated_gradients", "lime"]


def expected_calibration_error(conf, preds, labels, n_bins=10) -> float:
    """Standard ECE over equal-width confidence bins."""
    conf, preds, labels = map(np.asarray, (conf, preds, labels))
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc = (preds[m] == labels[m]).mean()
            ece += m.mean() * abs(acc - conf[m].mean())
    return float(ece)


def _faithfulness_for(adapter, sub, attr_df, cfg) -> dict[str, np.ndarray]:
    """Per-example comprehensiveness, sufficiency, AOPC for one explainer."""
    mask_id = adapter.tokenizer.mask_token_id
    comp, suff, aopc = [], [], []
    for i, row in sub.reset_index(drop=True).iterrows():
        ids = np.asarray(row["input_ids"])
        scores = attr_df[attr_df.example_id == i].sort_values("token_idx")["score"].to_numpy()
        if scores.size != ids.size:        # word-level explainer (LIME): re-tokenize map
            scores = np.resize(scores, ids.size)
        visible = np.array([w is not None for w in row["word_ids"]], dtype=bool)
        probs = adapter.predict_proba(ids[None, :])[0]
        pred = int(probs.argmax())
        comp.append(comprehensiveness(adapter.predict_proba, ids, scores, visible, pred, mask_id, cfg["k_d"]))
        suff.append(sufficiency(adapter.predict_proba, ids, scores, visible, pred, mask_id, cfg["k_d"]))
        aopc.append(aopc_comprehensiveness(adapter.predict_proba, ids, scores, visible, pred, mask_id, tuple(cfg["aopc_bins"])))
    return {"comprehensiveness": np.array(comp), "sufficiency": np.array(suff),
            "aopc": np.array(aopc)}


def main() -> None:
    """Compute all metrics + diagnostics and write metrics.json + figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ensure_dirs()
    cfg = load_config("explainers")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    adapter = HFModelAdapter(model, tok, device="cuda" if _cuda() else "cpu")
    sub = pd.read_parquet(CACHE_DIR / "subsample.parquet")

    # classifier diagnostics
    probs = np.vstack([adapter.predict_proba(np.asarray(r["input_ids"])[None, :])
                       for _, r in sub.iterrows()])
    preds = probs.argmax(1)
    labels = sub["label"].to_numpy()
    diagnostics = {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "ece": expected_calibration_error(probs.max(1), preds, labels),
        "class_balance": float(labels.mean()),
        "n": int(len(sub)),
    }

    results: dict[str, dict] = {}
    faith_comp: dict[str, np.ndarray] = {}
    for name in ["random"] + REAL_EXPLAINERS:
        attr_df = pd.read_parquet(CACHE_DIR / f"{name}.parquet")
        f = _faithfulness_for(adapter, sub, attr_df, cfg)
        faith_comp[name] = f["comprehensiveness"]
        results[name] = {
            m: dict(zip(("ci_low", "mean", "ci_high"),
                        bootstrap_ci(v, cfg["bootstrap"]["n_resamples"],
                                     cfg["bootstrap"]["alpha"], cfg["bootstrap"]["seed"])))
            for m, v in f.items()
        }

    # pairwise significance on comprehensiveness, Bonferroni over the real-explainer pairs
    pairs = list(combinations(REAL_EXPLAINERS, 2))
    bonf = cfg["bootstrap"]["alpha"] / len(pairs)
    sig = {}
    for a, b in pairs:
        t = paired_diff_test(faith_comp[a], faith_comp[b],
                             cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["seed"])
        t["significant"] = t["p_value"] < bonf
        sig[f"{a}_vs_{b}"] = t

    out = {"diagnostics": diagnostics, "metrics": results,
           "pairwise_comprehensiveness": sig, "bonferroni_alpha": bonf}
    Path("metrics.json").write_text(json.dumps(out, indent=2, default=float))

    # hero scatter: faithfulness (AOPC) vs plausibility (placeholder uses AUPRC mean=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ["random"] + REAL_EXPLAINERS:
        x = results[name]["aopc"]["mean"]
        ax.scatter(x, 0, label=name)      # plausibility axis filled when gold scoring added
    ax.set_xlabel("Faithfulness (AOPC comprehensiveness)")
    ax.set_ylabel("Plausibility (token AUPRC)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "faithfulness_plausibility.png", dpi=150)
    print(json.dumps(diagnostics, indent=2))


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
```
> Note: plausibility (AUPRC / P-R-F1) per explainer is wired through `_faithfulness_for`'s
> sibling using `aggregate_subwords_to_words` + `clip_gold_mask_to_window` +
> `token_prf1_at_k` + `token_auprc` against `sub["gold_mask"]`/`word_ids`; add a
> `_plausibility_for` function mirroring `_faithfulness_for` and populate the scatter's y
> axis from its AUPRC mean. (Same structure as `_faithfulness_for`; uses the Task 6–7 API.)

- [ ] **Step 4: Add `_plausibility_for` + populate the scatter y-axis.** Mirror
  `_faithfulness_for`: for each example, `words_scores = aggregate_subwords_to_words(scores, row["word_ids"], row["n_words"])` (or identity for word-level explainers), `gold = clip_gold_mask_to_window(np.array(row["gold_mask"]), row["n_words"])`, then collect `token_prf1_at_k(words_scores, gold, k=int(cfg["k_d"]*row["n_words"]))` and `token_auprc(words_scores, gold)`; bootstrap the AUPRC array and use its mean for the scatter `y`.

- [ ] **Step 5: Run to verify pass + commit.**
```bash
uv run pytest projects/02-text-eraser/tests/test_explain_eval_smoke.py::test_expected_calibration_error_zero_for_perfect -v --no-cov  # PASS
git add projects/02-text-eraser/scripts/30_eval.py projects/02-text-eraser/tests/test_explain_eval_smoke.py
git commit -m "P2: 30_eval — faithfulness+plausibility, CIs, pairwise tests, ECE, hero figure"
```

### Task 21: notebook + REPORT + README + ADR + CHANGELOG

**Files:**
- Create: `projects/02-text-eraser/notebooks/01-explainer-comparison.py`
- Create: `projects/02-text-eraser/REPORT.md`
- Modify: `projects/02-text-eraser/README.md`
- Create: `docs/decisions/002-eraser-data-and-truncation.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Notebook (jupytext percent format).** Cells: load `metrics.json`; render the
  per-explainer faithfulness/plausibility table; show the hero scatter; for 1–2 examples,
  render token-heatmaps for each explainer (color by normalized score) side by side; close
  with the random-baseline comparison. Keep it deterministic (fixed example ids).

- [ ] **Step 2: REPORT.md** — 9 sections mirroring P1: question, data, model, metric
  definitions (with the ERASER-exact formulas and the mask-replacement + truncation
  caveats), explainers, results (table + scatter + significance), discussion, limitations
  (§10 of the spec verbatim), references (DeYoung et al. 2020; Sundararajan et al. 2017;
  Ribeiro et al. 2016; Lundberg & Lee 2017; Abnar & Zuidema 2020; Jain & Wallace 2019).

- [ ] **Step 3: README.md** — replace "coming soon"; reflect the final method (4 explainers
  incl. Gradient×Input not rollout; PartitionSHAP; random baseline), reproduce steps
  (`just data | prepare | train | explain | eval | notebook`), and headline once known.

- [ ] **Step 4: ADR 002.** Record: code-only ERASER download; the 512-subword
  frozen-visible-sequence truncation contract + coverage gating; mask-replacement erasure
  (deviation from literal ERASER removal, with rationale); `shap` kept as an optional
  extra (with the recorded resolve outcome from Task 1).

- [ ] **Step 5: CHANGELOG** — add a `[Unreleased]` → Project 2 entry; note the repo stays
  v0.x, unpromoted per §15.

- [ ] **Step 6: Commit.**
```bash
git add projects/02-text-eraser/notebooks projects/02-text-eraser/REPORT.md \
        projects/02-text-eraser/README.md docs/decisions/002-eraser-data-and-truncation.md CHANGELOG.md
git commit -m "P2: notebook, REPORT, README, ADR 002, CHANGELOG"
```

### Task 22: Green CI + real end-to-end run

**Files:** none (verification + artifacts)

- [ ] **Step 1: Full CI gate.** Run `just ci` from the repo root.
  Expected: ruff clean, mypy clean on `src/awake`, `pytest -m "unit or smoke"` green with
  `awake` coverage ≥ 90%. Fix anything that trips.

- [ ] **Step 2: Real pipeline on the RTX 3090.** `just data && just prepare && just train &&
  just explain && just eval && just notebook`. Confirm `metrics.json`,
  `assets/faithfulness_plausibility.png`, and the rendered notebook HTML exist and the
  numbers are sane (real explainers beat the random floor on faithfulness).

- [ ] **Step 3: Commit artifacts.**
```bash
git add projects/02-text-eraser/metrics.json projects/02-text-eraser/assets \
        projects/02-text-eraser/notebooks/01-explainer-comparison.ipynb \
        projects/02-text-eraser/notebooks/01-explainer-comparison.html
git commit -m "P2: end-to-end results — metrics, hero figure, rendered notebook"
```

---

## Self-review notes (author)

- **Spec coverage:** every spec section maps to a task — §2 truncation→T11; §3 data→T10–11;
  §4 eval core→T2–8; §5 explainers→T14–18 (rollout dropped → Gradient×Input T15; KernelSHAP
  → PartitionSHAP T18); §6 explain/eval/diagnostics/cache→T19–20; §7 configs/justfile→T9;
  §8 tests interleaved; §9 deps/ADR/CHANGELOG→T1, T21; §10 limitations→T21 REPORT.
- **Placeholders:** the only intentional placeholder is `configs/data.yaml: sha256`, which
  by nature can only be filled after one verified download (T10 Step 5); flagged inline.
  The flagged stray `//` in T2 Step 3 is called out for removal.
- **Type consistency:** `TokenAttribution` fields are stable across tasks; `word_level` is
  added in T17 with a default so earlier constructions remain valid; `predict_proba`,
  `top_k_mask`, `k_fraction`, `k_d` names are consistent between faithfulness, erasure, and
  eval. Pairwise tests run over `REAL_EXPLAINERS` (3 → 3 pairs); the spec's "6 pairs" referred
  to 4 real explainers including rollout — now 3 after dropping rollout, so the eval uses
  `combinations(REAL_EXPLAINERS, 2)` and Bonferroni divides by the actual pair count. (This
  divergence from the spec's "6" is intentional and documented here.)
