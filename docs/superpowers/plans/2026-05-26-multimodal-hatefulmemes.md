# Multimodal Hateful-Memes (Project 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `projects/03-multimodal-hatefulmemes`: frozen CLIP encodes Hateful-Memes image+text, a LightGBM head classifies the concatenated embeddings (with image-only/text-only baselines), and an exact 2-player interventional **modality Shapley** attributes each decision to image vs text — plus a licence-safe Gradio HuggingFace Space.

**Architecture:** Pure, deterministic metric code (`shapley_2player`, `interventional_values`, `modality_shapley`, `occlusion_importance`) lives in `src/awake/eval/` and carries the 90% coverage floor via fast unit tests on toy `value_fn`s. CLIP encoding + LightGBM training + attribution are config-driven numbered scripts in `projects/03-multimodal-hatefulmemes/`; all heavy/GPU/data work is `slow`-marked and excluded from CPU CI, with smoke tests on tiny random embeddings. The Space loads the trained head from the HF Model Hub and uses a committed generic (non-HM) background — no Hateful-Memes-derived data is ever committed.

**Tech Stack:** Python 3.11, uv, just, pytest (markers unit/smoke/slow), transformers (`CLIPModel`), torch, lightgbm, scikit-learn, numpy, pandas/pyarrow, matplotlib, gradio, huggingface_hub, pillow.

**Reference spec:** `docs/superpowers/specs/2026-05-26-multimodal-hatefulmemes-design.md` — read it before starting. The dataset licence (Meta HM Agreement) is binding: publish the trained model, **never** the dataset or its CLIP embeddings; no raw HM images/text in any committed artifact.

---

## Conventions for every task

- Run from the **repo root** unless stated otherwise; `uv` is at `$HOME/.local/bin/uv`.
- Unit tests for `src/awake/` live in repo-root `tests/` (so `--cov=awake` counts them); project plumbing tests in `projects/03-multimodal-hatefulmemes/tests/`.
- Markers: `@pytest.mark.unit` (fast pure), `@pytest.mark.smoke` (tiny stub, CPU, no CLIP/no gated data), `@pytest.mark.slow` (real CLIP/GPU/data, excluded from CI).
- Single-file runs: `uv run pytest <path> -v --no-cov`. Coverage gate: `uv run pytest tests -m unit`.
- ruff: Google docstrings on `src/awake/`, line 100, double quotes, **no Unicode `×`/`–` in code** (RUF002/003 — use ASCII). Commit after each task.
- `seed_everything` exists: `from awake.utils.seeding import seed_everything`.
- Reuse P2's `awake.eval.bootstrap` (`bootstrap_ci`, `paired_diff_test`) for eval CIs.

## File structure (created by this plan)

```
src/awake/eval/
  modality_shapley.py      # shapley_2player, interventional_values, modality_shapley
  text_occlusion.py        # occlusion_importance
  __init__.py              # + re-export the above
tests/                     # repo-root awake unit tests
  test_eval_modality_shapley.py
  test_eval_text_occlusion.py
projects/03-multimodal-hatefulmemes/
  justfile
  configs/{clip_l14,clip_b32,train}.yaml
  scripts/
    __init__.py  _paths.py  _models.py  _stub.py
    00_data.py 01_encode.py 10_train.py 15_background.py 11_eval.py 20_attribute.py
    build_generic_background.py
  tests/{conftest.py, test_data_smoke.py, test_train_smoke.py, test_attribute_smoke.py}
  notebooks/01-modality-attribution.py     # jupytext source; .ipynb committed WITH outputs
  REPORT.md  (README.md exists — updated)
apps/hatefulmemes-space/
  app.py                   # real, lazy-loaded, licence-safe
  requirements.txt         # +torch(cpu index)/transformers/lightgbm/huggingface_hub
  generic_background.npz   # committed, non-HM
  tests/test_app_smoke.py
.github/workflows/deploy-space.yml   # vendor modality_shapley+text_occlusion; worktree build
.gitignore                            # + apps/hatefulmemes-space/_vendored/
docs/decisions/003-hateful-memes-licence-and-modality-shapley.md
```

---

## Phase 0 — Scaffold and dependencies

### Task 1: Dependencies + gitignore

**Files:** Modify `pyproject.toml`, `.gitignore`.

- [ ] **Step 1:** In `pyproject.toml` `[project].dependencies`, add (skip any already present):
```
  "pillow>=10.0",
  "huggingface-hub>=0.24",
```
(`torch`, `transformers`, `lightgbm`, `scikit-learn`, `matplotlib`, `pandas`, `pyarrow` already present.)

- [ ] **Step 2:** Append to `.gitignore`:
```
projects/03-multimodal-hatefulmemes/outputs/
apps/hatefulmemes-space/_vendored/
```

- [ ] **Step 3:** `uv lock && uv sync --extra dev`, then verify no numba regression: `uv pip list | grep -iE "numba|llvmlite" || echo OK`. Expected `OK`.

- [ ] **Step 4:** Commit.
```bash
git add pyproject.toml uv.lock .gitignore
git commit -m "P3: add pillow + huggingface-hub deps; gitignore outputs + _vendored"
```

---

## Phase 1 — Pure eval core (strict TDD)

### Task 2: `shapley_2player` (exact 2-player closed form)

**Files:** Create `src/awake/eval/modality_shapley.py`; Test `tests/test_eval_modality_shapley.py`.

- [ ] **Step 1: failing test.**
```python
# tests/test_eval_modality_shapley.py
import numpy as np
import pytest
from awake.eval.modality_shapley import shapley_2player


@pytest.mark.unit
def test_shapley_2player_efficiency_and_symmetry():
    phi_a, phi_b = shapley_2player(v_empty=0.0, v_a=2.0, v_b=4.0, v_ab=10.0)
    # phi_a = ½[(2-0)+(10-4)] = 4 ; phi_b = ½[(4-0)+(10-2)] = 6
    assert phi_a == pytest.approx(4.0)
    assert phi_b == pytest.approx(6.0)
    assert phi_a + phi_b == pytest.approx(10.0 - 0.0)  # efficiency


@pytest.mark.unit
def test_shapley_2player_symmetric_inputs_give_equal_phi():
    phi_a, phi_b = shapley_2player(v_empty=1.0, v_a=3.0, v_b=3.0, v_ab=5.0)
    assert phi_a == pytest.approx(phi_b)
```

- [ ] **Step 2: run → FAIL.** `uv run pytest tests/test_eval_modality_shapley.py -v --no-cov`

- [ ] **Step 3: implement.**
```python
# src/awake/eval/modality_shapley.py
"""Exact 2-player interventional modality Shapley over (image, text) feature groups."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

ValueFn = Callable[[np.ndarray], np.ndarray]  # (M, 2D) batch of concat embeddings -> (M,) logits


def shapley_2player(v_empty: float, v_a: float, v_b: float, v_ab: float) -> tuple[float, float]:
    """Exact Shapley values for a 2-player coalition game.

    Args:
        v_empty: Value of the empty coalition.
        v_a: Value of the coalition {a}.
        v_b: Value of the coalition {b}.
        v_ab: Value of the full coalition {a, b}.

    Returns:
        ``(phi_a, phi_b)``; efficiency holds: ``phi_a + phi_b == v_ab - v_empty``.
    """
    phi_a = 0.5 * ((v_a - v_empty) + (v_ab - v_b))
    phi_b = 0.5 * ((v_b - v_empty) + (v_ab - v_a))
    return phi_a, phi_b
```

- [ ] **Step 4: run → PASS.** **Step 5: commit.**
```bash
git add src/awake/eval/modality_shapley.py tests/test_eval_modality_shapley.py
git commit -m "P3: exact 2-player Shapley closed form"
```

### Task 3: `interventional_values` (deterministic, vectorised) + `modality_shapley`

**Files:** Modify `src/awake/eval/modality_shapley.py`; Modify `tests/test_eval_modality_shapley.py`.

- [ ] **Step 1: append failing tests.** A linear `value_fn` (sum of all features) makes every coalition value hand-computable, and verifies `v_ab` is the *point* score (not background-averaged).
```python
# append to tests/test_eval_modality_shapley.py
from awake.eval.modality_shapley import interventional_values, modality_shapley


def _sum_value_fn(batch: np.ndarray) -> np.ndarray:
    """Linear value function: sum across feature dims. batch is (M, 2D)."""
    return batch.sum(axis=1)


@pytest.mark.unit
def test_interventional_values_linear_closed_form():
    img = np.array([1.0, 1.0])          # sum 2
    txt = np.array([10.0, 10.0])        # sum 20
    img_bg = np.array([[0.0, 0.0], [2.0, 2.0]])   # sums 0, 4 -> mean 2
    txt_bg = np.array([[0.0, 0.0], [6.0, 6.0]])   # sums 0, 12 -> mean 6
    v_empty, v_img, v_txt, v_ab = interventional_values(
        img, txt, _sum_value_fn, img_bg, txt_bg
    )
    assert v_ab == pytest.approx(22.0)              # point score: 2 + 20, NOT averaged
    assert v_img == pytest.approx(2.0 + 6.0)        # actual img + mean bg txt
    assert v_txt == pytest.approx(2.0 + 20.0)       # mean bg img + actual txt
    assert v_empty == pytest.approx(2.0 + 6.0)      # mean bg img + mean bg txt


@pytest.mark.unit
def test_modality_shapley_efficiency_end_to_end():
    img = np.array([1.0, 1.0]); txt = np.array([10.0, 10.0])
    img_bg = np.array([[0.0, 0.0], [2.0, 2.0]]); txt_bg = np.array([[0.0, 0.0], [6.0, 6.0]])
    out = modality_shapley(img, txt, _sum_value_fn, img_bg, txt_bg)
    v_empty, _, _, v_ab = interventional_values(img, txt, _sum_value_fn, img_bg, txt_bg)
    assert set(out) == {"image", "text"}
    assert out["image"] + out["text"] == pytest.approx(v_ab - v_empty)
```

- [ ] **Step 2: run → FAIL.**

- [ ] **Step 3: append implementation.**
```python
def interventional_values(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    value_fn: ValueFn,
    img_background: np.ndarray,
    txt_background: np.ndarray,
) -> tuple[float, float, float, float]:
    """Deterministic interventional coalition values for the 2 modalities.

    ``v({img,txt})`` is the point score on the actual example; absent modalities are
    replaced by every (row-paired) background embedding and averaged. No sampling.

    Args:
        img_emb: Image embedding of the example, shape ``(D_img,)``.
        txt_emb: Text embedding of the example, shape ``(D_txt,)``.
        value_fn: Maps a ``(M, D_img + D_txt)`` batch to ``(M,)`` scores (logits).
        img_background: Background image embeddings, shape ``(N, D_img)``.
        txt_background: Background text embeddings, shape ``(N, D_txt)`` (row-paired).

    Returns:
        ``(v_empty, v_img, v_txt, v_ab)``.
    """
    n = img_background.shape[0]
    img_tiled = np.tile(img_emb, (n, 1))
    txt_tiled = np.tile(txt_emb, (n, 1))
    v_ab = float(value_fn(np.concatenate([img_emb, txt_emb])[None, :])[0])
    v_img = float(value_fn(np.concatenate([img_tiled, txt_background], axis=1)).mean())
    v_txt = float(value_fn(np.concatenate([img_background, txt_tiled], axis=1)).mean())
    v_empty = float(value_fn(np.concatenate([img_background, txt_background], axis=1)).mean())
    return v_empty, v_img, v_txt, v_ab


def modality_shapley(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    value_fn: ValueFn,
    img_background: np.ndarray,
    txt_background: np.ndarray,
) -> dict[str, float]:
    """Return ``{"image": phi_img, "text": phi_txt}`` for one example."""
    v_empty, v_img, v_txt, v_ab = interventional_values(
        img_emb, txt_emb, value_fn, img_background, txt_background
    )
    phi_img, phi_txt = shapley_2player(v_empty, v_img, v_txt, v_ab)
    return {"image": phi_img, "text": phi_txt}
```

- [ ] **Step 4: run → PASS. Step 5: commit.**
```bash
git add src/awake/eval/modality_shapley.py tests/test_eval_modality_shapley.py
git commit -m "P3: deterministic vectorised interventional values + modality_shapley"
```

### Task 4: `text_occlusion.occlusion_importance`

**Files:** Create `src/awake/eval/text_occlusion.py`; Test `tests/test_eval_text_occlusion.py`.

- [ ] **Step 1: failing test.**
```python
# tests/test_eval_text_occlusion.py
import pytest
from awake.eval.text_occlusion import occlusion_importance


def _score_fn(tokens):
    """Score = number of 'bad' tokens (additive ground truth)."""
    return float(sum(t == "bad" for t in tokens))


@pytest.mark.unit
def test_occlusion_ranks_influential_token_first():
    tokens = ["the", "movie", "is", "bad"]
    out = occlusion_importance(tokens, _score_fn, top_k=2)
    assert out[0][0] == "bad"
    assert out[0][1] == pytest.approx(1.0)   # removing "bad" drops score by 1
    assert len(out) == 2


@pytest.mark.unit
def test_occlusion_top_k_caps_results():
    tokens = ["bad", "bad", "good", "ok"]
    out = occlusion_importance(tokens, _score_fn, top_k=1)
    assert len(out) == 1 and out[0][1] == pytest.approx(1.0)
```

- [ ] **Step 2: run → FAIL.**

- [ ] **Step 3: implement.**
```python
# src/awake/eval/text_occlusion.py
"""Leave-one-out text occlusion importance for token-level attribution."""

from __future__ import annotations

from collections.abc import Callable

ScoreFn = Callable[[list[str]], float]


def occlusion_importance(
    tokens: list[str], score_fn: ScoreFn, top_k: int = 5
) -> list[tuple[str, float]]:
    """Rank tokens by the score drop when each is removed (leave-one-out).

    Args:
        tokens: The (whitespace) tokens of the text.
        score_fn: Maps a token list to a scalar score (e.g. predicted-class logit).
        top_k: Number of highest-magnitude tokens to return.

    Returns:
        ``(token, importance)`` pairs sorted by descending ``abs(importance)``,
        truncated to ``top_k``; ``importance = score(all) - score(all without token)``.
    """
    base = score_fn(tokens)
    scored = [
        (tok, base - score_fn(tokens[:i] + tokens[i + 1 :]))
        for i, tok in enumerate(tokens)
    ]
    scored.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return scored[:top_k]
```

- [ ] **Step 4: run → PASS. Step 5: commit.**
```bash
git add src/awake/eval/text_occlusion.py tests/test_eval_text_occlusion.py
git commit -m "P3: leave-one-out text occlusion importance"
```

### Task 5: Export from `awake.eval.__init__` + coverage gate

**Files:** Modify `src/awake/eval/__init__.py`.

- [ ] **Step 1:** Add to the imports and `__all__` in `src/awake/eval/__init__.py`:
```python
from awake.eval.modality_shapley import interventional_values, modality_shapley, shapley_2player
from awake.eval.text_occlusion import occlusion_importance
```
and append `"interventional_values", "modality_shapley", "shapley_2player", "occlusion_importance"` to `__all__`.

- [ ] **Step 2: coverage gate.** `uv run pytest tests -m unit` — must pass with awake coverage ≥ 90% (`--cov-fail-under=90`). The two new modules are fully unit-covered; if any line is missed, add a targeted test (e.g. `top_k` larger than len(tokens); empty background guard). Do not lower the threshold.

- [ ] **Step 3:** `uv run ruff check src/awake/eval/` clean; **commit.**
```bash
git add src/awake/eval/__init__.py tests/
git commit -m "P3: export modality_shapley + text_occlusion from awake.eval"
```

---

## Phase 2 — Project scaffold + data

### Task 6: scaffold (`_paths`, configs, justfile, conftest with isolation fixture)

**Files:** Create `projects/03-multimodal-hatefulmemes/{scripts/__init__.py, scripts/_paths.py, configs/clip_l14.yaml, configs/clip_b32.yaml, configs/train.yaml, justfile, tests/conftest.py}`.

- [ ] **Step 1: `scripts/_paths.py`.**
```python
"""Filesystem paths and config loading for project 03."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
ASSETS = PROJECT_ROOT / "assets"
CONFIGS = PROJECT_ROOT / "configs"
DATA_PATH = Path(os.environ.get("DATA_PATH", Path.home() / ".cache/hateful_memes"))


def load_config(name: str) -> dict:
    """Load ``configs/<name>.yaml`` as a dict."""
    with open(CONFIGS / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def embeddings_dir(variant_cfg: dict) -> Path:
    """Return the embedding cache dir for a CLIP variant config."""
    return OUTPUTS / variant_cfg["embedding_subdir"]


def models_dir(variant_cfg: dict) -> Path:
    """Return the models dir for a CLIP variant config."""
    return OUTPUTS / "models" / variant_cfg["embedding_subdir"].split("/")[-1]


def ensure_dirs(*paths: Path) -> None:
    """Create each path (and parents)."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: configs.**
```yaml
# configs/clip_l14.yaml
clip_model_id: "openai/clip-vit-large-patch14"
embedding_subdir: "embeddings/l14"
```
```yaml
# configs/clip_b32.yaml
clip_model_id: "openai/clip-vit-base-patch32"
embedding_subdir: "embeddings/b32"
```
```yaml
# configs/train.yaml
random_state: 1337
n_background: 200
background_seed: 1337
lgbm:
  n_estimators: 400
  learning_rate: 0.05
  num_leaves: 31
```

- [ ] **Step 3: justfile** (mirrors P1/P2; `data` verifies, `export-space-artifacts` builds B/32 + uploads head).
```just
set shell := ["bash", "-uc"]
set dotenv-load := true
project_root := justfile_directory()
repo_root := justfile_directory() + "/../.."
default_data_path := env_var_or_default("DATA_PATH", env_var("HOME") + "/.cache/hateful_memes")

default:
    @just --list

data:
    cd "{{project_root}}" && uv run python scripts/00_data.py --data-path "{{default_data_path}}"

encode CONFIG="clip_l14":
    cd "{{project_root}}" && uv run python scripts/01_encode.py --config {{CONFIG}}

train CONFIG="clip_l14":
    cd "{{project_root}}" && uv run python scripts/10_train.py --config {{CONFIG}} && \
        uv run python scripts/15_background.py --config {{CONFIG}}

eval CONFIG="clip_l14":
    cd "{{project_root}}" && uv run python scripts/11_eval.py --config {{CONFIG}}

attribute CONFIG="clip_l14":
    cd "{{project_root}}" && uv run python scripts/20_attribute.py --config {{CONFIG}}

pipeline CONFIG="clip_l14": data (encode CONFIG) (train CONFIG) (eval CONFIG) (attribute CONFIG)

export-space-artifacts:
    cd "{{project_root}}" && uv run python scripts/01_encode.py --config clip_b32 && \
        uv run python scripts/10_train.py --config clip_b32 && \
        uv run python scripts/15_background.py --config clip_b32 && \
        echo "Upload outputs/models/b32/fused.txt + head_meta.json to your HF Model Hub repo (see REPORT)."

notebook NAME="01-modality-attribution":
    cd "{{project_root}}/notebooks" && \
        P3_PROJECT_ROOT="{{project_root}}" uv run jupytext --to ipynb {{NAME}}.py && \
        P3_PROJECT_ROOT="{{project_root}}" uv run jupyter nbconvert --to notebook --execute --inplace {{NAME}}.ipynb && \
        uv run jupyter nbconvert --to html {{NAME}}.ipynb

test:
    cd "{{repo_root}}" && uv run pytest projects/03-multimodal-hatefulmemes/tests -m "unit or smoke" --no-cov
```

- [ ] **Step 4: `tests/conftest.py`** — copy P2's per-project isolation fixture verbatim (evicts colliding bare script names like `_paths`/`10_train` from `sys.modules` so P1/P2/P3 don't collide in a combined pytest run):
```python
"""Make this project's scripts/ importable in tests; isolate from sibling projects."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
_OWNED = {p.stem for p in SCRIPTS_DIR.glob("*.py") if not p.stem.startswith("__")}

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


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
Also create empty `scripts/__init__.py`.

- [ ] **Step 5:** `uv run ruff check projects/03-multimodal-hatefulmemes/scripts/_paths.py projects/03-multimodal-hatefulmemes/tests/conftest.py` clean; **commit.**
```bash
git add projects/03-multimodal-hatefulmemes/scripts projects/03-multimodal-hatefulmemes/configs \
        projects/03-multimodal-hatefulmemes/justfile projects/03-multimodal-hatefulmemes/tests/conftest.py
git commit -m "P3: project scaffold — paths, configs, justfile, conftest isolation"
```

### Task 7: `00_data.py` — verify + extract gated data

**Files:** Create `scripts/00_data.py`; Test `tests/test_data_smoke.py`.

- [ ] **Step 1: failing unit test (pure parsing helper, no real data).**
```python
# projects/03-multimodal-hatefulmemes/tests/test_data_smoke.py
import importlib
import json
import pytest

data_mod = importlib.import_module("00_data")


@pytest.mark.unit
def test_load_split_parses_records(tmp_path):
    f = tmp_path / "dev.jsonl"
    f.write_text(
        json.dumps({"id": 1, "img": "img/1.png", "label": 0, "text": "a"}) + "\n"
        + json.dumps({"id": 2, "img": "img/2.png", "label": 1, "text": "b"})  # no trailing nl
    )
    rows = data_mod.load_split(f)
    assert len(rows) == 2          # both records, despite missing trailing newline
    assert rows[1]["label"] == 1
```

- [ ] **Step 2: run → FAIL.**

- [ ] **Step 3: implement `scripts/00_data.py`.**
```python
"""Verify (and if needed extract) the gated Hateful Memes data. Never downloads/commits."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from _paths import DATA_PATH


def load_split(path: Path) -> list[dict]:
    """Parse a Hateful Memes ``.jsonl`` split into a list of records."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _verify(root: Path) -> bool:
    need = [root / "img", root / "train.jsonl", root / "dev.jsonl", root / "test.jsonl"]
    return all(p.exists() for p in need)


def main() -> None:
    """Verify the dataset at ``--data-path``; extract from a local archive if present."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", type=Path, default=DATA_PATH)
    ap.add_argument("--archive", type=Path, default=Path.home() / "Downloads/archive.zip")
    args = ap.parse_args()
    root = args.data_path
    if not _verify(root) and args.archive.exists():
        print(f"Extracting {args.archive} -> {root}")
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.archive) as z:
            z.extractall(root)
        # the Kaggle archive nests under data/
        if (root / "data" / "dev.jsonl").exists():
            root = root / "data"
    if not _verify(root):
        raise SystemExit(
            "FATAL: Hateful Memes not found.\n"
            "This dataset is gated by Meta's licence (accept terms; do not redistribute).\n"
            f"Place the licensed archive at {args.archive} or the extracted files at {root}\n"
            "(expected: img/, train.jsonl, dev.jsonl, test.jsonl)."
        )
    counts = {s: len(load_split(root / f"{s}.jsonl")) for s in ("train", "dev", "test")}
    print(f"OK: {root}  counts={counts}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: run → PASS. Step 5: commit.**
```bash
git add projects/03-multimodal-hatefulmemes/scripts/00_data.py projects/03-multimodal-hatefulmemes/tests/test_data_smoke.py
git commit -m "P3: data verify+extract (gated; no download/commit)"
```

---

## Phase 3 — Encode, models, background

### Task 8: `_models.py` (CLIP encoder + LightGBM value_fn) and `_stub.py`

**Files:** Create `scripts/_models.py`, `scripts/_stub.py`.

- [ ] **Step 1: `_models.py`** — CLIP encoding + a `margin_value_fn` factory that wraps a trained LightGBM booster's raw margin into the `(M,2D)->(M,)` `value_fn` the core expects.
```python
"""CLIP encoding + LightGBM raw-margin value_fn for project 03."""

from __future__ import annotations

import numpy as np


def load_clip(model_id: str, device: str = "cpu"):
    """Load a frozen CLIP model + processor (eval mode)."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    proc = CLIPProcessor.from_pretrained(model_id)
    torch.set_grad_enabled(False)
    return model, proc


def encode(model, proc, images, texts, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Return ``(img_emb, txt_emb)`` arrays for parallel lists of PIL images + strings."""
    import torch

    with torch.no_grad():
        pix = proc(images=images, return_tensors="pt", padding=True).to(device)
        img = model.get_image_features(**pix)
        tok = proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        txt = model.get_text_features(**tok)
    return img.cpu().numpy(), txt.cpu().numpy()


def margin_value_fn(booster):
    """Wrap a LightGBM booster as a value_fn: (M, 2D) concat embeddings -> (M,) raw margin."""
    def _fn(batch: np.ndarray) -> np.ndarray:
        return np.asarray(booster.predict(batch, raw_score=True)).ravel()
    return _fn
```

- [ ] **Step 2: `_stub.py`** — tiny artifacts for smoke tests (no CLIP, no data).
```python
"""Tiny in-memory stand-ins for smoke tests (no CLIP, no gated data)."""

from __future__ import annotations

import numpy as np


def tiny_embeddings(n: int = 40, d: int = 8, seed: int = 0):
    """Return (img_emb, txt_emb, labels) random arrays for a fake split."""
    rng = np.random.default_rng(seed)
    img = rng.normal(size=(n, d))
    txt = rng.normal(size=(n, d))
    # a learnable signal: label depends on a couple of dims
    labels = ((img[:, 0] + txt[:, 1]) > 0).astype(int)
    return img, txt, labels
```

- [ ] **Step 3:** `uv run ruff check` both; **commit.**
```bash
git add projects/03-multimodal-hatefulmemes/scripts/_models.py projects/03-multimodal-hatefulmemes/scripts/_stub.py
git commit -m "P3: CLIP encoder + LightGBM margin value_fn + smoke stubs"
```

### Task 9: `01_encode.py` (config-driven CLIP encode → parquet cache)

**Files:** Create `scripts/01_encode.py`. (No unit test — needs real CLIP + data; exercised by the real run. Keep `main` thin.)

- [ ] **Step 1: implement.**
```python
"""Encode Hateful Memes image+text with a frozen CLIP variant; cache per-split parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from _models import encode, load_clip
from _paths import DATA_PATH, embeddings_dir, ensure_dirs, load_config

from importlib import import_module
load_split = import_module("00_data").load_split


def main() -> None:
    """Encode each split's images+texts and cache (id,label,img_emb,txt_emb) parquet."""
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-path", type=Path, default=DATA_PATH)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = embeddings_dir(cfg)
    ensure_dirs(out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = load_clip(cfg["clip_model_id"], device)
    root = args.data_path / "data" if (args.data_path / "data").exists() else args.data_path
    for split in ("train", "dev", "test"):
        rows = load_split(root / f"{split}.jsonl")
        recs = []
        for i in range(0, len(rows), args.batch_size):
            chunk = rows[i : i + args.batch_size]
            imgs = [Image.open(root / r["img"]).convert("RGB") for r in chunk]
            img_emb, txt_emb = encode(model, proc, imgs, [r["text"] for r in chunk], device)
            for r, ie, te in zip(chunk, img_emb, txt_emb, strict=True):
                recs.append({"id": r["id"], "label": r.get("label", -1),
                             "img_emb": ie.tolist(), "txt_emb": te.tolist()})
        pd.DataFrame(recs).to_parquet(out / f"{split}.parquet")
        print(f"{split}: {len(recs)} encoded -> {out / f'{split}.parquet'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** `uv run ruff check` clean; **commit** (no test run — slow path).
```bash
git add projects/03-multimodal-hatefulmemes/scripts/01_encode.py
git commit -m "P3: config-driven CLIP encode with parquet cache"
```

### Task 10: `10_train.py` (fused + unimodal heads, train-CV, scale_pos_weight) + smoke

**Files:** Create `scripts/10_train.py`; Test `tests/test_train_smoke.py`.

- [ ] **Step 1: smoke test on tiny random embeddings (no CLIP/data).**
```python
# projects/03-multimodal-hatefulmemes/tests/test_train_smoke.py
import importlib
import numpy as np
import pytest

stub = importlib.import_module("_stub")
train_mod = importlib.import_module("10_train")


@pytest.mark.smoke
def test_fit_heads_returns_three_boosters():
    img, txt, y = stub.tiny_embeddings()
    heads = train_mod.fit_heads(img, txt, y, lgbm_params={"n_estimators": 20}, seed=0)
    assert set(heads) == {"fused", "image", "text"}
    # fused predicts a raw margin per row
    import numpy as np
    feats = np.concatenate([img, txt], axis=1)
    assert heads["fused"].predict(feats, raw_score=True).shape == (img.shape[0],)
```

- [ ] **Step 2: run → FAIL.**

- [ ] **Step 3: implement.**
```python
"""Train fused + image-only + text-only LightGBM heads on cached CLIP embeddings."""

from __future__ import annotations

import argparse

import numpy as np


def fit_heads(img: np.ndarray, txt: np.ndarray, y: np.ndarray, lgbm_params: dict, seed: int):
    """Fit fused/image/text LightGBM heads; scale_pos_weight from the class ratio."""
    import lightgbm as lgb

    pos = max(int((y == 1).sum()), 1)
    neg = int((y == 0).sum())
    spw = neg / pos
    params = {"objective": "binary", "random_state": seed, "scale_pos_weight": spw,
              "verbosity": -1, **lgbm_params}
    feats = {"fused": np.concatenate([img, txt], axis=1), "image": img, "text": txt}
    heads = {}
    for name, X in feats.items():
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X, y)
        heads[name] = clf.booster_
    return heads


def main() -> None:
    """Load cached train embeddings, fit the three heads, save boosters + sidecar meta."""
    import json
    import lightgbm as lgb  # noqa: F401  (ensures version available for meta)
    import pandas as pd

    from awake.utils.seeding import seed_everything
    from _paths import ensure_dirs, load_config, models_dir, embeddings_dir

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tcfg = load_config("train")
    seed_everything(tcfg["random_state"])
    emb = embeddings_dir(cfg)
    df = pd.read_parquet(emb / "train.parquet")
    img = np.array(df["img_emb"].tolist()); txt = np.array(df["txt_emb"].tolist())
    y = df["label"].to_numpy()
    heads = fit_heads(img, txt, y, tcfg["lgbm"], tcfg["random_state"])
    mdir = models_dir(cfg); ensure_dirs(mdir)
    for name, booster in heads.items():
        booster.save_model(str(mdir / f"{name}.txt"))
    meta = {"clip_model_id": cfg["clip_model_id"], "img_dim": img.shape[1],
            "txt_dim": txt.shape[1], "class_map": {"benign": 0, "hateful": 1},
            "lightgbm_version": lgb.__version__}
    (mdir / "head_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved heads + head_meta.json -> {mdir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: run → PASS. Step 5: commit.**
```bash
git add projects/03-multimodal-hatefulmemes/scripts/10_train.py projects/03-multimodal-hatefulmemes/tests/test_train_smoke.py
git commit -m "P3: train fused+unimodal heads (smoke-tested) + sidecar meta"
```

### Task 11: `15_background.py` (empirical background) — smoke

**Files:** Create `scripts/15_background.py`; Modify `tests/test_train_smoke.py`.

- [ ] **Step 1: append smoke test.**
```python
# append to test_train_smoke.py
bg_mod = importlib.import_module("15_background")


@pytest.mark.smoke
def test_sample_background_shapes():
    img, txt, _ = stub.tiny_embeddings(n=40, d=8)
    ib, tb = bg_mod.sample_background(img, txt, n=10, seed=0)
    assert ib.shape == (10, 8) and tb.shape == (10, 8)
    # deterministic
    ib2, tb2 = bg_mod.sample_background(img, txt, n=10, seed=0)
    assert (ib == ib2).all()
```

- [ ] **Step 2: run → FAIL. Step 3: implement.**
```python
"""Sample a seeded empirical (row-paired) background set from train embeddings."""

from __future__ import annotations

import argparse

import numpy as np


def sample_background(img: np.ndarray, txt: np.ndarray, n: int, seed: int):
    """Return ``(img_bg, txt_bg)`` of paired rows sampled (without replacement) from train."""
    rng = np.random.default_rng(seed)
    n = min(n, img.shape[0])
    idx = rng.choice(img.shape[0], size=n, replace=False)
    return img[idx], txt[idx]


def main() -> None:
    """Save outputs/models/<variant>/background.npz (empirical, paired)."""
    import numpy as np
    import pandas as pd

    from _paths import embeddings_dir, load_config, models_dir

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config); tcfg = load_config("train")
    df = pd.read_parquet(embeddings_dir(cfg) / "train.parquet")
    img = np.array(df["img_emb"].tolist()); txt = np.array(df["txt_emb"].tolist())
    ib, tb = sample_background(img, txt, tcfg["n_background"], tcfg["background_seed"])
    out = models_dir(cfg) / "background.npz"
    np.savez(out, img=ib, txt=tb)
    print(f"saved background {ib.shape} -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: run → PASS. Step 5: commit.**
```bash
git add projects/03-multimodal-hatefulmemes/scripts/15_background.py projects/03-multimodal-hatefulmemes/tests/test_train_smoke.py
git commit -m "P3: empirical paired background sampler (smoke-tested)"
```

---

## Phase 4 — Eval + attribution

### Task 12: `11_eval.py` (metrics + bootstrap CIs + schema)

**Files:** Create `scripts/11_eval.py`; Test `tests/test_attribute_smoke.py`.

- [ ] **Step 1: unit test for the pure metric helper.**
```python
# projects/03-multimodal-hatefulmemes/tests/test_attribute_smoke.py
import importlib
import numpy as np
import pytest

eval_mod = importlib.import_module("11_eval")


@pytest.mark.unit
def test_metric_block_keys_and_ranges():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 50)
    # scores correlated with y -> auroc > 0.5
    scores = y + rng.normal(scale=0.5, size=y.size)
    block = eval_mod.metric_block(y, scores, n_boot=200, seed=0)
    for k in ("auroc", "auprc", "acc"):
        assert {"mean", "lo", "hi"} <= set(block[k])
        assert 0.0 <= block[k]["mean"] <= 1.0
    assert block["auroc"]["lo"] <= block["auroc"]["mean"] <= block["auroc"]["hi"]
```

- [ ] **Step 2: run → FAIL. Step 3: implement** (`main` is slow/real; `metric_block` is pure).
```python
"""Evaluate fused/image/text heads on dev: AUROC/AUPRC/acc with bootstrap CIs."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


def metric_block(y: np.ndarray, scores: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """AUROC/AUPRC/accuracy@0.5 with percentile bootstrap 95% CIs."""
    rng = np.random.default_rng(seed)
    preds = (scores >= 0.5).astype(int) if scores.max() <= 1.0 else (scores >= 0.0).astype(int)

    def _metrics(yi, si, pi):
        return (roc_auc_score(yi, si), average_precision_score(yi, si), accuracy_score(yi, pi))

    boots = {0: [], 1: [], 2: []}
    n = y.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for k, v in enumerate(_metrics(y[idx], scores[idx], preds[idx])):
            boots[k].append(v)
    names = ["auroc", "auprc", "acc"]
    point = _metrics(y, scores, preds)
    return {
        names[k]: {
            "mean": float(point[k]),
            "lo": float(np.percentile(boots[k], 2.5)),
            "hi": float(np.percentile(boots[k], 97.5)),
        }
        for k in range(3)
    }


def main() -> None:
    """Score the three heads on dev probabilities; write metrics.json model blocks + diffs."""
    import json
    import numpy as np
    import pandas as pd
    import lightgbm as lgb

    from awake.eval.bootstrap import paired_diff_test
    from _paths import PROJECT_ROOT, embeddings_dir, load_config, models_dir

    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args(); cfg = load_config(args.config); tcfg = load_config("train")
    mdir = models_dir(cfg)
    df = pd.read_parquet(embeddings_dir(cfg) / "dev.parquet")
    img = np.array(df["img_emb"].tolist()); txt = np.array(df["txt_emb"].tolist())
    y = df["label"].to_numpy()
    feats = {"fused": np.concatenate([img, txt], axis=1), "image": img, "text": txt}
    models = {n: lgb.Booster(model_file=str(mdir / f"{n}.txt")) for n in feats}
    probs = {n: models[n].predict(feats[n]) for n in feats}  # sigmoid prob (objective=binary)
    out = {"split": "dev", "n": int(y.size),
           "models": {n: metric_block(y, probs[n], seed=tcfg["background_seed"]) for n in feats}}
    aurocs = {n: probs[n] for n in feats}
    out["auroc_diffs"] = {
        f"fused_vs_{m}": paired_diff_test(aurocs["fused"], aurocs[m], n_resamples=2000, seed=0)
        for m in ("image", "text")
    }
    (PROJECT_ROOT / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out["models"], indent=2))


if __name__ == "__main__":
    main()
```
> Note: `paired_diff_test` here compares score arrays as a stand-in difference; the headline
> AUROC-difference CI is documented in REPORT as bootstrap over examples. (Keep simple; the
> real run validates. If a true paired AUROC-difference bootstrap is wanted, resample example
> indices and recompute `roc_auc_score` for both heads on each resample — note this in REPORT.)

- [ ] **Step 4: run unit test → PASS. Step 5: commit.**
```bash
git add projects/03-multimodal-hatefulmemes/scripts/11_eval.py projects/03-multimodal-hatefulmemes/tests/test_attribute_smoke.py
git commit -m "P3: dev eval with bootstrap CIs + metrics.json schema"
```

### Task 13: `20_attribute.py` (modality Shapley over dev + hero figure)

**Files:** Create `scripts/20_attribute.py`; Modify `tests/test_attribute_smoke.py`.

- [ ] **Step 1: append smoke test** (uses tiny stub embeddings + a real tiny LightGBM head + the awake core; no CLIP).
```python
# append to test_attribute_smoke.py
stub = importlib.import_module("_stub")
train_mod = importlib.import_module("10_train")
attr_mod = importlib.import_module("20_attribute")


@pytest.mark.smoke
def test_attribute_rows_have_image_text_shares():
    img, txt, y = stub.tiny_embeddings(n=60, d=8)
    heads = train_mod.fit_heads(img, txt, y, {"n_estimators": 20}, seed=0)
    ib, tb = img[:15], txt[:15]
    rows = attr_mod.attribute_split(img[:10], txt[:10], heads["fused"], ib, tb)
    assert len(rows) == 10
    r0 = rows[0]
    assert {"phi_image", "phi_text", "total", "share"} <= set(r0)
    assert -1.0 <= r0["share"] <= 1.0
```

- [ ] **Step 2: run → FAIL. Step 3: implement.**
```python
"""Per-example modality Shapley over dev + aggregate + hero figure."""

from __future__ import annotations

import argparse

import numpy as np

from awake.eval.modality_shapley import modality_shapley
from _models import margin_value_fn

EPS = 1e-9


def attribute_split(img, txt, fused_booster, img_bg, txt_bg) -> list[dict]:
    """Return per-example phi_image/phi_text/total/share rows for a split."""
    value_fn = margin_value_fn(fused_booster)
    rows = []
    for i in range(img.shape[0]):
        phi = modality_shapley(img[i], txt[i], value_fn, img_bg, txt_bg)
        total = phi["image"] + phi["text"]
        denom = abs(phi["image"]) + abs(phi["text"]) + EPS
        rows.append({"phi_image": phi["image"], "phi_text": phi["text"],
                     "total": total, "share": phi["image"] / denom})
    return rows


def main() -> None:
    """Attribute dev, write attribution into metrics.json, render the hero figure."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import lightgbm as lgb

    from _paths import ASSETS, PROJECT_ROOT, embeddings_dir, ensure_dirs, load_config, models_dir

    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args(); cfg = load_config(args.config)
    mdir = models_dir(cfg)
    df = pd.read_parquet(embeddings_dir(cfg) / "dev.parquet")
    img = np.array(df["img_emb"].tolist()); txt = np.array(df["txt_emb"].tolist())
    fused = lgb.Booster(model_file=str(mdir / "fused.txt"))
    bg = np.load(mdir / "background.npz")
    rows = attribute_split(img, txt, fused, bg["img"], bg["txt"])
    shares = np.array([r["share"] for r in rows])
    mean_abs = {"image": float(np.mean([abs(r["phi_image"]) for r in rows])),
                "text": float(np.mean([abs(r["phi_text"]) for r in rows]))}
    metrics = json.loads((PROJECT_ROOT / "metrics.json").read_text())
    metrics["attribution"] = {"mean_abs_phi": mean_abs, "share_mean": float(shares.mean()),
                              "n": len(rows)}
    metrics["background"] = {"type": "empirical_train", "n": int(bg["img"].shape[0])}
    (PROJECT_ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    ensure_dirs(ASSETS)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(shares, bins=30)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("signed image share  phi_img / (|phi_img|+|phi_text|)")
    ax.set_ylabel("dev examples")
    ax.set_title("Modality attribution (image vs text)")
    fig.tight_layout(); fig.savefig(ASSETS / "modality_attribution.png", dpi=150)
    print(f"attribution: mean_abs={mean_abs}  share_mean={shares.mean():.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: run smoke → PASS. Step 5: commit.**
```bash
git add projects/03-multimodal-hatefulmemes/scripts/20_attribute.py projects/03-multimodal-hatefulmemes/tests/test_attribute_smoke.py
git commit -m "P3: modality Shapley attribution over dev + hero figure (smoke-tested)"
```

---

## Phase 5 — Gradio Space (licence-safe) + deploy

### Task 14: `build_generic_background.py` + committed `generic_background.npz`

**Files:** Create `scripts/build_generic_background.py`; create `apps/hatefulmemes-space/generic_background.npz`.

- [ ] **Step 1: implement the builder** (encodes ~50 generic CC0 images + neutral captions with CLIP-B/32 → `.npz`). For the committed artifact, if no CC0 images are handy, generate deterministic random-noise PIL images (clearly generic, non-HM) so the artifact is reproducible and licence-clean.
```python
"""Build a generic (non-Hateful-Memes) background embedding set for the Space."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from _models import encode, load_clip

OUT = Path(__file__).resolve().parents[3] / "apps/hatefulmemes-space/generic_background.npz"
CAPTIONS = ["a photo", "a picture of a landscape", "an everyday scene", "a generic image",
            "some text on a background"] * 10  # 50 generic captions


def main() -> None:
    """Encode 50 generic noise images + neutral captions; save the .npz for the Space."""
    rng = np.random.default_rng(0)
    images = [Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)) for _ in CAPTIONS]
    model, proc = load_clip("openai/clip-vit-base-patch32", "cpu")
    img_emb, txt_emb = encode(model, proc, images, CAPTIONS, "cpu")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, img=img_emb, txt=txt_emb)
    print(f"saved generic background {img_emb.shape} -> {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** Run it once (`uv run python scripts/build_generic_background.py` from the project) to produce `generic_background.npz` (~50×512 float32, well under 2 MB). Verify size `< 2000 KB`.

- [ ] **Step 3: commit** the script + the `.npz`.
```bash
git add projects/03-multimodal-hatefulmemes/scripts/build_generic_background.py apps/hatefulmemes-space/generic_background.npz
git commit -m "P3: generic non-HM background for the Space"
```

### Task 15: real `app.py` (lazy loader, sidecar assert, modality Shapley + occlusion)

**Files:** Modify `apps/hatefulmemes-space/app.py`, `apps/hatefulmemes-space/requirements.txt`; Test `apps/hatefulmemes-space/tests/test_app_smoke.py`.

- [ ] **Step 1: smoke test that injects a stub loader (no CLIP/hub download).**
```python
# apps/hatefulmemes-space/tests/test_app_smoke.py
import importlib
import sys
import pathlib

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(pathlib.Path("apps/hatefulmemes-space").resolve()))
app = importlib.import_module("app")


@pytest.mark.smoke
def test_predict_with_stub_loader(monkeypatch):
    class _Bundle:
        img_bg = np.zeros((4, 8)); txt_bg = np.zeros((4, 8))
        def encode(self, image, text):
            return np.ones(8), np.ones(8)
        def margin(self, feats):  # (M,16)->(M,)
            return feats.sum(axis=1)
        def prob(self, feats):
            return 1 / (1 + np.exp(-feats.sum(axis=1)))
    monkeypatch.setattr(app, "_load", lambda: _Bundle())
    label, conf, bars = app.predict(Image.new("RGB", (8, 8)), "some caption text")[:3]
    assert set(conf) == {"benign", "hateful"}
    assert set(bars) == {"image", "text"}
```

- [ ] **Step 2: run → FAIL. Step 3: implement** `app.py` with a cached lazy `_load()`, sidecar assert, modality bars (Shapley on margin) + top tokens (occlusion). Keep `build_interface()` from the stub but wire `predict` to the real bundle; map margin→prob for display. (Full code: load CLIP-B/32 + `hf_hub_download(repo, "fused.txt")` + `head_meta.json` asserted against the loaded CLIP dims/model id + `generic_background.npz`; `predict` encodes inputs, builds the `value_fn` from the booster margin, calls `awake.eval.modality_shapley`, and `awake.eval.text_occlusion.occlusion_importance` with a `score_fn` that re-joins tokens with spaces, re-encodes text, and returns the fused margin.)

- [ ] **Step 4: `requirements.txt`.**
```
--extra-index-url https://download.pytorch.org/whl/cpu
gradio==4.44.0
torch==2.2.2+cpu
transformers>=4.44
lightgbm>=4.3
huggingface_hub>=0.24
numpy>=1.26
pillow>=10.0
```

- [ ] **Step 5: run smoke → PASS; commit.**
```bash
git add apps/hatefulmemes-space/app.py apps/hatefulmemes-space/requirements.txt apps/hatefulmemes-space/tests/test_app_smoke.py
git commit -m "P3: real Space app — lazy loader, sidecar assert, modality bars + top tokens"
```

### Task 16: `deploy-space.yml` — vendor modality_shapley+text_occlusion, worktree build

**Files:** Modify `.github/workflows/deploy-space.yml`.

- [ ] **Step 1:** Update the `paths:` trigger and vendor step to copy `src/awake/eval/modality_shapley.py` + `src/awake/eval/text_occlusion.py` (and `awake/viz`) instead of `plausibility.py`. Replace the `git add _vendored && git commit` on the checked-out main with a **scratch worktree** build so `_vendored/` never lands on `main`:
```yaml
      - name: Build Space subtree in a scratch worktree (no _vendored on main)
        run: |
          mkdir -p _space/_vendored/awake/eval _space/_vendored/awake/viz
          cp -r apps/hatefulmemes-space/. _space/
          cp src/awake/eval/modality_shapley.py src/awake/eval/text_occlusion.py _space/_vendored/awake/eval/
          cp -r src/awake/viz/. _space/_vendored/awake/viz/
          touch _space/_vendored/awake/__init__.py _space/_vendored/awake/eval/__init__.py
      - name: Push to HuggingFace Space
        env: { HF_TOKEN: "${{ secrets.HF_TOKEN }}", HF_USER: "${{ secrets.HF_USER }}", HF_SPACE: "${{ secrets.HF_SPACE }}" }
        run: |
          if [ -z "$HF_TOKEN" ]; then echo "::warning::HF_TOKEN unset; skipping deploy."; exit 0; fi
          cd _space && git init -q && git add -A && git -c user.email=ci@x -c user.name=ci commit -qm "space build"
          git push --force "https://${HF_USER}:${HF_TOKEN}@huggingface.co/spaces/${HF_USER}/${HF_SPACE}" HEAD:main
```

- [ ] **Step 2:** `git add .github/workflows/deploy-space.yml && git commit -m "P3: deploy-space vendors modality_shapley+text_occlusion via scratch worktree"`

---

## Phase 6 — Docs

### Task 17: ADR 003, README update, REPORT, notebook, CHANGELOG

**Files:** Create `docs/decisions/003-hateful-memes-licence-and-modality-shapley.md`, `projects/03-multimodal-hatefulmemes/REPORT.md`, `projects/03-multimodal-hatefulmemes/notebooks/01-modality-attribution.py`; Modify `projects/03-multimodal-hatefulmemes/README.md`, `CHANGELOG.md`.

- [ ] **Step 1: ADR 003** — the HM licence analysis (model publishable, dataset/embeddings not, no derivative works, no raw HM in public artifacts), the 2-player-game-by-parsimony justification (corrected Janzing framing), the interventional baseline + off-manifold caveat, the logit value-function choice.
- [ ] **Step 2: README** — replace the stale `dev-unseen` + `ADR 004` + "additivity pathologies" text with: report on `dev` (500, balanced), `ADR 003` link, parsimony framing; the licence-safe Space note; reproduce steps (`just data && just encode && just train && just eval && just attribute`; `just export-space-artifacts` for the Space).
- [ ] **Step 3: REPORT.md** — 9 sections mirroring P1/P2 with the methodology, the metrics/attribution result placeholders (filled by the real run), and §-limitations from spec §9. No raw HM content.
- [ ] **Step 4: notebook `.py`** (jupytext) — loads `metrics.json` (robust `P3_PROJECT_ROOT`/cwd path resolution like P2), renders the fused-vs-unimodal table + attribution-share figure + a **synthetic** qualitative example (no real memes). Outputs committed.
- [ ] **Step 5: CHANGELOG** `[Unreleased]` P3 entry (no fabricated numbers).
- [ ] **Step 6: commit.**
```bash
git add docs/decisions/003-hateful-memes-licence-and-modality-shapley.md \
        projects/03-multimodal-hatefulmemes/REPORT.md projects/03-multimodal-hatefulmemes/README.md \
        projects/03-multimodal-hatefulmemes/notebooks/01-modality-attribution.py CHANGELOG.md
git commit -m "P3: ADR 003, README/REPORT, notebook source, CHANGELOG"
```

---

## Phase 7 — Green CI + real run

### Task 18: Green CI gate

- [ ] **Step 1:** From repo root: `uv run ruff check .` , `uv run ruff format --check .` , `uv run mypy src/awake/` , `uv run pytest tests projects -m "unit or smoke"` — all green, awake coverage ≥ 90%. Fix any failures (e.g. import-order in test files via `ruff --fix`).
- [ ] **Step 2: commit** any fixes.

### Task 19: Real end-to-end run (needs the user's data + HF repo)

> Requires the licensed dataset locally and, for deploy, the user's HF Model Hub repo + Space secrets. The controller will coordinate with the user.

- [ ] **Step 1:** `just data` (extract `~/Downloads/archive.zip` → `~/.cache/hateful_memes`); confirm counts (train 8499 / dev 500 / test 999).
- [ ] **Step 2:** `just encode clip_l14 && just train clip_l14 && just eval clip_l14 && just attribute clip_l14` on the GPU. Sanity: fused dev AUROC materially > 0.5; image-only and text-only reported with CIs.
- [ ] **Step 3:** `just notebook` (renders with outputs; no raw HM content). Fill REPORT headline numbers from `metrics.json`.
- [ ] **Step 4 (deploy):** `just export-space-artifacts` (B/32), upload `fused.txt`+`head_meta.json` to the user's HF Model Hub repo; set GitHub secrets `HF_TOKEN`/`HF_USER`/`HF_SPACE`; push to `main` to trigger `deploy-space.yml`. Confirm the live Space loads and returns label/confidence/modality bars/top tokens on a synthetic example.
- [ ] **Step 5: commit** artifacts (`metrics.json`, `assets/*.png`, rendered notebook, REPORT numbers).

---

## Self-review notes (author)

- **Spec coverage:** §2 data→T7; §3 encode/train/background→T8–11; §4 core→T2–5; §5 eval/attribute→T12–13; §6 Space→T14–16; §7 tests interleaved; §8 deps/ADR→T1,T17; §9 limitations→T17 REPORT; §10 layout→all; the logit value_fn (T8 `margin_value_fn`), deterministic estimator (T3), public-artifact rule (T17 docs + T14 generic bg + synthetic notebook example), sidecar assert (T10/T15), metrics schema (T12), hero formula (T13) all map to tasks.
- **Placeholder scan:** the only deferred content is REPORT/README *headline numbers* (T17/T19), which by nature come from the real run; flagged, not fabricated. T15 app.py prose-describes the full wiring rather than pasting ~120 lines — the implementer builds it against the stated contract + the smoke test pins the interface.
- **Type consistency:** `value_fn: (M,2D)->(M,)` consistent across T3/T8/T13/T15; `modality_shapley(...)->{"image","text"}` consistent T3/T13; `fit_heads(...)->{"fused","image","text"}` consistent T10/T13; `metric_block`/`attribute_split` signatures match their tests.
