# Project Build Playbook — `interpretability-foundations`

How to take a portfolio project from scaffold to **shipped, merged, and (if it has a Space)
deployed**, end-to-end and autonomously. Derived from building Project 2 (`02-text-eraser`)
and Project 3 (`03-multimodal-hatefulmemes`). Apply it verbatim to Project 4
(`04-vqa-aokvqa`) and Project 5 (`05-mechanistic-pythia`).

**Standing mandate:** finish a project end-to-end without checking in for routine decisions.
Only stop for the explicit exceptions in §8. When you do proceed, follow this playbook.

---

## 0. The lifecycle (phases)

Each project moves through these phases. Skills are from the `superpowers` plugin.

1. **Familiarize** — read the project's `README.md` scaffold (it pins most of the method),
   `apps/` if a Space exists, P1/P2/P3 as templates, and verify data/model/compute
   availability on disk *before* designing.
2. **Brainstorm → spec** — `superpowers:brainstorming`. The READMEs are prescriptive, so most
   "what" is decided; spend questions only on genuine forks (then apply the §1 defaults and
   proceed). Terminal output: a spec in `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`.
3. **Four-way design review** — before writing the plan, run the spec through four reviewers
   in parallel (see §5). Consolidate by cross-reviewer agreement, fold fixes into the spec,
   commit. Re-run the review once if the first round had blockers/majors.
4. **Plan** — `superpowers:writing-plans`. Bite-sized TDD tasks; **pure reusable core first**
   (it carries the 90% coverage floor), then data → train → eval → artifacts → app → docs →
   green-CI → real-run. Complete code in every step; honest placeholders only for headline
   numbers that come from the real run.
5. **Execute** — `superpowers:subagent-driven-development`. Fresh `sonnet` implementer per
   task; review between tasks (light direct review for small pure tasks, full subagent
   spec+quality review for integration tasks). Continuous — don't check in between tasks.
6. **Real run** — extract/verify data, encode/train, eval, attribute, render the notebook
   *with outputs*, fill REPORT/README/CHANGELOG with the real numbers (never fabricate).
7. **Final review** — one `opus` whole-implementation review of the branch diff vs `main`.
   Apply the cheap follow-ups before merge.
8. **Finish branch** — `superpowers:finishing-a-development-branch`. Merge to `main` (no-ff),
   re-run the full gate on the merge result, push.
9. **Deploy** — only if the project has an `apps/*-space/`. See §7.

Worked examples: `docs/superpowers/specs/2026-05-26-text-eraser-design.md` +
`…-multimodal-hatefulmemes-design.md` and the matching `docs/superpowers/plans/`.

---

## 1. Autonomous decision defaults (don't ask — assume these)

These pre-answer every question asked during P2/P3. Deviate only if the spec/README or a
reviewer consensus contradicts them.

- **Scope:** full end-to-end (pipeline + notebook + REPORT + Space if one exists + deploy).
- **Data:** code-only, governed by the dataset licence. Look for the dataset under
  `~/Downloads` (archives) and `~/.cache`. If present (even gated), run the real pipeline.
  If absent, build to "ready-for-data," commit honest placeholders, and note the manual
  fetch in the README/ADR — do **not** fabricate numbers or commit data.
- **Methodology forks:** follow the project README's stated method; where it's ambiguous,
  take the reviewer-consensus option (e.g. attribution on the **logit/raw margin**, not
  probability; **interventional/background-averaged** baselines as the primary estimand with
  a mean-baseline ablation; report metrics with **bootstrap CIs** and no post-hoc
  significance claims; select hyperparameters on train/CV, keep the eval split for final
  reporting only).
- **Backbone/model substitutions:** if the spec's model can't run in this environment (see
  §3), substitute the nearest robust equivalent, document it in an ADR + REPORT, and proceed.
- **Reviews:** always do the four-way design review and the final opus review. Re-review the
  spec once if round 1 surfaced blockers.
- **Merge:** when CI is green and the final review has no blockers, merge to `main` and push.
- **Honesty:** match every reported number to `metrics.json`; state caveats (overlapping CIs,
  off-manifold baselines, truncation, single dataset). Discard review findings that are
  factually wrong after you verify them.

---

## 2. Repo conventions (non-negotiable)

- **Tooling:** `uv` (at `~/.local/bin/uv` — export PATH; not on the default PATH) + `just`
  (**not installed** on this box → run the underlying `uv run …` commands directly). `gh`
  is **not installed** either.
- **Layout per project:** `projects/NN-name/{configs,scripts,tests,notebooks}`, numbered
  scripts (`00_…`,`01_…`,`10_…`,`20_…`), `outputs/` gitignored, `metrics.json` + `assets/`
  at the project root (committed), `REPORT.md`, `README.md`.
- **Shared library:** put reusable, pure, tokenizer/model-agnostic logic in `src/awake/eval/`
  (it carries the **90% coverage floor** via fast unit tests on toy `predict_fn`s/value_fns).
  Update `src/awake/eval/__init__.py` exports. Model/data-bound glue stays in `projects/.../scripts`.
- **Test markers:** `unit` (fast pure), `smoke` (tiny stub model/random arrays, CPU, **no real
  data, no model downloads**), `slow` (real GPU/data, excluded from CI). CI runs
  `pytest tests projects apps -m "unit or smoke"`; `--import-mode=importlib` is set in
  `pyproject.toml`.
- **Per-project test isolation (critical):** every `projects/*/tests/conftest.py` and
  `apps/*/tests/` must use the autouse fixture that prepends its own `scripts/` to `sys.path`
  and evicts its owned bare module names (`_paths`, `10_train`, …) from `sys.modules` before
  & after each test — otherwise sibling projects collide on identical filenames. Copy it
  verbatim from an existing project conftest.
- **Project test docstrings:** `projects/*/tests/**` and `apps/*/tests/**` are **not** covered
  by the root `tests/** = ["D"]` ruff ignore, so their modules + test functions need
  Google-style docstrings.
- **Notebooks:** jupytext `.py` source + executed `.ipynb` **committed WITH outputs** + `.html`.
  `nbstripout` is removed from the recipe/pre-commit (we keep outputs so GitHub renders
  results). Resolve the project root in the notebook via an env var first
  (`<PREFIX>_PROJECT_ROOT`), then `__file__`, then cwd candidates (nbconvert's cwd varies).
- **ruff:** Google docstrings on `src/awake/`, line length 100, double quotes, **no Unicode
  `×`/`–`/`—` in code** (RUF002/003 — use ASCII; fine in Markdown). Run `ruff check --fix`
  + `ruff format` to auto-fix import order/format.
- **ADRs:** `docs/decisions/NNN-*.md`, sequential. Record real design decisions + their
  rationale + consequences.
- **Commits/branches:** work on a `pNN-<name>` branch (never commit project work straight to
  `main`). Conventional, scoped commit messages ending with the
  `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>` trailer. Commit after each task.

---

## 3. Environment gotchas (this machine, 2026 — bleeding-edge stack)

This box runs unusually new versions; several "standard" choices break. Known traps + fixes:

- **transformers 5.9 / torch 2.12 / CUDA 13:**
  - **DeBERTa-v3 fine-tuning diverges to NaN** (disentangled-attention numerical bug) even
    with warmup + gradient clipping at low LR. → Use a standard-attention encoder
    (**RoBERTa-base** worked) and document the substitution. (P2.)
  - **`CLIPModel.get_image_features()`/`get_text_features()` return an object**, not a tensor;
    its `.pooler_output` **is** the projected joint embedding (verified `allclose` to
    `visual_projection(...)`). Handle both: `x.pooler_output if hasattr(x,"pooler_output") else x`.
- **Training stability:** always add LR warmup + `clip_grad_norm_` + per-step loss logging to
  custom training loops; verify the trained model isn't degenerate (check test accuracy + a
  couple of obvious predictions) before trusting downstream attribution.
- **CPU torch wheels:** pin to a version that exists on the CPU index (`2.2.x+cpu` is **gone**;
  use `torch==2.12.0+cpu` with `--extra-index-url https://download.pytorch.org/whl/cpu`).
- **Optional heavy deps (shap/numba/llvmlite):** isolate in an extra; pin
  `numba>=0.59, llvmlite>=0.42` for Python 3.11; omit slow/optional-only modules from the
  coverage denominator via `[tool.coverage.run] omit`.
- **parquet round-trips:** a list column with `None` (e.g. `word_ids`) comes back as a float
  array with `NaN` — normalise (`None` if `w is None or isnan(w) else int(w)`) before use, or
  `visible_mask`/aggregation breaks silently.
- **`sentencepiece`/`protobuf`** are needed for SentencePiece-tokenizer models (DeBERTa, T5,
  …) — add them to deps if the model uses one.

---

## 4. Data & licence governance

- **Never commit** the dataset, its files, or **derived artifacts that embed the data**
  (e.g. CLIP embeddings of licensed images = derivative works). `outputs/` is gitignored.
- **You may publish trained models** (weights/heads) — most research-dataset licences (incl.
  Meta Hateful Memes §2) grant model IP to the user.
- **No raw dataset content in committed artifacts** — notebook outputs, REPORT, figures, and
  the Space use **synthetic stand-ins**, not real samples.
- **Gated datasets** (PhysioNet/MIMIC, Meta HM, etc.): the `00_…` script *verifies/extracts*
  a locally-provided licensed copy; it never downloads. Read the bundled `LICENSE.txt` and
  encode the constraints in an ADR.

---

## 5. The four-way design review

Run on the written spec (copy it to `/tmp`, prepend a rigorous-reviewer prompt). In parallel:
- **Two internal subagents** (`general-purpose`, `sonnet`): one **methodology** reviewer, one
  **engineering** reviewer. Give each the spec path + a focused, skeptical brief.
- **Two external CLIs**, headless, in the background:
  `codex exec --skip-git-repo-check "$(cat prompt+spec)"` and `gemini -p "$(cat prompt+spec)"`.
  Both are installed (under `~/.nvm/.../bin`). Codex will web-search references; Gemini is fast.
- **Consolidate** by how many reviewers independently flag each item (cross-agreement = signal).
  Verify findings against the actual files — **discard ones that are wrong** (e.g. a reviewer
  claiming code is missing during a *design* review, or a flag that contradicts a user
  decision). Fold the rest into the spec; commit.
- Use the **same four-way pattern for the final implementation review** (use `opus` for it),
  on the branch diff vs `main`, and verify CI + numbers as part of it.

---

## 6. Definition of done (quality gates)

A project is done only when **all** hold:
- `uv run ruff check .` clean; `uv run ruff format --check .` clean; `uv run mypy src/awake/`
  clean; `uv run pytest tests projects apps -m "unit or smoke"` green with **awake coverage
  ≥ 90%**.
- Real run executed (if data available): `metrics.json` + `assets/*` + executed notebook
  committed; REPORT/README/CHANGELOG numbers **match `metrics.json`**; honest caveats stated.
- ADR(s) + CHANGELOG `[Unreleased]` entry written; repo stays v0.x unless told otherwise.
- Final review has no open blockers/majors.
- Merged to `main` (no-ff), gate re-run green on the merge, pushed to `origin/main`.
- If a Space exists: deployed and verified live (§7).

---

## 7. HuggingFace Space deployment runbook

(Only for projects with `apps/*-space/`.) **Requires the user's HF account** — see §8.

- **Credentials, never in chat.** The user caches a **write** token once
  (`printf '%s' 'hf_…' > ~/.cache/huggingface/token`); then every `hf` command auto-auths
  with no token in the command. Confirm with `hf auth whoami`. A read token authenticates but
  **can't create/upload** ("token has incorrect permissions") — needs **write**. If a token
  is ever pasted into chat, tell the user to **revoke it immediately**.
- **CLI is `hf`** (not deprecated `huggingface-cli`). Prefix with `~/.local/bin/uv run hf`.
  Single-line commands (multi-line `\` paste-wrap breaks). `hf repos create <id> --repo-type
  {model,space} [--space-sdk gradio]`; `hf upload <id> <local> <path> --repo-type …`.
- **Deploy without GitHub secrets:** assemble the Space dir (app + `_vendored/awake/<modules>`
  with **empty** `awake/__init__.py` + `awake/eval/__init__.py` so imports don't drag the rest)
  and `hf upload … --repo-type space` directly. (The `deploy-space.yml` Action is the
  alternative but needs `HF_TOKEN`/`HF_USER`/`HF_SPACE` repo secrets + a push to `main`.)
- **Set Space variables via API:** `huggingface_hub.add_space_variable(space_id, k, v)`
  (e.g. `HF_MODEL_REPO`). Keep the model repo **public** so the Space needs no runtime token.
- **Space build pins (this env):** in the Space `README.md` front-matter set
  `python_version: "3.11"` (Python 3.13 removed `audioop`, breaking gradio/pydub) and
  `sdk_version: "5.49.1"` (gradio 4.44 crashes at launch on a `get_api_info` "bool is not
  iterable" schema bug and needs `HfFolder` removed in hub 1.0; gradio 5 fixes both and works
  with modern hub/transformers). `requirements.txt`: `torch==2.12.0+cpu` via the CPU
  extra-index; load the model **lazily** at first predict (not import) so smoke tests can stub it.
- **Monitor + debug:** poll `huggingface_hub.get_space_runtime(...).stage`
  (`BUILDING`→`APP_STARTING`→`RUNNING`, or `BUILD_ERROR`/`RUNTIME_ERROR`). Fetch logs via
  `GET https://huggingface.co/api/spaces/<id>/logs/{build,run}` with `Authorization: Bearer
  <get_token()>` and read the SSE `data:` lines. Fix → re-`hf upload` the changed file →
  re-poll.
- **Verify end-to-end:** `gradio_client.Client(<space>).predict(handle_file(synthetic.png),
  "caption", api_name="/predict")` returns label + modality/confidence dicts + tokens.
  Confirm `RUNNING` **stays** (it can flip to `RUNTIME_ERROR` right after a transient RUNNING).

---

## 8. When to STILL stop and ask the user

Proceed autonomously on everything else. Stop only for:
- **Credentials/secrets only they hold** — HF write token, GitHub Actions secrets, any login
  (`gcloud`/`hf auth`). Provide exact `!`-runnable commands and wait.
- **Genuinely irreversible / outward-facing first-time actions** beyond an established
  pattern — e.g. publishing to a *new* external destination, force-pushing shared history,
  deleting work you didn't create.
- **A spec/data contradiction that changes the deliverable** and isn't resolvable from the
  README + reviewer consensus (rare).
- **3+ failed fix attempts on the same bug** → question the architecture / surface it (per
  `superpowers:systematic-debugging`).

Everything else — scope, methodology forks, model substitutions, reviews, merges, doc
content, CI fixes, deploy build-debugging — is yours to decide and execute.

---

## 9. Per-project quick checklist

```
[ ] Explore README scaffold + apps/ + verify data/model/compute on disk
[ ] Brainstorm → spec  → docs/superpowers/specs/<date>-<proj>-design.md  (commit on pNN branch)
[ ] Four-way review → consolidate → revise spec → (re-review if blockers) → commit
[ ] Plan → docs/superpowers/plans/<date>-<proj>.md  (commit)
[ ] Subagent-driven build: pure core (TDD, 90% floor) → scripts → eval → app → docs
[ ] Per-project conftest isolation fixture; project-test docstrings
[ ] Real run (if data present): encode/train/eval/attribute → metrics.json + assets + notebook(with outputs)
[ ] Fill REPORT/README/CHANGELOG with real numbers (match metrics.json; honest caveats); ADR(s)
[ ] Final opus review on branch diff; apply cheap follow-ups
[ ] Green gate: ruff + format + mypy + pytest(unit|smoke) ≥90%
[ ] Merge pNN → main (no-ff), re-gate on merge, push origin main
[ ] If Space: build artifacts, hf upload model + Space, set vars, pin py3.11/gradio5, verify live
```
