"""Intervention passes: direction-only number patches + controls (ADR 006 Decision 5).

For each stimulus, layer/point, and condition, one patched forward pass (batch 1) modifies
ONLY the subject-position residual and records next-token verb logits at the final stem
position. Conditions: ``number`` (opposite-number donor), ``same_number`` (C1),
``random:{seed}`` (C2, norm-matched to the primary patch, orthogonal to u), and
``full_residual`` (C3, matched-tokenization subset only). A ``noop`` pass (C0) runs once per
stimulus and must reproduce the cached baseline logits within ``noop_tolerance``.
GPU; run in tmux (see README v1.1).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from _models import load_pythia
from _paths import OUTPUTS, STIM_DIR, ensure_dirs, load_config

from awake.eval.causal import orthogonal_unit_vector

BASE_DIR = OUTPUTS / "stimuli" / "baseline"
INT_DIR = OUTPUTS / "stimuli" / "interventions"
DIRECTION_DIR = OUTPUTS / "probe" / "noun_number_direction"

CONDITIONS = ["number", "same_number", "random:0", "random:1", "random:2", "full_residual"]


def _hook_module(model, point: str):
    """Return the module to hook for a depth point (embedding / block_i / ln_f)."""
    base = model.gpt_neox
    if point == "embedding":
        return base.embed_in
    if point == "ln_f":
        return base.final_layer_norm
    return base.layers[int(point.split("_")[1])]


def main() -> None:  # pragma: no cover - slow path (GPU)
    """Run intervention passes for one or more splits (default: all)."""
    import sys

    import torch

    cfg = load_config("causal")
    probe_cfg = load_config("probe")
    splits = sys.argv[1:] or ["pilot", "dev", "test"]
    ensure_dirs(INT_DIR)

    model, tok = load_pythia(cfg["model_id"], cfg["model_revision"], device="cuda")
    d_model = probe_cfg.get("d_model", 768)

    vval = json.loads((STIM_DIR / "tokenizer_validation.json").read_text())
    kept = vval["kept_pairs"]
    kept_forms = [f for pair in kept for f in pair]
    verb_ids = {f: tok.encode(" " + f, add_special_tokens=False)[0] for f in kept_forms}
    vid_list = [verb_ids[f] for f in kept_forms]

    dman = json.loads((DIRECTION_DIR / "manifest.json").read_text())
    points = cfg["intervention"]["causal_points"] + cfg["intervention"]["terminal_points"]
    u_by_point = {p: np.load(DIRECTION_DIR / f"{p}.npz")["u"].astype(np.float64) for p in points}
    # deterministic C2 random directions: orthogonal to u, per (point, seed)
    rand_dir = {
        (p, s): orthogonal_unit_vector(seed=s * 1000 + points.index(p), d=d_model, u=u_by_point[p])
        for p in points
        for s in cfg["intervention"]["random_seeds"]
    }

    stimuli = pd.read_parquet(STIM_DIR / "stimuli.parquet")
    noop_tol = cfg["intervention"]["noop_tolerance"]
    cache_dtype = np.float32

    for split in splits:
        sub = stimuli[stimuli["split"] == split]
        out_dir = INT_DIR / split
        ensure_dirs(out_dir)

        # preload cached subject-position residuals for the split
        cache: dict[str, dict[str, np.ndarray]] = {}
        for r in sub.itertuples():
            z = np.load(BASE_DIR / split / f"{r.stim_id}.npz")
            cache[r.stim_id] = {p: z[p].astype(np.float64) for p in points}

        def forward_with_patch(
            prompt_text: str, subject_last: int, final_pos: int, point: str, new_row=None
        ) -> np.ndarray:
            """One forward; if ``new_row`` given, replace the subject row at ``point``.

            Verb logits are computed with the TIED embedding head
            (``final_layer_norm output @ embed_in.weight.T``): the pinned pythia-160m
            config declares ``tie_word_embeddings: false``, which makes transformers 5.9
            build an untied random ``embed_out``; ``model.out.logits`` is not used.
            """
            enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
            captured_lnf: dict[str, torch.Tensor] = {}

            def capture_lnf(_m, _i, module_out):
                captured_lnf["t"] = (
                    module_out[0] if isinstance(module_out, tuple) else module_out
                ).detach()

            lnf_handle = model.gpt_neox.final_layer_norm.register_forward_hook(capture_lnf)
            handle = None
            if point is not None:

                def hook(_m, _i, module_out):
                    t = module_out[0] if isinstance(module_out, tuple) else module_out
                    if new_row is not None:
                        t[0, subject_last] = torch.as_tensor(
                            new_row, device=t.device, dtype=t.dtype
                        )
                    return module_out

                handle = _hook_module(model, point).register_forward_hook(hook)
            try:
                with torch.no_grad():
                    model(**{k: v.to("cuda") for k, v in enc.items()})
            finally:
                if handle is not None:
                    handle.remove()
                lnf_handle.remove()
            lg = (
                (captured_lnf["t"][0, final_pos] @ model.gpt_neox.embed_in.weight.T)[vid_list]
                .detach()
                .to("cpu")
                .numpy()
            )
            return lg

        rows = []
        noop_max = 0.0
        base_logits = pd.read_parquet(BASE_DIR / split / "logits.parquet").set_index("stim_id")
        for r in sub.itertuples():
            base_row = cache[r.stim_id]
            # C0: no-op forward (hook active, no change) vs cached baseline
            base_lg = base_logits.loc[r.stim_id, [f"logit_{f}" for f in kept_forms]].to_numpy()
            noop_lg = forward_with_patch(
                r.prompt_text, r.subject_last, r.final_token_pos, points[0], new_row=None
            )
            noop_max = max(noop_max, float(np.max(np.abs(noop_lg - base_lg))))
            assert np.allclose(noop_lg, base_lg, atol=noop_tol), (
                f"C0 no-op diverged from baseline for {r.stim_id}: "
                f"max |diff| = {np.max(np.abs(noop_lg - base_lg))}"
            )

            for point in points:
                u = u_by_point[point]
                h_r = base_row[point]
                # primary opposite-number donor patch
                h_d = cache[r.donor_stim_id][point]
                a_r = float(u @ h_r)
                a_d = float(u @ h_d)
                delta_num = (a_d - a_r) * u
                norm_delta = float(np.abs(a_d - a_r))

                a_c1 = float(u @ cache[r.donor_c1_stim_id][point])
                norm_c1 = float(np.abs(a_c1 - a_r))
                conds: list[tuple[str, np.ndarray, str | None, float | None]] = [
                    ("number", h_r + delta_num, r.donor_stim_id, norm_delta),
                    ("same_number", h_r + (a_c1 - a_r) * u, r.donor_c1_stim_id, norm_c1),
                ]
                for s in cfg["intervention"]["random_seeds"]:
                    conds.append(
                        (f"random:{s}", h_r + norm_delta * rand_dir[(point, s)], None, norm_delta)
                    )
                if r.c3_eligible:
                    conds.append(("full_residual", h_d, r.donor_stim_id, None))
                for name, new_row, donor_id, norm in conds:
                    lg = forward_with_patch(
                        r.prompt_text,
                        r.subject_last,
                        r.final_token_pos,
                        point,
                        new_row=new_row.astype(cache_dtype),
                    )
                    row = {
                        "stim_id": r.stim_id,
                        "point": point,
                        "condition": name,
                        "donor_stim_id": donor_id,
                        "norm_delta": norm,
                    }
                    for f, v in zip(kept_forms, lg, strict=True):
                        row[f"logit_{f}"] = float(v)
                    rows.append(row)
            if r.Index % 100 == 0:
                print(f"{split}: {r.Index}/{len(sub)} stimuli done", flush=True)

        pd.DataFrame(rows).to_parquet(out_dir / "rows.parquet", index=False)
        freeze = {}
        freeze_path = STIM_DIR / "design_freeze.json"
        if freeze_path.exists():
            freeze = json.loads(freeze_path.read_text())
        (out_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "conditions": CONDITIONS,
                    "noop_max_abs_logit_diff": float(noop_max),
                    "noop_tolerance": noop_tol,
                    "random_seeds": cfg["intervention"]["random_seeds"],
                    "direction_manifest": dman.get("points"),
                    "freeze_git_sha": freeze.get("git_sha"),
                    "freeze_manifest_sha256": freeze.get("manifest_sha256"),
                },
                indent=2,
            )
        )
        print(f"{split}: {len(rows)} intervention rows; C0 max |diff| = {noop_max:.2e}")
    print("intervention passes complete")


if __name__ == "__main__":  # pragma: no cover
    main()
