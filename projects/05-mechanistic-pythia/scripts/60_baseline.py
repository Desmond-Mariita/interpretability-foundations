"""Baseline pass: per-stimulus subject-position residuals (all points) + verb logits.

One forward per stimulus (batch 1; intervention correctness beats batching). Writes per
split: ``outputs/stimuli/baseline/<split>/<stim_id>.npz`` (residuals at the subject's last
overlapping subword for every depth point + ln_f, float32) and
``outputs/stimuli/baseline/<split>/logits.parquet`` (next-token logits at the final stem
position for the validated verb pairs). GPU; run in tmux (see README v1.1).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from _models import load_pythia
from _paths import OUTPUTS, STIM_DIR, ensure_dirs, load_config

BASE_DIR = OUTPUTS / "stimuli" / "baseline"


def main() -> None:  # pragma: no cover - slow path (GPU)
    """Run the baseline pass for one or more splits (default: all in the stimuli table)."""
    import sys

    import torch

    cfg = load_config("causal")
    probe_cfg = load_config("probe")
    splits = sys.argv[1:] or ["pilot", "dev", "test"]
    ensure_dirs(BASE_DIR)

    model, tok = load_pythia(cfg["model_id"], cfg["model_revision"], device="cuda")

    # validated verb pairs + token ids (leading-space convention)
    vval = json.loads((STIM_DIR / "tokenizer_validation.json").read_text())
    kept = vval["kept_pairs"]
    verb_ids: dict[str, int] = {}
    for sing, plur in kept:
        verb_ids[sing] = tok.encode(" " + sing, add_special_tokens=False)[0]
        verb_ids[plur] = tok.encode(" " + plur, add_special_tokens=False)[0]

    stimuli = pd.read_parquet(STIM_DIR / "stimuli.parquet")
    kept_forms = [f for pair in kept for f in pair]

    for split in splits:
        sub = stimuli[stimuli["split"] == split]
        out_dir = BASE_DIR / split
        ensure_dirs(out_dir)
        rows = []
        for r in sub.itertuples():
            enc = tok(r.prompt_text, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"][0].tolist()
            assert ids == list(r.input_ids), f"tokenizer drift for {r.stim_id}"
            captured: dict[str, torch.Tensor] = {}
            handles = []
            base = model.gpt_neox

            def mk(name, captured=captured):
                def hook(_m, _i, module_out):
                    captured[name] = (
                        module_out[0] if isinstance(module_out, tuple) else module_out
                    ).detach()

                return hook

            handles.append(base.embed_in.register_forward_hook(mk("embedding")))
            for i in range(probe_cfg["n_blocks"]):
                handles.append(base.layers[i].register_forward_hook(mk(f"block_{i}")))
            handles.append(base.final_layer_norm.register_forward_hook(mk("ln_f")))
            try:
                with torch.no_grad():
                    out = model(**{k: v.to("cuda") for k, v in enc.items()})
            finally:
                for h in handles:
                    h.remove()

            pos = r.subject_last
            resid = {}
            for name, t in captured.items():
                resid[name] = t[0, pos].to(torch.float32).cpu().numpy().astype(np.float32)
            np.savez(out_dir / f"{r.stim_id}.npz", **resid)

            vids = [verb_ids[f] for f in kept_forms]
            lg = out.logits[0, r.final_token_pos, vids].detach().to("cpu").numpy()
            row = {"stim_id": r.stim_id}
            for f, v in zip(kept_forms, lg, strict=True):
                row[f"logit_{f}"] = float(v)
            rows.append(row)
        pd.DataFrame(rows).to_parquet(out_dir / "logits.parquet", index=False)
        print(f"{split}: cached {len(rows)} stimuli")
    print("baseline pass complete")


if __name__ == "__main__":  # pragma: no cover
    main()
