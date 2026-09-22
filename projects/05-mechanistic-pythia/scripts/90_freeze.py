"""Write the v1.1 design-freeze manifest (ADR 006 Decision 6).

Assembles outputs/stimuli/design_freeze.json from the current git SHA, the committed
config, the split manifest, the tokenizer validation, and the direction manifest -- and
verifies the stimulus table hash still matches the committed split manifest. Run once,
immediately before any dev/test model evaluation; any post-freeze bug fix invalidates the
confirmatory run and requires a new versioned freeze.
"""

from __future__ import annotations

import hashlib
import json
import subprocess

import pandas as pd
from _paths import OUTPUTS, STIM_DIR, ensure_dirs, load_config

from awake.eval.causal import canonical_stimulus_key, table_hash


def git_sha() -> str:
    """Current HEAD SHA (the freeze commit)."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def sha256_file(path) -> str:
    """SHA-256 of a file's bytes.

    Args:
        path: Filesystem path to hash.

    Returns:
        Hex digest string.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:  # pragma: no cover - provenance tooling (needs git + artifacts)
    """Assemble and write ``outputs/stimuli/design_freeze.json`` (verify hashes first)."""
    cfg = load_config("causal")
    ensure_dirs(STIM_DIR)

    manifest = json.loads((STIM_DIR / "split_manifest.json").read_text())
    vval = json.loads((STIM_DIR / "tokenizer_validation.json").read_text())
    dman = json.loads((OUTPUTS / "probe" / "noun_number_direction" / "manifest.json").read_text())

    # re-verify the on-disk stimulus table against the committed split hash
    stimuli = pd.read_parquet(STIM_DIR / "stimuli.parquet")
    keys = [
        canonical_stimulus_key(
            split=r["split"],
            lemma=r["lemma"],
            subject_surface=r["subject_surface"],
            subject_number=r["subject_number"],
            template_id=r["template_id"],
            attractor_surface=r["attractor_surface"],
            attractor_number=1
            if r["attractor_number_name"] == "Plur"
            else (-1 if r["attractor_number_name"] == "Sing" else 0),
            prompt_text=r["prompt_text"],
        )
        for _, r in stimuli.iterrows()
    ]
    live_hash = table_hash(keys)
    assert live_hash == manifest["stimuli"]["table_sha256"], "stimulus table drifted from manifest"

    freeze = {
        "version": "v1.1-design-freeze-1",
        "git_sha": git_sha(),
        "created_utc_note": "recorded at freeze time",
        "model_id": cfg["model_id"],
        "model_revision": cfg["model_revision"],
        "tokenizer_revision": cfg["model_revision"],
        "ud_tag": cfg["ud_tag"],
        "config": {
            "causal_sha256": sha256_file(STIM_DIR.parent.parent / "configs" / "causal.yaml"),
            "probe_sha256": sha256_file(STIM_DIR.parent.parent / "configs" / "probe.yaml"),
            "data_sha256": sha256_file(STIM_DIR.parent.parent / "configs" / "data.yaml"),
        },
        "stimulus": {
            "table_sha256": manifest["stimuli"]["table_sha256"],
            "row_hash_sha256": manifest["stimuli"]["row_hash_sha256"],
            "n_rows": manifest["stimuli"]["n_rows"],
            "n_by_split": manifest["stimuli"]["n_by_split"],
            "n_lemmas": manifest["lexicon"]["n_lemmas"],
            "strata": manifest["lexicon"]["by_stratum"],
            "lemma_split_hash": hashlib.sha256(
                json.dumps(manifest["splits"], sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "template_families": cfg["families"],
            "templates": cfg["templates"],
        },
        "verb_pairs": {
            "kept": vval["kept_pairs"],
            "excluded": [p["pair"] for p in vval["pairs"] if not p["ok"]],
        },
        "direction": {
            "manifest_sha256": sha256_file(
                OUTPUTS / "probe" / "noun_number_direction" / "manifest.json"
            ),
            "points": {
                p: {"sha256": d["sha256"], "flipped": d["flipped"]}
                for p, d in dman["points"].items()
            },
        },
        "intervention": {
            "definition": "h'_r = h_r + (u.h_d - u.h_r) * u at the subject's last subword",
            "causal_points": cfg["intervention"]["causal_points"],
            "terminal_points": cfg["intervention"]["terminal_points"],
            "conditions": ["number", "same_number", "random:{0,1,2}", "full_residual"],
            "random_seeds": cfg["intervention"]["random_seeds"],
            "donor_pairing_seed": cfg["intervention"]["donor_pairing_seed"],
            "noop_tolerance": cfg["intervention"]["noop_tolerance"],
            "scoring_convention": (
                "E = y_d x (M_patched - M_baseline); y_d = donor number for "
                "number/full_residual, recipient number for same_number, opposite of "
                "recipient for random (the specificity yardstick)"
            ),
        },
        "gate": cfg["gate"],
        "primary_metric": "per-layer mean donor-directed shift E (opposite-number direction patch)",
        "bootstrap": {
            "unit": cfg["bootstrap"]["cluster"],
            "n_resamples": cfg["bootstrap"]["n_resamples"],
            "seed": cfg["bootstrap"]["seed"],
        },
        "manifest_sha256": "",  # self-hash filled below
    }
    freeze["manifest_sha256"] = hashlib.sha256(
        json.dumps(freeze, sort_keys=True).encode("utf-8")
    ).hexdigest()
    (STIM_DIR / "design_freeze.json").write_text(json.dumps(freeze, indent=2))
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
