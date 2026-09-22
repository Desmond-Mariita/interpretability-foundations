"""Repair committed metrics.json: corrected AUROC-diff estimand + share semantics.

The original full-run artifacts (CLIP embeddings, boosters, per-example dev
scores) were not cached, so a paired-bootstrap re-computation is impossible
without re-running the gated-data pipeline. What IS derivable from the
committed aggregates is corrected here exactly:

- ``auroc_diffs``: the previous values were paired mean differences over
  per-example probability arrays, mislabeled as AUROC differences. They are
  recorded under ``provenance.removed_prob_array_diffs`` and replaced by the
  exact AUROC point differences (fused minus unimodal) computed from the
  committed AUROC means. The paired bootstrap CIs are left null: they require
  per-example scores and are populated by ``just eval`` on re-run.
- ``attribution``: ``share_mean`` (a mean of signed ratios, i.e. a direction
  summary) is renamed ``signed_share_mean``; the magnitude semantics are made
  explicit via ``magnitude_share_of_mean_abs``. Quantities needing per-example
  data (mean signed phi, mean per-example magnitude share, interaction) are
  left null and populated by ``just attribute`` on re-run.
- ``background``: the seed from ``configs/train.yaml`` is recorded.
"""

from __future__ import annotations

import json

import yaml
from _paths import CONFIGS, PROJECT_ROOT
from _stats import magnitude_share_image

METRICS = PROJECT_ROOT / "metrics.json"


def repair() -> dict:
    """Return the corrected metrics document; also write it to metrics.json."""
    with open(METRICS) as f:
        old = json.load(f)
    models = old["models"]

    def auroc_mean(name: str) -> float:
        return float(models[name]["auroc"]["mean"])

    old_diffs = old.get("auroc_diffs", {})
    corrected_diffs = {}
    for name in ("image", "text"):
        corrected_diffs[f"fused_vs_{name}"] = {
            "point_diff": auroc_mean("fused") - auroc_mean(name),
            "ci_low": None,
            "ci_high": None,
            "n_resamples": 2000,
            "n_valid": None,
            "seed": 0,
        }

    old_attr = old.get("attribution", {})
    mean_abs = old_attr["mean_abs_phi"]
    with open(CONFIGS / "train.yaml") as f:
        tcfg = yaml.safe_load(f)
    repaired = {
        "split": old["split"],
        "n": old["n"],
        "models": models,
        "auroc_diffs": corrected_diffs,
        "attribution": {
            "n": int(old_attr.get("n", old["n"])),
            "mean_abs_phi": mean_abs,
            "mean_signed_phi": {"image": None, "text": None},
            "signed_share_mean": float(old_attr["share_mean"]),
            "magnitude_share_mean": None,
            # magnitude_share_image(x, y) = |x| / (|x| + |y|); swapped args give the
            # complementary text share (the two sum to 1).
            "magnitude_share_of_mean_abs": {
                "image": magnitude_share_image(mean_abs["image"], mean_abs["text"]),
                "text": magnitude_share_image(mean_abs["text"], mean_abs["image"]),
            },
            "interaction": None,
            "per_example": None,
        },
        "background": {
            "type": "empirical_train",
            "n": int(old["background"]["n"]),
            "seed": int(tcfg["background_seed"]),
        },
        "provenance": {
            "script": "scripts/25_repair_metrics.py",
            "basis": (
                "committed aggregates from the original full run; per-example run "
                "artifacts were not cached (see REPAIR_REPORT.md)"
            ),
            "auroc_diffs": {
                "point_diff": "exact: difference of committed AUROC means",
                "paired_ci": (
                    "unavailable from cached aggregates; populated by `just eval` on re-run"
                ),
            },
            "removed_prob_array_diffs": {
                "note": (
                    "previous `auroc_diffs` were paired mean differences over "
                    "per-example probability arrays (mislabeled as AUROC); kept "
                    "here for traceability only and not used in any claim"
                ),
                **{k: v["mean_diff"] for k, v in old_diffs.items()},
            },
        },
    }
    with open(METRICS, "w") as f:
        json.dump(repaired, f, indent=2)
    return repaired


if __name__ == "__main__":
    out = repair()
    print(json.dumps(out["auroc_diffs"], indent=2))
    print(json.dumps(out["attribution"], indent=2))
