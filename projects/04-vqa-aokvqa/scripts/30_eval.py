"""Aggregate generation + ablation outputs into metrics.json and the hero figure.

``compute_subset_metrics`` is pure (lists in, dict out) and smoke-tested; ``main``
reads parquet, splits filtered/unfiltered, writes metrics.json, and renders the
multi-panel hero figure.
"""

from __future__ import annotations

import math

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
        [
            float(o is not None and a is not None and o == a)
            for o, a in zip(original, ablated, strict=True)
        ]
    )


def compute_subset_metrics(
    gen: dict, abl: dict, gold: list[int], n_resamples: int = 2000, seed: int = 0
) -> dict:
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
            n_resamples=n_resamples,
            seed=seed,
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
        agree_vec = np.array(
            [
                float(ax is not None and ay is not None and ax == ay)
                for ax, ay in zip(gen[x]["answer_idx"], gen[y]["answer_idx"], strict=True)
            ]
        )
        lo, _, hi = bootstrap_ci(1.0 - agree_vec, n_resamples=n_resamples, seed=seed)
        divergence[f"{x}_vs_{y}"] = {
            "overall": div["overall"],
            "overall_ci": [lo, hi],
            "contingency": div["contingency"],
        }
    return {"pipelines": pipelines, "divergence": divergence}


def _gen_dict(df) -> dict:
    return {
        "answer_idx": _none_list(df["answer_idx"]),
        "expl_leaks": list(df["expl_leaks"]),
        "parsed_by": list(df["parsed_by"]),
    }


def _none_list(series) -> list:
    """Pandas reads missing ints as NaN; map back to None."""
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
        "n": len(prep),
        "n_filtered": int((~prep["leakage_flag"]).sum()),
        "prompt_variant": "main",
        "model_revisions": model_revisions(cfg["models"]),
        "b7_completed": (GEN / "B7.parquet").exists(),
        "subsets": {
            "unfiltered": _subset(pd.Series(True, index=prep.index)),
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
    axes[0].bar(labels, deltas)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title("self-rationale recoverability gain (Delta)")
    axes[1].bar(labels, accs)
    axes[1].set_title("accuracy")
    axes[1].set_ylim(0, 1)
    axes[2].bar(labels, parses)
    axes[2].set_title("parse rate")
    axes[2].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(ASSETS / "hero.png", dpi=150)
    print("wrote metrics.json + hero.png")


if __name__ == "__main__":  # pragma: no cover
    main()
