"""Aggregate generation + ablation outputs into metrics.json and the hero figure.

``compute_subset_metrics`` is pure (lists in, dict out) and smoke-tested; ``main``
reads parquet, splits filtered/unfiltered, writes metrics.json, and renders the
multi-panel hero figure.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

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
        # Paired-parsed-only sensitivity (drop pairs unparseable on either side); the
        # primary (None=inconsistent) policy above remains the headline (see ADR-004).
        with_e_po = consistency_rate(ans, abl[p]["expl"], paired_only=True)
        no_e_po = consistency_rate(ans, abl[p]["noexpl"], paired_only=True)
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
                "paired_only": {
                    "with_expl": with_e_po,
                    "no_expl": no_e_po,
                    "delta": with_e_po - no_e_po,
                },
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


def render_hero_figure(metrics: dict, out_path: Path) -> None:
    """Render the four-panel hero figure (delta, accuracy, leak, parse) from aggregates.

    One metric per panel keeps the constructs visually separate: the recoverability
    delta is not answer accuracy, and neither is the explanation answer-leak rate.
    Confidence bars are drawn only for the delta (the only paired-bootstrap interval
    computed in the run); the other panels are point estimates and are labelled as
    such. Pipeline colors are entity-fixed from the portfolio palette, never cycled.

    Args:
        metrics: ``{"pipelines": {p: {"accuracy": float, "parse_rate_answer": float,
            "expl_leak_rate": float, "consistency": {"delta": float,
            "delta_ci": [lo, hi]}}}}`` for the unfiltered subset.
        out_path: Destination PNG path.

    Raises:
        ValueError: If any of the required pipeline keys (A, B, B7) is missing.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from awake.viz.style import PALETTE, apply_style

    apply_style()
    labels = ("A", "B", "B7")
    pipes = metrics.get("pipelines", {})
    if set(labels) - set(pipes):
        raise ValueError(f"metrics['pipelines'] must contain keys {labels}")
    colors = {p: PALETTE[i] for i, p in enumerate(labels)}  # entity-fixed, never cycled

    panels = (
        ("recoverability delta (95% CI)", [pipes[p]["consistency"]["delta"] for p in labels]),
        ("answer accuracy", [pipes[p]["accuracy"] for p in labels]),
        ("explanation answer-leak rate", [pipes[p]["expl_leak_rate"] for p in labels]),
        ("answer parse rate", [pipes[p]["parse_rate_answer"] for p in labels]),
    )
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.6))
    for ax, (title, values) in zip(axes, panels, strict=True):
        if title.startswith("recoverability"):
            yerr = [
                [
                    pipes[p]["consistency"]["delta"] - pipes[p]["consistency"]["delta_ci"][0]
                    for p in labels
                ],
                [
                    pipes[p]["consistency"]["delta_ci"][1] - pipes[p]["consistency"]["delta"]
                    for p in labels
                ],
            ]
            ax.bar(
                labels,
                values,
                width=0.55,
                yerr=yerr,
                capsize=4,
                error_kw={"ecolor": "#444444"},
                color=[colors[p] for p in labels],
            )
            ax.axhline(0, color="k", lw=0.8)
            label_ys = [pipes[p]["consistency"]["delta_ci"][1] + 0.05 for p in labels]
            ax.set_ylim(0, max(label_ys) * 1.06)
        else:
            ax.bar(labels, values, width=0.55, color=[colors[p] for p in labels])
            ax.set_ylim(0, 1)
            label_ys = [v + 0.03 for v in values]
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=8)
        for x, v, y in zip(range(3), values, label_ys, strict=True):
            ax.text(x, y, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("A-OKVQA validation (n=1145): four separate constructs per pipeline", fontsize=11)
    fig.text(
        0.005,
        0.01,
        "Point estimates; 95% CI drawn only for the paired-bootstrap delta. "
        "A = BLIP-2 caption → Qwen2.5-7B · B = Qwen2.5-VL-3B · B7 = Qwen2.5-VL-7B.",
        fontsize=8,
        va="bottom",
    )
    fig.subplots_adjust(bottom=0.19, wspace=0.28)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main_from_snapshot() -> None:  # pragma: no cover - render-only path
    """Render the hero figure from the committed aggregate snapshot (no model rerun)."""
    import json

    from _paths import ASSETS

    snapshot = json.loads((ASSETS / "metrics_snapshot.json").read_text())
    render_hero_figure(snapshot["unfiltered"], ASSETS / "hero.png")
    print("wrote hero.png from metrics_snapshot.json")


def main() -> None:  # pragma: no cover - slow/real-run path
    """Read parquet, compute filtered/unfiltered metrics, write metrics.json + hero.png."""
    import json

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

    # Hero figure: 4 panels (delta, accuracy, leak rate, parse rate), one construct per
    # panel, rendered from the same unfiltered aggregates written to metrics.json.
    u = metrics["subsets"]["unfiltered"]["pipelines"]
    render_hero_figure(
        {
            "pipelines": {
                p: {
                    "accuracy": u[p]["accuracy"],
                    "parse_rate_answer": u[p]["parse_rate"]["answer"],
                    "expl_leak_rate": u[p]["expl_leak_rate"],
                    "consistency": u[p]["consistency"],
                }
                for p in PIPELINES
            }
        },
        ASSETS / "hero.png",
    )
    print("wrote metrics.json + hero.png")


if __name__ == "__main__":  # pragma: no cover
    if "--from-snapshot" in sys.argv:
        main_from_snapshot()
    else:
        main()
