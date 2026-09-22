"""Assemble the v1.1 causal evaluation: competence gate, outcomes, CIs, and figures.

Pure parts (item construction, gate statistics, donor-scoring convention, bootstrap
wiring, schema + consistency checks) are smoke-tested; ``main`` reads the cached baseline
logits + intervention rows, runs the pre-registered competence gate on dev (ADR 006
Decision 4), and -- only if it passes -- computes the confirmatory test-split outcomes,
writes ``causal_metrics.json`` (outputs/ + assets/), and renders the v1.1 figures.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from _paths import ASSETS, OUTPUTS, STIM_DIR, ensure_dirs, load_config

from awake.eval.causal import (
    agreement_margin,
    cluster_mean_bootstrap,
    donor_directed_shift,
)

BASE_DIR = OUTPUTS / "stimuli" / "baseline"
INT_DIR = OUTPUTS / "stimuli" / "interventions"

# Validated light-mode categorical slots (dataviz reference palette); fixed order.
SERIES_COLORS = {
    "number": "#2a78d6",
    "same_number": "#eb6834",
    "random": "#1baf7a",
    "full_residual": "#eda100",
}


# ---------------------------------------------------------------------------
# Pure: items and the donor-scoring convention
# ---------------------------------------------------------------------------


def logit_contrast(row: pd.Series, sing: str, plur: str) -> float:
    """``M = logit(plural_verb) - logit(singular_verb)`` for one verb pair."""
    return float(row[f"logit_{plur}"] - row[f"logit_{sing}"])


def scoring_label(condition: str, recipient_number: int, donor_number: int | None) -> int:
    """The donor label ``y_d`` used to score a condition's shift (pre-registered).

    - ``number`` / ``full_residual``: the donor's number (opposite of the recipient's).
    - ``same_number``: the donor's number (equals the recipient's).
    - ``random:{seed}``: the opposite of the recipient's number -- the direction the
      primary patch targets, so an equal response to a norm-matched non-number
      perturbation shows up on the same axis (the specificity yardstick).

    Raises:
        ValueError: If ``donor_number`` is None for a donor-based condition.
    """
    if condition == "random" or condition.startswith("random:"):
        return -recipient_number
    if donor_number is None:
        raise ValueError(f"condition {condition!r} requires a donor number label")
    return donor_number


def build_items(
    stimuli: pd.DataFrame,
    baseline: pd.DataFrame,
    interventions: pd.DataFrame,
    kept_pairs: list[list[str]],
) -> pd.DataFrame:
    """Join stimuli, baseline logits, and intervention rows into per-item rows with M and E.

    One row per (stimulus, verb pair, point, condition) intervention. ``E`` is computed
    with the donor-scoring convention above. Baseline-only rows (for the gate) are
    returned separately via the caller using :func:`build_baseline_items`.
    """
    meta = stimuli.set_index("stim_id")
    base = baseline.set_index("stim_id")
    donor_number = meta["subject_number"].to_dict()
    rows = []
    for r in interventions.itertuples():
        m = meta.loc[r.stim_id]
        b = base.loc[r.stim_id]
        y_d = scoring_label(
            r.condition,
            int(m["subject_number"]),
            int(donor_number.get(r.donor_stim_id, m["subject_number"])),
        )
        for sing, plur in kept_pairs:
            m_base = logit_contrast(b, sing, plur)
            m_patch = logit_contrast(pd.Series(r._asdict()), sing, plur)
            rows.append(
                {
                    "stim_id": r.stim_id,
                    "lemma": m["lemma"],
                    "stratum": m["stratum"],
                    "template_id": m["template_id"],
                    "attractor_relation": m["attractor_relation"],
                    "subject_number": int(m["subject_number"]),
                    "verb_sing": sing,
                    "verb_plur": plur,
                    "point": r.point,
                    "condition": r.condition,
                    "donor_stim_id": r.donor_stim_id,
                    "m_baseline": m_base,
                    "m_patch": m_patch,
                    "e": donor_directed_shift(m_patch, m_base, y_d),
                    "y_d": y_d,
                }
            )
    return pd.DataFrame(rows)


def build_baseline_items(
    stimuli: pd.DataFrame, baseline: pd.DataFrame, kept_pairs: list[list[str]]
) -> pd.DataFrame:
    """Baseline-only rows for the competence gate: per (stimulus, verb) M and A."""
    meta = stimuli.set_index("stim_id")
    base = baseline.set_index("stim_id")
    rows = []
    for sid, b in base.iterrows():
        m = meta.loc[sid]
        for sing, plur in kept_pairs:
            m_val = logit_contrast(b, sing, plur)
            rows.append(
                {
                    "stim_id": sid,
                    "lemma": m["lemma"],
                    "stratum": m["stratum"],
                    "template_id": m["template_id"],
                    "attractor_relation": m["attractor_relation"],
                    "subject_number": int(m["subject_number"]),
                    "verb_sing": sing,
                    "verb_plur": plur,
                    "m": m_val,
                    "a": agreement_margin(m_val, int(m["subject_number"])),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pure: competence gate
# ---------------------------------------------------------------------------


def gate_statistics(base_items: pd.DataFrame, n_resamples: int, seed: int) -> dict:
    """Pre-registered competence-gate statistics on dev baseline items (ADR 006 D4).

    Per (lemma, verb) row: ``d = mean over templates of (M(plural subject) - M(singular
    subject))``. Gate 1: lemma-cluster bootstrap CI of mean d strictly above 0.
    Gate 2: lemma-cluster bootstrap CI of the directional rate (``d > 0``) strictly
    above chance (0.5).
    """
    piv = (
        base_items.groupby(["lemma", "verb_sing", "subject_number"])["m"]
        .mean()
        .reset_index()
        .pivot_table(index=["lemma", "verb_sing"], columns="subject_number", values="m")
    )
    if piv.shape[1] != 2:
        raise ValueError("gate requires both subject numbers on every (lemma, verb) row")
    d = (piv[1] - piv[-1]).reset_index()
    groups = d["lemma"].to_numpy()
    gate1 = {}
    lo, mean, hi = cluster_mean_bootstrap(d[0].to_numpy(), groups, n_resamples, seed=seed)
    gate1 = {"statistic": float(mean), "ci": [lo, hi], "pass": lo > 0.0}
    directional = (d[0] > 0).astype(float).to_numpy()
    lo2, mean2, hi2 = cluster_mean_bootstrap(directional, groups, n_resamples, seed=seed)
    gate2 = {"statistic": float(mean2), "ci": [lo2, hi2], "chance": 0.5, "pass": lo2 > 0.5}
    return {
        "gate1": gate1,
        "gate2": gate2,
        "passed": gate1["pass"] and gate2["pass"],
        "n_lemma_verb_rows": len(d),
    }


# ---------------------------------------------------------------------------
# Pure: outcomes
# ---------------------------------------------------------------------------


def per_point_effect(items: pd.DataFrame, n_resamples: int, seed: int, points: list[str]) -> dict:
    """Primary estimand: per-point mean donor-directed shift with lemma-cluster CI."""
    out = {}
    for point in points:
        sub = items[items["point"] == point]
        lo, mean, hi = cluster_mean_bootstrap(
            sub["e"].to_numpy(), sub["lemma"].to_numpy(), n_resamples, seed=seed
        )
        out[point] = {"mean_e": float(mean), "ci": [lo, hi], "n_items": len(sub)}
    return out


def rate_with_ci(items: pd.DataFrame, value_col: str, n_resamples: int, seed: int) -> dict:
    """Cluster-bootstrap mean of a 0/1 column (e.g. donor-consistent rate)."""
    lo, mean, hi = cluster_mean_bootstrap(
        items[value_col].to_numpy(), items["lemma"].to_numpy(), n_resamples, seed=seed
    )
    return {"rate": float(mean), "ci": [lo, hi], "n_items": len(items)}


# ---------------------------------------------------------------------------
# Pure: snapshot schema + report consistency
# ---------------------------------------------------------------------------

SNAPSHOT_KEYS = {
    "schema_version",
    "run_id",
    "model",
    "model_revision",
    "ud_tag",
    "design_freeze",
    "stimulus",
    "verb_pairs",
    "competence_gate",
    "primary",
    "controls",
    "secondary",
    "selectivity_v10",
    "conclusion",
}


def validate_snapshot_schema(snap: dict) -> bool:
    """True iff the causal_metrics.json snapshot carries the required top-level keys."""
    return set(snap) >= SNAPSHOT_KEYS


def assert_report_snapshot_consistency(report_numbers: dict, snap: dict) -> bool:
    """True iff every number the v1.1 report cites matches the snapshot (tolerance 1e-9)."""
    for path, value in report_numbers.items():
        node = snap
        for key in path.split("."):
            if key not in node:
                return False
            node = node[key]
        if not np.isclose(float(node), float(value), atol=1e-9, rtol=1e-9):
            return False
    return True


# ---------------------------------------------------------------------------
# Figures (rendered by main)
# ---------------------------------------------------------------------------


def figure1_causal_by_layer(
    primary: dict, controls: dict, points: list[str], out_path
) -> None:  # pragma: no cover - figure rendering
    """Figure 1: layerwise donor-directed shift (primary + controls, CI bands, one axis)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    xs = list(range(len(points)))
    ax.axhline(0.0, color="#7f7f7f", lw=0.8, ls="--")

    def draw(series: dict, name: str, color: str):
        means = [series[p]["mean_e"] for p in points]
        los = [series[p]["ci"][0] for p in points]
        his = [series[p]["ci"][1] for p in points]
        ax.plot(xs, means, color=color, lw=2, marker="o", markersize=6, zorder=3)
        ax.fill_between(xs, los, his, color=color, alpha=0.12, lw=0, zorder=2)
        ax.annotate(
            name,
            (xs[-1], means[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            color="#2b2b29",
            fontsize=8,
            va="center",
        )

    draw(primary, "opposite-number direction", SERIES_COLORS["number"])
    draw(controls["same_number"], "same-number control", SERIES_COLORS["same_number"])
    draw(controls["random"], "random-direction control", SERIES_COLORS["random"])
    draw(controls["full_residual"], "full-residual control", SERIES_COLORS["full_residual"])

    xt = ["emb"] + [str(i) for i in range(len(points) - 1)]
    ax.set_xticks(xs)
    ax.set_xticklabels(xt, fontsize=7)
    ax.set_xlabel("residual-stream depth point (emb = token embeddings)")
    ax.set_ylabel("mean donor-directed shift E = y_d x (M_patched - M_baseline)")
    ax.set_title("Causal number-direction intervention by depth (test split)", fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure2_selectivity_vs_causal(
    selectivity: dict, primary: dict, points: list[str], out_path
) -> None:  # pragma: no cover - figure rendering
    """Figure 2: selectivity vs causal effect in SEPARATE panels (no dual axis)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)
    xs = list(range(len(points)))
    xt = ["emb"] + [str(i) for i in range(len(points) - 1)]

    ax = axes[0]
    sel = [selectivity[p] for p in points]
    ax.plot(xs, sel, color=SERIES_COLORS["number"], lw=2, marker="o", markersize=6)
    ax.axhline(0.0, color="#7f7f7f", lw=0.8, ls="--")
    ax.set_ylabel("noun_number selectivity\n(probe - control balanced acc)")
    ax.set_title(
        "Decodability vs causal contribution -- separate scales, no implied alignment",
        fontsize=9,
    )

    ax = axes[1]
    means = [primary[p]["mean_e"] for p in points]
    los = [primary[p]["ci"][0] for p in points]
    his = [primary[p]["ci"][1] for p in points]
    ax.plot(xs, means, color=SERIES_COLORS["same_number"], lw=2, marker="s", markersize=6)
    ax.fill_between(xs, los, his, color=SERIES_COLORS["same_number"], alpha=0.12, lw=0)
    ax.axhline(0.0, color="#7f7f7f", lw=0.8, ls="--")
    ax.set_ylabel("mean donor-directed shift E")
    ax.set_xticks(xs)
    ax.set_xticklabels(xt, fontsize=7)
    ax.set_xlabel("residual-stream depth point (emb = token embeddings)")
    for a in axes:
        a.grid(alpha=0.25)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover - slow path (reads cached outputs)
    """Gate on dev; if passed, confirmatory outcomes on test; write snapshot + figures."""
    cfg = load_config("causal")
    points = cfg["intervention"]["causal_points"]
    n_res = cfg["bootstrap"]["n_resamples"]
    seed = cfg["bootstrap"]["seed"]
    ensure_dirs(OUTPUTS / "stimuli" / "eval")

    vval = json.loads((STIM_DIR / "tokenizer_validation.json").read_text())
    kept = vval["kept_pairs"]
    stimuli = pd.read_parquet(STIM_DIR / "stimuli.parquet")

    # --- competence gate on dev baseline ---
    dev_stim = stimuli[stimuli["split"] == "dev"]
    dev_base = pd.read_parquet(BASE_DIR / "dev" / "logits.parquet")
    gate = gate_statistics(build_baseline_items(dev_stim, dev_base, kept), n_res, seed)
    print(
        f"gate1: mean paired subject-number effect {gate['gate1']['statistic']:.4f} "
        f"CI {[round(v, 4) for v in gate['gate1']['ci']]} pass={gate['gate1']['pass']}"
    )
    print(
        f"gate2: directional rate {gate['gate2']['statistic']:.4f} "
        f"CI {[round(v, 4) for v in gate['gate2']['ci']]} pass={gate['gate2']['pass']}"
    )

    snap = {
        "schema_version": 1,
        "run_id": "p5-v11-causal",
        "model": cfg["model_id"],
        "model_revision": cfg["model_revision"],
        "ud_tag": cfg["ud_tag"],
        "design_freeze": {},
        "stimulus": {},
        "verb_pairs": {"kept": kept, "excluded": []},
        "competence_gate": gate,
        "primary": {},
        "controls": {},
        "secondary": {},
        "selectivity_v10": {},
        "conclusion": "gate not passed; causal intervention not interpretable",
    }
    freeze_path = STIM_DIR / "design_freeze.json"
    if freeze_path.exists():
        snap["design_freeze"] = json.loads(freeze_path.read_text())
    manifest = json.loads((STIM_DIR / "split_manifest.json").read_text())
    snap["stimulus"] = {
        "table_sha256": manifest["stimuli"]["table_sha256"],
        "n_rows": manifest["stimuli"]["n_rows"],
        "n_by_split": manifest["stimuli"]["n_by_split"],
        "n_lemmas": manifest["lexicon"]["n_lemmas"],
        "strata": manifest["lexicon"]["by_stratum"],
    }

    # v1.0 selectivity (reproduced run) for the decodability-vs-causality comparison
    metrics_path = OUTPUTS / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        nn = m["properties"]["noun_number"]["points"]
        snap["selectivity_v10"] = {pt["point"]: pt["selectivity"] for pt in nn}

    if not gate["passed"]:
        (OUTPUTS / "causal_metrics.json").write_text(json.dumps(snap, indent=2))
        print("STOP: competence gate failed (boundary result). No causal figures.")
        return

    # --- confirmatory test-split outcomes ---
    test_stim = stimuli[stimuli["split"] == "test"]
    test_base = pd.read_parquet(BASE_DIR / "test" / "logits.parquet")
    interv = pd.read_parquet(INT_DIR / "test" / "rows.parquet")
    items = build_items(test_stim, test_base, interv, kept)

    num_items = items[items["condition"] == "number"]
    primary = per_point_effect(num_items, n_res, seed, points)
    # terminal points (ln_f) reported separately, not on the causal depth axis
    terminal_points = cfg["intervention"]["terminal_points"]
    primary_terminal = per_point_effect(num_items, n_res, seed, terminal_points)
    snap["primary"] = {
        "points": primary,
        "terminal_points": primary_terminal,
        "n_items": len(num_items),
    }

    # causal-architecture diagnostic (ADR 006 / brief section 14): a subject-position
    # patch after the final block has no downstream route to a LATER prediction position,
    # so block_11 / ln_f effects must come only from templates where the subject IS the
    # final position (simple). Reported per template; not part of the primary estimand.
    for diag_point in ("block_11", "ln_f"):
        diag = {}
        for tid, sub in num_items[num_items["point"] == diag_point].groupby("template_id"):
            lo, mean, hi = cluster_mean_bootstrap(
                sub["e"].to_numpy(), sub["lemma"].to_numpy(), n_res, seed=seed
            )
            diag[tid] = {"mean_e": float(mean), "ci": [lo, hi], "n_items": len(sub)}
        snap["secondary"][f"terminal_{diag_point}_by_template"] = diag

    controls = {}
    for cond in ("same_number", "random", "full_residual"):
        sub = items[items["condition"].str.startswith(cond)]
        controls[cond] = {
            "points": per_point_effect(sub, n_res, seed, points),
            "n_items": len(sub),
            "n_seeds": sub["condition"].nunique() if cond == "random" else 1,
        }
    snap["controls"] = controls

    for cond in ("number", "same_number", "full_residual", "random"):
        sub = items[items["condition"].str.startswith(cond)]
        snap["secondary"][f"{cond}_donor_consistent_rate"] = rate_with_ci(
            sub.assign(flag=(sub["e"] > 0).astype(float)), "flag", n_res, seed
        )
    snap["secondary"]["flip_rate"] = rate_with_ci(
        num_items.assign(flag=((num_items["m_patch"] * num_items["m_baseline"]) < 0).astype(float)),
        "flag",
        n_res,
        seed,
    )
    # descriptive subgroups (secondary)
    snap["secondary"]["subgroups"] = {}
    for key, sub in {
        "template_simple": num_items[num_items["template_id"] == "simple"],
        "template_near": num_items[num_items["template_id"] == "near"],
        "template_attractor": num_items[num_items["template_id"] == "attractor"],
        "stratum_transparent": num_items[num_items["stratum"] == "suffix_transparent"],
        "stratum_nontransparent": num_items[num_items["stratum"] == "nontransparent"],
        "attractor_same": num_items[num_items["attractor_relation"] == "same"],
        "attractor_opposite": num_items[num_items["attractor_relation"] == "opposite"],
    }.items():
        if len(sub):
            lo, mean, hi = cluster_mean_bootstrap(
                sub["e"].to_numpy(), sub["lemma"].to_numpy(), n_res, seed=seed
            )
            snap["secondary"]["subgroups"][key] = {
                "mean_e": float(mean),
                "ci": [lo, hi],
                "n_items": len(sub),
            }

    # baseline agreement margin by stratum (descriptive)
    base_items = build_baseline_items(test_stim, test_base, kept)
    for key, sub in {
        "stratum_transparent": base_items[base_items["stratum"] == "suffix_transparent"],
        "stratum_nontransparent": base_items[base_items["stratum"] == "nontransparent"],
    }.items():
        lo, mean, hi = cluster_mean_bootstrap(
            sub["a"].to_numpy(), sub["lemma"].to_numpy(), n_res, seed=seed
        )
        snap["secondary"]["subgroups"][f"baseline_agreement_{key}"] = {
            "mean_a": float(mean),
            "ci": [lo, hi],
            "n_items": len(sub),
        }

    # conclusion (claim-boundary language, ADR 006 Decision 7)
    sig_layers = [p for p in points if primary[p]["ci"][0] > 0]
    if sig_layers:
        snap["conclusion"] = (
            f"Replacing only the subject representation's projection on the decoded "
            f"noun-number direction with the projection from an opposite-number donor "
            f"shifted Pythia-160M's verb-number preference toward the donor at "
            f"{sig_layers}, relative to no-op, same-number, and norm-matched "
            f"random-direction controls. This is evidence that the decoded direction "
            f"makes a causal contribution to agreement behaviour under the tested "
            f"intervention."
        )
    else:
        snap["conclusion"] = (
            "Noun number is highly linearly decodable, but manipulating the decoded "
            "direction did not produce a direction-specific change in agreement "
            "behaviour under the tested intervention. This separates decodability "
            "from causal use."
        )

    (OUTPUTS / "causal_metrics.json").write_text(json.dumps(snap, indent=2))
    (ASSETS / "causal_metrics.json").write_text(json.dumps(snap, indent=2))

    # figures
    ensure_dirs(ASSETS)
    figure1_causal_by_layer(
        primary,
        {cond: controls[cond]["points"] for cond in controls},
        points,
        ASSETS / "fig_causal_by_layer.png",
    )
    figure2_selectivity_vs_causal(
        snap["selectivity_v10"], primary, points, ASSETS / "fig_selectivity_vs_causal.png"
    )
    print("wrote causal_metrics.json + figures")


if __name__ == "__main__":  # pragma: no cover
    main()
