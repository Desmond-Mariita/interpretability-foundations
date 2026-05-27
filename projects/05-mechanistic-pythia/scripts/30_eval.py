"""Assemble metrics.json (per property x per point, cluster-bootstrap CIs) + figures."""

from __future__ import annotations

import numpy as np

from awake.eval.bootstrap import cluster_bootstrap_ci
from awake.eval.probing import balanced_accuracy, emergence_point


def _bal(y_true, y_pred):
    return balanced_accuracy(list(y_true), list(y_pred))


def _selectivity_ci(
    gold, probe, control_golds, control_preds, groups, n_resamples, seed, alpha=0.05
):
    """Grouped-bootstrap CI for selectivity vs the K-mean control (paired on each resample).

    The per-resample statistic mirrors the point estimate ``balanced_acc(probe) -
    mean_k balanced_acc(control_k)``, where each control's balanced accuracy is measured against
    that control's OWN gold labels (a control probe is scored on the control task, not the
    property). Resamples sentence groups with replacement; deterministic under ``seed``.
    """
    gold = np.asarray(gold)
    probe = np.asarray(probe)
    control_golds = [np.asarray(g) for g in control_golds]
    control_preds = [np.asarray(c) for c in control_preds]
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        drawn = uniq[rng.integers(0, uniq.size, uniq.size)]
        idx = np.concatenate([by_group[g] for g in drawn])
        probe_ba = _bal(gold[idx], probe[idx])
        ctrl_ba = float(
            np.mean(
                [
                    _bal(cg[idx], cp[idx])
                    for cg, cp in zip(control_golds, control_preds, strict=True)
                ]
            )
        )
        diffs[i] = probe_ba - ctrl_ba
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return lo, hi


def assemble_property_metrics(per_token: dict, n_resamples: int, seed: int) -> dict:
    """Per-point balanced-acc + control + selectivity with cluster-bootstrap CIs + emergence.

    ``per_token`` carries ``gold`` (property labels), ``control_gold`` (a list of K control-label
    vectors, one per control seed; shared across points since control labels are per-word-type),
    ``sent_id`` (cluster groups), and ``points[p]`` with ``probe`` preds and ``control`` (K pred
    vectors). Each control's balanced accuracy is scored against its OWN control gold.
    """
    gold = per_token["gold"]
    control_golds = per_token["control_gold"]
    groups = per_token["sent_id"]
    points_out, sel, sel_ci = [], {}, {}
    for point, preds in per_token["points"].items():
        plo, pmean, phi = cluster_bootstrap_ci(
            gold, preds["probe"], groups, _bal, n_resamples=n_resamples, seed=seed
        )
        # control: each seed scored against its control gold; mean over seeds; CI from seed 0
        ctrl_bas = [_bal(cg, cp) for cg, cp in zip(control_golds, preds["control"], strict=True)]
        cmean = float(np.mean(ctrl_bas))
        cspread = float(np.std(ctrl_bas))
        clo, _cmean, chi = cluster_bootstrap_ci(
            control_golds[0], preds["control"][0], groups, _bal, n_resamples=n_resamples, seed=seed
        )
        # selectivity CI consistent with the K-mean-control point estimate
        slo, shi = _selectivity_ci(
            gold,
            preds["probe"],
            control_golds,
            preds["control"],
            groups,
            n_resamples=n_resamples,
            seed=seed,
        )
        points_out.append(
            {
                "point": point,
                "balanced_acc": pmean,
                "balanced_acc_ci": [plo, phi],
                "control_balanced_acc": cmean,
                "control_ci": [clo, chi],
                "control_seed_spread": cspread,
                "selectivity": pmean - cmean,
                "selectivity_ci": [slo, shi],
            }
        )
        sel[point] = pmean - cmean
        sel_ci[point] = (slo, shi)
    return {"points": points_out, "emergence": emergence_point(sel, sel_ci)}


def main() -> None:  # pragma: no cover - slow/real-run path
    """Load outputs/probe/<property>.pkl, assemble metrics.json, render per-property + hero figs."""
    import json
    import pickle

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import transformers
    from _paths import ASSETS, OUTPUTS, ensure_dirs, load_config

    cfg = load_config("probe")
    points = ["embedding", *[f"block_{i}" for i in range(cfg["n_blocks"])], "ln_f"]
    depth = [p for p in points if p != "ln_f"]
    chosen_c = json.loads((OUTPUTS / "probe" / "chosen_C.json").read_text())
    ensure_dirs(ASSETS)

    props = {}
    for prop in cfg["properties"]:
        with open(OUTPUTS / "probe" / f"{prop}.pkl", "rb") as fh:
            d = pickle.load(fh)
        m = assemble_property_metrics(
            d["per_token"], cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["seed"]
        )
        m.update(d["info"])
        props[prop] = m

    metrics = {
        "model": cfg["model_id"],
        "model_revision": cfg["model_revision"],
        "tokenizer_revision": cfg["model_revision"],
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
        "repro": {
            "probe_random_state": cfg["probe"]["random_state"],
            "train_cap": cfg["train_token_cap"],
            "train_cap_seed": cfg["train_cap_seed"],
            "bootstrap_seed": cfg["bootstrap"]["seed"],
            "control_seeds": cfg["control"]["seeds"],
            "chosen_C": chosen_c,
        },
        "depth_points": depth,
        "extra_points": ["ln_f"],
        "properties": props,
    }
    (OUTPUTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    xs = list(range(len(depth)))
    xt = ["emb"] + [str(i) for i in range(len(depth) - 1)]
    for prop in cfg["properties"]:
        pm = {pt["point"]: pt for pt in props[prop]["points"]}
        probe = [pm[p]["balanced_acc"] for p in depth]
        ctrl = [pm[p]["control_balanced_acc"] for p in depth]
        sel = [pm[p]["selectivity"] for p in depth]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, probe, "-o", color="C0", label="probe (balanced acc)")
        ax.plot(xs, ctrl, "-s", color="C1", label="control (balanced acc)")
        ax.axhline(0.5, color="gray", ls=":", lw=0.8, label="chance (0.5)")
        ax.set_ylim(0.4, 1.0)
        ax.set_xlabel(
            "residual-stream depth point (emb = token embeddings; not transformer compute)"
        )
        ax.set_ylabel("balanced accuracy")
        ax.set_xticks(xs)
        ax.set_xticklabels(xt, fontsize=7)
        ax2 = ax.twinx()
        ax2.plot(xs, sel, "-^", color="C2", alpha=0.6, label="selectivity")
        ax2.set_ylabel("selectivity (probe - control)")
        ax2.axhline(0.0, color="C2", ls="--", lw=0.5)
        em = props[prop]["emergence"]
        up = " [UNDERPOWERED]" if props[prop].get("underpowered") else ""
        ax.set_title(
            f"{prop}: probe vs control by depth -- peak {em['peak']}, "
            f"emerges {em['earliest_within_peak_ci']}{up}",
            fontsize=9,
        )
        ax.legend(loc="lower right", fontsize=7)
        fig.tight_layout()
        fig.savefig(ASSETS / f"probe_{prop}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for prop in cfg["properties"]:
        pm = {pt["point"]: pt for pt in props[prop]["points"]}
        sel = [pm[p]["selectivity"] for p in depth]
        ax.plot(xs, sel, "-o", label=prop)
        peak = props[prop]["emergence"]["peak"]
        pi = depth.index(peak)
        ax.scatter([pi], [sel[pi]], s=120, edgecolor="k", facecolor="none", zorder=5)
    ax.axhline(0.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("residual-stream depth point (emb = token embeddings)")
    ax.set_ylabel("selectivity (balanced-acc gain over control)")
    ax.set_xticks(xs)
    ax.set_xticklabels(xt, fontsize=7)
    ax.set_title("Where each property emerges: selectivity by depth (peak circled)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "hero.png", dpi=150)
    plt.close(fig)
    print("wrote metrics.json + figures")


if __name__ == "__main__":  # pragma: no cover
    main()
