"""Assemble metrics.json (per property x per point, cluster-bootstrap CIs) + figures."""

from __future__ import annotations

import numpy as np

from awake.eval.bootstrap import cluster_bootstrap_ci
from awake.eval.probing import balanced_accuracy, emergence_point


def _bal(y_true, y_pred):
    return balanced_accuracy(list(y_true), list(y_pred))


def _selectivity_ci(gold, probe, controls, groups, n_resamples, seed, alpha=0.05):
    """Grouped-bootstrap CI for selectivity vs the K-mean control (paired on each resample).

    The per-resample statistic mirrors the point estimate ``balanced_acc - mean_k(control_k)``,
    so the CI is consistent with ``selectivity = pmean - cmean`` for any number of control seeds.
    Resamples sentence groups with replacement; deterministic under ``seed``.
    """
    gold = np.asarray(gold)
    probe = np.asarray(probe)
    controls = [np.asarray(c) for c in controls]
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        drawn = uniq[rng.integers(0, uniq.size, uniq.size)]
        idx = np.concatenate([by_group[g] for g in drawn])
        probe_ba = _bal(gold[idx], probe[idx])
        ctrl_ba = float(np.mean([_bal(gold[idx], c[idx]) for c in controls]))
        diffs[i] = probe_ba - ctrl_ba
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return lo, hi


def assemble_property_metrics(per_token: dict, n_resamples: int, seed: int) -> dict:
    """Per-point balanced-acc + control + selectivity with cluster-bootstrap CIs + emergence."""
    gold = per_token["gold"]
    groups = per_token["sent_id"]
    points_out, sel, sel_ci = [], {}, {}
    for point, preds in per_token["points"].items():
        plo, pmean, phi = cluster_bootstrap_ci(gold, preds["probe"], groups, _bal,
                                               n_resamples=n_resamples, seed=seed)
        # control: average balanced acc over seeds; CI from the first seed's grouped bootstrap
        ctrl_bas = [_bal(gold, c) for c in preds["control"]]
        cmean = float(np.mean(ctrl_bas))
        cspread = float(np.std(ctrl_bas))
        clo, _cmean, chi = cluster_bootstrap_ci(gold, preds["control"][0], groups, _bal,
                                                n_resamples=n_resamples, seed=seed)
        # selectivity CI consistent with the K-mean-control point estimate
        slo, shi = _selectivity_ci(gold, preds["probe"], preds["control"], groups,
                                   n_resamples=n_resamples, seed=seed)
        points_out.append({"point": point, "balanced_acc": pmean, "balanced_acc_ci": [plo, phi],
                           "control_balanced_acc": cmean, "control_ci": [clo, chi],
                           "control_seed_spread": cspread, "selectivity": pmean - cmean,
                           "selectivity_ci": [slo, shi]})
        sel[point] = pmean - cmean
        sel_ci[point] = (slo, shi)
    return {"points": points_out, "emergence": emergence_point(sel, sel_ci)}


def main() -> None:  # pragma: no cover - slow/real-run path
    """Load outputs/probe/<property>.npz, assemble metrics.json, render per-property + hero figs."""
    raise NotImplementedError  # implemented during the real run (mechanical I/O + matplotlib)


if __name__ == "__main__":  # pragma: no cover
    main()
