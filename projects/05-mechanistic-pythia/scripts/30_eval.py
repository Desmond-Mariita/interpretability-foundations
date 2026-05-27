"""Assemble metrics.json (per property x per point, cluster-bootstrap CIs) + figures."""

from __future__ import annotations

import numpy as np

from awake.eval.bootstrap import cluster_bootstrap_ci, paired_cluster_bootstrap
from awake.eval.probing import balanced_accuracy, emergence_point


def _bal(y_true, y_pred):
    return balanced_accuracy(list(y_true), list(y_pred))


def assemble_property_metrics(per_token: dict, n_resamples: int, seed: int) -> dict:
    """Per-point balanced-acc + control + selectivity with cluster-bootstrap CIs + emergence."""
    gold = per_token["gold"]
    groups = per_token["sent_id"]
    points_out, sel, sel_ci = [], {}, {}
    for point, preds in per_token["points"].items():
        plo, pmean, phi = cluster_bootstrap_ci(gold, preds["probe"], groups, _bal,
                                               n_resamples=n_resamples, seed=seed)
        # control: average balanced acc over seeds; CI from the first seed's paired difference
        ctrl_means = [_bal(gold, c) for c in preds["control"]]
        cmean = float(np.mean(ctrl_means))
        slo, _smean, shi = paired_cluster_bootstrap(gold, preds["probe"], preds["control"][0],
                                                    groups, _bal, n_resamples=n_resamples, seed=seed)
        points_out.append({"point": point, "balanced_acc": pmean, "balanced_acc_ci": [plo, phi],
                           "control_balanced_acc": cmean, "selectivity": pmean - cmean,
                           "selectivity_ci": [slo, shi]})
        sel[point] = pmean - cmean
        sel_ci[point] = (slo, shi)
    return {"points": points_out, "emergence": emergence_point(sel, sel_ci)}


def main() -> None:  # pragma: no cover - slow/real-run path
    """Load outputs/probe/<property>.npz, assemble metrics.json, render per-property + hero figs."""
    raise NotImplementedError  # implemented during the real run (mechanical I/O + matplotlib)


if __name__ == "__main__":  # pragma: no cover
    main()
