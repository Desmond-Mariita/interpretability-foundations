"""Smoke tests for the probing driver on tiny synthetic activations (no model, no sklearn-heavy)."""

import importlib
import sys

import pytest
from _stub import tiny_acts


def _linear_fit(x_train, y_train):
    """Trivial injectable fitter: threshold on dim-0 sign (recovers the planted signal)."""

    def predict(x):
        return (x[:, 0] > x_train[:, 0].mean()).astype(int)

    return predict


@pytest.mark.smoke
def test_probe_property_selectivity_rises_with_depth():
    """Verify selectivity at block_1 exceeds selectivity at embedding by at least 0.2."""
    mod = importlib.import_module("20_probe")
    acts, meta = tiny_acts()
    res = mod.probe_property(
        acts,
        meta,
        label_fn=lambda m: [int(u == "NOUN") for u in m["upos"]],
        subset_fn=lambda m: [True] * len(m["upos"]),
        fit_predict=_linear_fit,
        control_seeds=[0, 1],
        base_rate=1 / 3,
    )
    pts = {r["point"]: r for r in res}
    assert pts["block_1"]["selectivity"] >= pts["embedding"]["selectivity"] + 0.2


@pytest.mark.smoke
def test_extract_module_imports_without_torch():
    """Confirm 10_extract can be imported without triggering a torch import."""
    # Remove torch from cache so we can detect if 10_extract loads it at module level.
    torch_was_present = sys.modules.pop("torch", None)
    try:
        sys.modules.pop("10_extract", None)
        importlib.import_module("10_extract")
        assert "torch" not in sys.modules
    finally:
        if torch_was_present is not None:
            sys.modules["torch"] = torch_was_present


@pytest.mark.smoke
def test_assemble_metrics_shape_and_emergence():
    """Verify assemble_property_metrics returns expected keys and a valid emergence peak."""
    mod = importlib.import_module("30_eval")
    # synthetic per-token preds for 2 points, one property, 2 sentences
    per_token = {
        "gold": [1, 0, 1, 0],
        "control_gold": [[0, 1, 0, 1]],  # one control seed; scored against its OWN labels
        "sent_id": ["0", "0", "1", "1"],
        "points": {
            "embedding": {"probe": [1, 0, 0, 1], "control": [[1, 0, 0, 1]]},
            "block_0": {
                "probe": [1, 0, 1, 0],
                "control": [[0, 1, 0, 1]],
            },  # = control_gold -> ba 1.0
        },
    }
    out = mod.assemble_property_metrics(per_token, n_resamples=100, seed=0)
    assert {"points", "emergence"} <= set(out)
    pts = {p["point"]: p for p in out["points"]}
    assert {"selectivity_ci", "control_ci", "control_seed_spread"} <= set(pts["block_0"])
    assert out["emergence"]["peak"] in ("embedding", "block_0")
