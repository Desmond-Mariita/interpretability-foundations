"""Recover the per-layer ``noun_number`` probe direction for the v1.1 intervention.

Refits the v1.0 probe (train split, StandardScaler, LR C=0.01 balanced, cap 60k, seed 0)
at every depth point, converts the standardized coefficient to the residual-space normal
``w_x = w_z / scale`` and unit direction ``u = w_x / ||w_x||``, aligns the sign so that
*positive projection = plural*, and persists scaler/coefficient/direction/class-mapping
artifacts with SHA-256 hashes (ADR 006 Decision 5, configs/causal.yaml).

The refit is cross-checked against the v1.0 probe predictions persisted by ``20_probe.py``
(outputs/probe/noun_number.pkl) and against the v1.0 recorded balanced accuracies.
"""

from __future__ import annotations

import hashlib
import json
import pickle

import numpy as np
from _paths import ACTS, OUTPUTS, ensure_dirs, load_config

from awake.eval.causal import (
    align_direction_to_class,
    standardized_to_residual_direction,
    unit_direction,
)
from awake.eval.probing import balanced_accuracy

DIRECTION_DIR = OUTPUTS / "probe" / "noun_number_direction"

# v1.0 recorded noun_number probe balanced accuracies (REPORT.md section 4.3)
V10_RECORDED_BA = {"embedding": 0.983, "block_11": 0.977}
V10_TOLERANCE = 0.002


def fit_point(
    x_train: np.ndarray,
    y_train: np.ndarray,
    C: float,
    max_iter: int,
    random_state: int,
    positive_class: int,
) -> dict:
    """Fit StandardScaler + balanced LR and return residual-space direction artifacts.

    Returns a dict with ``mean``, ``scale``, ``w_z``, ``b``, ``w_x``, ``u``, ``classes``,
    ``mapping``, ``flipped``. ``u`` is sign-aligned so positive projection = plural.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(
        C=C, class_weight="balanced", max_iter=max_iter, random_state=random_state
    ).fit(scaler.transform(x_train), y_train)
    w_z = np.asarray(clf.coef_[0], dtype=np.float64)
    b = float(clf.intercept_[0])
    w_x = standardized_to_residual_direction(w_z, scaler.scale_)
    aligned = align_direction_to_class(
        unit_direction(w_x), list(clf.classes_), positive_class=positive_class
    )
    return {
        "mean": scaler.mean_.astype(np.float64),
        "scale": scaler.scale_.astype(np.float64),
        "w_z": w_z,
        "b": b,
        "w_x": w_x,
        "u": aligned["u"],
        "classes": [int(c) for c in clf.classes_],
        "mapping": aligned["mapping"],
        "flipped": aligned["flipped"],
    }


def main() -> None:  # pragma: no cover - slow path (needs extracted activations)
    """Load train activations, refit per point, cross-check vs v1.0, persist artifacts."""
    import pandas as pd

    cfg = load_config("causal")
    probe_cfg = load_config("probe")
    points = ["embedding", *[f"block_{i}" for i in range(probe_cfg["n_blocks"])], "ln_f"]
    ensure_dirs(DIRECTION_DIR)

    # v1.0 noun_number train subset (same rule as 20_probe._label_subset/_stratified_cap)
    meta = {}
    for split in ("train", "dev", "test"):
        mdf = pd.read_parquet(ACTS / split / "meta.parquet")
        meta[split] = {c: list(mdf[c]) for c in ("words", "upos", "number", "sent_id")}
    sub = {
        s: [
            u == "NOUN" and n in ("Plur", "Sing")
            for u, n in zip(meta[s]["upos"], meta[s]["number"], strict=True)
        ]
        for s in meta
    }
    y = {
        s: np.array([int(n == "Plur") for n, k in zip(meta[s]["number"], sub[s], strict=True) if k])
        for s in meta
    }
    cap = np.ones(y["train"].size, bool)
    if y["train"].size > cfg["probe"]["train_token_cap"]:
        rng = np.random.default_rng(cfg["probe"]["train_cap_seed"])
        cap = np.zeros(y["train"].size, bool)
        for cls in np.unique(y["train"]):
            idx = np.flatnonzero(y["train"] == cls)
            n = min(
                idx.size,
                max(1, round(cfg["probe"]["train_token_cap"] * idx.size / y["train"].size)),
            )
            keep = rng.choice(idx, size=n, replace=False)
            cap[keep] = True

    # verify the v1.0-chosen C from the reproduced run
    chosen_c = json.loads((OUTPUTS / "probe" / "chosen_C.json").read_text())
    assert chosen_c.get("noun_number") == cfg["probe"]["C"], (
        f"reproduced chosen C for noun_number is {chosen_c.get('noun_number')}, "
        f"expected {cfg['probe']['C']} (configs/causal.yaml)"
    )

    # v1.0 per-token predictions for the cross-check
    with open(OUTPUTS / "probe" / "noun_number.pkl", "rb") as fh:
        v10 = pickle.load(fh)
    v10_preds = {p: np.asarray(v10["per_token"]["points"][p]["probe"]) for p in points}

    manifest = {
        "model_revision": cfg["model_revision"],
        "ud_tag": cfg["ud_tag"],
        "property": "noun_number",
        "C": cfg["probe"]["C"],
        "max_iter": cfg["probe"]["max_iter"],
        "random_state": cfg["probe"]["random_state"],
        "train_token_cap": cfg["probe"]["train_token_cap"],
        "train_cap_seed": cfg["probe"]["train_cap_seed"],
        "positive_class": cfg["probe"]["positive_class"],
        "n_train": int(y["train"][cap].size),
        "points": {},
    }
    print(f"{'point':<12} {'v1.0 BA':>8} {'refit BA':>8} {'match':>6} {'flip':>5}")
    for p in points:
        x_tr = np.load(ACTS / "train" / f"{p}.npy").astype(np.float64)[sub["train"]][cap]
        x_te = np.load(ACTS / "test" / f"{p}.npy").astype(np.float64)[sub["test"]]
        art = fit_point(
            x_tr,
            y["train"][cap],
            cfg["probe"]["C"],
            cfg["probe"]["max_iter"],
            cfg["probe"]["random_state"],
            cfg["probe"]["positive_class"],
        )
        pred = (x_te - art["mean"]) / art["scale"] @ art["w_z"] + art["b"] > 0
        refit_ba = balanced_accuracy(list(y["test"]), list(pred.astype(int)))
        v10_ba = balanced_accuracy(list(y["test"]), list(v10_preds[p]))
        assert refit_ba == v10_ba, f"refit diverges from v1.0 probe at {p}"
        if p in V10_RECORDED_BA:
            assert abs(refit_ba - V10_RECORDED_BA[p]) <= V10_TOLERANCE, (
                f"reconstruction check failed at {p}: refit {refit_ba:.4f} vs recorded "
                f"{V10_RECORDED_BA[p]} (tolerance {V10_TOLERANCE})"
            )
        print(
            f"{p:<12} {v10_ba:>8.4f} {refit_ba:>8.4f} {'OK':>6} "
            f"{'yes' if art['flipped'] else 'no':>5}"
        )

        np.savez(
            DIRECTION_DIR / f"{p}.npz",
            mean=art["mean"],
            scale=art["scale"],
            w_z=art["w_z"],
            b=art["b"],
            w_x=art["w_x"],
            u=art["u"],
        )
        manifest["points"][p] = {
            "sha256": hashlib.sha256((DIRECTION_DIR / f"{p}.npz").read_bytes()).hexdigest(),
            "classes": art["classes"],
            "mapping": art["mapping"],
            "flipped": art["flipped"],
            "refit_balanced_acc": refit_ba,
            "v10_balanced_acc": v10_ba,
        }

    (DIRECTION_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", DIRECTION_DIR)


if __name__ == "__main__":  # pragma: no cover
    main()
