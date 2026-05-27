"""Per-point linear probing + Hewitt-Liang control on extracted activations.

``probe_property`` is pure-ish (numpy in, dict out) given an injectable ``fit_predict`` callable
(default = standardise + sklearn LR; stub in tests). ``main`` wires real data + sklearn.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from awake.eval.probing import assign_control_labels, balanced_accuracy, control_vector


def probe_property(
    acts_by_split: dict,
    meta_by_split: dict,
    label_fn: Callable[[dict], list[int]],
    subset_fn: Callable[[dict], list[bool]],
    fit_predict: Callable,
    control_seeds: list[int],
    base_rate: float,
    extra_type_words: list[str] | None = None,
) -> list[dict]:
    """Return per-point {point, balanced_acc, control_balanced_acc, selectivity} for one property.

    ``acts_by_split``/``meta_by_split`` map 'train'/'test' -> {point: (n,d)} / meta dict. For the
    smoke stub a single split is reused for train and test.

    ``extra_type_words`` lists additional surface forms (e.g. dev-split words seen during the
    dev C-grid search) whose types must also appear in the control map so ``control_vector``
    never KeyErrors on them; spec section 5 requires the control map to cover the train+dev+test
    union of word types.
    """
    # Support both split-keyed dicts {"train": ..., "test": ...} and bare meta/acts dicts
    # (the smoke stub passes acts/meta directly, so fall back to the whole dict as "all").
    _meta_fallback = meta_by_split.get("all", meta_by_split)
    train_m = meta_by_split.get("train", _meta_fallback)
    test_m = meta_by_split.get("test", _meta_fallback)
    _acts_fallback = acts_by_split.get("all", acts_by_split)
    tr_acts = acts_by_split.get("train", _acts_fallback)
    te_acts = acts_by_split.get("test", _acts_fallback)

    tr_sub, te_sub = np.array(subset_fn(train_m), bool), np.array(subset_fn(test_m), bool)
    y_tr = np.array(label_fn(train_m))[tr_sub]
    y_te = np.array(label_fn(test_m))[te_sub]
    words_tr = [w for w, k in zip(train_m["words"], tr_sub, strict=True) if k]
    words_te = [w for w, k in zip(test_m["words"], te_sub, strict=True) if k]

    # control label map over the union, token-rate-matched on train frequencies
    counts: dict[str, int] = {}
    for w in words_tr:
        counts[w] = counts.get(w, 0) + 1
    all_types = set(words_tr) | set(words_te) | set(extra_type_words or [])

    results = []
    for point in tr_acts:
        x_tr = tr_acts[point][tr_sub].astype(np.float64)
        x_te = te_acts[point][te_sub].astype(np.float64)
        predict = fit_predict(x_tr, y_tr)
        probe_ba = balanced_accuracy(list(y_te), list(predict(x_te)))
        ctrl_bas = []
        for seed in control_seeds:
            cmap = assign_control_labels(all_types, counts, base_rate, seed)
            c_tr = np.array(control_vector(words_tr, cmap))
            c_te = np.array(control_vector(words_te, cmap))
            cpred = fit_predict(x_tr, c_tr)
            ctrl_bas.append(balanced_accuracy(list(c_te), list(cpred(x_te))))
        ctrl_ba = float(np.mean(ctrl_bas))
        results.append(
            {
                "point": point,
                "balanced_acc": probe_ba,
                "control_balanced_acc": ctrl_ba,
                "selectivity": probe_ba - ctrl_ba,
            }
        )
    return results


def sklearn_fitter(C: float, max_iter: int, random_state: int):  # pragma: no cover - slow
    """Default fit_predict: StandardScaler(train) + balanced LogisticRegression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def fit_predict(x_train, y_train):
        scaler = StandardScaler().fit(x_train)
        clf = LogisticRegression(
            C=C, class_weight="balanced", max_iter=max_iter, random_state=random_state
        ).fit(scaler.transform(x_train), y_train)

        def predict(x):
            return clf.predict(scaler.transform(x))

        return predict

    return fit_predict


def _label_subset(prop: str):  # pragma: no cover - slow path
    """Return (label_fn, subset_fn) for a property given a meta dict."""
    if prop == "is_noun":
        return (lambda m: [int(u == "NOUN") for u in m["upos"]], lambda m: [True] * len(m["upos"]))
    if prop == "is_verb":
        return (lambda m: [int(u == "VERB") for u in m["upos"]], lambda m: [True] * len(m["upos"]))
    # noun_number: among NOUN tokens with Number in {Plur, Sing}; label Plur=1
    return (
        lambda m: [int(n == "Plur") for n in m["number"]],
        lambda m: [
            u == "NOUN" and n in ("Plur", "Sing")
            for u, n in zip(m["upos"], m["number"], strict=True)
        ],
    )


def _stratified_cap(y, cap: int, seed: int):  # pragma: no cover - slow path
    """Boolean mask keeping <= cap rows, proportionally per class (seeded)."""
    import numpy as np

    y = np.asarray(y)
    if y.size <= cap:
        return np.ones(y.size, bool)
    rng = np.random.default_rng(seed)
    keep = np.zeros(y.size, bool)
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        n = min(idx.size, max(1, round(cap * idx.size / y.size)))
        keep[rng.choice(idx, size=n, replace=False)] = True
    return keep


def main() -> None:  # pragma: no cover - slow path
    """Load per-point acts, choose C on dev per property, probe+control on test, persist preds."""
    import json
    import pickle

    import numpy as np
    import pandas as pd
    from _paths import ACTS, OUTPUTS, ensure_dirs, load_config

    from awake.eval.probing import base_rate, majority_class, type_overlap

    cfg = load_config("probe")
    points = ["embedding", *[f"block_{i}" for i in range(cfg["n_blocks"])], "ln_f"]

    def fitter(c):
        return sklearn_fitter(c, cfg["probe"]["max_iter"], cfg["probe"]["random_state"])

    acts, meta = {}, {}
    for split in ("train", "dev", "test"):
        d = ACTS / split
        acts[split] = {p: np.load(d / f"{p}.npy") for p in points}
        mdf = pd.read_parquet(d / "meta.parquet")
        meta[split] = {c: list(mdf[c]) for c in ("words", "upos", "number", "sent_id")}

    ensure_dirs(OUTPUTS / "probe")
    chosen_c = {}
    for prop in cfg["properties"]:
        label_fn, subset_fn = _label_subset(prop)
        sub = {s: np.array(subset_fn(meta[s]), bool) for s in ("train", "dev", "test")}

        def keep(split, seq, sub=sub):
            return [v for v, k in zip(seq, sub[split], strict=True) if k]

        y = {s: np.array(label_fn(meta[s]))[sub[s]] for s in ("train", "dev", "test")}
        words = {s: keep(s, meta[s]["words"]) for s in ("train", "dev", "test")}
        sent_te = keep("test", meta["test"]["sent_id"])

        cap = _stratified_cap(y["train"], cfg["train_token_cap"], cfg["train_cap_seed"])
        y_tr = y["train"][cap]
        words_tr = [w for w, k in zip(words["train"], cap, strict=True) if k]
        underpowered = prop == "noun_number" and int(y_tr.size) < cfg["noun_number_min_train"]

        # Choose C on dev at a representative mid-depth point.
        rep = "block_6"
        x_tr_rep = acts["train"][rep][sub["train"]][cap].astype(np.float64)
        x_dev_rep = acts["dev"][rep][sub["dev"]].astype(np.float64)
        best_c, best_ba = cfg["probe"]["C_grid"][0], -1.0
        for c in cfg["probe"]["C_grid"]:
            pred = fitter(c)(x_tr_rep, y_tr)
            ba = balanced_accuracy(list(y["dev"]), list(pred(x_dev_rep)))
            if ba > best_ba:
                best_ba, best_c = ba, c
        chosen_c[prop] = best_c
        fit_predict = fitter(best_c)

        counts: dict[str, int] = {}
        for w in words_tr:
            counts[w] = counts.get(w, 0) + 1
        all_types = set(words_tr) | set(words["dev"]) | set(words["test"])
        br = float(base_rate(list(y_tr)))
        cmaps = [assign_control_labels(all_types, counts, br, s) for s in cfg["control"]["seeds"]]
        c_tr = [np.array(control_vector(words_tr, cm)) for cm in cmaps]
        c_te = [np.array(control_vector(words["test"], cm)) for cm in cmaps]

        per_token = {
            "gold": [int(v) for v in y["test"]],
            "control_gold": [[int(v) for v in ct] for ct in c_te],
            "sent_id": sent_te,
            "points": {},
        }
        for p in points:
            x_tr = acts["train"][p][sub["train"]][cap].astype(np.float64)
            x_te = acts["test"][p][sub["test"]].astype(np.float64)
            probe_pred = [int(v) for v in fit_predict(x_tr, y_tr)(x_te)]
            ctrl_preds = [[int(v) for v in fit_predict(x_tr, ct)(x_te)] for ct in c_tr]
            per_token["points"][p] = {"probe": probe_pred, "control": ctrl_preds}

        maj = majority_class(list(y_tr))
        info = {
            "base_rate": br,
            "majority_baseline": float(np.mean(y["test"] == maj)),
            "train_n": int(y_tr.size),
            "test_n": int(y["test"].size),
            "chosen_C": best_c,
            "underpowered": bool(underpowered),
            "type_overlap": type_overlap(words_tr, words["test"]),
            "control_seeds": cfg["control"]["seeds"],
        }
        with open(OUTPUTS / "probe" / f"{prop}.pkl", "wb") as fh:
            pickle.dump({"per_token": per_token, "info": info}, fh)
        print(
            f"{prop}: C={best_c} train_n={y_tr.size} test_n={y['test'].size} "
            f"base_rate={br:.3f}{' UNDERPOWERED' if underpowered else ''}"
        )
    (OUTPUTS / "probe" / "chosen_C.json").write_text(json.dumps(chosen_c))


if __name__ == "__main__":  # pragma: no cover
    main()
