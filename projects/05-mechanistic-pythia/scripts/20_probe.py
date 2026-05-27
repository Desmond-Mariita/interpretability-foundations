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
        results.append({"point": point, "balanced_acc": probe_ba,
                        "control_balanced_acc": ctrl_ba, "selectivity": probe_ba - ctrl_ba})
    return results


def sklearn_fitter(C: float, max_iter: int, random_state: int):  # pragma: no cover - slow
    """Default fit_predict: StandardScaler(train) + balanced LogisticRegression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def fit_predict(x_train, y_train):
        scaler = StandardScaler().fit(x_train)
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter,
                                 random_state=random_state).fit(scaler.transform(x_train), y_train)

        def predict(x):
            return clf.predict(scaler.transform(x))
        return predict
    return fit_predict


def main() -> None:  # pragma: no cover - slow path
    """Load per-point acts, choose C on dev per property, probe+control on test, store raw preds."""
    # Full wiring: load ACTS/<split>/<point>.npy + meta; for each property build label_fn/subset_fn;
    # grid-search C on dev (balanced acc on dev probe); run probe_property on test with the chosen C
    # and config control seeds; persist per-token (gold, probe_pred, control_preds, sent_id) to
    # outputs/probe/<property>.npz for 30_eval's cluster bootstrap. (Mechanical; see spec section 5/7.)
    raise NotImplementedError  # implemented during the real-run task, exercised by `slow`/real run


if __name__ == "__main__":  # pragma: no cover
    main()
