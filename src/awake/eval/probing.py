"""Pure, I/O-free core for the layerwise-probing project (P5).

No models, no sklearn, no file I/O: every function takes Python lists/dicts and returns a number
or dict, so the methodology is locked down by fast unit tests. See
docs/superpowers/specs/2026-05-27-mechanistic-pythia-design.md (v2.1). All metric callables use the
sklearn argument order (y_true, y_pred). P5 deliberately does NOT define `accuracy` here -- it reuses
the existing `awake.eval.accuracy`.
"""

from __future__ import annotations

import random


def balanced_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Mean of per-class recall (chance = 0.5 regardless of prevalence).

    Args:
        y_true: Gold binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Mean over classes present in ``y_true`` of (correct / gold) for that class;
        ``0.0`` for empty input.

    Raises:
        ValueError: If the inputs differ in length.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not y_true:
        return 0.0
    recalls = []
    for cls in {*y_true}:
        gold = [p for t, p in zip(y_true, y_pred, strict=True) if t == cls]
        recalls.append(sum(p == cls for p in gold) / len(gold))
    return sum(recalls) / len(recalls)


def base_rate(labels: list[int]) -> float:
    """Share of positive labels (``0.0`` for empty)."""
    return sum(labels) / len(labels) if labels else 0.0


def majority_class(train_labels: list[int]) -> int:
    """Majority class (1 if more than half positive, else 0)."""
    return 1 if base_rate(train_labels) > 0.5 else 0


def selectivity(probe_metric: float, control_metric: float) -> float:
    """probe_metric - control_metric."""
    return probe_metric - control_metric


def assign_control_labels(
    all_types: set[str],
    train_counts: dict[str, int],
    base_rate: float,
    seed: int,
) -> dict[str, int]:
    """Hewitt-Liang control: a deterministic random binary label for every type.

    Labels every type in ``all_types`` (the train+dev+test union, so no scored token is ever
    unseen). The positive share is matched in TOKEN space: types are visited in a seeded-random
    order and assigned 1 until their cumulative TRAIN token mass (from ``train_counts``, 0 for
    types absent from train) reaches ``base_rate`` of the total train mass; the rest get 0.

    Args:
        all_types: Every word type that will be scored (train union dev union test).
        train_counts: Train-token frequency per type (missing -> 0).
        base_rate: Target positive share in token space.
        seed: RNG seed (determinism).

    Returns:
        ``{type: 0|1}`` for every type in ``all_types``.
    """
    types = sorted(all_types)
    random.Random(seed).shuffle(types)
    total = sum(train_counts.values())
    target = base_rate * total
    labels: dict[str, int] = {}
    acc = 0.0
    for t in types:
        if acc < target:
            labels[t] = 1
            acc += train_counts.get(t, 0)
        else:
            labels[t] = 0
    return labels


def control_vector(words: list[str], type_to_label: dict[str, int]) -> list[int]:
    """Map each token's exact-surface-form type to its control label.

    Args:
        words: Per-token surface forms.
        type_to_label: Map built over the train+dev+test union.

    Returns:
        Per-token control labels.

    Raises:
        KeyError: If a word's type is absent from ``type_to_label`` (a programming error,
            since the map must be built over the full union).
    """
    return [type_to_label[w] for w in words]


_DEPTH_PREFIXES = ("embedding", "block_")


def type_overlap(train_words: list[str], test_words: list[str]) -> dict:
    """Token-weighted train/test type overlap.

    Returns:
        ``{"seen_type_token_rate": frac of test tokens whose type appears in train,
        "oov_type_token_rate": 1 - that}``; both ``0.0`` for empty test.
    """
    if not test_words:
        return {"seen_type_token_rate": 0.0, "oov_type_token_rate": 0.0}
    train_types = set(train_words)
    seen = sum(w in train_types for w in test_words) / len(test_words)
    return {"seen_type_token_rate": seen, "oov_type_token_rate": 1.0 - seen}


def _is_depth_point(name: str) -> bool:
    return name == "embedding" or name.startswith("block_")


def emergence_point(
    sel_by_point: dict[str, float],
    sel_ci_by_point: dict[str, tuple[float, float]],
) -> dict:
    """Peak and earliest-emergence depth point (the 'ln_f' extra point is excluded).

    Args:
        sel_by_point: Selectivity per point (may include non-depth points like 'ln_f').
        sel_ci_by_point: ``(lo, hi)`` CI per point.

    Returns:
        ``{"peak": <point with max selectivity among depth points>,
        "earliest_within_peak_ci": <earliest depth point whose CI overlaps the peak's CI>}``.
        Overlap: ``lo_j <= hi_peak and lo_peak <= hi_j``. Lowest-index tie-break (depth points
        are ordered embedding, block_0, block_1, ...).
    """
    def order(name: str) -> int:
        return 0 if name == "embedding" else 1 + int(name.split("_")[1])

    depth = sorted((p for p in sel_by_point if _is_depth_point(p)), key=order)
    peak = max(depth, key=lambda p: (sel_by_point[p], -order(p)))
    lo_pk, hi_pk = sel_ci_by_point[peak]
    for p in depth:
        lo, hi = sel_ci_by_point[p]
        if lo <= hi_pk and lo_pk <= hi:
            return {"peak": peak, "earliest_within_peak_ci": p}
    return {"peak": peak, "earliest_within_peak_ci": peak}
