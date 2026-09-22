"""Pure, I/O-free core for the P5 v1.1 causal number-intervention study.

Everything here operates on numpy arrays / plain Python values: probe-direction algebra,
the direction-only patch, agreement-margin and donor-shift conventions, deterministic
stimulus hashing, and a lemma-cluster bootstrap for per-layer causal effects. No torch, no
sklearn, no file I/O -- the methodology is locked down by fast unit tests. See
docs/decisions/006-pythia-number-agreement-causal-intervention.md for the design.
"""

from __future__ import annotations

import hashlib

import numpy as np

# ---------------------------------------------------------------------------
# Probe-direction recovery (standardized -> residual space)
# ---------------------------------------------------------------------------


def standardized_to_residual_direction(w_z: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Convert a standardized-space coefficient to the residual-space normal ``w_x = w_z / scale``.

    Componentwise division by the StandardScaler ``scale_`` (mean centering only translates
    the intercept; see :func:`residual_space_intercept`).

    Args:
        w_z: Standardized-space logistic-regression coefficient, shape ``(d,)``.
        scale: StandardScaler ``scale_``, shape ``(d,)`` (same as ``w_z``).

    Returns:
        ``w_x``, shape ``(d,)``.

    Raises:
        ValueError: If shapes differ.
    """
    w_z = np.asarray(w_z, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if w_z.shape != scale.shape:
        raise ValueError("w_z and scale must have equal shape")
    return w_z / scale


def unit_direction(w_x: np.ndarray) -> np.ndarray:
    """Normalize ``w_x`` to a unit direction ``u = w_x / ||w_x||``.

    Raises:
        ValueError: If the vector is zero (a degenerate probe direction).
    """
    w_x = np.asarray(w_x, dtype=np.float64)
    norm = float(np.linalg.norm(w_x))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero probe direction")
    return w_x / norm


def align_direction_to_class(u: np.ndarray, classes: list[int], positive_class: int = 1) -> dict:
    """Align the direction sign so that *positive projection = ``positive_class``*.

    sklearn's LogisticRegression predicts ``classes_[1]`` when the score is positive. If the
    fitted ``classes_`` ordering is ``[0, 1]`` the direction is already aligned (positive
    projection -> class 1); if it is reversed, the direction is sign-flipped.

    Args:
        u: Unit direction (or any coefficient vector), shape ``(d,)``.
        classes: The classifier's ``classes_`` (a list of two ints).
        positive_class: The class the positive projection should map to (default 1).

    Returns:
        Dict with ``u`` (sign-aligned direction), ``mapping`` (``{0: class_of_negative,
        1: class_of_positive}``), and ``flipped`` (bool).

    Raises:
        ValueError: If ``classes`` is not a two-element list or ``positive_class`` is absent.
    """
    classes = list(classes)
    if len(classes) != 2 or positive_class not in classes:
        raise ValueError(f"classes must contain exactly two entries including {positive_class}")
    flipped = bool(classes[1] != positive_class)
    u = np.asarray(u, dtype=np.float64)
    return {
        "u": -u if flipped else u.copy(),
        "mapping": {
            0: int(classes[0] if not flipped else classes[1]),
            1: int(positive_class),
        },
        "flipped": flipped,
    }


def residual_space_score(
    h: np.ndarray, w_x: np.ndarray, b: float, mean: np.ndarray, scale: np.ndarray
) -> float:
    """Equivalent residual-space classifier score: ``w_x . h + b_x``.

    The standardized classifier computes ``w_z . z + b`` with ``z = (h - mean) / scale``.
    With ``w_x = w_z / scale`` the same score equals ``w_x . h + b_x`` where the
    transformed intercept is ``b_x = b - w_x . mean``. Used by unit tests to
    verify direction recovery (the raw ``w_z`` is never used directly in residual space).

    Args:
        h: Residual vector, shape ``(d,)``.
        w_x: Residual-space coefficient ``w_z / scale``, shape ``(d,)``.
        b: Standardized-space intercept.
        mean: StandardScaler ``mean_``, shape ``(d,)``.
        scale: StandardScaler ``scale_``, shape ``(d,)``.

    Returns:
        ``w_x . h + b - w_x . mean`` -- must equal
        ``w_z . ((h - mean) / scale) + b`` up to floating-point error.

    Raises:
        ValueError: If shapes differ.
    """
    h = np.asarray(h, dtype=np.float64)
    w_x = np.asarray(w_x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if h.shape != w_x.shape or w_x.shape != mean.shape or w_x.shape != scale.shape:
        raise ValueError("h, w_x, mean, and scale must have equal shape")
    return float(w_x @ h + b - float(np.sum(w_x * mean)))


# ---------------------------------------------------------------------------
# Direction-only intervention
# ---------------------------------------------------------------------------


def direction_patch(h_r: np.ndarray, h_d: np.ndarray, u: np.ndarray) -> dict:
    """Replace the recipient's projection on ``u`` with the donor's, leaving the rest unchanged.

    ``a_r = u . h_r``, ``a_d = u . h_d``, ``h' = h_r + (a_d - a_r) * u``.

    Args:
        h_r: Recipient residual, shape ``(d,)``.
        h_d: Donor residual, shape ``(d,)``.
        u: Unit direction, shape ``(d,)``.

    Returns:
        Dict with ``a_r``, ``a_d``, ``delta`` (``(a_d - a_r) * u``), ``h_prime``, and
        ``norm_delta`` (``|a_d - a_r|`` since ``u`` is unit).

    Raises:
        ValueError: If shapes differ.
    """
    h_r = np.asarray(h_r, dtype=np.float64)
    h_d = np.asarray(h_d, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if h_r.shape != h_d.shape or h_r.shape != u.shape:
        raise ValueError("h_r, h_d, and u must have equal shape")
    a_r = float(u @ h_r)
    a_d = float(u @ h_d)
    delta = (a_d - a_r) * u
    return {
        "a_r": a_r,
        "a_d": a_d,
        "delta": delta,
        "h_prime": h_r + delta,
        "norm_delta": float(np.abs(a_d - a_r)),
    }


def patch_changes_only_projection(h_r: np.ndarray, h_prime: np.ndarray, u: np.ndarray) -> bool:
    """True iff ``h_prime - h_r`` lies along ``u`` and the orthogonal component is unchanged.

    Args:
        h_r: Original residual, shape ``(d,)``.
        h_prime: Patched residual, shape ``(d,)``.
        u: Unit direction, shape ``(d,)``.
    """
    h_r = np.asarray(h_r, dtype=np.float64)
    h_prime = np.asarray(h_prime, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    delta = h_prime - h_r
    if not np.allclose(delta, (u @ delta) * u, atol=1e-12):
        return False
    orth_h_r = h_r - (u @ h_r) * u
    orth_hp = h_prime - (u @ h_prime) * u
    return bool(np.allclose(orth_h_r, orth_hp, atol=1e-12))


def orthogonal_unit_vector(seed: int, d: int, u: np.ndarray) -> np.ndarray:
    """A deterministic random unit vector orthogonal to ``u`` (Gram-Schmidt + normalize).

    Args:
        seed: RNG seed (determinism).
        d: Dimension.
        u: Unit direction to be orthogonal to, shape ``(d,)``.

    Returns:
        Unit vector ``v`` with ``v . u == 0`` (up to numerical tolerance).

    Raises:
        ValueError: If ``u`` has the wrong shape or is zero.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.shape != (d,):
        raise ValueError("u must have shape (d,)")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d)
    v = v - float(u @ v) * u
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError("degenerate orthogonal vector; try another seed")
    return v / norm


# ---------------------------------------------------------------------------
# Behavioural conventions
# ---------------------------------------------------------------------------


def tied_logits(h: np.ndarray, w_emb: np.ndarray) -> np.ndarray:
    """Next-token logits under the GPT-NeoX tied-embedding head: ``h @ w_emb.T``.

    The pinned pythia-160m config declares ``tie_word_embeddings: false``, which makes
    transformers 5.9 build an untied ``embed_out`` Linear that the checkpoint does not
    provide (it contains no output-head weights), leaving a randomly initialized head.
    The true GPT-NeoX architecture computes logits with the tied embedding matrix, so the
    behavioural readout must use ``final_layer_norm(h) @ embed_in.weight.T`` (recorded in
    ADR 006 / the v1.1 report); ``model.out.logits`` is not used for this model.

    Args:
        h: Post-final-layernorm hidden state at the prediction position, shape ``(d,)``.
        w_emb: Embedding matrix, shape ``(vocab, d)``.

    Returns:
        Logits over the vocabulary, shape ``(vocab,)``.
    """
    h = np.asarray(h, dtype=np.float64)
    w_emb = np.asarray(w_emb, dtype=np.float64)
    if h.shape != w_emb.shape[1:]:
        raise ValueError("h must match the embedding row dimension")
    return w_emb @ h


def agreement_margin(m: float, y: int) -> float:
    """Grammatical agreement margin ``A = y * M``.

    Args:
        m: Verb-number logit contrast ``M = logit(plural_verb) - logit(singular_verb)``.
        y: Subject number label (``+1`` plural, ``-1`` singular).

    Returns:
        ``y * m``; higher means greater preference for the grammatically matching number.
    """
    return y * m


def donor_directed_shift(m_patched: float, m_baseline: float, y_donor: int) -> float:
    """Donor-directed shift ``E = y_d * (M_patched - M_baseline)``.

    Args:
        m_patched: Post-intervention logit contrast.
        m_baseline: Baseline logit contrast.
        y_donor: Donor subject-number label (``+1`` plural, ``-1`` singular).

    Returns:
        Positive means the intervention shifted the verb preference toward the donor number.
    """
    return y_donor * (m_patched - m_baseline)


def directional_rate(values: np.ndarray) -> float:
    """Fraction of values strictly above zero (``0.0`` for empty)."""
    values = np.asarray(values)
    if values.size == 0:
        return 0.0
    return float((values > 0).mean())


# ---------------------------------------------------------------------------
# Deterministic stimulus identity
# ---------------------------------------------------------------------------


def canonical_stimulus_key(
    split: str,
    lemma: str,
    subject_surface: str,
    subject_number: int,
    template_id: str,
    attractor_surface: str,
    attractor_number: int,
    prompt_text: str,
) -> str:
    """Canonical string for a stimulus row (used for SHA-256 stimulus hashing).

    Field order is fixed; the number fields are ints (``+1``/``-1``); empty attractor fields
    serialize as ``""`` so the non-attractor families hash consistently.
    """
    return "|".join(
        [
            split,
            lemma,
            subject_surface,
            str(subject_number),
            template_id,
            attractor_surface,
            str(attractor_number),
            prompt_text,
        ]
    )


def stimulus_hash(key: str) -> str:
    """SHA-256 hex digest of a canonical stimulus key (deterministic stimulus identity)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def table_hash(keys: list[str]) -> str:
    """SHA-256 over the sorted set of canonical stimulus keys (order-independent split hash)."""
    joined = "\n".join(sorted(keys)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


# ---------------------------------------------------------------------------
# Lemma-cluster bootstrap for causal effects
# ---------------------------------------------------------------------------


def cluster_mean_bootstrap(
    values: np.ndarray,
    groups: np.ndarray,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Cluster (group-resampled) percentile bootstrap CI for the mean of ``values``.

    Resamples GROUPS (e.g. noun lemmas) with replacement -- the correct unit of independence
    when rows within a group are correlated (a lemma appears in multiple templates/verb
    pairs). Deterministic under ``seed``.

    Args:
        values: 1-D per-row values.
        groups: 1-D per-row group labels (same length as ``values``).
        n_resamples: Number of bootstrap resamples.
        alpha: Two-sided significance level; CI at ``1 - alpha`` confidence.
        seed: RNG seed.

    Returns:
        ``(lo, mean, hi)`` at the ``1 - alpha`` level (``mean`` = full-sample mean).

    Raises:
        ValueError: If ``values`` and ``groups`` differ in length.
    """
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    if values.shape != groups.shape:
        raise ValueError("values and groups must have equal length")
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    uniq = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        drawn = uniq[rng.integers(0, uniq.size, uniq.size)]
        idx = np.concatenate([by_group[g] for g in drawn])
        stats[i] = values[idx].mean()
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, float(values.mean()), hi
