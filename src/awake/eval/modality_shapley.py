"""Exact 2-player interventional modality Shapley over (image, text) feature groups."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

ValueFn = Callable[[np.ndarray], np.ndarray]  # (M, 2D) batch of concat embeddings -> (M,) logits


def shapley_2player(v_empty: float, v_a: float, v_b: float, v_ab: float) -> tuple[float, float]:
    """Exact Shapley values for a 2-player coalition game.

    Args:
        v_empty: Value of the empty coalition.
        v_a: Value of the coalition {a}.
        v_b: Value of the coalition {b}.
        v_ab: Value of the full coalition {a, b}.

    Returns:
        ``(phi_a, phi_b)``; efficiency holds: ``phi_a + phi_b == v_ab - v_empty``.
    """
    phi_a = 0.5 * ((v_a - v_empty) + (v_ab - v_b))
    phi_b = 0.5 * ((v_b - v_empty) + (v_ab - v_a))
    return phi_a, phi_b
