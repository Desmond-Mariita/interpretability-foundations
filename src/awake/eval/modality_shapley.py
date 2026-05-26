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


def interventional_values(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    value_fn: ValueFn,
    img_background: np.ndarray,
    txt_background: np.ndarray,
) -> tuple[float, float, float, float]:
    """Deterministic interventional coalition values for the 2 modalities.

    ``v({img,txt})`` is the point score on the actual example; absent modalities are
    replaced by every (row-paired) background embedding and averaged. No sampling.

    Args:
        img_emb: Image embedding of the example, shape ``(D_img,)``.
        txt_emb: Text embedding of the example, shape ``(D_txt,)``.
        value_fn: Maps a ``(M, D_img + D_txt)`` batch to ``(M,)`` scores (logits).
        img_background: Background image embeddings, shape ``(N, D_img)``.
        txt_background: Background text embeddings, shape ``(N, D_txt)`` (row-paired).

    Returns:
        ``(v_empty, v_img, v_txt, v_ab)``.
    """
    n = img_background.shape[0]
    img_tiled = np.tile(img_emb, (n, 1))
    txt_tiled = np.tile(txt_emb, (n, 1))
    v_ab = float(value_fn(np.concatenate([img_emb, txt_emb])[None, :])[0])
    v_img = float(value_fn(np.concatenate([img_tiled, txt_background], axis=1)).mean())
    v_txt = float(value_fn(np.concatenate([img_background, txt_tiled], axis=1)).mean())
    v_empty = float(value_fn(np.concatenate([img_background, txt_background], axis=1)).mean())
    return v_empty, v_img, v_txt, v_ab


def modality_shapley(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    value_fn: ValueFn,
    img_background: np.ndarray,
    txt_background: np.ndarray,
) -> dict[str, float]:
    """Return ``{"image": phi_img, "text": phi_txt}`` for one example."""
    v_empty, v_img, v_txt, v_ab = interventional_values(
        img_emb, txt_emb, value_fn, img_background, txt_background
    )
    phi_img, phi_txt = shapley_2player(v_empty, v_img, v_txt, v_ab)
    return {"image": phi_img, "text": phi_txt}
