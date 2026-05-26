"""Tiny in-memory stand-ins for smoke tests (no CLIP, no gated data)."""

from __future__ import annotations

import numpy as np


def tiny_embeddings(n: int = 40, d: int = 8, seed: int = 0):
    """Return (img_emb, txt_emb, labels) random arrays for a fake split."""
    rng = np.random.default_rng(seed)
    img = rng.normal(size=(n, d))
    txt = rng.normal(size=(n, d))
    # a learnable signal: label depends on a couple of dims
    labels = ((img[:, 0] + txt[:, 1]) > 0).astype(int)
    return img, txt, labels
