"""Tiny synthetic fixtures for P5 smoke tests (no model, no dataset)."""

from __future__ import annotations

import numpy as np


def tiny_acts(n: int = 60, d: int = 16, seed: int = 0):
    """Return (acts, meta): per-point residual matrices with is_noun linearly encoded at depth.

    Signal strength grows with depth so selectivity should rise with depth.

    acts: {point: (n, d) float32}; meta: {"upos": [...], "number": [...], "words": [...],
    "sent_id": [...]}.
    """
    rng = np.random.default_rng(seed)
    upos = ["NOUN" if i % 3 == 0 else ("VERB" if i % 3 == 1 else "DET") for i in range(n)]
    is_noun = np.array([u == "NOUN" for u in upos], dtype=float)
    points = {}
    for li, name in enumerate(["embedding", "block_0", "block_1"]):
        noise = rng.normal(size=(n, d))
        # signal strength grows with depth; deep points encode is_noun in dim 0
        signal = (li / 2.0) * is_noun[:, None] * np.concatenate([[3.0], np.zeros(d - 1)])[None, :]
        points[name] = (noise + signal).astype(np.float32)
    sent_id = [str(i // 5) for i in range(n)]  # 5 tokens per fake sentence
    words = [f"w{i}" for i in range(n)]
    number = ["Plur" if (u == "NOUN" and i % 2 == 0) else ("Sing" if u == "NOUN" else "")
              for i, u in enumerate(upos)]
    return points, {"upos": upos, "number": number, "words": words, "sent_id": sent_id}
