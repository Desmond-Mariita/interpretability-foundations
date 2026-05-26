"""Uniform-random attribution baseline (the floor in every comparison)."""

from __future__ import annotations

import numpy as np

from awake.eval.attribution import TokenAttribution


class RandomExplainer:
    """Assigns uniform random scores; reference floor for all metrics."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        """Store the RNG seed."""
        self.seed = seed

    def attribute(self, example: dict) -> TokenAttribution:
        """Return random per-token scores for the example's visible tokens."""
        rng = np.random.default_rng(self.seed + hash(str(example["input_ids"])) % 10_000)
        n = len(example["input_ids"])
        return TokenAttribution(
            tokens=list(example["tokens"]),
            offsets=list(example["offsets"]),
            scores=rng.random(n),
            visible_mask=np.asarray(example["visible_mask"], dtype=bool),
            predicted_class=int(example["predicted_class"]),
            class_scores=np.asarray(example["class_scores"], dtype=float),
        )
