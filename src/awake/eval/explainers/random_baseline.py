"""Deterministic uniform scores per canonical word (not per subword)."""

from __future__ import annotations

import numpy as np

from awake.eval.explainers._canonical import fixed_example, word_result


class RandomExplainer:
    """Word-level random ranking with process-independent input-derived seed."""

    name = "random"

    def __init__(self, tokenizer=None, seed: int = 0, max_length: int = 512):
        """Store the seed; tokenization belongs exclusively to preparation."""
        self.seed = seed

    def attribute(self, example: dict):
        """Preserve the fixed class and assign exchangeable random word scores."""
        visible, pred, probs = fixed_example(example)
        seed = self.seed + int(visible.fingerprint[:16], 16)
        rng = np.random.default_rng(seed)
        return word_result(visible, pred, probs, rng.random(len(visible.words)))
