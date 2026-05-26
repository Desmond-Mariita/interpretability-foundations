"""Uniform-random attribution baseline (the floor in every comparison)."""

from __future__ import annotations

import numpy as np

from awake.eval.attribution import TokenAttribution


class RandomExplainer:
    """Assigns uniform random scores; reference floor for all metrics.

    Accepts the same ``{"text": ...}`` example the other explainers receive (it
    tokenizes the text to get the token grid), and also a pre-tokenized example
    carrying ``input_ids``/``tokens``/``offsets``/``visible_mask`` directly.
    """

    name = "random"

    def __init__(self, tokenizer=None, seed: int = 0, max_length: int = 512) -> None:
        """Store an optional tokenizer (for the text path), RNG seed, and max length."""
        self.tokenizer = tokenizer
        self.seed = seed
        self.max_length = max_length

    def attribute(self, example: dict) -> TokenAttribution:
        """Return random per-token scores over the example's tokens."""
        if "input_ids" in example:
            tokens = list(example["tokens"])
            offsets = list(example["offsets"])
            visible = np.asarray(example["visible_mask"], dtype=bool)
            n = len(example["input_ids"])
            predicted_class = int(example.get("predicted_class") or 0)
            class_scores = np.asarray(example.get("class_scores", [0.5, 0.5]), dtype=float)
            key = str(example["input_ids"])
        else:
            if self.tokenizer is None:
                raise ValueError("RandomExplainer needs a tokenizer when given raw text")
            enc = self.tokenizer(
                example["text"],
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
                return_tensors=None,
            )
            ids = enc["input_ids"]
            offsets = [tuple(o) for o in enc["offset_mapping"]]
            word_ids = enc.word_ids()
            visible = np.array([w is not None for w in word_ids], dtype=bool)
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            n = len(ids)
            predicted_class = 0
            class_scores = np.array([0.5, 0.5])
            key = str(ids)
        rng = np.random.default_rng(self.seed + hash(key) % 10_000)
        return TokenAttribution(
            tokens=tokens,
            offsets=offsets,
            scores=rng.random(n),
            visible_mask=visible,
            predicted_class=predicted_class,
            class_scores=class_scores,
        )
