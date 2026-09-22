"""Shared fixed-input/target boundary for the Project 2 core explainers."""

from __future__ import annotations

import numpy as np
import torch

from awake.eval.attribution import TokenAttribution
from awake.eval.visible_words import VisibleWords


def fixed_example(example: dict) -> tuple[VisibleWords, int, np.ndarray]:
    """Require a precomputed canonical input and intact predicted class."""
    visible = example["visible"]
    if not isinstance(visible, VisibleWords):
        raise ValueError("explainer requires canonical VisibleWords")
    probs = np.asarray(example["class_scores"], dtype=float)
    pred = int(example["predicted_class"])
    if probs.ndim != 1 or not np.isfinite(probs).all() or pred != int(probs.argmax()):
        raise ValueError("target must be the intact predicted class")
    return visible, pred, probs


def tensors(visible: VisibleWords, device: str):
    """Return original IDs and attention mask, without retokenization."""
    return (
        torch.tensor([visible.input_ids], device=device),
        torch.tensor([visible.attention_mask], device=device),
    )


def word_result(visible, pred, probs, scores, diagnostics=None) -> TokenAttribution:
    """Validate and return scores in canonical word order."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(visible.words),) or not np.isfinite(scores).all():
        raise ValueError("canonical word score mismatch")
    return TokenAttribution(
        tokens=visible.words,
        offsets=visible.spans,
        scores=scores,
        visible_mask=np.ones(len(scores), dtype=bool),
        predicted_class=pred,
        class_scores=probs,
        word_level=True,
        diagnostics=diagnostics or {},
    )
