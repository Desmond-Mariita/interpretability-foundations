"""Core attribution datatype and explainer/model protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class TokenAttribution:
    """Per-token importance scores for one model-visible sequence.

    Attributes:
        tokens: Subword (or whitespace, for LIME) token strings.
        offsets: Character ``(start, end)`` spans into the original text.
        scores: Per-token importance toward the predicted class.
        visible_mask: True for real tokens, False for special/pad tokens.
        predicted_class: The class index the metrics are scored against.
        class_scores: Full predicted probability vector for the input.
        word_level: True if scores are already whitespace-word level (LIME);
            False for subword-level.
    """

    tokens: list[str]
    offsets: list[tuple[int, int]]
    scores: np.ndarray
    visible_mask: np.ndarray
    predicted_class: int
    class_scores: np.ndarray
    word_level: bool = False

    def __post_init__(self) -> None:
        """Validate that all per-token sequences have the same length."""
        n = len(self.tokens)
        if not (len(self.offsets) == n == len(self.scores) == len(self.visible_mask)):
            raise ValueError(
                f"length mismatch: tokens={n}, offsets={len(self.offsets)}, "
                f"scores={len(self.scores)}, visible_mask={len(self.visible_mask)}"
            )

    @property
    def n_tokens(self) -> int:
        """Number of tokens in this attribution."""
        return len(self.tokens)


@runtime_checkable
class ModelAdapter(Protocol):
    """Wraps a fine-tuned classifier over its frozen visible tokenization."""

    def predict_proba(self, token_ids_batch: np.ndarray) -> np.ndarray:
        """Return ``(batch, n_classes)`` probabilities for token-id rows."""
        ...


@runtime_checkable
class Explainer(Protocol):
    """Produces a :class:`TokenAttribution` for one prepared example."""

    name: str

    def attribute(self, example: dict) -> TokenAttribution:
        """Explain one example dict (keys: ``input_ids``, ``text``, ...)."""
        ...
