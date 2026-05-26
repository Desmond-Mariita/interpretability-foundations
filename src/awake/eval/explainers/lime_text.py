"""LIME text explainer; produces whitespace-word-level attributions.

LIME perturbs whitespace tokens, so its scores are already word-level and use
the identity alignment path in plausibility scoring (word_level=True).
"""

from __future__ import annotations

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer

from awake.eval.attribution import TokenAttribution


class LimeExplainer:
    """Wraps lime.lime_text over an HF classifier (word-level scores)."""

    name = "lime"

    def __init__(
        self, model, tokenizer, device: str = "cpu", num_samples: int = 1000
    ) -> None:
        """Store model/tokenizer, device, and the LIME sample budget."""
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.num_samples = num_samples
        self.explainer = LimeTextExplainer(class_names=["neg", "pos"], bow=False)

    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return (n_texts, n_classes) probability matrix for a list of texts.

        Args:
            texts: Input strings to classify.

        Returns:
            A float32 array of shape (len(texts), n_classes).
        """
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def attribute(self, example: dict) -> TokenAttribution:
        """Run LIME and map per-word weights back to word order.

        Args:
            example: Dict with keys ``text`` (str) and ``predicted_class``
                (int or None). When None, the argmax of the model output is
                used as the target class.

        Returns:
            A :class:`TokenAttribution` with ``word_level=True`` and one score
            per whitespace-split word.
        """
        text = example["text"]
        words = text.split()
        probs = self._predict_proba([text])[0]
        if example.get("predicted_class") is None:
            pred = int(probs.argmax())
        else:
            pred = int(example["predicted_class"])
        exp = self.explainer.explain_instance(
            text,
            self._predict_proba,
            num_features=len(words),
            num_samples=self.num_samples,
            labels=(pred,),
        )
        scores = np.zeros(len(words))
        # LIME returns (token_index_in_split, weight) pairs when bow=False
        for idx, weight in exp.as_map()[pred]:
            if idx < len(words):
                scores[idx] = weight
        return TokenAttribution(
            tokens=words,
            offsets=[(0, 0)] * len(words),
            scores=scores,
            visible_mask=np.ones(len(words), dtype=bool),
            predicted_class=pred,
            class_scores=probs,
            word_level=True,
        )
