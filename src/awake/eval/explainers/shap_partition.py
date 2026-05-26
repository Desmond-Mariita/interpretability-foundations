"""SHAP PartitionExplainer for text (stable on long inputs; optional dep)."""

from __future__ import annotations

import numpy as np
import torch

from awake.eval.attribution import TokenAttribution


class ShapPartitionExplainer:
    """Hierarchical PartitionExplainer over the classifier's predicted class."""

    name = "shap_partition"

    def __init__(self, model, tokenizer, device: str = "cpu", max_evals: int = 500) -> None:
        """Store model/tokenizer, device, and the SHAP evaluation budget."""
        import shap

        self.shap = shap
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_evals = max_evals

    def _predict(self, texts):
        enc = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def attribute(self, example: dict) -> TokenAttribution:
        """Compute SHAP values for the predicted class via a text masker.

        Args:
            example: Dict with keys ``text`` (str) and ``predicted_class``
                (int or None). When ``predicted_class`` is None the argmax of
                the model's output probabilities is used.

        Returns:
            A word-level :class:`~awake.eval.attribution.TokenAttribution`.
        """
        text = example["text"]
        words = text.split()
        probs = self._predict([text])[0]
        if example.get("predicted_class") is None:
            pred = int(probs.argmax())
        else:
            pred = int(example["predicted_class"])
        masker = self.shap.maskers.Text(self.tokenizer)
        explainer = self.shap.PartitionExplainer(self._predict, masker)
        sv = explainer([text], max_evals=self.max_evals, silent=True)
        scores = np.asarray(sv.values[0][:, pred]).ravel()[: len(words)]
        if scores.size < len(words):
            scores = np.pad(scores, (0, len(words) - scores.size))
        return TokenAttribution(
            tokens=words,
            offsets=[(0, 0)] * len(words),
            scores=scores,
            visible_mask=np.ones(len(words), dtype=bool),
            predicted_class=pred,
            class_scores=probs,
            word_level=True,
        )
