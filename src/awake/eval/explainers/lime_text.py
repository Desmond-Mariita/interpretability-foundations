"""LIME local surrogate over explicit canonical word-presence features.

Uses lime.lime_base, cosine locality and ridge regression. Absent words are
masked in the frozen token sequence, not deleted/retokenized. This is the
word-mask LIME variant, not the old default LimeTextExplainer protocol.
"""

from __future__ import annotations

import numpy as np
import torch
from lime.lime_base import LimeBase
from sklearn.metrics import pairwise_distances

from awake.eval.explainers._canonical import fixed_example, word_result


class LimeExplainer:
    """Explain the original predicted-class logit with positional word features."""

    name = "lime"

    def __init__(self, model, tokenizer, device="cpu", num_samples=1000, batch_size=8, seed=0):
        """Store the frozen model and deterministic neighbourhood configuration."""
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seed = seed

    def attribute(self, example: dict):
        """Fit a local logit surrogate without accessing original hidden text."""
        visible, pred, probs = fixed_example(example)
        n = len(visible.words)
        rng = np.random.RandomState(self.seed)
        masks = np.ones((self.num_samples, n), dtype=int)
        for row in masks[1:]:
            row[rng.choice(n, rng.randint(1, n + 1), replace=False)] = 0
        outputs = []
        for start in range(0, len(masks), self.batch_size):
            batch = visible.perturb(
                masks[start : start + self.batch_size], self.tokenizer.mask_token_id
            )
            ids = torch.as_tensor(batch, device=self.device)
            attn = torch.tensor([visible.attention_mask], device=self.device).expand(len(batch), -1)
            with torch.no_grad():
                outputs.append(self.model(input_ids=ids, attention_mask=attn).logits.cpu().numpy())
        logits = np.concatenate(outputs)
        if int(logits[0].argmax()) != pred:
            raise ValueError("intact target changed")
        distances = pairwise_distances(masks, masks[0:1], metric="cosine").ravel() * 100
        base = LimeBase(lambda d: np.sqrt(np.exp(-(d**2) / 25**2)), random_state=self.seed)
        intercept, pairs, score, local_prediction = base.explain_instance_with_data(
            masks,
            logits,
            distances,
            pred,
            n,
            feature_selection="none",
        )
        if sorted(i for i, _ in pairs) != list(range(n)):
            raise ValueError("LIME feature identities do not match canonical words")
        weights = np.empty(n)
        for i, weight in pairs:
            weights[i] = weight
        return word_result(
            visible,
            pred,
            probs,
            np.abs(weights),
            {
                "weighted_r2": float(score),
                "intercept": float(intercept),
                "local_prediction": float(np.asarray(local_prediction).item()),
                "intact_logit": float(logits[0, pred]),
            },
        )
