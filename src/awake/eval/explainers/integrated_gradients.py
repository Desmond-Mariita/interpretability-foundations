"""Integrated Gradients for a fixed logit with a mask-token reference."""

from __future__ import annotations

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients

from awake.eval.explainers._canonical import fixed_example, tensors, word_result


class IntegratedGradientsExplainer:
    """Keep specials, partial-word context, positions and attention fixed."""

    name = "integrated_gradients"

    def __init__(self, model, tokenizer, device="cpu", n_steps=50, internal_batch_size=4):
        """Store the model, mask reference and integration budget."""
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size

    def _forward(self, input_ids, attention_mask):
        """Return logits, matching Gradient x Input and word-mask LIME."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def attribute(self, example: dict):
        """Integrate editable embeddings and record signed completeness residual."""
        visible, pred, probs = fixed_example(example)
        ids, attn = tensors(visible, self.device)
        baseline = ids.clone()
        editable = np.asarray(visible.token_word) >= 0
        baseline[:, editable] = self.tokenizer.mask_token_id
        with torch.no_grad():
            logits = self._forward(ids, attn)
            if int(logits.argmax(-1).item()) != pred:
                raise ValueError("intact target changed")
            endpoint = float(logits[0, pred].detach())
            reference = float(self._forward(baseline, attn)[0, pred].detach())
        lig = LayerIntegratedGradients(self._forward, self.model.get_input_embeddings())
        atts, delta = lig.attribute(
            ids,
            baselines=baseline,
            target=pred,
            additional_forward_args=(attn,),
            n_steps=self.n_steps,
            internal_batch_size=self.internal_batch_size,
            return_convergence_delta=True,
        )
        sal = atts.sum(dim=-1)[0].detach().cpu().numpy()
        residual = float(sal.sum() - (endpoint - reference))
        return word_result(
            visible,
            pred,
            probs,
            visible.aggregate(sal),
            {
                "completeness_residual": residual,
                "captum_delta": float(delta.detach().cpu().item()),
                "endpoint_logit_difference": endpoint - reference,
                "attribution_sum": float(sal.sum()),
                "n_steps": self.n_steps,
            },
        )
