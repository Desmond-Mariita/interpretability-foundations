"""Gradient times input for the fixed predicted-class logit."""

from __future__ import annotations

import torch

from awake.eval.explainers._canonical import fixed_example, tensors, word_result


class GradientXInputExplainer:
    """Differentiate embedding-layer outputs in the original input-ID forward.

    A hook avoids the different automatically inferred position IDs that some
    models use for their ``inputs_embeds`` path (notably padded RoBERTa inputs).
    """

    name = "grad_x_input"

    def __init__(self, model, tokenizer, device: str = "cpu") -> None:
        """Store the frozen model and device."""
        self.model = model.to(device).eval()
        self.device = device

    def attribute(self, example: dict):
        """Return max-absolute subword contributions per canonical word."""
        visible, pred, probs = fixed_example(example)
        ids, attn = tensors(visible, self.device)
        captured = []

        def hook(_module, _inputs, output):
            value = output.detach().requires_grad_(True)
            captured.append(value)
            return value

        handle = self.model.get_input_embeddings().register_forward_hook(hook)
        try:
            logits = self.model(input_ids=ids, attention_mask=attn).logits
            if int(logits.argmax(-1).item()) != pred:
                raise ValueError("intact target changed")
            grad = torch.autograd.grad(logits[0, pred], captured[0])[0]
            sal = (grad * captured[0]).sum(-1)[0].detach().cpu().numpy()
        finally:
            handle.remove()
        return word_result(visible, pred, probs, visible.aggregate(sal))
