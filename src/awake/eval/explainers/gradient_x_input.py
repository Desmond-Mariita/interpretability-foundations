"""Gradient x Input attribution (valid on DeBERTa's disentangled attention).

Replaces attention rollout, which is ill-defined for DeBERTa's disentangled
attention matrices.
"""

from __future__ import annotations

import numpy as np
import torch

from awake.eval.attribution import TokenAttribution


class GradientXInputExplainer:
    """Saliency = sum over embedding dims of (grad x input embedding)."""

    name = "grad_x_input"

    def __init__(self, model, tokenizer, device: str = "cpu") -> None:
        """Store model/tokenizer and the compute device."""
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    def attribute(self, example: dict) -> TokenAttribution:
        """Compute gradient x input saliency toward the predicted class."""
        enc = self.tokenizer(
            example["text"],
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        word_ids = enc.word_ids()
        ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        emb_layer = self.model.get_input_embeddings()
        embeds = emb_layer(ids).clone().detach().requires_grad_(True)
        logits = self.model(inputs_embeds=embeds, attention_mask=attn).logits
        if example.get("predicted_class") is None:
            pred = int(logits.argmax(dim=-1))
        else:
            pred = int(example["predicted_class"])
        logits[0, pred].backward()
        sal = (embeds.grad * embeds).sum(dim=-1)[0].detach().cpu().numpy()
        visible = np.array([w is not None for w in word_ids], dtype=bool)
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        return TokenAttribution(
            tokens=self.tokenizer.convert_ids_to_tokens(ids[0].tolist()),
            offsets=[tuple(o) for o in offsets],
            scores=sal,
            visible_mask=visible,
            predicted_class=pred,
            class_scores=probs,
        )
