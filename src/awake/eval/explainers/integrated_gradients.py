"""Integrated Gradients on the embedding layer via captum."""

from __future__ import annotations

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients

from awake.eval.attribution import TokenAttribution


class IntegratedGradientsExplainer:
    """LayerIntegratedGradients over the model's input embeddings."""

    name = "integrated_gradients"

    def __init__(self, model, tokenizer, device: str = "cpu", n_steps: int = 50) -> None:
        """Store model/tokenizer, device, and the IG step count."""
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.n_steps = n_steps

    def _forward(self, input_ids, attention_mask):
        """Run the model and return softmax probabilities."""
        return torch.softmax(
            self.model(input_ids=input_ids, attention_mask=attention_mask).logits, dim=-1
        )

    def attribute(self, example: dict) -> TokenAttribution:
        """Attribute the predicted class with IG; pad baseline as reference."""
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
        probs = self._forward(ids, attn)[0].detach().cpu().numpy()
        if example.get("predicted_class") is None:
            pred = int(probs.argmax())
        else:
            pred = int(example["predicted_class"])
        baseline = torch.full_like(ids, self.tokenizer.pad_token_id)
        lig = LayerIntegratedGradients(self._forward, self.model.get_input_embeddings())
        atts = lig.attribute(
            ids,
            baselines=baseline,
            target=pred,
            additional_forward_args=(attn,),
            n_steps=self.n_steps,
        )
        sal = atts.sum(dim=-1)[0].detach().cpu().numpy()
        visible = np.array([w is not None for w in word_ids], dtype=bool)
        return TokenAttribution(
            tokens=self.tokenizer.convert_ids_to_tokens(ids[0].tolist()),
            offsets=[tuple(o) for o in offsets],
            scores=sal,
            visible_mask=visible,
            predicted_class=pred,
            class_scores=probs,
        )
