"""HFModelAdapter: ModelAdapter over a fine-tuned HF sequence classifier."""

from __future__ import annotations

import numpy as np
import torch


class HFModelAdapter:
    """Adapts an HF classifier to ``awake.eval.ModelAdapter.predict_proba``."""

    def __init__(self, model, tokenizer, device: str = "cpu") -> None:
        """Store the model/tokenizer and move the model to ``device``."""
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def predict_proba(self, token_ids_batch: np.ndarray) -> np.ndarray:
        """Softmax probabilities for a ``(batch, seq)`` array of token ids."""
        ids = torch.as_tensor(token_ids_batch, dtype=torch.long, device=self.device)
        attn = (ids != self.tokenizer.pad_token_id).long()
        logits = self.model(input_ids=ids, attention_mask=attn).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()
