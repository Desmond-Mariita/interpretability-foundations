"""CLIP encoding + LightGBM raw-margin value_fn for project 03."""

from __future__ import annotations

import numpy as np


def load_clip(model_id: str, device: str = "cpu"):
    """Load a frozen CLIP model + processor (eval mode)."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    proc = CLIPProcessor.from_pretrained(model_id)
    torch.set_grad_enabled(False)
    return model, proc


def _to_tensor(out):
    """Extract a plain tensor from either a Tensor or BaseModelOutputWithPooling."""
    import torch

    if isinstance(out, torch.Tensor):
        return out
    # Newer transformers returns BaseModelOutputWithPooling; pooler_output is the CLS embedding.
    return out.pooler_output


def encode(model, proc, images, texts, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Return ``(img_emb, txt_emb)`` arrays for parallel lists of PIL images + strings."""
    import torch

    with torch.no_grad():
        pix = proc(images=images, return_tensors="pt", padding=True).to(device)
        img = _to_tensor(model.get_image_features(**pix))
        tok = proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        txt = _to_tensor(model.get_text_features(**tok))
    return img.cpu().numpy(), txt.cpu().numpy()


def margin_value_fn(booster):
    """Wrap a LightGBM booster as a value_fn: (M, 2D) concat embeddings -> (M,) raw margin."""

    def _fn(batch: np.ndarray) -> np.ndarray:
        return np.asarray(booster.predict(batch, raw_score=True)).ravel()

    return _fn
