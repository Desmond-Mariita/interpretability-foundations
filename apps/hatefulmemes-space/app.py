"""Gradio Space for Project 3: CLIP + LightGBM meme hate-speech detector.

Wires a CLIP-ViT-B/32 encoder + LightGBM head (loaded from HF Hub at first
inference) to a 2-player interventional modality Shapley explainer and a
leave-one-out token importance scorer.  Import is zero-network: all heavy work
is deferred to :func:`_load` which is called lazily on the first :func:`predict`
invocation.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# awake import: works whether the package is installed (local dev) or vendored
# (HF Space deploy workflow copies src/awake -> apps/hatefulmemes-space/_vendored/awake).
# ---------------------------------------------------------------------------
try:
    import awake.eval.modality_shapley
    import awake.eval.text_occlusion
except ImportError:
    _vendored = pathlib.Path(__file__).parent / "_vendored"
    sys.path.insert(0, str(_vendored))
    import awake.eval.modality_shapley
    import awake.eval.text_occlusion  # noqa: F401

from PIL import Image

from awake.eval.modality_shapley import modality_shapley
from awake.eval.text_occlusion import occlusion_importance

# ---------------------------------------------------------------------------
# Lazy bundle loader
# ---------------------------------------------------------------------------

_BUNDLE: Any = None

_HERE = pathlib.Path(__file__).parent
_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


class _RealBundle:
    """Wraps CLIP + LightGBM head loaded from disk / HF Hub.

    Attributes:
        img_bg: Background image embeddings array, shape ``(N, D_img)``.
        txt_bg: Background text embeddings array, shape ``(N, D_txt)``.
    """

    def __init__(self):
        """Load CLIP, the LightGBM head, and the generic background."""
        import lightgbm as lgb
        import torch
        from transformers import CLIPModel, CLIPProcessor

        # -- CLIP -----------------------------------------------------------
        self._device = "cpu"
        self._clip_model = CLIPModel.from_pretrained(_CLIP_MODEL_ID).to(self._device).eval()
        for p in self._clip_model.parameters():
            p.requires_grad_(False)
        self._clip_proc = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)
        torch.set_grad_enabled(False)

        # -- LightGBM head from HF Hub -------------------------------------
        from huggingface_hub import hf_hub_download

        repo_id = os.environ.get("HF_MODEL_REPO")
        if not repo_id:
            raise RuntimeError(
                "HF_MODEL_REPO is not set. Set it as a Space variable to the HuggingFace "
                "Model Hub repo that hosts the B/32 LightGBM head (fused.txt + head_meta.json)."
            )
        model_path = hf_hub_download(repo_id=repo_id, filename="fused.txt")
        meta_path = hf_hub_download(repo_id=repo_id, filename="head_meta.json")

        with open(meta_path) as fh:
            meta = json.load(fh)

        if meta["clip_model_id"] != _CLIP_MODEL_ID:
            raise RuntimeError(
                f"head_meta.json clip_model_id={meta['clip_model_id']!r} does not match"
                f" loaded CLIP {_CLIP_MODEL_ID!r}"
            )

        # Probe embedding dims with a dummy forward pass
        _dummy_img = Image.new("RGB", (8, 8))
        _dummy_txt = "test"
        _img_e, _txt_e = self._raw_encode(_dummy_img, _dummy_txt)
        img_dim, txt_dim = int(_img_e.shape[0]), int(_txt_e.shape[0])

        if meta.get("img_dim") != img_dim or meta.get("txt_dim") != txt_dim:
            raise RuntimeError(
                f"head_meta.json img_dim={meta.get('img_dim')}, txt_dim={meta.get('txt_dim')}"
                f" but CLIP returns img_dim={img_dim}, txt_dim={txt_dim}"
            )

        self._booster = lgb.Booster(model_file=model_path)

        # -- Generic background -------------------------------------------
        bg = np.load(_HERE / "generic_background.npz")
        self.img_bg = bg["img"]
        self.txt_bg = bg["txt"]

    def _raw_encode(self, image: Image.Image, text: str):
        """Encode one image+text pair; returns 1-D numpy arrays."""
        import torch

        with torch.no_grad():
            pix = self._clip_proc(images=image, return_tensors="pt").to(self._device)
            img_out = self._clip_model.get_image_features(**pix)
            if not isinstance(img_out, type(img_out)) or hasattr(img_out, "pooler_output"):
                img_e = img_out.pooler_output.squeeze(0).cpu().numpy()
            else:
                img_e = img_out.squeeze(0).cpu().numpy()

            tok = self._clip_proc(text=text, return_tensors="pt", padding=True, truncation=True).to(
                self._device
            )
            txt_out = self._clip_model.get_text_features(**tok)
            if hasattr(txt_out, "pooler_output"):
                txt_e = txt_out.pooler_output.squeeze(0).cpu().numpy()
            else:
                txt_e = txt_out.squeeze(0).cpu().numpy()

        return img_e, txt_e

    def encode(self, image: Image.Image, text: str):
        """Encode one image + text; returns ``(img_emb, txt_emb)`` 1-D arrays.

        Args:
            image: PIL image.
            text: Caption string.

        Returns:
            Tuple of ``(img_emb, txt_emb)`` as 1-D numpy float32 arrays.
        """
        return self._raw_encode(image, text)

    def margin(self, feats: np.ndarray) -> np.ndarray:
        """Return LightGBM raw margin scores for a batch of concat embeddings.

        Args:
            feats: Array of shape ``(M, img_dim + txt_dim)``.

        Returns:
            Raw margin array of shape ``(M,)``.
        """
        return np.asarray(self._booster.predict(feats, raw_score=True)).ravel()

    def prob(self, feats: np.ndarray) -> np.ndarray:
        """Return sigmoid class probability for a batch of concat embeddings.

        Args:
            feats: Array of shape ``(M, img_dim + txt_dim)``.

        Returns:
            Probability-of-hateful array of shape ``(M,)`` in ``[0, 1]``.
        """
        return 1.0 / (1.0 + np.exp(-self.margin(feats)))


def _load() -> Any:
    """Return the cached model bundle, building it on the first call.

    Returns:
        A bundle object exposing ``img_bg``, ``txt_bg``, ``encode``,
        ``margin``, and ``prob``.
    """
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = _RealBundle()
    return _BUNDLE


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

EXAMPLE_CAPTION = "look at this picture — what do you see?"


def predict(
    image: Image.Image | None,
    caption: str,
) -> tuple[str, dict[str, float], dict[str, float], list[tuple[str, float]]]:
    """Classify a meme image + caption and return Shapley modality attributions.

    Calls ``_load()`` to obtain the model bundle on the first real inference;
    this is the only site that triggers network access.  When ``image`` is
    ``None`` or ``caption`` is blank a no-input sentinel is returned immediately.

    Args:
        image: PIL meme image uploaded by the user, or ``None``.
        caption: Accompanying caption text.

    Returns:
        A four-tuple ``(label, conf, bars, top_tokens)`` where:

        * ``label`` is ``"hateful"`` or ``"benign"`` (or ``"no input"``).
        * ``conf`` maps ``{"benign": float, "hateful": float}`` in ``[0, 1]``.
        * ``bars`` maps ``{"image": float, "text": float}`` normalised to sum 1
          (absolute Shapley shares).
        * ``top_tokens`` is a list of ``(token, importance)`` pairs.
    """
    _empty_conf = {"benign": 0.0, "hateful": 0.0}
    _empty_bars = {"image": 0.0, "text": 0.0}

    if image is None or not caption.strip():
        return "no input", _empty_conf, _empty_bars, []

    b = _load()

    img_e, txt_e = b.encode(image, caption)
    concat = np.concatenate([img_e, txt_e])

    # -- modality Shapley --------------------------------------------------
    value_fn = lambda batch: b.margin(batch)  # noqa: E731
    phi_dict = modality_shapley(img_e, txt_e, value_fn, b.img_bg, b.txt_bg)

    phi_img = phi_dict["image"]
    phi_txt = phi_dict["text"]
    total = abs(phi_img) + abs(phi_txt)
    if total == 0.0:
        bars = {"image": 0.5, "text": 0.5}
    else:
        bars = {"image": abs(phi_img) / total, "text": abs(phi_txt) / total}

    # -- class probability -------------------------------------------------
    p = float(b.prob(concat[None, :])[0])
    conf = {"benign": float(1.0 - p), "hateful": float(p)}
    label = "hateful" if p >= 0.5 else "benign"

    # -- token importance --------------------------------------------------
    tokens = caption.split()

    def score_fn(toks: list[str]) -> float:
        joined = " ".join(toks)
        _img_e, _txt_e = b.encode(image, joined)
        _concat = np.concatenate([_img_e, _txt_e])
        return float(b.margin(_concat[None, :])[0])

    top_tokens = occlusion_importance(tokens, score_fn, top_k=5)

    return label, conf, bars, top_tokens


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_interface():
    """Build the Gradio Blocks UI wired to ``predict``.

    Returns:
        A ``gr.Blocks`` instance ready to ``.launch()``; kept as a function so
        tests and the HF Space entrypoint share the same construction.
    """
    import gradio as gr

    with gr.Blocks(title="Modality attribution demo") as demo:
        gr.Markdown(
            "## Modality attribution demo\n\n"
            "Upload a meme image and enter a caption to see the classification "
            "and 2-player Shapley modality attribution from the CLIP + LightGBM head. "
            "Part of the "
            "[`interpretability-foundations`](https://github.com/Desmond-Mariita/interpretability-foundations) "
            "portfolio."
        )
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Image")
                caption_in = gr.Textbox(label="Caption", value=EXAMPLE_CAPTION)
                go = gr.Button("Attribute", variant="primary")
            with gr.Column():
                label_out = gr.Label(label="Prediction")
                conf_out = gr.Label(label="Class confidence")
                share_out = gr.Label(label="Modality share")
                tokens_out = gr.JSON(label="Top tokens (leave-one-out importance)")
        go.click(
            predict,
            inputs=[image_in, caption_in],
            outputs=[label_out, conf_out, share_out, tokens_out],
        )
    return demo


if __name__ == "__main__":
    pathlib.Path("outputs").mkdir(exist_ok=True)
    build_interface().launch()
