"""Hello-world Gradio app for the Project 3 HuggingFace Space.

Week 1 placeholder: the UI shape (image + caption -> prediction, confidence,
modality bars, top tokens) is wired up against a stub backend so deploy
plumbing can be exercised. The real CLIP-ViT-B/32 + LightGBM model and
2-player modality Shapley wire in during week 4.
"""

from __future__ import annotations

import random
from pathlib import Path

import gradio as gr
from PIL import Image

EXAMPLE_CAPTION = "this is an example caption — replace with the real demo example in week 4"


def predict(
    image: Image.Image | None, caption: str
) -> tuple[str, dict[str, float], dict[str, float]]:
    """Stub predictor returning illustrative outputs.

    The real implementation will load CLIP-ViT-B/32, encode the inputs, run the
    LightGBM head, and compute a 2-player Shapley attribution over the image
    and text feature groups.
    """
    if image is None or not caption.strip():
        return "no input", {"benign": 0.0, "hateful": 0.0}, {"image": 0.0, "text": 0.0}

    rng = random.Random(hash((image.size, caption)) & 0xFFFFFFFF)
    p_hateful = rng.uniform(0.0, 1.0)
    image_share = rng.uniform(0.2, 0.8)
    label = "hateful" if p_hateful >= 0.5 else "benign"

    return (
        f"{label} (placeholder)",
        {"benign": 1 - p_hateful, "hateful": p_hateful},
        {"image": image_share, "text": 1 - image_share},
    )


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Modality attribution demo") as demo:
        gr.Markdown(
            "## Modality attribution demo (hello-world)\n\n"
            "Week-1 placeholder UI for Project 3 of the "
            "[`interpretability-foundations`](https://github.com/Desmond-Mariita/interpretability-foundations) "
            "portfolio. The real model lands in week 4."
        )
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Image")
                caption_in = gr.Textbox(label="Caption", value=EXAMPLE_CAPTION)
                go = gr.Button("Attribute", variant="primary")
            with gr.Column():
                label_out = gr.Label(label="Prediction (placeholder)")
                conf_out = gr.Label(label="Class confidence (placeholder)")
                share_out = gr.Label(label="Modality share (placeholder)")
        go.click(
            predict,
            inputs=[image_in, caption_in],
            outputs=[label_out, conf_out, share_out],
        )
    return demo


if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    build_interface().launch()
