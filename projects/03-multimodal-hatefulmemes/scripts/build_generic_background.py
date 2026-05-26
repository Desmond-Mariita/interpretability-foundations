"""Build a generic (non-Hateful-Memes) background embedding set for the Space."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from _models import encode, load_clip
from PIL import Image

OUT = Path(__file__).resolve().parents[3] / "apps/hatefulmemes-space/generic_background.npz"
CAPTIONS = [
    "a photo",
    "a picture of a landscape",
    "an everyday scene",
    "a generic image",
    "some text on a background",
] * 10  # 50 generic captions


def main() -> None:
    """Encode 50 generic noise images + neutral captions; save the .npz for the Space."""
    rng = np.random.default_rng(0)
    images = [Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)) for _ in CAPTIONS]
    model, proc = load_clip("openai/clip-vit-base-patch32", "cpu")
    img_emb, txt_emb = encode(model, proc, images, CAPTIONS, "cpu")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, img=img_emb, txt=txt_emb)
    print(f"saved generic background {img_emb.shape} -> {OUT}")


if __name__ == "__main__":
    main()
