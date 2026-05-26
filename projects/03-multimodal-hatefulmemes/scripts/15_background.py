"""Sample a seeded empirical (row-paired) background set from train embeddings."""

from __future__ import annotations

import argparse

import numpy as np


def sample_background(img: np.ndarray, txt: np.ndarray, n: int, seed: int):
    """Return ``(img_bg, txt_bg)`` of paired rows sampled (without replacement) from train."""
    rng = np.random.default_rng(seed)
    n = min(n, img.shape[0])
    idx = rng.choice(img.shape[0], size=n, replace=False)
    return img[idx], txt[idx]


def main() -> None:
    """Save outputs/models/<variant>/background.npz (empirical, paired)."""
    import pandas as pd
    from _paths import embeddings_dir, load_config, models_dir

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tcfg = load_config("train")
    df = pd.read_parquet(embeddings_dir(cfg) / "train.parquet")
    img = np.array(df["img_emb"].tolist())
    txt = np.array(df["txt_emb"].tolist())
    ib, tb = sample_background(img, txt, tcfg["n_background"], tcfg["background_seed"])
    out = models_dir(cfg) / "background.npz"
    np.savez(out, img=ib, txt=tb)
    print(f"saved background {ib.shape} -> {out}")


if __name__ == "__main__":
    main()
