"""Encode Hateful Memes image+text with a frozen CLIP variant; cache per-split parquet."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

import pandas as pd
from _models import encode, load_clip
from _paths import DATA_PATH, embeddings_dir, ensure_dirs, load_config
from PIL import Image

load_split = import_module("00_data").load_split


def main() -> None:
    """Encode each split's images+texts and cache (id,label,img_emb,txt_emb) parquet."""
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-path", type=Path, default=DATA_PATH)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = embeddings_dir(cfg)
    ensure_dirs(out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = load_clip(cfg["clip_model_id"], device)
    root = args.data_path / "data" if (args.data_path / "data").exists() else args.data_path
    for split in ("train", "dev", "test"):
        rows = load_split(root / f"{split}.jsonl")
        recs = []
        for i in range(0, len(rows), args.batch_size):
            chunk = rows[i : i + args.batch_size]
            imgs = [Image.open(root / r["img"]).convert("RGB") for r in chunk]
            img_emb, txt_emb = encode(model, proc, imgs, [r["text"] for r in chunk], device)
            for r, ie, te in zip(chunk, img_emb, txt_emb, strict=True):
                recs.append(
                    {
                        "id": r["id"],
                        "label": r.get("label", -1),
                        "img_emb": ie.tolist(),
                        "txt_emb": te.tolist(),
                    }
                )
        pd.DataFrame(recs).to_parquet(out / f"{split}.parquet")
        print(f"{split}: {len(recs)} encoded -> {out / f'{split}.parquet'}")


if __name__ == "__main__":
    main()
