"""Train fused + image-only + text-only LightGBM heads on cached CLIP embeddings."""

from __future__ import annotations

import argparse

import numpy as np


def fit_heads(img: np.ndarray, txt: np.ndarray, y: np.ndarray, lgbm_params: dict, seed: int):
    """Fit fused/image/text LightGBM heads; scale_pos_weight from the class ratio."""
    import lightgbm as lgb

    pos = max(int((y == 1).sum()), 1)
    neg = int((y == 0).sum())
    spw = neg / pos
    params = {"objective": "binary", "random_state": seed, "scale_pos_weight": spw,
              "verbosity": -1, **lgbm_params}
    feats = {"fused": np.concatenate([img, txt], axis=1), "image": img, "text": txt}
    heads = {}
    for name, X in feats.items():
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X, y)
        heads[name] = clf.booster_
    return heads


def main() -> None:
    """Load cached train embeddings, fit the three heads, save boosters + sidecar meta."""
    import json

    import lightgbm as lgb
    import pandas as pd
    from _paths import embeddings_dir, ensure_dirs, load_config, models_dir

    from awake.utils.seeding import seed_everything

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tcfg = load_config("train")
    seed_everything(tcfg["random_state"])
    emb = embeddings_dir(cfg)
    df = pd.read_parquet(emb / "train.parquet")
    img = np.array(df["img_emb"].tolist())
    txt = np.array(df["txt_emb"].tolist())
    y = df["label"].to_numpy()
    heads = fit_heads(img, txt, y, tcfg["lgbm"], tcfg["random_state"])
    mdir = models_dir(cfg)
    ensure_dirs(mdir)
    for name, booster in heads.items():
        booster.save_model(str(mdir / f"{name}.txt"))
    meta = {"clip_model_id": cfg["clip_model_id"], "img_dim": int(img.shape[1]),
            "txt_dim": int(txt.shape[1]), "class_map": {"benign": 0, "hateful": 1},
            "lightgbm_version": lgb.__version__}
    (mdir / "head_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved heads + head_meta.json -> {mdir}")


if __name__ == "__main__":
    main()
