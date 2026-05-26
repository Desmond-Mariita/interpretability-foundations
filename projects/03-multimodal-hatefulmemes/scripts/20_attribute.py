"""Per-example modality Shapley over dev + aggregate + hero figure."""

from __future__ import annotations

import argparse

import numpy as np
from _models import margin_value_fn

from awake.eval.modality_shapley import modality_shapley

EPS = 1e-9


def attribute_split(img, txt, fused_booster, img_bg, txt_bg) -> list[dict]:
    """Return per-example phi_image/phi_text/total/share rows for a split."""
    value_fn = margin_value_fn(fused_booster)
    rows = []
    for i in range(img.shape[0]):
        phi = modality_shapley(img[i], txt[i], value_fn, img_bg, txt_bg)
        total = phi["image"] + phi["text"]
        denom = abs(phi["image"]) + abs(phi["text"]) + EPS
        rows.append(
            {
                "phi_image": phi["image"],
                "phi_text": phi["text"],
                "total": total,
                "share": phi["image"] / denom,
            }
        )
    return rows


def main() -> None:
    """Attribute dev, write attribution into metrics.json, render the hero figure."""
    import json

    import matplotlib

    matplotlib.use("Agg")
    import lightgbm as lgb
    import matplotlib.pyplot as plt
    import pandas as pd
    from _paths import ASSETS, PROJECT_ROOT, embeddings_dir, ensure_dirs, load_config, models_dir

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    mdir = models_dir(cfg)
    df = pd.read_parquet(embeddings_dir(cfg) / "dev.parquet")
    img = np.array(df["img_emb"].tolist())
    txt = np.array(df["txt_emb"].tolist())
    fused = lgb.Booster(model_file=str(mdir / "fused.txt"))
    bg = np.load(mdir / "background.npz")
    rows = attribute_split(img, txt, fused, bg["img"], bg["txt"])
    shares = np.array([r["share"] for r in rows])
    mean_abs = {
        "image": float(np.mean([abs(r["phi_image"]) for r in rows])),
        "text": float(np.mean([abs(r["phi_text"]) for r in rows])),
    }
    metrics = json.loads((PROJECT_ROOT / "metrics.json").read_text())
    metrics["attribution"] = {
        "mean_abs_phi": mean_abs,
        "share_mean": float(shares.mean()),
        "n": len(rows),
    }
    metrics["background"] = {"type": "empirical_train", "n": int(bg["img"].shape[0])}
    (PROJECT_ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    ensure_dirs(ASSETS)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(shares, bins=30)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("signed image share  phi_img / (|phi_img|+|phi_text|)")
    ax.set_ylabel("dev examples")
    ax.set_title("Modality attribution (image vs text)")
    fig.tight_layout()
    fig.savefig(ASSETS / "modality_attribution.png", dpi=150)
    print(f"attribution: mean_abs={mean_abs}  share_mean={shares.mean():.3f}")


if __name__ == "__main__":
    main()
