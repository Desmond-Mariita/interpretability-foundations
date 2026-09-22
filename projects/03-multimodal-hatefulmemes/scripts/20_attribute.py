"""Per-example modality Shapley over dev + aggregates + hero figure.

Semantics (see REPAIR_REPORT.md):
- ``phi_image`` / ``phi_text`` are the SIGNED contributions (raw-margin units);
  their sign is a direction, not a dominance measure.
- ``signed_share`` = ``phi_image / (|phi_image| + |phi_text|)`` is the mean
  signed direction of the image contribution (in [-1, 1]).
- ``magnitude_share`` = ``|phi_image| / (|phi_image| + |phi_text|)`` is the
  relative magnitude carried by the image (in [0, 1]; 0.5 when both are ~0).
- ``interaction`` = ``v(ab) - v(a) - v(b) + v(empty)`` is the non-additivity
  of the margin under the interventional game (not semantic synergy).
"""

from __future__ import annotations

import argparse

import numpy as np
from _models import margin_value_fn
from _stats import interaction, magnitude_share_image, signed_image_share

from awake.eval.bootstrap import bootstrap_ci
from awake.eval.modality_shapley import interventional_values, shapley_2player


def attribute_split(img, txt, fused_booster, img_bg, txt_bg) -> list[dict]:
    """Return per-example attribution rows (signed phi, shares, interaction) for a split."""
    value_fn = margin_value_fn(fused_booster)
    rows = []
    for i in range(img.shape[0]):
        v_empty, v_img, v_txt, v_ab = interventional_values(
            img[i], txt[i], value_fn, img_bg, txt_bg
        )
        phi_img, phi_txt = shapley_2player(v_empty, v_img, v_txt, v_ab)
        rows.append(
            {
                "phi_image": phi_img,
                "phi_text": phi_txt,
                "total": phi_img + phi_txt,
                "signed_share": signed_image_share(phi_img, phi_txt),
                "magnitude_share": magnitude_share_image(phi_img, phi_txt),
                "interaction": interaction(v_empty, v_img, v_txt, v_ab),
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
    from _paths import (
        ASSETS,
        OUTPUTS,
        PROJECT_ROOT,
        embeddings_dir,
        ensure_dirs,
        load_config,
        models_dir,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tcfg = load_config("train")
    mdir = models_dir(cfg)
    df = pd.read_parquet(embeddings_dir(cfg) / "dev.parquet")
    img = np.array(df["img_emb"].tolist())
    txt = np.array(df["txt_emb"].tolist())
    fused = lgb.Booster(model_file=str(mdir / "fused.txt"))
    bg = np.load(mdir / "background.npz")
    rows = attribute_split(img, txt, fused, bg["img"], bg["txt"])
    signed = np.array([r["signed_share"] for r in rows])
    magnitude = np.array([r["magnitude_share"] for r in rows])
    interaction_vals = np.array([r["interaction"] for r in rows])
    mean_abs = {
        "image": float(np.mean([abs(r["phi_image"]) for r in rows])),
        "text": float(np.mean([abs(r["phi_text"]) for r in rows])),
    }
    mean_signed = {
        "image": float(np.mean([r["phi_image"] for r in rows])),
        "text": float(np.mean([r["phi_text"] for r in rows])),
    }
    abs_total = mean_abs["image"] + mean_abs["text"]
    int_lo, int_mean, int_hi = bootstrap_ci(interaction_vals, n_resamples=2000, seed=0)
    metrics = json.loads((PROJECT_ROOT / "metrics.json").read_text())
    metrics["attribution"] = {
        "n": len(rows),
        "mean_abs_phi": mean_abs,
        "mean_signed_phi": mean_signed,
        "signed_share_mean": float(signed.mean()),
        "magnitude_share_mean": float(magnitude.mean()),
        "magnitude_share_of_mean_abs": {
            "image": float(mean_abs["image"] / abs_total) if abs_total > 0 else 0.5,
            "text": float(mean_abs["text"] / abs_total) if abs_total > 0 else 0.5,
        },
        "interaction": {
            "mean": float(int_mean),
            "mean_abs": float(np.mean(np.abs(interaction_vals))),
            "ci_low": float(int_lo),
            "ci_high": float(int_hi),
            "n_boot": 2000,
            "seed": 0,
        },
        "per_example": f"outputs/attribution/{mdir.name}/dev_attribution.json",
    }
    metrics["background"] = {
        "type": "empirical_train",
        "n": int(bg["img"].shape[0]),
        "seed": int(tcfg["background_seed"]),
    }
    (PROJECT_ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    # Per-example rows stay in a local run artifact (HM-derived scores); only
    # aggregates are committed to metrics.json.
    per_example_dir = OUTPUTS / "attribution" / mdir.name
    ensure_dirs(per_example_dir)
    (per_example_dir / "dev_attribution.json").write_text(json.dumps(rows, indent=2))
    ensure_dirs(ASSETS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(signed, bins=30)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("signed image share  phi_img / (|phi_img|+|phi_text|)")
    axes[0].set_ylabel("dev examples")
    axes[0].set_title("Direction: signed image share (>0 pushes margin up via image)")
    axes[1].hist(magnitude, bins=30)
    axes[1].axvline(0.5, color="k", lw=0.8)
    axes[1].set_xlabel("image magnitude share  |phi_img| / (|phi_img|+|phi_text|)")
    axes[1].set_title("Relative magnitude: share of |phi| carried by the image")
    fig.tight_layout()
    fig.savefig(ASSETS / "modality_attribution.png", dpi=150)
    print(
        f"attribution: mean_abs={mean_abs}  mean_signed={mean_signed}  "
        f"signed_share_mean={signed.mean():.3f}  magnitude_share_mean={magnitude.mean():.3f}"
    )


if __name__ == "__main__":
    main()
