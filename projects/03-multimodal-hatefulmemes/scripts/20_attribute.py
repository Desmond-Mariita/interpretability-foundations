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
import contextlib

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


def render_metrics_figure(metrics: dict, out_path) -> None:
    """Render the evidence-aligned metrics summary figure (2x2, one measure per panel).

    Panel A: per-head AUROC with marginal bootstrap 95% CIs (chance line at 0.5).
    Panel B: paired fused-minus-unimodal AUROC deltas with paired bootstrap CIs
    (zero line; CIs are skipped if unavailable).
    Panel C: aggregate magnitude share of the mean absolutes (stacked image/text).
    Panel D: interaction mean with bootstrap 95% CI over examples (zero line).

    Entity colors are fixed across panels (fused = deep blue, image = green,
    text = accent red; baselines in grey) and the shared portfolio style is used.
    """
    import matplotlib.pyplot as plt

    with contextlib.suppress(ImportError):
        import awake.viz.style as viz_style

        viz_style.apply_style()

    head_colors = {"fused": "#1f4e79", "image": "#2c7a4b", "text": "#c0392b"}
    baseline = "#7f7f7f"
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    # Panel A: per-head AUROC with marginal bootstrap CIs.
    names = ["fused", "image", "text"]
    means = [metrics["models"][n]["auroc"]["mean"] for n in names]
    los = [metrics["models"][n]["auroc"]["lo"] for n in names]
    his = [metrics["models"][n]["auroc"]["hi"] for n in names]
    err_lo = [m - lo for m, lo in zip(means, los, strict=True)]
    err_hi = [hi - m for m, hi in zip(means, his, strict=True)]
    axes[0, 0].bar(
        names,
        means,
        yerr=[err_lo, err_hi],
        capsize=4,
        color=[head_colors[n] for n in names],
        edgecolor="black",
        linewidth=0.5,
        error_kw={"elinewidth": 1.5, "ecolor": baseline},
    )
    axes[0, 0].axhline(0.5, color=baseline, lw=1, ls="--")
    axes[0, 0].set_ylim(0.4, 0.85)
    axes[0, 0].set_ylabel("AUROC")
    axes[0, 0].set_title(
        "A. Per-head AUROC (dev, n=500)\nmarginal bootstrap 95% CI; dashed = chance"
    )
    for x, (m, lo, hi) in enumerate(zip(means, los, his, strict=True)):
        axes[0, 0].text(x, m + 0.012, f"{m:.3f}", ha="center", fontsize=9)
        axes[0, 0].text(x, 0.415, f"[{lo:.3f}, {hi:.3f}]", ha="center", fontsize=8, color="0.35")

    # Panel B: paired fused-minus-unimodal AUROC deltas (forest plot).
    pairs = ["fused_vs_image", "fused_vs_text"]
    pair_labels = ["fused - image", "fused - text"]
    points = [metrics["auroc_diffs"][p]["point_diff"] for p in pairs]
    ci_lo = [metrics["auroc_diffs"][p]["ci_low"] for p in pairs]
    ci_hi = [metrics["auroc_diffs"][p]["ci_high"] for p in pairs]
    ypos = [1, 0]
    for y, (pt, lo, hi) in zip(ypos, zip(points, ci_lo, ci_hi, strict=True), strict=True):
        if lo is not None and hi is not None:
            axes[0, 1].plot(
                [lo, hi], [y, y], color=baseline, lw=4, alpha=0.8, solid_capstyle="round"
            )
        axes[0, 1].plot(pt, y, "o", color="#1f4e79", markersize=9, zorder=3)
        axes[0, 1].text(pt + 0.004, y + 0.12, f"{pt:+.3f}", fontsize=9)
        if lo is not None and hi is not None:
            axes[0, 1].text(
                pt + 0.004, y - 0.22, f"[{lo:+.3f}, {hi:+.3f}]", fontsize=8, color="0.35"
            )
    axes[0, 1].axvline(0, color="#7f7f7f", lw=0.8)
    axes[0, 1].set_yticks(ypos, pair_labels)
    axes[0, 1].set_xlabel("AUROC difference")
    axes[0, 1].set_title(
        "B. Paired AUROC deltas (fused - unimodal)\npaired bootstrap 95% CI over shared examples"
    )
    axes[0, 1].margins(y=0.3)

    # Panel C: aggregate magnitude share of the mean absolutes (stacked single bar).
    mshare = metrics["attribution"]["magnitude_share_of_mean_abs"]
    axes[1, 0].bar(
        [0],
        [mshare["image"]],
        color=head_colors["image"],
        edgecolor="black",
        linewidth=0.5,
        label="image",
        width=0.45,
    )
    axes[1, 0].bar(
        [0],
        [mshare["text"]],
        bottom=[mshare["image"]],
        color=head_colors["text"],
        edgecolor="black",
        linewidth=0.5,
        label="text",
        width=0.45,
    )
    axes[1, 0].text(0, mshare["image"] / 2, f"{mshare['image']:.3f}", ha="center", fontsize=10)
    axes[1, 0].text(
        0, mshare["image"] + mshare["text"] / 2, f"{mshare['text']:.3f}", ha="center", fontsize=10
    )
    axes[1, 0].set_xticks([])
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel("share of mean |phi|")
    ma = metrics["attribution"]["mean_abs_phi"]
    axes[1, 0].set_title(
        f"C. Aggregate magnitude share\n(mean |phi|: image {ma['image']:.3f} vs text {ma['text']:.3f})"
    )
    axes[1, 0].legend(loc="upper left", fontsize=8, frameon=False)

    # Panel D: interaction mean with bootstrap CI over examples (zero line).
    inter = metrics["attribution"]["interaction"]
    if isinstance(inter, dict):
        i_mean, i_lo, i_hi = inter["mean"], inter["ci_low"], inter["ci_high"]
        axes[1, 1].errorbar(
            0,
            i_mean,
            yerr=[[i_mean - i_lo], [i_hi - i_mean]],
            fmt="o",
            capsize=5,
            color="#5b3a8a",
            markersize=8,
            elinewidth=1.5,
        )
        axes[1, 1].axhline(0, color=baseline, lw=0.8)
        axes[1, 1].text(0.06, i_mean, f"{i_mean:+.3f}", fontsize=9, va="center")
        axes[1, 1].text(
            0.06,
            i_mean - (i_mean - i_lo) - 0.15 * abs(i_mean or 1.0),
            f"95% CI [{i_lo:+.3f}, {i_hi:+.3f}]",
            fontsize=8,
            color="0.35",
        )
        axes[1, 1].set_xlim(-0.25, 0.5)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_ylabel("interaction I (margin units)")
        axes[1, 1].set_title(
            "D. Interaction I = v(ab) - v(a) - v(b) + v(empty)\n"
            "mean over dev examples; bootstrap 95% CI"
        )
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "interaction unavailable\n(requires re-run)",
            ha="center",
            va="center",
            color="0.4",
        )
        axes[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


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
    render_metrics_figure(metrics, ASSETS / "metrics_summary.png")
    print(
        f"attribution: mean_abs={mean_abs}  mean_signed={mean_signed}  "
        f"signed_share_mean={signed.mean():.3f}  magnitude_share_mean={magnitude.mean():.3f}"
    )


if __name__ == "__main__":
    main()
