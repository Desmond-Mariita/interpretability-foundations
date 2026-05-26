"""Evaluate fused/image/text heads on dev: AUROC/AUPRC/acc with bootstrap CIs."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


def metric_block(y: np.ndarray, scores: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """AUROC/AUPRC/accuracy@0.5 with percentile bootstrap 95% CIs."""
    rng = np.random.default_rng(seed)
    preds = (scores >= 0.5).astype(int)

    def _metrics(yi, si, pi):
        return (roc_auc_score(yi, si), average_precision_score(yi, si), accuracy_score(yi, pi))

    boots: dict[int, list[float]] = {0: [], 1: [], 2: []}
    n = y.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for k, v in enumerate(_metrics(y[idx], scores[idx], preds[idx])):
            boots[k].append(v)
    names = ["auroc", "auprc", "acc"]
    point = _metrics(y, scores, preds)
    return {
        names[k]: {
            "mean": float(point[k]),
            "lo": float(np.percentile(boots[k], 2.5)),
            "hi": float(np.percentile(boots[k], 97.5)),
        }
        for k in range(3)
    }


def main() -> None:
    """Score the three heads on dev probabilities; write metrics.json model blocks + diffs."""
    import json

    import lightgbm as lgb
    import pandas as pd
    from _paths import PROJECT_ROOT, embeddings_dir, load_config, models_dir

    from awake.eval.bootstrap import paired_diff_test

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tcfg = load_config("train")
    mdir = models_dir(cfg)
    df = pd.read_parquet(embeddings_dir(cfg) / "dev.parquet")
    img = np.array(df["img_emb"].tolist())
    txt = np.array(df["txt_emb"].tolist())
    y = df["label"].to_numpy()
    feats = {"fused": np.concatenate([img, txt], axis=1), "image": img, "text": txt}
    models = {n: lgb.Booster(model_file=str(mdir / f"{n}.txt")) for n in feats}
    probs = {n: models[n].predict(feats[n]) for n in feats}
    out = {
        "split": "dev",
        "n": int(y.size),
        "models": {n: metric_block(y, probs[n], seed=tcfg["background_seed"]) for n in feats},
        "auroc_diffs": {
            f"fused_vs_{m}": paired_diff_test(probs["fused"], probs[m], n_resamples=2000, seed=0)
            for m in ("image", "text")
        },
    }
    (PROJECT_ROOT / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out["models"], indent=2))


if __name__ == "__main__":
    main()
