"""Score cached attributions: faithfulness, plausibility, CIs, figures."""

from __future__ import annotations

import json
import math
from itertools import combinations, pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from _model_adapter import HFModelAdapter
from _paths import ASSETS, CACHE_DIR, MODEL_DIR, ensure_dirs, load_config
from sklearn.metrics import f1_score

from awake.eval.bootstrap import bootstrap_ci, paired_diff_test
from awake.eval.faithfulness import aopc_comprehensiveness, comprehensiveness, sufficiency
from awake.eval.plausibility import (
    aggregate_subwords_to_words,
    clip_gold_mask_to_window,
    token_auprc,
    token_prf1_at_k,
)

REAL_EXPLAINERS = ["grad_x_input", "integrated_gradients", "lime"]


def _clean_word_ids(raw) -> list[int | None]:
    """Normalise a parquet-loaded word_ids row to ``list[int | None]``.

    Parquet stores the ``[None, 0, 1, ...]`` list as a float array, turning the
    special-token ``None`` entries into NaN; restore them to real ``None``/ints.
    """
    out: list[int | None] = []
    for w in raw:
        if w is None or (isinstance(w, float) and math.isnan(w)):
            out.append(None)
        else:
            out.append(int(w))
    return out


def expected_calibration_error(conf, _preds, labels, n_bins=10) -> float:
    """Standard ECE over equal-width confidence bins.

    ``conf`` is the predicted probability for the positive class (``probs[:,
    1]`` for binary tasks); accuracy is the empirical positive rate in each bin
    so that a perfectly calibrated model (conf == P(y=1)) achieves ECE == 0.
    ``_preds`` is accepted for API consistency but is not used.
    """
    conf, labels = np.asarray(conf), np.asarray(labels)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc = labels[m].mean()
            ece += m.mean() * abs(acc - conf[m].mean())
    return float(ece)


def _scores_for_row(attr_df, example_id, n_tokens):
    """Per-token scores for one example, resized to ``n_tokens``."""
    s = attr_df[attr_df.example_id == example_id].sort_values("token_idx")["score"].to_numpy()
    if s.size != n_tokens:
        s = np.resize(s, n_tokens)
    return s


def _faithfulness_for(adapter, sub, attr_df, cfg) -> dict[str, np.ndarray]:
    """Per-example comprehensiveness, sufficiency, AOPC for one explainer."""
    mask_id = adapter.tokenizer.mask_token_id
    comp, suff, aopc = [], [], []
    for i, row in sub.reset_index(drop=True).iterrows():
        ids = np.asarray(row["input_ids"])
        scores = _scores_for_row(attr_df, i, ids.size)
        visible = np.array([w is not None for w in _clean_word_ids(row["word_ids"])], dtype=bool)
        probs = adapter.predict_proba(ids[None, :])[0]
        pred = int(probs.argmax())
        comp.append(
            comprehensiveness(
                adapter.predict_proba, ids, scores, visible, pred, mask_id, cfg["k_d"]
            )
        )
        suff.append(
            sufficiency(adapter.predict_proba, ids, scores, visible, pred, mask_id, cfg["k_d"])
        )
        aopc.append(
            aopc_comprehensiveness(
                adapter.predict_proba, ids, scores, visible, pred, mask_id, tuple(cfg["aopc_bins"])
            )
        )
    return {
        "comprehensiveness": np.array(comp),
        "sufficiency": np.array(suff),
        "aopc": np.array(aopc),
    }


def _plausibility_for(sub, attr_df, cfg, word_level: bool) -> dict[str, np.ndarray]:
    """Per-example token-F1 and AUPRC vs the human rationale for one explainer."""
    f1s, auprcs = [], []
    for i, row in sub.reset_index(drop=True).iterrows():
        n_words = int(row["n_words"])
        gold = clip_gold_mask_to_window(np.asarray(row["gold_mask"]), n_words)
        if word_level:
            raw = _scores_for_row(attr_df, i, n_words)
            word_scores = np.resize(raw, n_words)
        else:
            wids = _clean_word_ids(row["word_ids"])
            raw = _scores_for_row(attr_df, i, len(wids))
            # The tokenizer's word segmentation can differ slightly from the
            # whitespace split behind n_words/gold; size to cover all word ids,
            # then align to the gold range (documented alignment approximation).
            max_wid = max((w for w in wids if w is not None), default=-1)
            word_scores = aggregate_subwords_to_words(raw, wids, max(n_words, max_wid + 1))
        # align predicted word scores and the gold mask to a common length
        length = min(len(word_scores), len(gold))
        word_scores, gold = word_scores[:length], gold[:length]
        k = max(1, round(cfg["k_d"] * length))
        _, _, f1 = token_prf1_at_k(word_scores, gold, k)
        f1s.append(f1)
        auprcs.append(token_auprc(word_scores, gold))
    return {"token_f1": np.array(f1s), "auprc": np.array(auprcs, dtype=float)}


def _cuda() -> bool:
    """True if a CUDA device is available."""
    import torch

    return torch.cuda.is_available()


def main() -> None:
    """Compute all metrics + diagnostics and write metrics.json + figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ensure_dirs()
    cfg = load_config("explainers")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    adapter = HFModelAdapter(model, tok, device="cuda" if _cuda() else "cpu")
    sub = pd.read_parquet(CACHE_DIR / "subsample.parquet")

    probs = np.vstack(
        [adapter.predict_proba(np.asarray(r["input_ids"])[None, :]) for _, r in sub.iterrows()]
    )
    preds = probs.argmax(1)
    labels = sub["label"].to_numpy()
    diagnostics = {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "ece": expected_calibration_error(probs[:, 1], preds, labels),
        "class_balance": float(labels.mean()),
        "n": len(sub),
    }

    word_level_map = {
        "random": False,
        "grad_x_input": False,
        "integrated_gradients": False,
        "lime": True,
    }
    results: dict[str, dict] = {}
    faith_comp: dict[str, np.ndarray] = {}
    plaus_auprc: dict[str, float] = {}
    for name in ["random", *REAL_EXPLAINERS]:
        attr_df = pd.read_parquet(CACHE_DIR / f"{name}.parquet")
        f = _faithfulness_for(adapter, sub, attr_df, cfg)
        p = _plausibility_for(sub, attr_df, cfg, word_level_map[name])
        faith_comp[name] = f["comprehensiveness"]
        plaus_auprc[name] = float(np.nanmean(p["auprc"]))
        metrics = {**f, **p}
        results[name] = {
            m: dict(
                zip(
                    ("ci_low", "mean", "ci_high"),
                    # drop NaNs (e.g. AUPRC on single-class windows) so the table mean
                    # matches the figure's nanmean rather than counting NaN as 0
                    bootstrap_ci(
                        v[~np.isnan(v)] if v.size else v,
                        cfg["bootstrap"]["n_resamples"],
                        cfg["bootstrap"]["alpha"],
                        cfg["bootstrap"]["seed"],
                    ),
                    strict=False,
                )
            )
            for m, v in metrics.items()
        }

    pairs = list(combinations(REAL_EXPLAINERS, 2))
    bonf = cfg["bootstrap"]["alpha"] / len(pairs)
    sig = {}
    for a, b in pairs:
        t = paired_diff_test(
            faith_comp[a],
            faith_comp[b],
            cfg["bootstrap"]["n_resamples"],
            cfg["bootstrap"]["seed"],
        )
        t["significant"] = bool(t["p_value"] < bonf)
        sig[f"{a}_vs_{b}"] = t

    out = {
        "diagnostics": diagnostics,
        "metrics": results,
        "pairwise_comprehensiveness": sig,
        "bonferroni_alpha": bonf,
    }
    Path("metrics.json").write_text(json.dumps(out, indent=2, default=float))

    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ["random", *REAL_EXPLAINERS]:
        ax.scatter(results[name]["aopc"]["mean"], plaus_auprc[name], label=name)
        ax.annotate(name, (results[name]["aopc"]["mean"], plaus_auprc[name]))
    ax.set_xlabel("Faithfulness (AOPC comprehensiveness)")
    ax.set_ylabel("Plausibility (token AUPRC)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "faithfulness_plausibility.png", dpi=150)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
