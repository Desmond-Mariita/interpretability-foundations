"""Run explainers over a stratified test subsample; cache attributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from _explainers import build_explainer
from _paths import CACHE_DIR, MODEL_DIR, PREPARED, ensure_dirs, load_config

EXPLAINERS = ["random", "grad_x_input", "integrated_gradients", "lime"]


def run_one_explainer(
    name,
    model,
    tok,
    df,
    out_dir: Path,
    model_sha: str,
    device: str,
    cfg: dict | None = None,
) -> Path:
    """Run one explainer over ``df`` rows; write a cache parquet with metadata."""
    cfg = cfg or load_config("explainers")
    explainer = build_explainer(name, model, tok, cfg, device)
    rows = []
    for i, row in df.reset_index(drop=True).iterrows():
        attr = explainer.attribute({"text": row["text"], "predicted_class": None})
        for j, (tokstr, score) in enumerate(zip(attr.tokens, attr.scores, strict=True)):
            rows.append(
                {
                    "example_id": int(i),
                    "token_idx": j,
                    "token_str": tokstr,
                    "score": float(score),
                }
            )
    table = pa.Table.from_pandas(pd.DataFrame(rows))
    table = table.replace_schema_metadata(
        {
            "model_sha256": model_sha,
            "explainer_name": name,
            "explainer_version": "1",
            "seed": str(cfg["bootstrap"]["seed"]),
        }
    )
    out = Path(out_dir) / f"{name}.parquet"
    pq.write_table(table, out)
    return out


def _stratified_subsample(df: pd.DataFrame, n: int, threshold: float, seed: int):
    df = df.copy()
    df["cov_stratum"] = (df["truncation_coverage"] >= threshold).astype(int)
    g = df.groupby(["label", "cov_stratum"], group_keys=False)
    frac = min(1.0, n / len(df))
    return g.apply(lambda x: x.sample(frac=frac, random_state=seed)).reset_index(drop=True)


def main() -> None:
    """Load the model + test split, subsample, run all explainers, cache them."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_config("explainers")
    data_cfg = load_config("data")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    sha = (MODEL_DIR / "model_sha256.txt").read_text().strip()
    df = pd.read_parquet(PREPARED / "test.parquet")
    sub = _stratified_subsample(
        df, cfg["subsample_n"], data_cfg["coverage_threshold"], cfg["bootstrap"]["seed"]
    )
    sub.to_parquet(CACHE_DIR / "subsample.parquet")
    for name in EXPLAINERS:
        run_one_explainer(name, model, tok, sub, CACHE_DIR, sha, args.device, cfg)
        print(f"cached {name}")


if __name__ == "__main__":
    main()
