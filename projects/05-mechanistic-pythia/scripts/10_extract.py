"""Extract per-point residual-stream activations for each split -> outputs/acts/<split>/."""

from __future__ import annotations

from _paths import ACTS, PREPARED, ensure_dirs, load_config


def main() -> None:  # pragma: no cover - slow path
    """Run Pythia over each split, hook residual points, write per-point float16 .npy + meta."""
    import numpy as np
    import pandas as pd
    from _models import extract_points, load_pythia

    cfg = load_config("probe")
    model, tok = load_pythia(cfg["model_id"], cfg["model_revision"])
    points = ["embedding", *[f"block_{i}" for i in range(cfg["n_blocks"])], "ln_f"]
    for split in ("train", "dev", "test"):
        df = pd.read_parquet(PREPARED / f"{split}.parquet")
        out_dir = ACTS / split
        ensure_dirs(out_dir)
        acc = {p: [] for p in points}
        meta = {"words": [], "upos": [], "number": [], "sent_id": []}
        for row in df.itertuples():
            pts, keep = extract_points(
                model, tok, list(row.words), list(row.space_after), cfg["n_blocks"]
            )
            if not pts:
                continue
            for p in points:
                acc[p].append(pts[p])
            for w, u, nu, k in zip(row.words, row.upos, row.number, keep, strict=True):
                if k:
                    meta["words"].append(w)
                    meta["upos"].append(u)
                    meta["number"].append(nu)
                    meta["sent_id"].append(row.sent_id)
        for p in points:
            np.save(out_dir / f"{p}.npy", np.concatenate(acc[p], axis=0))
        pd.DataFrame(meta).to_parquet(out_dir / "meta.parquet")
        print(f"{split}: {len(meta['words'])} aligned words")


if __name__ == "__main__":  # pragma: no cover
    main()
