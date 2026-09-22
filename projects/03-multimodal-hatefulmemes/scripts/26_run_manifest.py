"""Persist a run/provenance manifest for the P3 reproduction run.

Writes ``outputs/manifest.json`` (run identity, dataset identity, CLIP model
identity, environment identity, seeds, artifact hashes) and points
``metrics.json → provenance`` at it. This is the anti-artifact-loss record:
if the run outputs are ever discarded again, the manifest documents exactly
what existed, where, and with which hashes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from _paths import DATA_PATH, PROJECT_ROOT, embeddings_dir, load_config, models_dir


def sha256(path: Path) -> str:
    """Return the hex sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    """Return the current git commit sha of the repository."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=PROJECT_ROOT
    ).stdout.strip()


def env_identity() -> dict:
    """Record the software environment identity (importable packages only)."""
    import importlib.metadata as im

    import numpy

    def version(pkg: str) -> str:
        try:
            return im.version(pkg)
        except im.PackageNotFoundError:
            return "not-installed"

    out = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "pandas": version("pandas"),
        "scikit-learn": version("scikit-learn"),
        "lightgbm": version("lightgbm"),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "huggingface-hub": version("huggingface-hub"),
        "pyarrow": version("pyarrow"),
    }
    try:
        import torch

        out["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        out["torch_cuda_available"] = "torch-import-failed"
    return out


def dataset_identity() -> dict:
    """Record dataset source, counts, and checksums (jsonl only; images counted)."""
    root = DATA_PATH / "data" if (DATA_PATH / "data").exists() else DATA_PATH
    counts = {}
    for split in ("train", "dev", "test"):
        p = root / f"{split}.jsonl"
        with open(p) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        counts[split] = len(rows)
    n_images = sum(1 for _ in (root / "img").glob("*.png"))
    identity = {
        "root": str(root),
        "counts": counts,
        "n_images": n_images,
        "jsonl_sha256": {s: sha256(root / f"{s}.jsonl") for s in ("train", "dev", "test")},
    }
    license_file = root / "LICENSE.txt"
    if license_file.exists():
        identity["license_sha256"] = sha256(license_file)
        identity["license_first_line"] = license_file.read_text().splitlines()[0][:80]
    source_file = root / "SOURCE.txt"
    if source_file.exists():
        identity["source"] = source_file.read_text().strip()
    return identity


def clip_identity(cfg: dict) -> dict:
    """Record the CLIP model id and the exact HF revision resolved for it."""
    out = {"clip_model_id": cfg["clip_model_id"]}
    try:
        from huggingface_hub import model_info

        info = model_info(cfg["clip_model_id"])
        out["hf_revision"] = info.sha
    except Exception as exc:  # identity is best-effort; never fatal
        out["hf_revision"] = f"unresolved: {exc}"
    return out


def main() -> None:
    """Build and write the manifest, then point metrics.json provenance at it."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default=dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%SZ"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    tcfg = load_config("train")
    mdir = models_dir(cfg)
    manifest = {
        "run_id": args.run_id,
        "run_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_sha": git_sha(),
        "config": {"clip": cfg, "train": tcfg},
        "dataset": dataset_identity(),
        "clip": clip_identity(cfg),
        "environment": env_identity(),
        "artifacts": {
            "embeddings_dir": str(embeddings_dir(cfg)),
            "models_dir": str(mdir),
            "hashes": {
                f"{cfg['embedding_subdir']}/{s}.parquet": sha256(
                    embeddings_dir(cfg) / f"{s}.parquet"
                )
                for s in ("train", "dev", "test")
            }
            | {
                f"models/{mdir.name}/{name}.txt": sha256(mdir / f"{name}.txt")
                for name in ("fused", "image", "text")
            }
            | {
                f"models/{mdir.name}/background.npz": sha256(mdir / "background.npz"),
                "metrics.json": sha256(PROJECT_ROOT / "metrics.json"),
            },
        },
    }
    attr_file = PROJECT_ROOT / "outputs" / "attribution" / mdir.name / "dev_attribution.json"
    if attr_file.exists():
        manifest["artifacts"]["hashes"][f"attribution/{mdir.name}/dev_attribution.json"] = sha256(
            attr_file
        )
    out = PROJECT_ROOT / "outputs" / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    metrics_path = PROJECT_ROOT / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["provenance"] = {
        "manifest": "outputs/manifest.json",
        "run_id": args.run_id,
        "git_sha": manifest["git_sha"],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"manifest -> {out}")


if __name__ == "__main__":
    main()
