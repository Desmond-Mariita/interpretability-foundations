"""Train one prespecified v2 substrate; select on dev only, seal, back up, test once."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import platform
import shutil
from pathlib import Path

import numpy as np
import torch
from _contract import code_identity, file_hash, run_identity
from _paths import DATA_PATH, MODEL_DIR, OUTPUTS, PROJECT_ROOT
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from awake.utils.seeding import seed_everything

legacy = importlib.import_module("10_train")
prepare = importlib.import_module("01_prepare")
CONFIG = PROJECT_ROOT / "configs/training_v2.json"


def write_json(path, obj):
    """Write auditable JSON without NaN values."""
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def dataset_identity():
    """Fingerprint the archive and every extracted split/document, without fitting on test."""
    base = DATA_PATH / "movies"
    files = {str(p.relative_to(base)): file_hash(p) for p in sorted(base.rglob("*")) if p.is_file()}
    return {
        "archive_sha256": file_hash(DATA_PATH / "movies.tar.gz"),
        "files": files,
        "extracted_sha256": run_identity(files),
        "split_sha256": {s: files[f"{s}.jsonl"] for s in ("train", "val", "test")},
    }


def load_split(split):
    """Use the existing same-document ERASER inclusion rule."""
    records = []
    for line in (DATA_PATH / "movies" / f"{split}.jsonl").read_text().splitlines():
        ex = json.loads(line)
        docid = ex["annotation_id"]
        if not prepare.is_comparison(ex, docid):
            records.append(
                {
                    "id": docid,
                    "text": (DATA_PATH / "movies/docs" / docid).read_text(),
                    "label": prepare.LABELS[ex["classification"]],
                }
            )
    return records


def selection_key(metrics, epoch):
    """Highest dev accuracy, then lowest dev loss, then earliest epoch wins."""
    return (-metrics["accuracy"], metrics["loss"], epoch)


def loader(records, tok, cfg, shuffle=False):
    """Tokenize the intact review with the frozen tokenizer and sequence limit."""
    ds = legacy._TextDataset(
        [r["text"] for r in records], [r["label"] for r in records], tok, cfg["max_sequence_length"]
    )
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=0)


@torch.no_grad()
def evaluate(model, batches, device):
    """Compute established classifier metrics in float32."""
    model.eval()
    total, labels, probs = 0.0, [], []
    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total += out.loss.item() * len(batch["labels"])
        labels.extend(batch["labels"].cpu().tolist())
        probs.extend(out.logits.softmax(-1).cpu().tolist())
    preds = np.asarray(probs).argmax(1)
    return {
        "n": len(labels),
        "loss": total / len(labels),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }, probs


def main():
    """Run isolated smoke training or the committed primary protocol."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(CONFIG.read_text())
    code = code_identity(PROJECT_ROOT.parents[1])
    assert file_hash(PROJECT_ROOT.parents[1] / "uv.lock") == cfg["environment"]["uv_lock_sha256"]
    data = dataset_identity()
    assert data["archive_sha256"] == cfg["dataset"]["archive_sha256"]
    assert data["extracted_sha256"] == cfg["dataset"]["extracted_sha256"]
    if not torch.cuda.is_available() or "3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This protocol requires the manyee RTX 3090")
    out = OUTPUTS / "smoke-training-v2" if args.smoke else MODEL_DIR
    if (out / "run_manifest.json").exists():
        raise RuntimeError("Sealed run already exists; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    tok = AutoTokenizer.from_pretrained(
        cfg["tokenizer_identifier"], revision=cfg["tokenizer_revision"]
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_identifier"],
        revision=cfg["model_revision"],
        num_labels=2,
        id2label={0: "NEG", 1: "POS"},
        label2id=cfg["label_mapping"],
        attn_implementation="eager",
    ).to(cfg["device"])
    train, dev = load_split("train"), load_split("val")
    epochs = cfg["epochs"]
    if args.smoke:
        train, dev, epochs = train[:8], dev[:8], 1
    batches = loader(train, tok, cfg, True)
    dev_batches = loader(dev, tok, cfg)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        betas=tuple(cfg["adam_betas"]),
        eps=cfg["adam_epsilon"],
    )
    accumulation = cfg["gradient_accumulation"]
    steps = epochs * math.ceil(len(batches) / accumulation)
    scheduler = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["precision"] == "float16_autocast")
    best_key, best, history = None, None, []
    log_path = out / "training_log.jsonl"
    with log_path.open("w") as log:
        for epoch in range(1, epochs + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            for step, batch in enumerate(batches, 1):
                batch = {k: v.to(cfg["device"]) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                scaler.scale(loss / accumulation).backward()
                if step % accumulation == 0 or step == len(batches):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                    old_scale = scaler.get_scale()
                    scaler.step(opt)
                    scaler.update()
                    if scaler.get_scale() >= old_scale:
                        scheduler.step()
                    opt.zero_grad(set_to_none=True)
                record = {
                    "epoch": epoch,
                    "batch": step,
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "grad_scaler": scaler.get_scale(),
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                if step == 1 or step % 10 == 0:
                    print(json.dumps(record), flush=True)
            metrics, _ = evaluate(model, dev_batches, cfg["device"])
            history.append({"epoch": epoch, **metrics})
            print("DEV " + json.dumps(history[-1]), flush=True)
            key = selection_key(metrics, epoch)
            if best_key is None or key < best_key:
                best_key, best = key, history[-1]
                model.save_pretrained(out)
                tok.save_pretrained(out)
    write_json(out / "training_config.json", cfg)
    write_json(out / "dataset_identity.json", data)
    write_json(
        out / "dev_selection.json",
        {"rule": cfg["checkpoint_selection"], "selected": best, "all_epochs": history},
    )
    inventory = {
        p.name: file_hash(p)
        for p in sorted(out.iterdir())
        if p.suffix in {"safetensors", ".safetensors"}
        or p.name
        in {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
        }
    }
    assert "model.safetensors" in inventory and "tokenizer.json" in inventory
    fingerprint = run_identity(inventory)
    (out / "model_sha256.txt").write_text(fingerprint + "\n")
    tokenizer_files = {
        k: v for k, v in inventory.items() if k not in {"model.safetensors", "config.json"}
    }
    meta = {
        "schema": "p2-fresh-training-v2",
        "smoke": args.smoke,
        **code,
        "config_sha256": file_hash(CONFIG),
        "artifacts": inventory,
        "artifact_fingerprint": fingerprint,
        "weights_sha256": inventory["model.safetensors"],
        "tokenizer_fingerprint": run_identity(tokenizer_files),
        "model_revision": cfg["model_revision"],
        "tokenizer_revision": cfg["tokenizer_revision"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            **cfg["environment"],
        },
        "selected": best,
    }
    write_json(out / "train_meta.json", meta)
    # Freeze selection before reading the test split. Smoke never reads test examples.
    if not args.smoke:
        model = AutoModelForSequenceClassification.from_pretrained(out, local_files_only=True).to(
            cfg["device"]
        )
        test = load_split("test")
        metrics, probs = evaluate(model, loader(test, tok, cfg), cfg["device"])
        write_json(
            out / "classifier_test_metrics.json",
            {"schema": "p2-classifier-v2", **metrics, "weights_sha256": meta["weights_sha256"]},
        )
        write_json(
            out / "test_predictions.json",
            [
                {"id": r["id"], "label": r["label"], "probabilities": p}
                for r, p in zip(test, probs, strict=True)
            ],
        )
        print("TEST " + json.dumps(metrics), flush=True)
    manifest = {
        "schema": "p2-model-run-v2",
        **meta,
        "dataset_identity": {k: v for k, v in data.items() if k != "files"},
        "files": {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file()},
    }
    if not args.smoke:
        backup = Path(cfg["backup_root"]).expanduser() / ("seed42-" + fingerprint[:16])
        if backup.exists():
            raise RuntimeError("Refusing to replace an existing stable backup")
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(out, backup)
        for name, sha in manifest["files"].items():
            assert file_hash(backup / name) == sha
        manifest["backup"] = {
            "path": str(backup),
            "weights_sha256": meta["weights_sha256"],
            "inventory_sha256": run_identity(manifest["files"]),
            "verified": True,
        }
        write_json(backup / "run_manifest.json", manifest)
    write_json(out / "run_manifest.json", manifest)
    print("SEALED " + fingerprint, flush=True)


if __name__ == "__main__":
    main()
