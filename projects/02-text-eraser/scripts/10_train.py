"""Fine-tune a sequence classifier on prepared ERASER Movies; save + hash it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from _paths import MODEL_DIR, PREPARED, ensure_dirs, load_config
from torch.utils.data import DataLoader, Dataset

from awake.utils.seeding import seed_everything


class _TextDataset(Dataset):
    """Tokenised text classification dataset backed by a HuggingFace tokenizer."""

    def __init__(self, texts, labels, tok, max_len):
        """Tokenise texts and store labels as a tensor."""
        self.enc = tok(
            list(texts),
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels))

    def __len__(self):
        """Return number of examples."""
        return len(self.labels)

    def __getitem__(self, i):
        """Return a single example as a dict of tensors."""
        return {
            "input_ids": self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "labels": self.labels[i],
        }


def _hash_dir(path: Path) -> str:
    """Return a SHA-256 hex digest of all files in *path* (sorted, recursive)."""
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def train_loop(
    model,
    tok,
    texts,
    labels,
    out_dir: Path,
    epochs,
    batch_size,
    lr,
    max_len,
    fp16,
) -> dict:
    """Train with warmup + gradient clipping; save model + SHA. Returns metadata.

    DeBERTa-v3 diverges to NaN without a learning-rate warmup and gradient
    clipping, so both are applied here; per-step loss is logged for visibility.
    """
    from transformers import get_linear_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).train()
    ds = _TextDataset(texts, labels, tok, max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = max(1, epochs * len(dl))
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    step = 0
    for epoch in range(epochs):
        for batch in dl:
            opt.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0 or step == 1:
                print(f"epoch {epoch} step {step}/{total_steps} loss {out.loss.item():.4f}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    sha = _hash_dir(out_dir)
    (out_dir / "model_sha256.txt").write_text(sha)
    return {"sha256": sha}


def main() -> None:
    """Fine-tune DeBERTa on the prepared train split and save the checkpoint."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ensure_dirs()
    cfg = load_config("model")
    seed_everything(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    df = pd.read_parquet(PREPARED / "train.parquet")
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=cfg["num_labels"]
    )
    meta = train_loop(
        model,
        tok,
        df["text"],
        df["label"],
        MODEL_DIR,
        cfg["epochs"],
        cfg["batch_size"],
        cfg["lr"],
        512,
        cfg["fp16"],
    )
    (MODEL_DIR / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved model, sha={meta['sha256'][:12]}")


if __name__ == "__main__":
    main()
