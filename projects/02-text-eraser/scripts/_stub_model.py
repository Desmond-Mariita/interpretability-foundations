"""A tiny BERT-class model + tokenizer for CPU smoke tests (no downloads)."""

from __future__ import annotations

import os
import tempfile

from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
)


def build_stub_model_and_tokenizer():
    """Return a randomly-initialised 2-layer BERT classifier + a tiny tokenizer."""
    config = BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=2,
    )
    model = BertForSequenceClassification(config)
    # minimal fast tokenizer over a tiny vocab
    vocab = {f"[{t}]": i for i, t in enumerate(["PAD", "UNK", "CLS", "SEP", "MASK"])}
    for i in range(5, 64):
        vocab[f"w{i}"] = i
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "vocab.txt"), "w") as f:
        for tokstr in vocab:
            f.write(tokstr + "\n")
    tok = BertTokenizerFast(
        vocab_file=os.path.join(d, "vocab.txt"),
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        mask_token="[MASK]",
    )
    return model, tok
