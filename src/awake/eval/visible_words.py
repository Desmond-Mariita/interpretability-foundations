"""Canonical, span-verified visible words over a frozen classifier token sequence.

The original tokenizer IDs remain authoritative, including a partially retained
last word. Only fully covered whitespace words are editable/evaluated features.
No downstream explainer retokenizes a perturbed string.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass

import numpy as np

CONTRACT = "eraser-visible-words-v2"
TARGET_SCALAR = "original_predicted_class_logit"


@dataclass(frozen=True)
class VisibleWords:
    """Serializable input identity and canonical word-to-token mapping.

    ``token_word`` is -1 for fixed context (specials, whitespace-only tokens,
    and partially retained words). Word indices are positional, not lexical.
    """

    text: str
    input_ids: list[int]
    attention_mask: list[int]
    offsets: list[tuple[int, int]]
    special_tokens_mask: list[int]
    words: list[str]
    spans: list[tuple[int, int]]
    original_word_ids: list[int]
    token_word: list[int]
    original_word_count: int
    partial_word_ids: list[int]

    def __post_init__(self) -> None:
        """Reject mismatched or ambiguous scientific identities."""
        n = len(self.input_ids)
        if not n or any(
            len(a) != n
            for a in (self.attention_mask, self.offsets, self.special_tokens_mask, self.token_word)
        ):
            raise ValueError("token identity/length mismatch")
        if not self.words or not (
            len(self.words) == len(self.spans) == len(self.original_word_ids)
        ):
            raise ValueError("empty or mismatched canonical words")
        if self.original_word_ids != sorted(set(self.original_word_ids)) or any(
            w < 0 or w >= self.original_word_count for w in self.original_word_ids
        ):
            raise ValueError("invalid original word identity")
        for word, (start, end) in zip(self.words, self.spans, strict=True):
            if not (0 <= start < end <= len(self.text)) or self.text[start:end] != word:
                raise ValueError("word span identity mismatch")
        expected = []
        for (start, end), special in zip(self.offsets, self.special_tokens_mask, strict=True):
            if not 0 <= start <= end <= len(self.text):
                raise ValueError("token offset outside visible text")
            overlaps = [i for i, (a, b) in enumerate(self.spans) if end > a and start < b]
            if special:
                if start != end:
                    raise ValueError("special token has a text span")
                overlaps = []
            if len(overlaps) > 1:
                raise ValueError("token overlaps multiple canonical words")
            expected.append(overlaps[0] if overlaps else -1)
        if expected != self.token_word:
            raise ValueError("word/token identity mismatch")
        if any(i not in self.token_word for i in range(len(self.words))):
            raise ValueError("canonical word has no retained tokens")
        for i, (start, end) in enumerate(self.spans):
            covered = {
                c
                for j, (a, b) in enumerate(self.offsets)
                if self.token_word[j] == i
                for c in range(a, b)
            }
            if not set(range(start, end)) <= covered:
                raise ValueError("canonical word not completely visible")

    @property
    def fingerprint(self) -> str:
        """Stable identity; unlike Python hash(), invariant between processes."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def to_json(self) -> str:
        """Serialize deterministically for prepared rows and cache verification."""
        return json.dumps({"contract": CONTRACT, **asdict(self)}, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> VisibleWords:
        """Deserialize and validate; reject superseded contract versions."""
        data = json.loads(raw)
        if data.pop("contract", None) != CONTRACT:
            raise ValueError("stale visible-input contract; rerun prepare")
        return cls(**data)

    def clip_gold(self, gold: list[int]) -> np.ndarray:
        """Select only complete visible ERASER whitespace-word rationale labels."""
        if len(gold) != self.original_word_count or any(g not in (0, 1) for g in gold):
            raise ValueError("gold rationale length/value mismatch")
        return np.asarray(gold, dtype=int)[self.original_word_ids]

    def aggregate(self, scores: np.ndarray) -> np.ndarray:
        """Max absolute subword score per complete word; no invisible zero padding."""
        values = np.asarray(scores, dtype=float)
        if values.shape != (len(self.input_ids),) or not np.isfinite(values).all():
            raise ValueError("subword attribution length/value mismatch")
        mapping = np.asarray(self.token_word)
        return np.array([np.abs(values[mapping == i]).max() for i in range(len(self.words))])

    def perturb(self, keep_words: np.ndarray, mask_token_id: int) -> np.ndarray:
        """Mask all subwords of absent words; retain fixed context and positions."""
        keep = np.asarray(keep_words)
        if keep.ndim != 2 or keep.shape[1] != len(self.words) or not np.isin(keep, [0, 1]).all():
            raise ValueError("word perturbation mask length/value mismatch")
        ids = np.tile(self.input_ids, (len(keep), 1))
        for token, word in enumerate(self.token_word):
            if word >= 0:
                ids[keep[:, word] == 0, token] = mask_token_id
        return ids


def canonical_visible(text: str, tokenizer, max_length: int = 512) -> VisibleWords:
    """Freeze original IDs, derive visible prefix and complete words by offsets.

    A partial last word remains in the IDs and prefix, but is excluded from
    features, gold evaluation and perturbation budgets. This preserves the
    original intact classifier prediction and needs no classifier retraining.
    """
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    special = list(map(int, enc["special_tokens_mask"]))
    real = [o for o, s in zip(offsets, special, strict=True) if not s and o[1] > o[0]]
    end = max((b for _, b in real), default=0)
    matches = list(re.finditer(r"\S+", text))
    covered = {c for a, b in real for c in range(a, b)}
    complete, partial = [], []
    for i, m in enumerate(matches):
        chars = set(range(m.start(), m.end()))
        if chars <= covered:
            complete.append(i)
        elif chars & covered:
            partial.append(i)
    spans = [(matches[i].start(), matches[i].end()) for i in complete]
    token_word = []
    for (a, b), s in zip(offsets, special, strict=True):
        owners = [i for i, (ws, we) in enumerate(spans) if b > ws and a < we] if not s else []
        if len(owners) > 1:
            raise ValueError("ambiguous word/subword overlap")
        token_word.append(owners[0] if owners else -1)
    return VisibleWords(
        text=text[:end],
        input_ids=list(map(int, enc["input_ids"])),
        attention_mask=list(map(int, enc["attention_mask"])),
        offsets=offsets,
        special_tokens_mask=special,
        words=[matches[i].group() for i in complete],
        spans=spans,
        original_word_ids=complete,
        token_word=token_word,
        original_word_count=len(matches),
        partial_word_ids=partial,
    )


def top_words(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Rank canonical word magnitudes, with stable positional tie breaking."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not 0 <= fraction <= 1:
        raise ValueError("invalid word scores or budget")
    keep = np.zeros(scores.size, dtype=bool)
    k = math.ceil(fraction * scores.size)
    keep[np.argsort(-scores, kind="stable")[:k]] = True
    return keep
