"""Pure, I/O-free metric core for the A-OKVQA vision-ablation probe.

No model loading and no file I/O live here: every function takes parsed ints or
strings and returns a number or dict, so the metric definitions are locked down by
fast unit tests. See docs/superpowers/specs/2026-05-26-vqa-aokvqa-design.md (v2.1).
"""

from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")
_STRICT = re.compile(r"^\s*answer\s*[:\-]?\s*([a-d])\b", re.IGNORECASE | re.MULTILINE)


def normalize_text(s: str) -> str:
    """Lowercase, drop punctuation, and collapse whitespace.

    Args:
        s: Raw text.

    Returns:
        Normalized string (lowercased, punctuation replaced by spaces, runs of
        whitespace collapsed to single spaces, stripped).
    """
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s)
    return s.strip()


def extract_choice(model_output: str, choices: list[str]) -> tuple[int | None, str]:
    """Parse the chosen choice index from a model's free-text output (strict-then-text).

    Primary (strict): the first ``Answer: <A-D>`` match, mapped to a 0-based index;
    used only if that index is valid for ``choices``. Fallback (text): exactly one
    choice's normalized text is a substring of the normalized output; ambiguous or
    zero matches yield ``None``.

    Args:
        model_output: Raw generated text.
        choices: The multiple-choice option strings (index order defines A, B, C, ...).

    Returns:
        ``(index, parsed_by)`` where ``index`` is a 0-based int or ``None`` and
        ``parsed_by`` is one of ``"strict"``, ``"text"``, ``"none"``.
    """
    m = _STRICT.search(model_output)
    if m is not None:
        idx = "abcd".index(m.group(1).lower())
        if idx < len(choices):
            return idx, "strict"
    norm_out = normalize_text(model_output)
    hits = [i for i, c in enumerate(choices) if normalize_text(c) and normalize_text(c) in norm_out]
    if len(hits) == 1:
        return hits[0], "text"
    return None, "none"


def explanation_leaks_answer(explanation: str, chosen_choice_text: str) -> bool:
    """True iff the chosen choice's text appears verbatim (normalized) in the explanation.

    Detects the direct leak path for the probe: a self-explanation that simply restates
    the chosen answer. Matches the choice TEXT, never the bare letter.

    Args:
        explanation: The model's free-text explanation.
        chosen_choice_text: The text of the choice the model selected.

    Returns:
        ``True`` if ``chosen_choice_text`` is non-empty and its normalized form is a
        substring of the normalized explanation.
    """
    needle = normalize_text(chosen_choice_text)
    if not needle:
        return False
    return needle in normalize_text(explanation)


def rationale_leaks_answer(rationales: list[str], gold_choice_text: str) -> bool:
    """True iff any human rationale contains the gold choice text (normalized).

    Used to build the leakage-sensitivity split on A-OKVQA's ``rationales`` field.

    Args:
        rationales: The human-written rationales for an item.
        gold_choice_text: The text of the gold (correct) choice.

    Returns:
        ``True`` if any rationale leaks the gold choice text.
    """
    return any(explanation_leaks_answer(r, gold_choice_text) for r in rationales)


def parse_rate(parsed: list[int | None]) -> float:
    """Fraction of items that parsed to a choice index (not ``None``).

    Args:
        parsed: Per-item parsed indices (``None`` = unparseable).

    Returns:
        Share in [0, 1]; ``0.0`` for an empty list.
    """
    if not parsed:
        return 0.0
    return sum(p is not None for p in parsed) / len(parsed)


def accuracy(pred: list[int | None], gold: list[int]) -> float:
    """Top-1 accuracy where unparseable predictions count as wrong.

    Args:
        pred: Per-item predicted indices (``None`` = unparseable = wrong).
        gold: Per-item gold indices.

    Returns:
        Share correct in [0, 1] (denominator = all items); ``0.0`` for empty.

    Raises:
        ValueError: If ``pred`` and ``gold`` differ in length.
    """
    if len(pred) != len(gold):
        raise ValueError("pred and gold must have equal length")
    if not pred:
        return 0.0
    return sum(p is not None and p == g for p, g in zip(pred, gold)) / len(pred)


def consistency_rate(
    original: list[int | None],
    ablated: list[int | None],
    paired_only: bool = False,
) -> float:
    """Rate at which the ablated answer matches the original answer.

    Args:
        original: Per-item original parsed indices.
        ablated: Per-item ablated parsed indices.
        paired_only: If ``False`` (primary policy), a pair is consistent iff both
            parse AND are equal, and the denominator is all items (unparseable on
            either side counts as inconsistent). If ``True`` (documented sensitivity),
            pairs with ``None`` on either side are dropped and the denominator is the
            pairs where both sides parsed.

    Returns:
        Consistency share in [0, 1]; ``0.0`` when the denominator is empty.

    Raises:
        ValueError: If the two lists differ in length.
    """
    if len(original) != len(ablated):
        raise ValueError("original and ablated must have equal length")
    pairs = list(zip(original, ablated))
    if paired_only:
        pairs = [(o, a) for o, a in pairs if o is not None and a is not None]
    if not pairs:
        return 0.0
    return sum(o is not None and a is not None and o == a for o, a in pairs) / len(pairs)


def pipeline_divergence(
    a: list[int | None],
    b: list[int | None],
    gold: list[int],
) -> dict:
    """Inter-pipeline divergence plus a correctness-conditioned 2x2 contingency.

    Two pipelines "agree" on an item iff both parse and predict the same index;
    ``None`` on either side counts as a disagreement. Each item is also bucketed by
    (a-correct, b-correct) into one of four cells, and within each cell counted as
    agree/disagree.

    Args:
        a: Pipeline A per-item parsed indices.
        b: Pipeline B per-item parsed indices.
        gold: Per-item gold indices.

    Returns:
        ``{"overall": disagree_rate, "contingency": {cell: {"agree": int,
        "disagree": int}}}`` with cells ``both_correct``, ``a_correct_b_wrong``,
        ``a_wrong_b_correct``, ``both_wrong``.

    Raises:
        ValueError: If the three lists differ in length.
    """
    if not (len(a) == len(b) == len(gold)):
        raise ValueError("a, b, gold must have equal length")
    cells = {
        "both_correct": {"agree": 0, "disagree": 0},
        "a_correct_b_wrong": {"agree": 0, "disagree": 0},
        "a_wrong_b_correct": {"agree": 0, "disagree": 0},
        "both_wrong": {"agree": 0, "disagree": 0},
    }
    disagree = 0
    for ai, bi, gi in zip(a, b, gold):
        agree = ai is not None and bi is not None and ai == bi
        if not agree:
            disagree += 1
        a_ok = ai is not None and ai == gi
        b_ok = bi is not None and bi == gi
        if a_ok and b_ok:
            cell = "both_correct"
        elif a_ok and not b_ok:
            cell = "a_correct_b_wrong"
        elif b_ok and not a_ok:
            cell = "a_wrong_b_correct"
        else:
            cell = "both_wrong"
        cells[cell]["agree" if agree else "disagree"] += 1
    overall = disagree / len(a) if a else 0.0
    return {"overall": overall, "contingency": cells}
