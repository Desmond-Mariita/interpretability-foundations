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
