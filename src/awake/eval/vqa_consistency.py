"""Pure, I/O-free metric core for the A-OKVQA vision-ablation probe.

No model loading and no file I/O live here: every function takes parsed ints or
strings and returns a number or dict, so the metric definitions are locked down by
fast unit tests. See docs/superpowers/specs/2026-05-26-vqa-aokvqa-design.md (v2.1).
"""

from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


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
