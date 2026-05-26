"""Deterministic stand-in generator for smoke tests (no real models)."""

from __future__ import annotations

_LETTERS = "ABCD"


def stub_generate(prompt: str, item: dict) -> str:
    """Return a canned ``Answer: <letter>`` + explanation string for an item.

    Picks the item's gold choice so smoke runs produce a sensible, deterministic
    signal; the explanation restates the chosen choice text (exercises the leakage
    flag path).

    Args:
        prompt: The (ignored) prompt text; present to match the real generator's
            ``(prompt, item) -> str`` interface.
        item: Dict with ``choices`` and ``correct_choice_idx``.

    Returns:
        A two-line string ``"Answer: <L>\\nBecause it is <choice text>."``.
    """
    idx = int(item["correct_choice_idx"])
    choice_text = item["choices"][idx]
    return f"Answer: {_LETTERS[idx]}\nBecause it is {choice_text}."
