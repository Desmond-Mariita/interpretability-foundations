"""Pure prompt-formatting helpers (no models, no I/O)."""

from __future__ import annotations

_LETTERS = "ABCD"


def format_choices(choices: list[str]) -> str:
    """Render choices as a lettered block: ``A. <c0>\\nB. <c1>`` ..."""
    return "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))


def render(template: str, **kwargs: str) -> str:
    """Fill a prompt template's ``{name}`` placeholders, stripping trailing whitespace."""
    return template.format(**kwargs).strip()
