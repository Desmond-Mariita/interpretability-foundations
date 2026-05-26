"""Leave-one-out text occlusion importance for token-level attribution."""

from __future__ import annotations

from collections.abc import Callable

ScoreFn = Callable[[list[str]], float]


def occlusion_importance(
    tokens: list[str], score_fn: ScoreFn, top_k: int = 5
) -> list[tuple[str, float]]:
    """Rank tokens by the score drop when each is removed (leave-one-out).

    Args:
        tokens: The (whitespace) tokens of the text.
        score_fn: Maps a token list to a scalar score (e.g. predicted-class logit).
        top_k: Number of highest-magnitude tokens to return.

    Returns:
        ``(token, importance)`` pairs sorted by descending ``abs(importance)``,
        truncated to ``top_k``; ``importance = score(all) - score(all without token)``.
    """
    base = score_fn(tokens)
    scored = [
        (tok, base - score_fn(tokens[:i] + tokens[i + 1 :]))
        for i, tok in enumerate(tokens)
    ]
    scored.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return scored[:top_k]
