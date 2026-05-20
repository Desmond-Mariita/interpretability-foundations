from __future__ import annotations

import matplotlib as mpl
from cycler import cycler

PALETTE: tuple[str, ...] = (
    "#1f4e79",  # deep blue
    "#c0392b",  # accent red
    "#2c7a4b",  # green
    "#b8860b",  # gold
    "#5b3a8a",  # purple
    "#7f7f7f",  # grey (for baselines)
)


def apply_style() -> None:
    """Apply a consistent matplotlib style across notebooks and figures."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.prop_cycle": cycler(color=list(PALETTE)),
        }
    )
