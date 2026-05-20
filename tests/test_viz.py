from __future__ import annotations

import matplotlib as mpl
import pytest

from awake.viz import PALETTE, apply_style


@pytest.mark.unit
def test_apply_style_sets_palette() -> None:
    mpl.rcParams.update(mpl.rcParamsDefault)
    apply_style()
    cycle_colors = tuple(mpl.rcParams["axes.prop_cycle"].by_key()["color"])
    assert cycle_colors == PALETTE
    assert mpl.rcParams["axes.spines.top"] is False
