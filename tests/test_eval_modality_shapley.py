import numpy as np
import pytest
from awake.eval.modality_shapley import shapley_2player


@pytest.mark.unit
def test_shapley_2player_efficiency_and_symmetry():
    phi_a, phi_b = shapley_2player(v_empty=0.0, v_a=2.0, v_b=4.0, v_ab=10.0)
    # phi_a = ½[(2-0)+(10-4)] = 4 ; phi_b = ½[(4-0)+(10-2)] = 6
    assert phi_a == pytest.approx(4.0)
    assert phi_b == pytest.approx(6.0)
    assert phi_a + phi_b == pytest.approx(10.0 - 0.0)  # efficiency


@pytest.mark.unit
def test_shapley_2player_symmetric_inputs_give_equal_phi():
    phi_a, phi_b = shapley_2player(v_empty=1.0, v_a=3.0, v_b=3.0, v_ab=5.0)
    assert phi_a == pytest.approx(phi_b)
