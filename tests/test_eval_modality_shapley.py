import numpy as np
import pytest

from awake.eval.modality_shapley import interventional_values, modality_shapley, shapley_2player


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


def _sum_value_fn(batch: np.ndarray) -> np.ndarray:
    """Linear value function: sum across feature dims. batch is (M, 2D)."""
    return batch.sum(axis=1)


@pytest.mark.unit
def test_interventional_values_linear_closed_form():
    img = np.array([1.0, 1.0])  # sum 2
    txt = np.array([10.0, 10.0])  # sum 20
    img_bg = np.array([[0.0, 0.0], [2.0, 2.0]])  # sums 0, 4 -> mean 2
    txt_bg = np.array([[0.0, 0.0], [6.0, 6.0]])  # sums 0, 12 -> mean 6
    v_empty, v_img, v_txt, v_ab = interventional_values(img, txt, _sum_value_fn, img_bg, txt_bg)
    assert v_ab == pytest.approx(22.0)  # point score: 2 + 20, NOT averaged
    assert v_img == pytest.approx(2.0 + 6.0)  # actual img + mean bg txt
    assert v_txt == pytest.approx(2.0 + 20.0)  # mean bg img + actual txt
    assert v_empty == pytest.approx(2.0 + 6.0)  # mean bg img + mean bg txt


@pytest.mark.unit
def test_modality_shapley_efficiency_end_to_end():
    img = np.array([1.0, 1.0])
    txt = np.array([10.0, 10.0])
    img_bg = np.array([[0.0, 0.0], [2.0, 2.0]])
    txt_bg = np.array([[0.0, 0.0], [6.0, 6.0]])
    out = modality_shapley(img, txt, _sum_value_fn, img_bg, txt_bg)
    v_empty, _, _, v_ab = interventional_values(img, txt, _sum_value_fn, img_bg, txt_bg)
    assert set(out) == {"image", "text"}
    assert out["image"] + out["text"] == pytest.approx(v_ab - v_empty)
