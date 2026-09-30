import numpy as np
import pytest

from tensor_ops import mac


def test_hand_calculated_unusual_shape():
    a = np.array([[[2.0], [-1.5]]])
    b = np.array([[[4.0], [2.0]]])
    c = np.array([[[-3.0], [0.25]]])
    out = mac(a, b, c)
    assert out.shape == (1, 2, 1)
    np.testing.assert_allclose(out, [[[5.0], [-2.75]]], rtol=1e-12, atol=1e-12)


def test_same_object_for_all_inputs():
    x = np.array([[[0.0, 1.0, 2.0], [-1.0, 3.0, 0.5]]])
    before = x.copy()
    out = mac(x, x, x)
    np.testing.assert_allclose(out, [[[0.0, 2.0, 6.0], [0.0, 12.0, 0.75]]], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(x, before)
    assert not np.shares_memory(out, x)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_fortran_layout_is_rejected(position):
    args = [np.ones((2, 3, 4)) for _ in range(3)]
    args[position] = np.asfortranarray(args[position])
    assert not args[position].flags.c_contiguous
    with pytest.raises(ValueError):
        mac(*args)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_transposed_view_is_rejected(position):
    args = [np.ones((2, 3, 4)) for _ in range(3)]
    args[position] = np.ones((4, 3, 2)).transpose(2, 1, 0)
    assert args[position].shape == (2, 3, 4)
    with pytest.raises(ValueError):
        mac(*args)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_zero_dim_and_4d_inputs_are_rejected(position):
    for bad in (np.array(1.0), np.ones((1, 2, 3, 4))):
        args = [np.ones((2, 3, 4)) for _ in range(3)]
        args[position] = bad
        with pytest.raises(ValueError):
            mac(*args)


def test_mismatched_empty_shapes_are_rejected():
    with pytest.raises(ValueError):
        mac(np.ones((0, 3, 4)), np.ones((0, 3, 4)), np.ones((0, 4, 3)))


def test_empty_input_with_wrong_dtype_is_rejected():
    with pytest.raises(TypeError):
        mac(np.ones((0, 3, 4)), np.ones((0, 3, 4), dtype=np.float32), np.ones((0, 3, 4)))


def test_scalar_argument_is_not_converted():
    with pytest.raises(TypeError):
        mac(np.ones((1, 1, 1)), 2.0, np.ones((1, 1, 1)))


def test_keyword_arguments():
    a = np.full((1, 1, 2), 3.0)
    out = mac(c=np.full((1, 1, 2), 1.0), b=np.full((1, 1, 2), 2.0), a=a)
    np.testing.assert_array_equal(out, [[[7.0, 7.0]]])
