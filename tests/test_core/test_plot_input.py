"""
Author: Elko Gerville-Reache
Date Created: 2026-10-26
Date Modified: 2026-10-26
Description:
    Tests for plot_input module.
"""

import astropy.units as u
import numpy as np
import pytest

from visualastro.core.config import config, _UNSET
from visualastro.plotting.core.plot_input import (
    _extract_xy, _extract_xy_from_ndarray2D
)


class TestExtractXY:
    # ##########
    # two inputs
    # ##########
    def test_two_arrays(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])

        X, Y = _extract_xy(x, y)

        assert np.array_equal(X, x)
        assert np.array_equal(Y, y)

    def test_two_arrays_input_mutation(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])

        X, Y = _extract_xy(x, y)

        assert X is x
        assert Y is y

    def test_two_scalars(self):
        X, Y = _extract_xy(1.0, 2.0)

        assert X == 1.0
        assert Y == 2.0

    def test_two_quantities_keep_units(self):
        x = np.array([1, 2, 3]) * u.s
        y = np.array([4, 5, 6]) * u.m

        X, Y = _extract_xy(x, y)

        assert X.unit == u.s
        assert Y.unit == u.m

    def test_two_inputs_x_none_allowed(self):
        y = np.array([4, 5, 6])

        X, Y = _extract_xy(None, y)

        assert X is None
        assert Y is y

    # ##################
    # single 0D/1D input
    # ##################
    def test_single_scalar(self):
        X, Y = _extract_xy(1)

        assert X is None
        assert Y == 1

    def test_single_scalar_quantity(self):
        q = 5.0 * u.m

        X, Y = _extract_xy(q)

        assert X is None
        assert Y.unit == u.m

    def test_single_0d_array(self):
        a = np.array(5.0)

        X, Y = _extract_xy(a)

        assert X is None
        assert Y is a

    def test_single_1d_array(self):
        y = np.array([1, 2, 3])

        X, Y = _extract_xy(y)

        assert X is None
        assert Y is y

    def test_single_1d_quantity(self):
        y = np.array([1, 2, 3]) * u.m

        X, Y = _extract_xy(y)

        assert X is None
        assert Y.unit == u.m
        assert np.array_equal(Y.value, [1, 2, 3])

    def test_single_1d_list(self):
        y = [1, 2, 3]

        X, Y = _extract_xy(y)

        assert X is None
        assert Y is y

    # ###############
    # single 2D input
    # ###############
    def test_single_2d_default_config(self):
        """order='c' indices=(0,1)"""
        arr = np.arange(12).reshape(4, 3)

        X, Y = _extract_xy(arr)

        assert X is None
        assert np.array_equal(Y, arr)

    def test_single_2d_order_c(self):
        arr = np.arange(12).reshape(4, 3)

        X, Y = _extract_xy(arr, order='c', xy_indices=(0, 2))

        assert np.array_equal(X, arr[:, 0])
        assert np.array_equal(Y, arr[:, 2])

    def test_single_2d_order_fortran(self):
        arr = np.arange(12).reshape(3, 4)

        X, Y = _extract_xy(arr, order='fortran', xy_indices=(0, 2))

        assert np.array_equal(X, arr[0, :])
        assert np.array_equal(Y, arr[2, :])

    @pytest.mark.parametrize('order', ['f', 'F', 'fortran', 'FORTRAN', 'Fortran'])
    def test_single_2d_fortran_aliases(self, order):
        arr = np.arange(8).reshape(2, 4)

        X, Y = _extract_xy(arr, order=order, xy_indices=(0,1))

        assert np.array_equal(X, arr[0, :])
        assert np.array_equal(Y, arr[1, :])

    @pytest.mark.parametrize('order', ['c', 'C'])
    def test_single_2d_c_aliases(self, order):
        arr = np.arange(8).reshape(4, 2)

        X, Y = _extract_xy(arr, order=order, xy_indices=(0,1))

        assert np.array_equal(X, arr[:, 0])
        assert np.array_equal(Y, arr[:, 1])

    def test_single_2d_swapped_indices(self):
        arr = np.arange(8).reshape(4, 2)

        X, Y = _extract_xy(arr, xy_indices=(1, 0))

        assert np.array_equal(X, arr[:, 1])
        assert np.array_equal(Y, arr[:, 0])

    def test_single_2d_negative_indices(self):
        arr = np.arange(12).reshape(4, 3)

        X, Y = _extract_xy(arr, xy_indices=(-2, -1))

        assert np.array_equal(X, arr[:, -2])
        assert np.array_equal(Y, arr[:, -1])

    def test_single_2d_indices_as_list(self):
        arr = np.arange(12).reshape(4, 3)

        X, Y = _extract_xy(arr, xy_indices=[0, 2])

        assert np.array_equal(X, arr[:, 0])
        assert np.array_equal(Y, arr[:, 2])

    def test_single_2d_returns_views(self):
        arr = np.arange(12).reshape(4, 3)

        X, Y = _extract_xy(arr, xy_indices=(0,1))

        assert np.shares_memory(X, arr)
        assert np.shares_memory(Y, arr)

    def test_single_2d_quantity_keeps_units(self):
        arr = np.arange(8).reshape(4, 2) * u.m

        X, Y = _extract_xy(arr, xy_indices=(0,1))

        assert X.unit == u.m
        assert Y.unit == u.m
        assert np.array_equal(X.value, arr.value[:, 0])
        assert np.array_equal(Y.value, arr.value[:, 1])

    # ######
    # raises
    # ######
    def test_two_inputs_y_none_raises(self):
        with pytest.raises(ValueError, match='`y` must not be None'):
            _extract_xy(np.array([1, 2, 3]), None)

    def test_no_input_raises(self):
        with pytest.raises(ValueError, match='got 0'):
            _extract_xy()

    def test_three_inputs_raises(self):
        a = np.array([1, 2])
        with pytest.raises(ValueError, match='got 3'):
            _extract_xy(a, a, a)

    def test_single_none_raises(self):
        with pytest.raises(ValueError, match='must not be None'):
            _extract_xy(None)

class TestExtractXYFromNdarray2D:
    def test_order_c(self):
        arr = np.arange(12).reshape(4, 3)

        X, Y = _extract_xy_from_ndarray2D(arr, 'c', (0, 2))

        assert np.array_equal(X, arr[:, 0])
        assert np.array_equal(Y, arr[:, 2])

    def test_order_fortran(self):
        arr = np.arange(12).reshape(3, 4)

        X, Y = _extract_xy_from_ndarray2D(arr, 'fortran', (0, 2))

        assert np.array_equal(X, arr[0, :])
        assert np.array_equal(Y, arr[2, :])

    def test_c_and_fortran_are_transposes(self):
        arr = np.arange(12).reshape(4, 3)

        Xc, Yc = _extract_xy_from_ndarray2D(arr, 'c', (0, 1))
        Xf, Yf = _extract_xy_from_ndarray2D(arr.T, 'fortran', (0, 1))

        assert np.array_equal(Xc, Xf)
        assert np.array_equal(Yc, Yf)

    def test_returns_1d_views(self):
        arr = np.arange(12).reshape(4, 3)

        X, Y = _extract_xy_from_ndarray2D(arr, 'c', (0, 1))

        assert X.ndim == 1
        assert Y.ndim == 1
        assert np.shares_memory(X, arr)
        assert np.shares_memory(Y, arr)

    def test_same_index_for_x_and_y(self):
        arr = np.arange(8).reshape(4, 2)

        X, Y = _extract_xy_from_ndarray2D(arr, 'c', (1, 1))

        assert np.array_equal(X, Y)

    def test_negative_indices(self):
        arr = np.arange(12).reshape(4, 3)

        X, Y = _extract_xy_from_ndarray2D(arr, 'c', (-2, -1))

        assert np.array_equal(X, arr[:, -2])
        assert np.array_equal(Y, arr[:, -1])

    def test_quantity_keeps_units(self):
        arr = np.arange(8).reshape(4, 2) * u.m

        X, Y = _extract_xy_from_ndarray2D(arr, 'c', (0, 1))

        assert isinstance(X, u.Quantity)
        assert isinstance(Y, u.Quantity)
        assert X.unit == u.m
        assert Y.unit == u.m

    def test_non_square_shapes(self):
        arr = np.arange(10).reshape(5, 2)

        X, Y = _extract_xy_from_ndarray2D(arr, 'c', (0, 1))

        assert X.shape == (5,)
        assert Y.shape == (5,)

    @pytest.mark.parametrize(
        'obj',
        [
            [[1, 2], [3, 4]],
            ((1, 2), (3, 4)),
            5.0,
            None,
            'abc',
        ],
    )
    def test_non_ndarray_raises(self, obj):
        with pytest.raises(ValueError, match='2D array'):
            _extract_xy_from_ndarray2D(obj, 'c', (0, 1))

    @pytest.mark.parametrize('indices', [(0,), (0, 1, 2), ()])
    def test_wrong_length_indices_raises(self, indices):
        arr = np.arange(12).reshape(4, 3)

        with pytest.raises(ValueError, match='tuple\\[int, int\\]'):
            _extract_xy_from_ndarray2D(arr, 'c', indices)
