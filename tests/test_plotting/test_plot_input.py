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

from visualastro.plotting.core.plot_input import (
    _extract_xy,
    _extract_xy_from_ndarray2D,
    _extract_xyz,
    _resolve_implicit_x,
    _resolve_xy_plot_inputs
)


class TestExtractXY:
    kwargs = {}
    # ##########
    # two inputs
    # ##########
    def test_two_arrays(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])

        X, Y = _extract_xy(x, y, kwargs=self.kwargs)

        assert np.array_equal(X, x)
        assert np.array_equal(Y, y)

    def test_two_arrays_input_mutation(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])

        X, Y = _extract_xy(x, y, kwargs=self.kwargs)

        assert X is x
        assert Y is y

    def test_two_scalars(self):
        X, Y = _extract_xy(1.0, 2.0, kwargs=self.kwargs)

        assert X == 1.0
        assert Y == 2.0

    def test_two_quantities_keep_units(self):
        x = np.array([1, 2, 3]) * u.s
        y = np.array([4, 5, 6]) * u.m

        X, Y = _extract_xy(x, y, kwargs=self.kwargs)

        assert X.unit == u.s
        assert Y.unit == u.m

    def test_two_inputs_x_none_allowed(self):
        y = np.array([4, 5, 6])

        X, Y = _extract_xy(None, y, kwargs=self.kwargs)

        assert X is None
        assert Y is y

    # ##################
    # single 0D/1D input
    # ##################
    def test_single_scalar(self):
        X, Y = _extract_xy(1, kwargs=self.kwargs)

        assert X is None
        assert Y == 1

    def test_single_scalar_quantity(self):
        q = 5.0 * u.m

        X, Y = _extract_xy(q, kwargs=self.kwargs)

        assert X is None
        assert Y.unit == u.m

    def test_single_0d_array(self):
        a = np.array(5.0)

        X, Y = _extract_xy(a, kwargs=self.kwargs)

        assert X is None
        assert Y is a

    def test_single_1d_array(self):
        y = np.array([1, 2, 3])

        X, Y = _extract_xy(y, kwargs=self.kwargs)

        assert X is None
        assert Y is y

    def test_single_1d_quantity(self):
        y = np.array([1, 2, 3]) * u.m

        X, Y = _extract_xy(y, kwargs=self.kwargs)

        assert X is None
        assert Y.unit == u.m
        assert np.array_equal(Y.value, [1, 2, 3])

    def test_single_1d_list(self):
        y = [1, 2, 3]

        X, Y = _extract_xy(y, kwargs=self.kwargs)

        assert X is None
        assert Y is y

    # ###############
    # single 2D input
    # ###############
    def test_single_2d_default_config(self):
        """order='c' indices=(0,1)"""
        arr = np.arange(12).reshape(4, 3)

        X, Y = _extract_xy(arr, kwargs=self.kwargs)

        assert X is None
        assert np.array_equal(Y, arr)

    def test_single_2d_order_c(self):
        arr = np.arange(12).reshape(4, 3)

        X, Y = _extract_xy(arr, order='c', xy_indices=(0, 2), kwargs=self.kwargs)

        assert np.array_equal(X, arr[:, 0])
        assert np.array_equal(Y, arr[:, 2])

    def test_single_2d_order_fortran(self):
        arr = np.arange(12).reshape(3, 4)

        X, Y = _extract_xy(arr, order='fortran', xy_indices=(0, 2), kwargs=self.kwargs)

        assert np.array_equal(X, arr[0, :])
        assert np.array_equal(Y, arr[2, :])

    @pytest.mark.parametrize('order', ['f', 'F', 'fortran', 'FORTRAN', 'Fortran'])
    def test_single_2d_fortran_aliases(self, order):
        arr = np.arange(8).reshape(2, 4)

        X, Y = _extract_xy(arr, order=order, xy_indices=(0,1), kwargs=self.kwargs)

        assert np.array_equal(X, arr[0, :])
        assert np.array_equal(Y, arr[1, :])

    @pytest.mark.parametrize('order', ['c', 'C'])
    def test_single_2d_c_aliases(self, order):
        arr = np.arange(8).reshape(4, 2)

        X, Y = _extract_xy(arr, order=order, xy_indices=(0,1), kwargs=self.kwargs)

        assert np.array_equal(X, arr[:, 0])
        assert np.array_equal(Y, arr[:, 1])

    def test_single_2d_swapped_indices(self):
        arr = np.arange(8).reshape(4, 2)

        X, Y = _extract_xy(arr, xy_indices=(1, 0), kwargs=self.kwargs)

        assert np.array_equal(X, arr[:, 1])
        assert np.array_equal(Y, arr[:, 0])

    def test_single_2d_negative_indices(self):
        arr = np.arange(12).reshape(4, 3)

        X, Y = _extract_xy(arr, xy_indices=(-2, -1), kwargs=self.kwargs)

        assert np.array_equal(X, arr[:, -2])
        assert np.array_equal(Y, arr[:, -1])

    def test_single_2d_indices_as_list(self):
        arr = np.arange(12).reshape(4, 3)

        X, Y = _extract_xy(arr, xy_indices=[0, 2], kwargs=self.kwargs)

        assert np.array_equal(X, arr[:, 0])
        assert np.array_equal(Y, arr[:, 2])

    def test_single_2d_returns_views(self):
        arr = np.arange(12).reshape(4, 3)

        X, Y = _extract_xy(arr, xy_indices=(0,1), kwargs=self.kwargs)

        assert np.shares_memory(X, arr)
        assert np.shares_memory(Y, arr)

    def test_single_2d_quantity_keeps_units(self):
        arr = np.arange(8).reshape(4, 2) * u.m

        X, Y = _extract_xy(arr, xy_indices=(0,1), kwargs=self.kwargs)

        assert X.unit == u.m
        assert Y.unit == u.m
        assert np.array_equal(X.value, arr.value[:, 0])
        assert np.array_equal(Y.value, arr.value[:, 1])

    # ######
    # raises
    # ######
    def test_two_inputs_y_none_raises(self):
        with pytest.raises(ValueError, match='`y` must not be None'):
            _extract_xy(np.array([1, 2, 3]), None, kwargs=self.kwargs)

    def test_no_input_raises(self):
        with pytest.raises(ValueError, match='got 0'):
            _extract_xy(kwargs=self.kwargs)

    def test_three_inputs_raises(self):
        a = np.array([1, 2])
        with pytest.raises(ValueError, match='got 3'):
            _extract_xy(a, a, a, kwargs=self.kwargs)

    def test_single_none_raises(self):
        with pytest.raises(ValueError, match='must not be None'):
            _extract_xy(None, kwargs=self.kwargs)

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


class TestResolveImplicitX:
    def test_explicit_x_passes_through(self):
        x, y = np.arange(3), np.arange(3) * 2
        for mode in ('plot', 'scatter'):
            out = _resolve_implicit_x(x, y, mode=mode)
            assert out[0] is x
            assert out[1] is y

    def test_plot_implicit_x_returns_y_only(self):
        y = np.arange(4)
        out = _resolve_implicit_x(None, y, mode='plot')
        assert len(out) == 1
        assert out[0] is y

    def test_scatter_implicit_x_is_arange(self):
        y = np.array([5.0, 6.0, 7.0])
        x, y_out = _resolve_implicit_x(None, y, mode='scatter')
        np.testing.assert_array_equal(x, [0, 1, 2])
        assert y_out is y

    def test_scatter_scalar_y_gives_single_point(self):
        x, y = _resolve_implicit_x(None, 3.0, mode='scatter')
        np.testing.assert_array_equal(x, [0])
        assert y == 3.0

    def test_plot_allows_2d_y_with_1d_x(self):
        x, y = np.arange(5), np.zeros((5, 2))
        out = _resolve_implicit_x(x, y, mode='plot')
        assert out[1].shape == (5, 2)
        assert x is out[0]
        assert y is out[1]

    @pytest.mark.parametrize('x', [None, np.arange(5)])
    def test_scatter_rejects_2d_y(self, x):
        with pytest.raises(ValueError):
            _resolve_implicit_x(x, np.zeros((5, 2)), mode='scatter')

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            _resolve_implicit_x(None, np.arange(3), mode='line')


class TestResolveXYPlotInputs:
    def test_single_1d_plot(self):
        y = np.arange(4)
        out = _resolve_xy_plot_inputs(y, kwargs={}, mode='plot')
        assert len(out) == 1
        assert out[0] is y

    def test_single_1d_scatter(self):
        y = np.arange(4)
        x, y_out = _resolve_xy_plot_inputs(y, kwargs={}, mode='scatter')
        np.testing.assert_array_equal(x, np.arange(4))
        assert y_out is y

    @pytest.mark.parametrize('mode', ['plot', 'scatter'])
    def test_two_inputs_pass_through(self, mode):
        x, y = np.arange(3), np.arange(3) ** 2
        out = _resolve_xy_plot_inputs(x, y, kwargs={}, mode=mode)
        assert out[0] is x
        assert out[1] is y

    @pytest.mark.parametrize('mode', ['plot', 'scatter'])
    def test_2d_split_works_for_both_modes(self, mode):
        arr = np.column_stack([np.arange(5), np.arange(5) * 10])
        x, y = _resolve_xy_plot_inputs(
            arr, kwargs={}, mode=mode, order='c', xy_indices=(0, 1)
        )
        np.testing.assert_array_equal(x, arr[:, 0])
        np.testing.assert_array_equal(y, arr[:, 1])

    def test_2d_no_split_plot_returns_2d_y(self):
        arr = np.zeros((5, 3))
        out = _resolve_xy_plot_inputs(
            arr, kwargs={}, mode='plot', order='c', xy_indices=None
        )
        assert len(out) == 1
        assert out[0].shape == (5, 3)

    def test_2d_no_split_scatter_raises(self):
        with pytest.raises(ValueError):
            _resolve_xy_plot_inputs(
                np.zeros((5, 3)), kwargs={}, mode='scatter',
                order='c', xy_indices=None,
            )

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            _resolve_xy_plot_inputs(np.arange(3), kwargs={}, mode='line')


class TestExtractXYZ:
    """Test `_extract_xyz`."""
    def test_single_2d_array(self):
        a = np.arange(30).reshape(10, 3)
        result = _extract_xyz(a, order='c', index_spec=(0, 1, 2))
        assert len(result) == 1
        x, y, z = result[0]
        np.testing.assert_array_equal(x, a[:, 0])
        np.testing.assert_array_equal(y, a[:, 1])
        np.testing.assert_array_equal(z, a[:, 2])

    def test_list_of_2d_arrays(self):
        a1 = np.arange(30).reshape(10, 3)
        a2 = np.arange(30, 60).reshape(10, 3)
        result = _extract_xyz([a1, a2], order='c', index_spec=(0, 1, 2))
        assert len(result) == 2
        np.testing.assert_array_equal(result[0][0], a1[:, 0])
        np.testing.assert_array_equal(result[0][1], a1[:, 1])
        np.testing.assert_array_equal(result[0][2], a1[:, 2])
        np.testing.assert_array_equal(result[1][0], a2[:, 0])
        np.testing.assert_array_equal(result[1][1], a2[:, 1])
        np.testing.assert_array_equal(result[1][2], a2[:, 2])

    def test_three_1d_arrays(self):
        x = np.arange(5)
        y = np.arange(5, 10)
        z = np.arange(10, 15)
        result = _extract_xyz(x, y, z, order='c', index_spec=(0, 1, 2))
        assert result == [(x, y, z)]

    def test_three_quantity_arrays(self):
        x = np.arange(5) * u.m
        y = np.arange(5, 10) * u.m
        z = np.arange(10, 15) * u.m
        result = _extract_xyz(x, y, z, order='c', index_spec=(0, 1, 2))
        assert len(result) == 1
        assert all(arr.unit == u.m for arr in result[0])

    def test_three_wrapped_1d_arrays(self):
        # shapes (N,1)
        x = np.arange(5).reshape(-1, 1)
        y = np.arange(5, 10).reshape(-1, 1)
        z = np.arange(10, 15).reshape(-1, 1)
        result = _extract_xyz(x, y, z, order='c', index_spec=(0, 1, 2))
        assert len(result) == 1

    def test_three_sequences_of_1d_arrays(self):
        x1, x2 = np.arange(5), np.arange(5, 10)
        y1, y2 = np.arange(10, 15), np.arange(15, 20)
        z1, z2 = np.arange(20, 25), np.arange(25, 30)
        result = _extract_xyz(
            [x1, x2], [y1, y2], [z1, z2], order='c', index_spec=(0, 1, 2)
        )
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], (x1, y1, z1))
        np.testing.assert_array_equal(result[1], (x2, y2, z2))

    def test_three_scalars(self):
        result = _extract_xyz(1.0, 2.0, 3.0, order='c', index_spec=(0, 1, 2))
        assert len(result) == 1
        assert result[0] == (1.0, 2.0, 3.0)

    def test_two_positional_args_raises(self):
        with pytest.raises(ValueError):
            _extract_xyz(1.0, 2.0, order='c', index_spec=(0, 1, 2))
