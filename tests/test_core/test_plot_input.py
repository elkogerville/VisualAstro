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
from visualastro.core.plot_input import (
    _extract_xy,
    _extract_xy_from_ndarray2D,
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

# class TestExtractXY:
#     """
#     Test _extract_xy.
#
#     Supports:
#     - Two separate inputs:
#         - (array-like, array-like) for X and Y
#         - (scalar, scalar)
#         - (scalar Quantity, scalar Quantity)
#         - (array-like, scalar array-like / Quantity array-like mix)
#
#     - Single input:
#         - 1D array / Quantity array → X = arange(N), Y = values
#         - 2D array / Quantity array:
#             - shape (N, 2) interpreted as columns (C order)
#             - shape (2, N) interpreted as rows (Fortran order)
#         - list / tuple of scalars → X = arange(N), Y = values
#         - list of Quantities → X = arange(N), Y = values
#
#     - Type support:
#         - NumPy arrays (including scalar arrays)
#         - Python scalars (int, float)
#         - NumPy scalar types (np.float64, np.int64, etc.)
#         - astropy Quantity scalars and arrays
#         - lists and tuples of scalar or Quantity elements
#         - lists of lists of scalar or Quantity elements
#
#     - Edge cases:
#         - Empty 1D arrays (len 0)
#         - Mixed NumPy scalar inputs
#         - Mixed Quantity / ndarray combinations in two-argument mode
#
#     - Invalid inputs:
#         - More than two positional arguments → ValueError
#         - Jagged nested lists → ValueError
#         - Mixed-type sequences that cannot be safely interpreted → ValueError
#     """
#     def test_two_arrays(self):
#         x = np.array([1,2,3])
#         y = np.array([4,5,6])
#
#         X, Y = _extract_xy(x, y)
#
#         assert np.array_equal(X, x)
#         assert np.array_equal(Y, y)
#
#     def test_single_1d_array(self):
#         y = np.array([10, 20, 30])
#
#         X, Y = _extract_xy(y)
#
#         assert X is None
#         assert np.array_equal(Y, y)
#
#     def test_single_2d_array(self):
#         arr = np.array([
#             [1,2],
#             [4,5]
#         ])
#
#         X, Y = _extract_xy(arr, order='c', index_spec=(0,1))
#
#         assert np.array_equal(X, [1, 4])
#         assert np.array_equal(Y, [2, 5])
#
#         y = np.array([[10, 20, 30]])
#
#         X, Y = _extract_xy(y, order='c', index_spec=(0,1))
#
#         assert np.array_equal(X, np.array([10]))
#         assert np.array_equal(Y, np.array([20]))
#
#         arr = np.array([
#             [1,2,3,4,5],
#             [6,7,8,9,10]
#         ]) * u.um
#
#         X, Y = _extract_xy(arr, order='c', index_spec=(0,1))
#
#         assert np.array_equal(X, [1, 6]*u.um)
#         assert np.array_equal(Y, [2, 7]*u.um)
#
#     def test_fortran_order(self):
#         arr = np.array([
#             [1,2],
#             [4,5]
#         ])
#
#         X, Y = _extract_xy(arr, order='fortran', index_spec=(0,1))
#
#         assert np.array_equal(X, [1, 2])
#         assert np.array_equal(Y, [4, 5])
#
#         arr = np.array([
#             [1,2,3,4,5],
#             [6,7,8,9,10]
#         ])*u.Jy
#
#         X, Y = _extract_xy(arr, order='fortran', index_spec=(0,1))
#
#         assert np.array_equal(X, arr[0])
#         assert np.array_equal(Y, arr[1])
#
#     def test_single_quantity_array(self):
#         y = np.array([1,2,3]) * u.m
#
#         X, Y = _extract_xy(y)
#
#         assert X is None and np.all(Y == y)
#
#     def test_two_quantity_arrays(self):
#         x = [1,2,3] * u.um
#         y = [4,5,6]
#
#         X, Y = _extract_xy(x, y)
#         assert np.array_equal(X, x) and Y == y
#
#         x = [1,2,3] * u.um
#         y = [4,5,6] * u.Jy
#
#         X, Y = _extract_xy(x, y)
#         assert np.array_equal(X, x) and np.array_equal(Y, y)
#
#     def test_single_list(self):
#         y = [1,2,3]
#         X, Y = _extract_xy(y)
#
#         assert X is None and np.array_equal(Y, [1,2,3])
#
#     def test_two_lists(self):
#         x = [1,2,3]
#         y = [4,5,6]
#         X, Y = _extract_xy(x, y)
#
#         assert X == x and Y == y
#
#     def test_list_of_lists(self):
#         x = [1,2,3]
#         y = [[4,5,6], [7,8,9]]
#         X, Y = _extract_xy(x, y)
#
#         assert X == x and Y == y
#
#         x = [1,2,3]
#         y = [[4*u.um,5*u.um,6*u.um], [7*u.um,8*u.um,9*u.um]]
#         X, Y = _extract_xy(x, y)
#
#         assert X == x and Y == y
#
#         x = [1,2,3]
#         y = [[4,5,6]*u.um, [7,8,9]*u.um]
#         X, Y = _extract_xy(x, y)
#
#         assert X == x and Y == y
#
#     def test_list_of_quantities(self):
#         y = [1*u.um, 2*u.um, 3*u.um]
#         X, Y = _extract_xy(y)
#
#         assert X is None and Y == y
#
#     def test_two_lists_of_quantities(self):
#         x = [1*u.um, 2*u.um, 3*u.um]
#         y = [4*u.Jy, 5*u.Jy, 6*u.Jy]
#         X, Y = _extract_xy(x, y)
#
#         assert X == x and Y == y
#
#         x = [1, 2, 3]
#         X, Y = _extract_xy(x, y)
#
#         assert X == x and Y == y
#
#     def test_list_and_array(self):
#         x = [1,2,3]
#         y = np.array([4,5,6])
#         X, Y = _extract_xy(x, y)
#
#         assert X == x and np.array_equal(Y, y)
#
#         x = [1,2,3]
#         y = [4,5,6] * u.um
#         X, Y = _extract_xy(x, y)
#
#         assert X == x and np.array_equal(Y, y)
#
#     def test_scalars(self):
#         X, Y = _extract_xy(1)
#
#         assert X is None and np.array_equal(Y, 1)
#
#         X, Y = _extract_xy(1, 2)
#
#         assert np.array_equal(X, 1)
#         assert np.array_equal(Y, 2)
#
#     def test_quantity_scalars(self):
#         X, Y = _extract_xy(1*u.um)
#
#         assert X is None and np.array_equal(Y, 1*u.um)
#
#         X, Y = _extract_xy(1*u.um, 2*u.Jy)
#
#         assert np.array_equal(X, 1*u.um)
#         assert np.array_equal(Y, 2*u.Jy)
#
#     def test_numpy_scalar_inputs(self):
#         X, Y = _extract_xy(np.float64(1), np.float64(2))
#
#         assert np.array_equal(X, np.float64(1))
#         assert np.array_equal(Y, np.float64(2))
#
#     def test_quantity_ndarray(self):
#         x = np.array([1, 2, 3])
#         y = [4, 5, 6] * u.um
#
#         X, Y = _extract_xy(x, y)
#
#         assert np.array_equal(X, x)
#         assert np.array_equal(Y, y)
#
#     def test_numpy_scalar_list(self):
#         y = [np.float64(1), np.float64(2), np.float64(3)]
#
#         X, Y = _extract_xy(y)
#
#         assert X is None and np.array_equal(Y, y)
#
#     def test_jagged_list_raises(self):
#         y = [[1, 2], [3]]
#         with pytest.raises(ValueError):
#             _extract_xy(y)
#
#     def test_invalid_argument_raises(self):
#         with pytest.raises(ValueError):
#             _extract_xy(1,2,3)
#
#         with pytest.raises(ValueError):
#             y = [[1,2,3],[4,5,6]]
#             _extract_xy(y)
#
#
# class TestExtractXYZFromNDArray:
#     """Test `_extract_xyx_from_ndarray`."""
#     def test_c_order_default_index_spec(self):
#         a = np.arange(30).reshape(10, 3)
#         x, y, z = _extract_xyz_from_ndarray(a, order='c', index_spec=(0, 1, 2))
#         np.testing.assert_array_equal(x, a[:, 0])
#         np.testing.assert_array_equal(y, a[:, 1])
#         np.testing.assert_array_equal(z, a[:, 2])
#
#     def test_fortran_order(self):
#         a = np.arange(30).reshape(3, 10)
#         x, y, z = _extract_xyz_from_ndarray(a, order='fortran', index_spec=(0, 1, 2))
#         np.testing.assert_array_equal(x, a[0, :])
#         np.testing.assert_array_equal(y, a[1, :])
#         np.testing.assert_array_equal(z, a[2, :])
#
#     def test_custom_index_spec(self):
#         a = np.arange(80).reshape(10, 8)
#         x, y, z = _extract_xyz_from_ndarray(a, order='c', index_spec=(0, 4, 7))
#         np.testing.assert_array_equal(x, a[:, 0])
#         np.testing.assert_array_equal(y, a[:, 4])
#         np.testing.assert_array_equal(z, a[:, 7])
#
#     def test_quantity_preserves_units(self):
#         a = np.arange(30).reshape(10, 3) * u.m
#         x, y, z = _extract_xyz_from_ndarray(a, order='c', index_spec=(0, 1, 2))
#         assert x.unit == u.m
#         assert y.unit == u.m
#         assert z.unit == u.m
#         np.testing.assert_array_equal(x.value, a.value[:, 0])
#
#     def test_order_case_insensitive(self):
#         a = np.arange(30).reshape(10, 3)
#         x, y, z = _extract_xyz_from_ndarray(a, order='C', index_spec=(0, 1, 2))
#         np.testing.assert_array_equal(x, a[:, 0])
#
#     def test_1d_input_raises(self):
#         a = np.arange(10)
#         with pytest.raises(ValueError):
#             _extract_xyz_from_ndarray(a, order='c', index_spec=(0, 1, 2))
#
#     def test_3d_input_raises(self):
#         a = np.arange(60).reshape(5, 4, 3)
#         with pytest.raises(ValueError):
#             _extract_xyz_from_ndarray(a, order='c', index_spec=(0, 1, 2))
#
#     def test_wrong_index_spec_length_raises(self):
#         a = np.arange(30).reshape(10, 3)
#         with pytest.raises(ValueError):
#             _extract_xyz_from_ndarray(a, order='c', index_spec=(0, 1))
#
#     def test_non_array_input_raises(self):
#         with pytest.raises(ValueError):
#             _extract_xyz_from_ndarray([[1, 2, 3], [4, 5, 6]], order='c', index_spec=(0, 1, 2))
#
#
# class TestExtractXYZ:
#     """Test `_extract_xyz`."""
#     def test_single_2d_array(self):
#         a = np.arange(30).reshape(10, 3)
#         result = _extract_xyz(a, order='c', index_spec=(0, 1, 2))
#         assert len(result) == 1
#         x, y, z = result[0]
#         np.testing.assert_array_equal(x, a[:, 0])
#         np.testing.assert_array_equal(y, a[:, 1])
#         np.testing.assert_array_equal(z, a[:, 2])
#
#     def test_list_of_2d_arrays(self):
#         a1 = np.arange(30).reshape(10, 3)
#         a2 = np.arange(30, 60).reshape(10, 3)
#         result = _extract_xyz([a1, a2], order='c', index_spec=(0, 1, 2))
#         assert len(result) == 2
#         np.testing.assert_array_equal(result[0][0], a1[:, 0])
#         np.testing.assert_array_equal(result[0][1], a1[:, 1])
#         np.testing.assert_array_equal(result[0][2], a1[:, 2])
#         np.testing.assert_array_equal(result[1][0], a2[:, 0])
#         np.testing.assert_array_equal(result[1][1], a2[:, 1])
#         np.testing.assert_array_equal(result[1][2], a2[:, 2])
#
#     def test_three_1d_arrays(self):
#         x = np.arange(5)
#         y = np.arange(5, 10)
#         z = np.arange(10, 15)
#         result = _extract_xyz(x, y, z, order='c', index_spec=(0, 1, 2))
#         assert result == [(x, y, z)]
#
#     def test_three_quantity_arrays(self):
#         x = np.arange(5) * u.m
#         y = np.arange(5, 10) * u.m
#         z = np.arange(10, 15) * u.m
#         result = _extract_xyz(x, y, z, order='c', index_spec=(0, 1, 2))
#         assert len(result) == 1
#         assert all(arr.unit == u.m for arr in result[0])
#
#     def test_three_wrapped_1d_arrays(self):
#         # shapes (N,1)
#         x = np.arange(5).reshape(-1, 1)
#         y = np.arange(5, 10).reshape(-1, 1)
#         z = np.arange(10, 15).reshape(-1, 1)
#         result = _extract_xyz(x, y, z, order='c', index_spec=(0, 1, 2))
#         assert len(result) == 1
#
#     def test_three_sequences_of_1d_arrays(self):
#         x1, x2 = np.arange(5), np.arange(5, 10)
#         y1, y2 = np.arange(10, 15), np.arange(15, 20)
#         z1, z2 = np.arange(20, 25), np.arange(25, 30)
#         result = _extract_xyz(
#             [x1, x2], [y1, y2], [z1, z2], order='c', index_spec=(0, 1, 2)
#         )
#         assert len(result) == 2
#         np.testing.assert_array_equal(result[0], (x1, y1, z1))
#         np.testing.assert_array_equal(result[1], (x2, y2, z2))
#
#     def test_three_scalars(self):
#         result = _extract_xyz(1.0, 2.0, 3.0, order='c', index_spec=(0, 1, 2))
#         assert len(result) == 1
#         assert result[0] == (1.0, 2.0, 3.0)
#
#     def test_two_positional_args_raises(self):
#         with pytest.raises(ValueError):
#             _extract_xyz(1.0, 2.0, order='c', index_spec=(0, 1, 2))
#
#
# class TestNumericalUtils:
#
#     def test_to_array_keep_unit(self):
#         """Test that to_array returns either array or Quantity, for non-cube inputs."""
#         a = np.random.rand(10)
#         b = np.random.rand(10, 10, 10) * u.erg
#         d = DataCube(data=b)
#         e = FitsFile(data=b)
#
#         A = to_array(a)
#         AA = to_array(a, keep_unit=False)
#         B = to_array(b, keep_unit=False)
#         D = to_array(d, keep_unit=False)
#         E = to_array(e, keep_unit=False)
#
#         for result in (A, AA, B, D, E):
#             assert isinstance(result, np.ndarray)
#             assert not isinstance(result, u.Quantity)
#
#         assert not isinstance(to_array(a, keep_unit=True), u.Quantity)
#         assert isinstance(to_array(b, keep_unit=True), u.Quantity)
#         assert isinstance(to_array(d, keep_unit=True), u.Quantity)
#         assert isinstance(to_array(e, keep_unit=True), u.Quantity)
#
#     def test_to_array_keep_unit_spectral_cube(self, generate_test_spectralcube):
#         """Test that to_array handles SpectralCube input."""
#         c = generate_test_spectralcube
#         C = to_array(c, keep_unit=False)
#         assert isinstance(C, np.ndarray)
#         assert not isinstance(C, u.Quantity)
#         assert isinstance(to_array(c, keep_unit=True), u.Quantity)
#
#     def test_to_list(self):
#         """Test that inputs are converted to a list"""
#         a = [1, 2, 3]
#         b = (1, 2, 3)
#         c = np.random.rand(10)
#
#         assert to_list(a) is a
#         assert to_list(b) == list(b)
#         assert to_list(c) == [c]
#
#     def test_unwrap_if_single(self):
#         """
#         Test that _unwrap_if_single unwraps an object
#         if the len == 1.
#         """
#         A = [1]
#         B = [1, 2]
#         C = (1)
#         D = (1, 2)
#         E = (1,)
#         F = np.asarray([2.0])
#         G = np.random.rand(1)
#         H = np.random.rand(3)
#         I = np.random.rand(10, 10)
#
#         assert _unwrap_if_single(A) is A[0]
#         assert _unwrap_if_single(B) is B
#         assert _unwrap_if_single(C) is C
#         assert _unwrap_if_single(D) is D
#         assert _unwrap_if_single(E) is E[0]
#         assert _unwrap_if_single(F) == F[0]
#         assert _unwrap_if_single(G) == G
#         assert np.allclose(_unwrap_if_single(H),  H)
#         assert np.allclose(_unwrap_if_single(I),  I)
