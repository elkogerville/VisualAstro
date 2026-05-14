"""
Author: Elko Gerville-Reache
Date Created: 2026-03-31
Date Modified: 2026-04-01
Description:
    Tests for plot utils module.
Dependencies:
    - astropy
    - matplotlib
    - numpy
    - pytest
    - spectral_cube
"""

import astropy.units as u
from astropy.visualization import ImageNormalize
from matplotlib.colors import AsinhNorm, LogNorm, PowerNorm
import numpy as np
import pytest
from spectral_cube import SpectralCube

from tests.conftest import generate_test_spectralcube
from visualastro.core.numerical_utils import get_value
from visualastro.datamodels.datacube import DataCube
from visualastro.datamodels.fitsfile import FitsFile
from visualastro.plotting.core.plot_utils import (
    get_imshow_norm,
    get_vmin_vmax,
    _extract_xy
)


class TestExtractXY:
    """
    Test _extract_xy.

    Supports:
    - Two separate inputs:
        - (array-like, array-like) for X and Y
        - (scalar, scalar)
        - (scalar Quantity, scalar Quantity)
        - (array-like, scalar array-like / Quantity array-like mix)

    - Single input:
        - 1D array / Quantity array → X = arange(N), Y = values
        - 2D array / Quantity array:
            - shape (N, 2) interpreted as columns (C order)
            - shape (2, N) interpreted as rows (Fortran order)
        - list / tuple of scalars → X = arange(N), Y = values
        - list of Quantities → X = arange(N), Y = values

    - Type support:
        - NumPy arrays (including scalar arrays)
        - Python scalars (int, float)
        - NumPy scalar types (np.float64, np.int64, etc.)
        - astropy Quantity scalars and arrays
        - lists and tuples of scalar or Quantity elements
        - lists of lists of scalar or Quantity elements

    - Edge cases:
        - Empty 1D arrays (len 0)
        - Mixed NumPy scalar inputs
        - Mixed Quantity / ndarray combinations in two-argument mode

    - Invalid inputs:
        - More than two positional arguments → ValueError
        - Jagged nested lists → ValueError
        - Mixed-type sequences that cannot be safely interpreted → ValueError
    """
    def test_two_arrays(self):
        x = np.array([1,2,3])
        y = np.array([4,5,6])

        X, Y = _extract_xy(x, y)

        assert np.array_equal(X, x)
        assert np.array_equal(Y, y)

    def test_single_1d_array(self):
        y = np.array([10, 20, 30])

        X, Y = _extract_xy(y)

        assert np.array_equal(X, np.arange(3))
        assert np.array_equal(Y, y)

    def test_single_2d_array(self):
        arr = np.array([
            [1,2],
            [4,5]
        ])

        X, Y = _extract_xy(arr, order='c')

        assert np.array_equal(X, [1, 4])
        assert np.array_equal(Y, [2, 5])

        arr = np.array([
            [1,2,3,4,5],
            [6,7,8,9,10]
        ]) * u.um

        X, Y = _extract_xy(arr, order='c')

        assert np.array_equal(X, [1, 6]*u.um)
        assert np.array_equal(Y, [2, 7]*u.um)

    def test_fortran_order(self):
        arr = np.array([
            [1,2],
            [4,5]
        ])

        X, Y = _extract_xy(arr, order='fortran')

        assert np.array_equal(X, [1, 2])
        assert np.array_equal(Y, [4, 5])

        arr = np.array([
            [1,2,3,4,5],
            [6,7,8,9,10]
        ])*u.Jy

        X, Y = _extract_xy(arr, order='fortran')

        assert np.array_equal(X, arr[0])
        assert np.array_equal(Y, arr[1])

    def test_single_quantity_array(self):
        y = np.array([1,2,3]) * u.m

        X, Y = _extract_xy(y)

        assert np.array_equal(X, np.arange(3))
        assert np.all(Y == y)

    def test_two_quantity_arrays(self):
        x = [1,2,3] * u.um
        y = [4,5,6]

        X, Y = _extract_xy(x, y)
        assert np.array_equal(X, x) and Y == y

        x = [1,2,3] * u.um
        y = [4,5,6] * u.Jy

        X, Y = _extract_xy(x, y)
        assert np.array_equal(X, x) and np.array_equal(Y, y)

    def test_single_list(self):
        y = [1,2,3]
        X, Y = _extract_xy(y)

        assert np.array_equal(X, np.arange(3))
        assert np.array_equal(Y, [1,2,3])

    def test_two_lists(self):
        x = [1,2,3]
        y = [4,5,6]
        X, Y = _extract_xy(x, y)

        assert X == x and Y == y

    def test_list_of_lists(self):
        x = [1,2,3]
        y = [[4,5,6], [7,8,9]]
        X, Y = _extract_xy(x, y)

        assert X == x and Y == y

        x = [1,2,3]
        y = [[4*u.um,5*u.um,6*u.um], [7*u.um,8*u.um,9*u.um]]
        X, Y = _extract_xy(x, y)

        assert X == x and Y == y

        x = [1,2,3]
        y = [[4,5,6]*u.um, [7,8,9]*u.um]
        X, Y = _extract_xy(x, y)

        assert X == x and Y == y

    def test_list_of_quantities(self):
        y = [1*u.um, 2*u.um, 3*u.um]
        X, Y = _extract_xy(y)

        assert np.array_equal(X, np.arange(3))
        assert Y == y

    def test_two_lists_of_quantities(self):
        x = [1*u.um, 2*u.um, 3*u.um]
        y = [4*u.Jy, 5*u.Jy, 6*u.Jy]
        X, Y = _extract_xy(x, y)

        assert X == x and Y == y

        x = [1, 2, 3]
        X, Y = _extract_xy(x, y)

        assert X == x and Y == y

    def test_list_and_array(self):
        x = [1,2,3]
        y = np.array([4,5,6])
        X, Y = _extract_xy(x, y)

        assert X == x and np.array_equal(Y, y)

        x = [1,2,3]
        y = [4,5,6] * u.um
        X, Y = _extract_xy(x, y)

        assert X == x and np.array_equal(Y, y)

    def test_scalars(self):
        X, Y = _extract_xy(1)

        assert np.array_equal(X, 0)
        assert np.array_equal(Y, 1)

        X, Y = _extract_xy(1, 2)

        assert np.array_equal(X, 1)
        assert np.array_equal(Y, 2)

    def test_quantity_scalars(self):
        X, Y = _extract_xy(1*u.um)

        assert np.array_equal(X, 0)
        assert np.array_equal(Y, 1*u.um)

        X, Y = _extract_xy(1*u.um, 2*u.Jy)

        assert np.array_equal(X, 1*u.um)
        assert np.array_equal(Y, 2*u.Jy)

    def test_numpy_scalar_inputs(self):
        X, Y = _extract_xy(np.float64(1), np.float64(2))

        assert np.array_equal(X, np.float64(1))
        assert np.array_equal(Y, np.float64(2))

    def test_quantity_ndarray(self):
        x = np.array([1, 2, 3])
        y = [4, 5, 6] * u.um

        X, Y = _extract_xy(x, y)

        assert np.array_equal(X, x)
        assert np.array_equal(Y, y)

    def test_numpy_scalar_list(self):
        y = [np.float64(1), np.float64(2), np.float64(3)]

        X, Y = _extract_xy(y)

        assert np.array_equal(X, np.arange(3))
        assert np.array_equal(Y, y)

    def test_jagged_list_raises(self):
        y = [[1, 2], [3]]
        with pytest.raises(ValueError):
            _extract_xy(y)

    def test_invalid_argument_raises(self):
        with pytest.raises(ValueError):
            _extract_xy(1,2,3)

        with pytest.raises(ValueError):
            y = [[1,2,3],[4,5,6]]
            _extract_xy(y)

class TestGetImshowNorm:

    def test_none_norm(self):
        assert get_imshow_norm(None, 0.0, 2.0) is None
        assert get_imshow_norm(None, None, None) is None

    def test_supported_norms(self):
        assert isinstance(get_imshow_norm('asinh', 0.1, 9), ImageNormalize)
        assert isinstance(get_imshow_norm('log', 0.1, 9), LogNorm)
        assert isinstance(get_imshow_norm( 'power', 0.1, 9), PowerNorm)
        assert isinstance(get_imshow_norm('asinhnorm', 0.1, 9), AsinhNorm)

    def test_boolean_images(self):
        assert get_imshow_norm(None, 0, 1) is None
        assert get_imshow_norm(None, 0.0, 1.0) is None
        assert get_imshow_norm('asinh', 0, 1) is None

    def test_case_insensitive(self):
        assert isinstance(get_imshow_norm('ASINH', 0.1, 1.2), ImageNormalize)

    def test_invalid_norm(self):
        with pytest.raises(ValueError):
            get_imshow_norm('invalid', 0.2, 99)

    def test_valid_norm_no_vmin_vmax(self):
        with pytest.raises(ValueError):
            get_imshow_norm('asinh', None, None)

class TestVminVmax:

    def _validate_vmin_vmax(self, cube):
        """Test cases for get_vmin_vmax"""
        percentile = (3, 99)
        pmin = np.nanpercentile(get_value(cube), percentile[0])
        pmax = np.nanpercentile(get_value(cube), percentile[1])

        vmin, vmax = get_vmin_vmax(cube, percentile, None, None)
        assert vmin == pytest.approx(pmin, 0.2)
        assert vmax == pytest.approx(pmax, 0.2)

        vmin, vmax = get_vmin_vmax(
            cube, percentile=(3, 99), vmin=10, vmax=20
        )
        assert vmin == 10 and vmax == 20

        vmin, vmax = get_vmin_vmax(
            cube, percentile=(3, 99), vmin=10, vmax=None
        )
        assert vmin == 10 and vmax == pytest.approx(pmax, 0.2)

        with pytest.raises(ValueError):
            get_vmin_vmax(
                cube, percentile=None, vmin=None, vmax=None
            )

    def test_get_vmin_vmax(self, generate_test_spectralcube):
        """Tests for get_vmin_vmax"""
        cube = generate_test_spectralcube
        datacube = DataCube(data=cube)
        boolarray = np.zeros((10, 10), dtype=bool)
        array = np.random.rand(10, 10)

        assert isinstance(cube, SpectralCube)
        assert isinstance(datacube, DataCube)

        self._validate_vmin_vmax(cube)
        self._validate_vmin_vmax(datacube)
        self._validate_vmin_vmax(array)
        self._validate_vmin_vmax(array * u.um)
        self._validate_vmin_vmax(FitsFile(array * u.um))

        vmin, vmax = get_vmin_vmax(
            boolarray, percentile=(3,99), vmin=None, vmax=None
        )
        assert vmin == 0 and vmax == 1

        vmin, vmax = get_vmin_vmax(
            boolarray, percentile=(3,99), vmin=10, vmax=20
        )
        assert vmin == 0 and vmax == 1

        vmin, vmax = get_vmin_vmax(
            boolarray, percentile=(3,99), vmin=None, vmax=20
        )
        assert vmin == 0 and vmax == 1
