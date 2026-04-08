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
Module Structure:
    - DataCube
        Data class for 3D datacubes, spectral_cubes, or timeseries data.
"""

import astropy.units as u
from astropy.visualization import ImageNormalize
from matplotlib.colors import AsinhNorm, LogNorm, PowerNorm
import numpy as np
import pytest
from spectral_cube.stokes_spectral_cube import SpectralCube
from tests.conftest import generate_test_spectralcube
from visualastro.core.numerical_utils import get_value
from visualastro.dataclasses.datacube import DataCube
from visualastro.plotting.plot_utils import (
    get_imshow_norm, get_vmin_vmax
)


class TestPlotUtils:

    def test_get_imshow_norm(self):

        # norm is None
        assert get_imshow_norm(None, 0.0, 2.0) is None
        assert get_imshow_norm(None, None, None) is None

        assert isinstance(get_imshow_norm('asinh', 0.1, 9), ImageNormalize)
        assert isinstance(get_imshow_norm('log', 0.1, 9), LogNorm)
        assert isinstance(get_imshow_norm( 'power', 0.1, 9), PowerNorm)
        assert isinstance(get_imshow_norm('asinhnorm', 0.1, 9), AsinhNorm)

        # for plotting boolean maps:
        assert get_imshow_norm(None, 0, 1) is None
        assert get_imshow_norm(None, 0.0, 1.0) is None
        assert get_imshow_norm('asinh', 0, 1) is None

        # check case insensitive
        assert isinstance(get_imshow_norm('ASINH', 0.1, 1.2), ImageNormalize)

        # test invalid norm raises ValueError
        with pytest.raises(ValueError):
            get_imshow_norm('invalid', 0.2, 99)

        # test valid norm with no vmin/vmax raises ValueError
        with pytest.raises(ValueError):
            get_imshow_norm('asinh', None, None)

    def _validate_vmin_vmax(self, cube):

        vmin, vmax = get_vmin_vmax(cube)
        assert isinstance(vmin, (float, np.floating))
        assert isinstance(vmax, (float, np.floating))

        vmin, vmax = get_vmin_vmax(
            cube, percentile=None, vmin=None, vmax=None
        )
        assert vmin is None and vmax is None

        vmin, vmax = get_vmin_vmax(
            cube, percentile=None, vmin=10, vmax=None
        )
        assert vmin == 10 and vmax is None

        vmin, vmax = get_vmin_vmax(
            cube, percentile=[3, 99], vmin=10, vmax=None
        )
        percentile = np.nanpercentile(get_value(cube), 99)

        assert vmin == 10 and percentile == pytest.approx(vmax, 0.2)


    def test_get_vmin_vmax(self, generate_test_spectralcube):

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

        vmin, vmax = get_vmin_vmax(boolarray)
        assert vmin == 0 and vmax == 1

        vmin, vmax = get_vmin_vmax(boolarray, percentile=None)
        assert vmin == 0 and vmax == 1

        vmin, vmax = get_vmin_vmax(boolarray, vmin=2, vmax=10)
        assert vmin == 0 and vmax == 1
