"""
Author: Elko Gerville-Reache
Date Created: 2026-03-31
Date Modified: 2026-03-31
Description:
    Tests for plot utils module.
Dependencies:
    - astropy
    - numpy
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
from spectral_cube import SpectralCube
from tests.conftest import generate_test_cube
from visualastro.core.numerical_utils import (
    interpolate,
    to_array,
    to_list,
    _unwrap_if_single
)
from visualastro.dataclasses.datacube import DataCube
from visualastro.dataclasses.fitsfile import FitsFile
from visualastro.plotting.plot_utils import get_imshow_norm


class TestPlotUtils:

    def test_get_imshow_norm(self):

        assert isinstance(get_imshow_norm(0.1, 9, 'asinh'), ImageNormalize)
        assert isinstance(get_imshow_norm(0.1, 9, 'log'), LogNorm)
        assert isinstance(get_imshow_norm(0.1, 9, 'powernorm'), PowerNorm)
        assert isinstance(get_imshow_norm(0.1, 9, 'asinhnorm'), AsinhNorm)
        assert get_imshow_norm(0.1, 1.2, 'none') is None
        assert get_imshow_norm(0., 1.2, None) is None
        # for plotting boolean maps:
        assert get_imshow_norm(0, 1, None) is None
        assert isinstance(get_imshow_norm(0.1, 1.2, 'ASINH'), ImageNormalize)

        # Test invalid norm raises ValueError
        with pytest.raises(ValueError, match='unsupported norm'):
            get_imshow_norm(0.2, 99, 'invalid')
