"""
Author: Elko Gerville-Reache
Date Created: 2026-07-10
Date Modified: 2026-07-17
Description:
    Tests for image utils plotting module.
"""

import astropy.units as u
from astropy.visualization import ImageNormalize
from matplotlib.colors import AsinhNorm, LogNorm, PowerNorm
import numpy as np
import pytest

from visualastro.core.numerical_utils import get_value
from visualastro.core.optional_deps import (
    SpectralCube,
    _HAS_SPECTRAL_CUBE,
    _require_dependency
)
from visualastro.datamodels.datacube import DataCube
from visualastro.datamodels.fitsfile import FitsFile
from visualastro.plotting.core.image_utils import get_imshow_norm, get_vmin_vmax

from tests.conftest import generate_test_spectralcube


class TestGetImshowNorm:
    """Tests for `get_imshow_norm`."""
    def test_none_norm(self):
        assert get_imshow_norm(None, 0.0, 2.0) is None
        assert get_imshow_norm(None, None, None) is None

    def test_supported_norms(self):
        assert isinstance(get_imshow_norm('asinh', 0.1, 9), ImageNormalize)
        assert isinstance(get_imshow_norm('log', 0.1, 9), LogNorm)
        assert isinstance(get_imshow_norm('power', 0.1, 9), PowerNorm)
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
    """Tests for `get_vmin_vmax`."""
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

    def test_get_vmin_vmax_spectralcube(self, generate_test_spectralcube):
        """Tests for get_vmin_vmax with SpectralCube input."""
        cube = generate_test_spectralcube
        datacube = DataCube(data=cube)
        assert isinstance(cube, SpectralCube)
        assert isinstance(datacube, DataCube)
        self._validate_vmin_vmax(cube)
        self._validate_vmin_vmax(datacube)

    def test_get_vmin_vmax_boolean(self, generate_test_spectralcube):
        """Tests for get_vmin_vmax with boolean array inputs."""
        boolarray = np.zeros((10, 10), dtype=bool)
        array = np.random.rand(10, 10)

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
