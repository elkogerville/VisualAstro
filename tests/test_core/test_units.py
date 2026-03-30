"""
Author: Elko Gerville-Reache
Date Created: 2026-03-30
Date Modified: 2026-03-30
Description:
    Tests for numerical utils module.
Dependencies:
    - astropy
    - numpy
    - spectral_cube
Module Structure:
    - DataCube
        Data class for 3D datacubes, spectral_cubes, or timeseries data.
"""

import astropy.units as u
import numpy as np
from spectral_cube import SpectralCube
from tests.conftest import generate_test_cube, generate_test_spectralcube
from visualastro.core.units import get_unit, get_units, to_unit
from visualastro.dataclasses.datacube import DataCube


class TestGetUnits:
    def test_get_unit(self, generate_test_cube):
        hdu = generate_test_cube
        data = hdu.data
        header = hdu.header
        cube = SpectralCube.read(hdu)
        unit = cube.unit
        structured_unit = u.Unit('AU,AU/day')
        quantity = data * u.m
        dc = DataCube(data=cube)

        assert get_unit(data) is None
        assert get_unit(header) == unit
        assert get_unit(cube) == unit
        assert get_unit(unit) is unit
        assert get_unit(structured_unit) is structured_unit
        assert get_unit(quantity) == u.m
        assert get_unit(dc) == unit

    def test_get_units(self, generate_test_cube):
        hdu = generate_test_cube
        cube = SpectralCube.read(hdu)
        cube2 = cube * u.sr
        cube3 = np.random.rand(10,10) * u.AA
        units = get_units([cube, cube2, cube3])
        assert all(units) == all([u.MJy/u.sr, u.MJy, u.AA])

        assert get_units(cube) == u.MJy/u.sr

    def test_to_unit(self):
        unit = u.AA
        structured_unit = u.Unit('AU,AU/day')
        assert to_unit('AA') == u.AA
        assert to_unit(unit) is unit
        assert to_unit(structured_unit) is structured_unit
        assert to_unit(np.random.rand(10)*u.m) == u.m
