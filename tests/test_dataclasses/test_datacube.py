"""
Author: Elko Gerville-Reache
Date Created: 2026-03-14
Date Modified: 2026-03-14
Description:
    Tests for DataCube datastructure.
Dependencies:
    - astropy
    - matplotlib
    - numpy
    - spectral_cube
    - specutils
    - tqdm
Module Structure:
    - DataCube
        Data class for 3D datacubes, spectral_cubes, or timeseries data.
"""

from astropy.io.fits import Header
import astropy.units as u
from astropy.wcs import WCS
import numpy as np
from numpy.typing import NDArray
from spectral_cube import SpectralCube
from tests.conftest import generate_test_cube
from visualastro.dataclasses.datacube import DataCube


class TestDataCubeInit:
    """

    """

    def assert_valid_cube(
        self,
        cube: DataCube,
        data: NDArray,
        header: Header,
        wcs: WCS,
        unit: u.UnitBase
    ):
        assert isinstance(cube.value, np.ndarray)
        assert isinstance(cube.quantity, u.Quantity)
        assert isinstance(cube.header, Header)
        assert isinstance(cube.wcs, WCS)

        assert np.array_equal(cube.value, data)
        assert cube.data is not data

        assert cube.unit == unit
        assert cube.shape == data.shape
        assert cube.wcs.wcs.compare(wcs.wcs)

    def test_quantity_init(self, generate_test_cube):
        """
        """
        hdu = generate_test_cube
        data = hdu.data
        header = hdu.header
        unit = u.Unit(header['BUNIT'])
        wcs = WCS(header)

        cube = DataCube(data=data, header=header)

        assert isinstance(cube.data, u.Quantity)
        self.assert_valid_cube(cube, data, header, wcs, unit)

    def test_spectralcube_init(self, generate_test_cube):
        """
        """
        hdu = generate_test_cube
        data = hdu.data
        header = hdu.header
        wcs = WCS(header)
        unit = u.Unit(header['BUNIT'])
        spectral_cube = SpectralCube.read(hdu)

        cube = DataCube(data=spectral_cube, header=header)

        assert isinstance(cube.data, SpectralCube)
        self.assert_valid_cube(cube, data, header, wcs, unit)

    def test_ndarray_init(self, generate_test_cube):
        """
        """
        hdu = generate_test_cube
        data = hdu.data
        header = hdu.header
        header['BUNIT'] = ''
        unit = u.Unit(header['BUNIT'])
        wcs = WCS(header)

        cube = DataCube(data=data, header=header)

        assert isinstance(cube.data, np.ndarray)
        self.assert_valid_cube(
            cube, data, header, wcs, unit
        )
