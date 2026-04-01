"""
Author: Elko Gerville-Reache
Date Created: 2026-03-14
Date Modified: 2026-03-29
Description:
    Tests for DataCube datastructure.
Dependencies:
    - astropy
    - numpy
    - spectral_cube
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
    """Test suite for DataCube __init__"""

    def assert_valid_cube(
        self,
        cube: DataCube,
        data: NDArray,
        header: Header,
        wcs: WCS,
        unit: u.UnitBase
    ):
        """
        Test DataCube attribute types.
        """
        assert isinstance(cube.value, np.ndarray)
        assert isinstance(cube.quantity, u.Quantity)
        assert isinstance(cube.header, Header)
        assert isinstance(cube.wcs, WCS)

        assert np.array_equal(cube.value, data)

        assert cube.unit == unit
        assert cube.shape == data.shape
        assert cube.wcs.wcs.compare(wcs.wcs)

    def assert_DataCube_attributes(self, cube):
        """
        Test important DataCube attribute properties.
        """
        if isinstance(cube.header, list):
            assert cube.header[0] is cube.primary_header
            assert cube.header[0] is not cube.nowcs_header
            assert len(cube.header) == cube.value.shape[0]
            if cube.wcs is not None:
                assert isinstance(cube.wcs, list)
                assert len(cube.wcs) == len(cube.header)

        else:
            assert cube.header is cube.primary_header
            assert cube.header is not cube.nowcs_header
            if cube.wcs is not None:
                assert isinstance(cube.wcs, WCS)

    def test_quantity_init(self, generate_test_cube):
        """
        Test the initialization of a DataCube from a Quantity.
        """
        hdu = generate_test_cube
        data = hdu.data
        header = hdu.header
        unit = u.Unit(header['BUNIT'])
        wcs = WCS(header)

        cube = DataCube(data=data, header=header)

        assert isinstance(cube.data, u.Quantity)
        self.assert_valid_cube(cube, data, header, wcs, unit)
        self.assert_DataCube_attributes(cube)

    def test_spectralcube_init(self, generate_test_cube):
        """
        Test the initialization of a DataCube from a SpectralCube.
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
        self.assert_DataCube_attributes(cube)

    def test_ndarray_init(self, generate_test_cube):
        """
        Test the initialization of a DataCube from a np.ndarray.
        """
        hdu = generate_test_cube
        data = hdu.data
        header = hdu.header
        header['BUNIT'] = ''
        unit = u.Unit(header['BUNIT'])
        wcs = WCS(header)

        cube = DataCube(data=data, header=header)

        assert isinstance(cube.data, np.ndarray)
        self.assert_valid_cube(cube, data, header, wcs, unit)
        self.assert_DataCube_attributes(cube)

    def test_ndarray_2_quantity(self, generate_test_cube):
        """
        Test that DataCube assigns the BUNIT to data if
        data has no unit
        """
        hdu = generate_test_cube
        data = np.asarray(hdu.data)
        header = hdu.header
        assert not isinstance(data, u.Quantity)
        assert header['BUNIT'] == 'MJy / sr'

        cube = DataCube(data=data, header=header)
        self.assert_DataCube_attributes(cube)

        assert cube.unit == u.Unit(header['BUNIT'])
