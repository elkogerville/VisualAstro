"""
Author: Elko Gerville-Reache
Date Created: 2026-03-29
Date Modified: 2026-10-25
Description:
    Tests for data module.
"""


import astropy.units as u
import numpy as np

from tests.conftest import generate_test_cube
from visualastro.core.data import to_array, to_list
from visualastro.datamodels.datacube import DataCube
from visualastro.datamodels.fitsfile import FitsFile


class TestDataUtils:

    def test_to_array_keep_unit(self, generate_test_cube):
        """Test that to_array returns either array or Quantity, for non-cube inputs."""
        a = np.random.rand(10)
        b = np.random.rand(10, 10, 10) * u.erg
        d = DataCube(data=b)
        e = FitsFile(data=b)

        A = to_array(a)
        AA = to_array(a, keep_unit=False)
        B = to_array(b, keep_unit=False)
        D = to_array(d, keep_unit=False)
        E = to_array(e, keep_unit=False)

        for result in (A, AA, B, D, E):
            assert isinstance(result, np.ndarray)
            assert not isinstance(result, u.Quantity)

        assert not isinstance(to_array(a, keep_unit=True), u.Quantity)
        assert isinstance(to_array(b, keep_unit=True), u.Quantity)
        assert isinstance(to_array(d, keep_unit=True), u.Quantity)
        assert isinstance(to_array(e, keep_unit=True), u.Quantity)

    def test_to_array_keep_unit_spectral_cube(self, generate_test_spectralcube):
        """Test that to_array handles SpectralCube input."""
        c = generate_test_spectralcube
        C = to_array(c, keep_unit=False)
        assert isinstance(C, np.ndarray)
        assert not isinstance(C, u.Quantity)
        assert isinstance(to_array(c, keep_unit=True), u.Quantity)

    def test_to_list(self):
        """Test that inputs are converted to a list"""
        a = [1, 2, 3]
        b = (1, 2, 3)
        c = np.random.rand(10)

        assert to_list(a) is a
        assert to_list(b) == list(b)
        assert to_list(c) == [c]
