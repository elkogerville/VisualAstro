"""
Author: Elko Gerville-Reache
Date Created: 2026-03-29
Date Modified: 2026-03-29
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
from astropy.wcs import WCS
import numpy as np
from spectral_cube import SpectralCube
from tests.conftest import generate_test_cube
from visualastro.core.numerical_utils import (
    interpolate,
    to_array,
    _unwrap_if_single
)
from visualastro.dataclasses.datacube import DataCube
from visualastro.dataclasses.fitsfile import FitsFile

class TestToArray:

    def test_toarray_keep_unit(self, generate_test_cube):
        """Test that to array returns either array or Quantity."""
        a = np.random.rand(10)
        b = np.random.rand(10) * u.erg
        hdu = generate_test_cube
        c = SpectralCube.read(hdu)
        d = np.random.rand(10, 10, 10) * u.erg
        e = DataCube(data=d)
        f = FitsFile(data=d)

        assert isinstance(to_array(a), np.ndarray)
        assert isinstance(to_array(a, keep_unit=False), np.ndarray)
        assert isinstance(to_array(b, keep_unit=False), np.ndarray)
        assert isinstance(to_array(c, keep_unit=False), np.ndarray)
        assert isinstance(to_array(e, keep_unit=False), np.ndarray)
        assert isinstance(to_array(f, keep_unit=False), np.ndarray)

        assert not isinstance(to_array(a, keep_unit=True), u.Quantity)
        assert isinstance(to_array(b, keep_unit=True), u.Quantity)
        assert isinstance(to_array(c, keep_unit=True), u.Quantity)
        assert isinstance(to_array(e, keep_unit=True), u.Quantity)
        assert isinstance(to_array(f, keep_unit=True), u.Quantity)


class TestInterpolate:

    def test_basic_linear_interpolation(self):
        """Test basic linear interpolation."""
        xp = np.array([0, 1, 2])
        yp = np.array([0, 1, 4])
        x_interp, y_interp = interpolate(xp, yp, x_range=(0, 2), N_samples=5)

        assert len(x_interp) == 5
        assert len(y_interp) == 5
        assert x_interp[0] == 0
        assert x_interp[-1] == 2
        assert np.allclose(y_interp[0], 0)
        assert np.allclose(y_interp[-1], 4)

    def test_cubic_interpolation(self):
        """Test cubic interpolation."""
        xp = np.array([0, 1, 2, 3])
        yp = np.array([0, 1, 4, 9])
        x_interp, y_interp = interpolate(
            xp, yp, x_range=(0, 3), N_samples=10, method='cubic'
        )

        assert len(x_interp) == 10
        assert len(y_interp) == 10

    def test_cubic_spline_interpolation(self):
        """Test cubic spline interpolation."""
        xp = np.array([0, 1, 2, 3])
        yp = np.array([0, 1, 4, 9])
        x_interp, y_interp = interpolate(
            xp, yp, x_range=(0, 3), N_samples=10, method='cubic_spline'
        )

        assert len(x_interp) == 10
        assert len(y_interp) == 10

    def test_interpolate_with_units(self):
        """Test interpolation conserves units."""
        xp = np.array([0, 1, 2, 3]) * u.AA
        yp = np.array([0, 1, 4, 9]) * u.MJy
        x_interp, y_interp = interpolate(
            xp, yp, x_range=(0, 3), N_samples=10, method='cubic_spline'
        )

        assert x_interp.unit == u.AA
        assert y_interp.unit == u.MJy

class TestNumericalUtils:

    def test_unwrap_if_single(self):
        """
        Test that _unwrap_if_single unwraps an object
        if the len == 1.
        """
        A = [1]
        B = [1, 2]
        C = (1)
        D = (1, 2)
        E = (1,)
        F = np.ndarray(1)
        G = np.random.rand(1)
        H = np.random.rand(3)
        I = np.random.rand(10, 10)

        assert _unwrap_if_single(A) is A[0]
        assert _unwrap_if_single(B) is B
        assert _unwrap_if_single(C) is C
        assert _unwrap_if_single(D) is D
        assert _unwrap_if_single(E) is E[0]
        assert _unwrap_if_single(F) == F[0]
        assert _unwrap_if_single(G) == G
        assert np.allclose(_unwrap_if_single(H),  H)
        assert np.allclose(_unwrap_if_single(I),  I)
