"""
Author: Elko Gerville-Reache
Date Created: 2026-03-29
Date Modified: 2026-04-17
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
from tests.conftest import generate_test_cube
from visualastro.core.numerical_utils import (
    interpolate,
    to_array,
    to_list,
    _unwrap_if_single
)
from visualastro.core.validation import allclose
from visualastro.dataclasses.datacube import DataCube
from visualastro.dataclasses.fitsfile import FitsFile
from visualastro.plotting.plot_utils import _normalize_plotting_input,


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
        xp: u.Quantity = np.array([0, 1, 2, 3]) * u.AA
        yp: u.Quantity = np.array([0, 1, 4, 9]) * u.MJy
        x_interp, y_interp = interpolate(
            xp, yp, x_range=(0, 3), N_samples=10, method='cubic_spline'
        )

        assert x_interp.unit == u.AA
        assert y_interp.unit == u.MJy


class TestNormalizePlottingInputs:

    def test_normalize_plotting_input(self):
        # test 1: Single arrays
        X = np.array([1, 2, 3])
        Y = np.array([4, 5, 6])
        x = _normalize_plotting_input(X)
        y = _normalize_plotting_input(Y)
        assert isinstance(x, list) and allclose(x, [X])
        assert isinstance(y, list) and allclose(y, [Y])

        # test 2: Quantity arrays
        X = np.array([1, 2, 3]) * u.um
        Y = np.array([4, 5, 6]) * u.um
        x = _normalize_plotting_input(X)
        y = _normalize_plotting_input(Y)
        assert isinstance(x, list) and allclose(x, [X])
        assert isinstance(y, list) and allclose(y, [Y])

        # test 3: List of scalar Quantities
        X = [1*u.um, 2*u.um, 3*u.um]
        Y = [4*u.um, 5*u.um, 6*u.um]
        x = _normalize_plotting_input(X)
        y = _normalize_plotting_input(Y)
        assert allclose(x, list(X)) and allclose(y, list(Y))

        # test 4: tuple of scalar Quantities
        X = (1*u.um, 2*u.um, 3*u.um)
        Y = (4*u.um, 5*u.um, 6*u.um)
        x = _normalize_plotting_input(X)
        y = _normalize_plotting_input(Y)
        assert allclose(x, list(X)) and allclose(y, list(Y))

        # Test 5: List of Quantity arrays (multiple datasets)
        X = [np.array([1, 2])*u.um, np.array([3, 4])*u.um]
        Y = [np.array([5, 6])*u.um, np.array([7, 8])*u.um]
        x = _normalize_plotting_input(X)
        y = _normalize_plotting_input(Y)
        assert allclose(x, list(X)) and allclose(y, list(Y))


class TestNumericalUtils:

    def test_to_array_keep_unit(self, generate_test_cube):
        """Test that to array returns either array or Quantity."""
        a = np.random.rand(10)
        b = np.random.rand(10, 10, 10) * u.erg
        hdu = generate_test_cube
        c = SpectralCube.read(hdu)
        d = DataCube(data=b)
        e = FitsFile(data=b)

        A = to_array(a)
        AA = to_array(a, keep_unit=False)
        B = to_array(b, keep_unit=False)
        C = to_array(c, keep_unit=False)
        D = to_array(d, keep_unit=False)
        E = to_array(e, keep_unit=False)

        assert isinstance(A, np.ndarray)
        assert isinstance(AA, np.ndarray)
        assert isinstance(B, np.ndarray)
        assert isinstance(C, np.ndarray)
        assert isinstance(D, np.ndarray)
        assert isinstance(E, np.ndarray)

        assert not isinstance(A, u.Quantity)
        assert not isinstance(AA, u.Quantity)
        assert not isinstance(B, u.Quantity)
        assert not isinstance(C, u.Quantity)
        assert not isinstance(D, u.Quantity)
        assert not isinstance(E, u.Quantity)

        assert not isinstance(to_array(a, keep_unit=True), u.Quantity)
        assert isinstance(to_array(b, keep_unit=True), u.Quantity)
        assert isinstance(to_array(c, keep_unit=True), u.Quantity)
        assert isinstance(to_array(d, keep_unit=True), u.Quantity)
        assert isinstance(to_array(e, keep_unit=True), u.Quantity)

    def test_to_list(self):
        """Test that inputs are converted to a list"""
        a = [1, 2, 3]
        b = (1, 2, 3)
        c = np.random.rand(10)

        assert to_list(a) is a
        assert to_list(b) == list(b)
        assert to_list(c) == [c]

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
        F = np.asarray([2.0])
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
