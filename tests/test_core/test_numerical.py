"""
Author: Elko Gerville-Reache
Date Created: 2026-03-29
Date Modified: 2026-10-26
Description:
    Tests for numerical module.
"""

import astropy.units as u
import numpy as np

from visualastro.core.numerical import interpolate


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
