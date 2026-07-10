"""
Author: Elko Gerville-Reache
Date Created: 2026-03-31
Date Modified: 2026-07-10
Description:
    Tests for plot utils module.
"""

import astropy.units as u
import numpy as np

from visualastro.core.validation import allclose
from visualastro.plotting.core.utils import _normalize_plotting_input


class TestNormalizePlottingInputs:

    def test_normalize_plotting_input(self):
        # test 1: single arrays
        X = np.array([1, 2, 3])
        Y = np.array([4, 5, 6])
        x = _normalize_plotting_input(X)
        y = _normalize_plotting_input(Y)
        assert allclose(x[0], X) and allclose(y[0], Y)

        # test 2: Quantity arrays
        X = np.array([1, 2, 3]) * u.um
        Y = np.array([4, 5, 6]) * u.um
        x = _normalize_plotting_input(X)
        y = _normalize_plotting_input(Y)
        assert allclose(x[0], X) and allclose(y[0], Y)

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
