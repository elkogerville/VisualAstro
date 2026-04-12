"""
Author: Elko Gerville-Reache
Date Created: 2026-04-17
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

from visualastro.core.validation import allclose

class TestValidation:

    def test_allclose(self):

        # single arrays
        assert allclose([1, 2, 3], [1, 2, 3])
        assert allclose(np.array([1, 2, 3]), np.array([1, 2, 3]))

        # Quantity arrays
        assert allclose([1, 2, 3]*u.um, [1, 2, 3]*u.um)
        assert not allclose([1, 2, 3]*u.um, [1, 2, 3]*u.m)

        # list of scalars
        assert allclose([1*u.um, 2*u.um], [1*u.um, 2*u.um])
        assert not allclose([1*u.um, 2*u.um], [1*u.m, 2*u.m])
        assert allclose([1, 2], [1, 2])
        assert allclose([1, 2], (1, 2))    # list vs tuple

        # List of arrays
        assert allclose([[1, 2], [3, 4]], [[1, 2], [3, 4]])
        assert allclose(
            [np.array([1, 2])*u.um, np.array([3, 4])*u.um],
            [np.array([1, 2])*u.um, np.array([3, 4])*u.um]
        )

        # mixed structures
        assert not allclose([[1, 2]], [1, 2])  # Different nesting
        assert not allclose([1, 2, 3], [1, 2]) # Length mismatch

        # tolerance handling
        assert allclose([1.0, 2.0], [1.000001, 2.000001])
        assert allclose([1*u.um, 2*u.um], [1.000001*u.um, 2.000001*u.um])
