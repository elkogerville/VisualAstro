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

import numpy as np
from visualastro.core.numerical_utils import _unwrap_if_single

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
