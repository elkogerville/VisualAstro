"""
Author: Elko Gerville-Reache
Date Created: 2026-03-29
Date Modified: 2026-10-25
Description:
    Tests for sequences module.
"""

import astropy.units as u
import numpy as np
import pytest

from visualastro.core.sequences import _broadcast, _unwrap_if_single


class TestSequences:

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


class TestBroadcast:
    def test_scalar_and_none_are_repeated(self):
        assert _broadcast(150, 3) == [150, 150, 150]
        assert _broadcast(None, 2) == [None, None]

    def test_sequence_passes_through(self):
        assert _broadcast([1, 2, 3], 3) == [1, 2, 3]
        assert _broadcast(np.array([1, 2, 3]), 3) == [1, 2, 3]

    def test_quantity_scalar_and_array(self):
        scalar = _broadcast(5 * u.AA, 2)
        assert scalar == [5 * u.AA, 5 * u.AA]

        arr = _broadcast([1, 2, 3] * u.AA, 3)
        assert len(arr) == 3
        assert arr[1] == 2 * u.AA

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            _broadcast([1, 2], 3, 'y')

    def test_default_name_in_error(self):
        with pytest.raises(ValueError):
            _broadcast([1, 2], 3)
