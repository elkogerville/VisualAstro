"""
Author: Elko Gerville-Reache
Date Created: 2026-10-10
Date Modified: 2026-10-10
Description:
    Tests for stats module.
"""

import numpy as np
import pytest
import astropy.units as u
from astropy.units import UnitConversionError

from visualastro.core.stats import normalize


class TestNormalize:
    def test_list_default_max(self):
        assert np.all(normalize([1, 2, 4]) == [0.25, 0.5, 1.0])

    def test_tuple_returns_list(self):
        result = normalize((1, 2, 4))
        assert isinstance(result, list)
        assert result == [0.25, 0.5, 1.0]

    def test_ndarray_max(self):
        result = normalize(np.array([1.0, 2.0, 4.0]))
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [0.25, 0.5, 1.0])

    def test_methods(self):
        data = np.array([1.0, 2.0, 4.0])
        cases = [
            ('max', [0.25, 0.5, 1.0]),
            ('min', [1.0, 2.0, 4.0]),
            ('mean', [3 / 7, 6 / 7, 12 / 7]),
            ('median', [0.5, 1.0, 2.0]),
        ]
        for method, expected in cases:
            result = normalize(data, method=method)
            np.testing.assert_allclose(result, expected, err_msg=f'method={method}')

    def test_nan_ignored(self):
        result = normalize(np.array([1.0, np.nan, 2.0]))
        np.testing.assert_allclose(result, [0.5, np.nan, 1.0])

    def test_negative_reference_flips_sign(self):
        result = normalize(np.array([-4.0, -2.0, -1.0]), method='max')
        np.testing.assert_allclose(result, [4.0, 2.0, 1.0])

    # ---------- units ----------
    def test_quantity_array_dimensionless(self):
        result = normalize([1, 5, 1000] * u.m)
        assert isinstance(result, u.Quantity)
        assert result.unit.is_equivalent(u.dimensionless_unscaled)
        np.testing.assert_allclose(
            result.to_value(u.dimensionless_unscaled), [0.001, 0.005, 1.0]
        )

    def test_list_of_quantities(self):
        result = normalize([1 * u.m, 5 * u.m, 1000 * u.m])
        assert isinstance(result, list)
        values = [r.to_value(u.dimensionless_unscaled) for r in result]
        assert all(r.unit == u.dimensionless_unscaled for r in result)
        np.testing.assert_allclose(values, [0.001, 0.005, 1.0])

    def test_list_mixed_compatible_units(self):
        result = normalize([1 * u.m, 100 * u.cm])
        assert isinstance(result, list)
        values = [r.to_value(u.dimensionless_unscaled) for r in result]
        np.testing.assert_allclose(values, [1.0, 1.0])

    def test_list_incompatible_units_raises(self):
        with pytest.raises(UnitConversionError):
            normalize([1 * u.m, 2 * u.s])

    # ---------- errors ----------
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            normalize([1, 2, 3], method='std')  # type: ignore[arg-type]

    def test_none_method_raises(self):
        with pytest.raises(ValueError):
            normalize([1, 2, 3], method=None)  # type: ignore[arg-type]

    def test_zero_reference_raises(self):
        with pytest.raises(ValueError, match='Cannot normalize'):
            normalize([0, 0, 0])

    def test_zero_mean_raises(self):
        with pytest.raises(ValueError, match='Cannot normalize'):
            normalize(np.array([-1.0, 1.0]), method='mean')

    def test_all_nan_raises(self):
        with pytest.warns(RuntimeWarning):
            with pytest.raises(ValueError, match='Cannot normalize'):
                normalize(np.array([np.nan, np.nan]))

    def test_inf_reference_raises(self):
        with pytest.raises(ValueError, match='Cannot normalize'):
            normalize(np.array([1.0, np.inf]))

    def test_input_not_mutated(self):
        data = np.array([1.0, 2.0, 4.0])
        original = data.copy()
        normalize(data)
        np.testing.assert_array_equal(data, original)
