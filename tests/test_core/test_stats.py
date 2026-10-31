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

from visualastro.core.stats import (
    normalize, percent_difference, relative_error
)


class TestNormalize:
    """Test `visualastro.core.stats.normalize`."""
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


class TestPercentDifference:
    """Test `visualastro.core.stats.percent_difference`."""
    def test_scalars(self):
        np.testing.assert_allclose(percent_difference(1.0, 2.0), 200 / 3)

    def test_arrays(self):
        result = percent_difference(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 4.0]))
        np.testing.assert_allclose(result, [200 / 3, 0.0, 200 / 7])

    def test_symmetric(self):
        a, b = np.array([1.0, 5.0]), np.array([3.0, 2.0])
        np.testing.assert_allclose(percent_difference(a, b), percent_difference(b, a))

    def test_identical_is_zero(self):
        np.testing.assert_allclose(percent_difference(np.array([1.0, 7.0]), np.array([1.0, 7.0])), 0.0)

    def test_negative_values_match_positive(self):
        np.testing.assert_allclose(percent_difference(-1.0, -2.0), percent_difference(1.0, 2.0))

    def test_broadcasting(self):
        result = percent_difference(np.array([1.0, 2.0]), 2.0)
        np.testing.assert_allclose(result, [200 / 3, 0.0])

    def test_both_zero_is_nan(self):
        assert np.isnan(percent_difference(0.0, 0.0))

    def test_zero_mean_nonzero_diff_is_inf(self):
        assert np.isinf(percent_difference(1.0, -1.0))

    def test_same_units(self):
        np.testing.assert_allclose(percent_difference(1 * u.m, 2 * u.m), 200 / 3)

    def test_unit_scale_conversion(self):
        np.testing.assert_allclose(percent_difference(1 * u.m, 100 * u.cm), 0.0, atol=1e-12)

    def test_returns_dimensionless_array(self):
        result = percent_difference(1 * u.m, 2 * u.m)
        assert not isinstance(result, u.Quantity)

    def test_incompatible_units_raises(self):
        with pytest.raises(UnitConversionError):
            percent_difference(1 * u.m, 1 * u.s)


class TestRelativeError:
    """Test `visualastro.core.stats.relative_error`."""
    def test_scalars(self):
        np.testing.assert_allclose(relative_error(2.0, 1.0), 1.0)

    def test_arrays(self):
        result = relative_error(np.array([2.0, 4.0]), np.array([1.0, 2.0]))
        np.testing.assert_allclose(result, [1.0, 1.0])

    def test_signed(self):
        np.testing.assert_allclose(relative_error(0.5, 1.0), -0.5)

    def test_not_symmetric(self):
        assert relative_error(2.0, 1.0) != relative_error(1.0, 2.0)

    def test_negative_reference_flips_sign(self):
        np.testing.assert_allclose(relative_error(-2.0, -1.0), 1.0)
        np.testing.assert_allclose(relative_error(-0.5, -1.0), -0.5)

    def test_broadcasting(self):
        result = relative_error(np.array([2.0, 3.0]), 1.0)
        np.testing.assert_allclose(result, [1.0, 2.0])

    def test_both_zero_is_nan(self):
        assert np.isnan(relative_error(0.0, 0.0))

    def test_zero_reference_is_inf(self):
        assert np.isinf(relative_error(1.0, 0.0))

    def test_same_units(self):
        np.testing.assert_allclose(relative_error(2 * u.m, 1 * u.m), 1.0)

    def test_unit_scale_conversion(self):
        np.testing.assert_allclose(relative_error(1 * u.m, 100 * u.cm), 0.0, atol=1e-12)

    def test_returns_dimensionless_array(self):
        result = relative_error(2 * u.m, 1 * u.m)
        assert not isinstance(result, u.Quantity)

    def test_incompatible_units_raises(self):
        with pytest.raises(UnitConversionError):
            relative_error(1 * u.m, 1 * u.s)
