"""
Author: Elko Gerville-Reache
Date Created: 2026-05-13
Date Modified: 2026-05-13
Description:
    Functions related to statistical analysis.
"""

from typing import Literal

import astropy.units as u
import numpy as np
from numpy.typing import NDArray

from visualastro.core.config import config
from visualastro.core.units import ensure_common_unit


def normalize(
    data: NDArray | u.Quantity | list | tuple,
    method: Literal['max', 'min', 'mean', 'median'] = 'max',
) -> NDArray | u.Quantity | list:
    """
    Rescale data so that a chosen statistic (default: the maximum) equals 1.

    Parameters
    ----------
    data : NDArray | u.Quantity | list | tuple
        Input data. Lists and tuples must be flat. A list of `Quantity`
        objects must share compatible units.
    method : {'max', 'min', 'mean', 'median'}, optional, default='max'
        Statistic used as the reference, computed ignoring NaNs.

    Returns
    -------
    NDArray | u.Quantity | list
        Normalized data, `data / reference`. Tuples are returned as
        lists. `Quantity` input yields dimensionless values.

    Raises
    ------
    ValueError
        If `method` is not supported, or the reference value is zero
        or non-finite.
    TypeError
        If `data` is not an array, `Quantity`, list, or tuple.
    astropy.units.UnitConversionError
        If a list contains `Quantity` objects with incompatible units.

    Notes
    -----
    A negative reference (e.g. `method='max'` on all-negative data)
    flips the sign of the result.
    """
    if not isinstance(data, (np.ndarray, u.Quantity, list, tuple)):
        raise TypeError(f"Unsupported input type: {type(data).__name__}")

    norm_method = {
        'max': np.nanmax,
        'min': np.nanmin,
        'mean': np.nanmean,
        'median': np.nanmedian,
    }.get(str(method).lower())
    if norm_method is None:
        raise ValueError(
            "method must be one of: 'max', 'min', 'mean', 'median', "
            f"got: {method!r}"
        )

    is_seq = isinstance(data, (list, tuple))
    has_unit = isinstance(data, u.Quantity)
    if is_seq:
        has_unit = any(isinstance(d, u.Quantity) for d in data)
        arr = u.Quantity(data) if has_unit else np.asarray(data, dtype=float)
    else:
        arr = data

    norm = norm_method(arr)
    value = norm.value if isinstance(norm, u.Quantity) else norm
    if not np.isfinite(value) or value == 0:
        raise ValueError(f"Cannot normalize: reference value is {norm}.")

    result = arr / norm
    if is_seq:
        return result.tolist() if not has_unit else list(result)
    return result


def percent_difference(a: NDArray, b: NDArray) -> NDArray:
    """
    Compute the percent difference between two arrays.

    The percent difference is defined as the absolute difference between
    `a` and `b` divided by their mean, expressed as a percentage:

        percent_difference = |a - b| / ((a + b) / 2) * 100

    Parameters
    ----------
    a : np.ndarray
        First input array. Must be convertable to an array
        with `np.asarray`.
    b : np.ndarray
        Second input array. Must be broadcastable with `a`.
        Must be convertable to an array with `np.asarray`.

    Returns
    -------
    numpy.ndarray
        Percent difference between `a` and `b`, element-wise.
        Returns `nan` where both `a` and `b` are zero.

    Notes
    -----
    Uses `numpy.errstate` to suppress division by zero and invalid
    value warnings. Elements where the mean of `a` and `b` is zero
    will produce `nan` in the output.

    Examples
    --------
    >>> percent_difference(1.0, 2.0)
    66.666...
    >>> percent_difference(np.array([1, 2, 3]), np.array([2, 2, 4]))
    array([66.666...,  0.    , 28.571...])
    >>> percent_difference(0.0, 0.0)
    nan
    """
    unit = ensure_common_unit([a, b], on_mismatch=config.unit_mismatch)

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    with np.errstate(invalid='ignore', divide='ignore'):
        result = (np.abs(a - b) / (a + b) / 2) * 100

    if unit is not None:
        result = result * unit

    return result


def relative_error(a: NDArray | u.Quantity, b: NDArray | u.Quantity) -> NDArray | u.Quantity:
    """
    Compute element-wise relative error between two arrays.

    Parameters
    ----------
    a : NDArray or Quantity
        Approximation or predicted values.
    b : NDArray or Quantity
        Reference or ground truth values. Must be broadcastable with `a`.

    Returns
    -------
    NDArray or Quantity
        Relative error computed as (a - b) / b. Shape matches broadcasted
        input. Preserves units if inputs are Quantities.

    Raises
    ------
    UnitsError
        If `a` and `b` have incompatible units and `config.unit_mismatch='raise'`.

    Notes
    -----
    Division by zero produces nan without raising warnings.
    """
    unit = ensure_common_unit([a, b], on_mismatch=config.unit_mismatch)

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    with np.errstate(invalid='ignore', divide='ignore'):
        error = (a - b) / b

    if unit is not None:
        error = error * unit

    return error
