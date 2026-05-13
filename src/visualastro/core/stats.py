"""
Author: Elko Gerville-Reache
Date Created: 2026-05-13
Date Modified: 2026-05-13
Description:
    Functions related to statistical analysis.
Dependencies:
    - astropy
    - numpy
    - scipy
    - spectral-cube
"""

import astropy.units as u
import numpy as np
from numpy.typing import NDArray

from visualastro.core.config import config
from visualastro.core.units import ensure_common_unit
from visualastro.core.validation import _type_name


def normalize(
    data: NDArray | u.Quantity | list | tuple | float
) -> NDArray | u.Quantity | list:
    """
    Normalize input data with the formula: norm = data / np.nanmax(data).

    Parameters
    ----------
    data : NDArray | u.Quantity | list | tuple
        Input data to normalize.

    Returns
    -------
    NDArray | u.Quantity | list :
        Normalized data. Tuples are converted to lists.
    """
    if isinstance(data, (np.ndarray, u.Quantity)):
        return data / np.nanmax(data)

    if isinstance(data, (list, tuple)):
        return [d/np.nanmax(data) for d in data]

    raise ValueError(
        f'Unsupported input type! got{_type_name(data)}.'
    )


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
