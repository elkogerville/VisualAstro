"""
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2026-10-06
Description:
    Array properties inspection functions.
"""

from collections.abc import Sequence
from typing import Any, TypeVar

from astropy import units as u
import numpy as np


T = TypeVar('T')


def _is_scalar_quantity(obj) -> bool:
    """Check if `obj` is a scalar Quantity (0-dimensional)."""
    return isinstance(obj, u.Quantity) and obj.ndim == 0


def _is_scalar(obj) -> bool:
    """Check if `obj` is a scalar or scalar Quantity."""
    if isinstance(obj, str):
        return False

    if np.isscalar(obj):
        return True

    if isinstance(obj, np.ndarray) and obj.shape == ():
        return True
    return _is_scalar_quantity(obj)


def _is_iterable(obj) -> bool:
    """Check that an object is an iterable container (array-like)."""
    if isinstance(obj, (list, tuple)):
        return True
    if isinstance(obj, (np.ndarray, u.Quantity)):
        return obj.ndim > 0
    return False


def _is_array_like(obj) -> bool:
    """Check if object is array-like (list, ndarray, or array Quantity)."""
    if _is_scalar_quantity(obj):
        return False
    return isinstance(obj, (list, tuple, np.ndarray)) or (hasattr(obj, 'unit') and np.ndim(obj) >= 1)


def _is_ndarray_or_quantity_array(obj) -> bool:
    """
    Check if an object is either a `NDArray` or a `u.Quantity` array.
    `u.Quantity` scalars return `False`.
    """
    if isinstance(obj, (list, tuple)):
        return False
    return _is_array_like(obj)


def _is_sequence_of_sequences(obj: Sequence) -> bool:
    """
    Check if an object is a list[array-like]
    or a tuple[array-like]. Array-like includes
    `lists`, `tuples`, `np.ndarray` and `u.Quantity` arrays.
    """
    return (
        isinstance(obj, (list, tuple)) and
        all(_is_array_like(o) for o in obj)
    )


def _is_1d(obj: Any) -> bool:
    """Check that an object is a 1D Sequence."""
    if isinstance(obj, (np.ndarray, u.Quantity)):
        return obj.ndim == 1

    if isinstance(obj, (list, tuple)):
        return len(obj) > 0 and all(_is_scalar(o) for o in obj)

    return False


def _is_2d(obj: Any) -> bool:
    """Check that an object is a 2D Sequence."""
    if isinstance(obj, (np.ndarray, u.Quantity)):
        return obj.ndim == 2

    if isinstance(obj, (list, tuple)):
        for o in obj:
            if not hasattr(o, '__len__') or getattr(o, 'isscalar', False):
                return False
        return True

    return False


def _is_wrapped_1d(obj) -> bool:
    """
    Check that an object is either a Sequence of sequences of len 1,
    or is a 2D array with shape (N,1) or (1,N).

    In other words, does `obj[0]` still contain all the data values of `obj`,
    minus the extra unused axis.

    Examples
    --------
    >>> obj = [1,2,3]
    >>> _is_wrapped_1d(obj)
    False

    >>> obj = [[1,2,3]]
    >>> _is_wrapped_1d(obj)
    True

    >>> obj = np.random.rand(10)
    >>> _is_wrapped_1d(obj)
    False
    >>> _is_wrapped_1d(obj.reshape(10, 1))
    True
    """
    if _is_sequence_of_sequences(obj) and len(obj) == 1:
        return True

    if isinstance(obj, (np.ndarray, u.Quantity)) and obj.ndim == 2:
        shape = obj.shape
        if shape[0] == 1 or shape[1] == 1:
            return True

    return False
