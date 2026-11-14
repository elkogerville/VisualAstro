"""
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2026-11-14
Description:
    Public data / array related functions.
"""

from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import Any, Literal, TypeVar, overload

from astropy import units as u
import numpy as np
from numpy.typing import ArrayLike, NDArray

from visualastro.core.array_types import _is_scalar
from visualastro.optional_dependencies._spectralcube import (
    SpectralCube, _HAS_SPECTRAL_CUBE
)


T = TypeVar('T')

# Type Checking Arrays and Objects
# --------------------------------
def get_data(obj):
    """
    Return the `data` attribute of an object if present;
    otherwise return the object unchanged. In visualastro,
    the data extension represents the high level datastructure
    used to hold data. This is usually a `np.ndarray`,
    `u.Quantity`, or `SpectralCube`.

    Parameters
    ----------
    obj : any
        An object that may expose a `data` attribute (e.g. a DataCube,
        FITS-like object), or a raw NumPy array.

    Returns
    -------
    array-like
        `obj.data` if the attribute exists; otherwise `obj` itself.
    """
    if isinstance(obj, np.ma.MaskedArray):
        return obj.data
    if isinstance(obj, (np.ndarray, u.Quantity)):
        return obj
    return obj.data if hasattr(obj, 'data') else obj


@overload
def get_value(obj: u.Quantity) -> NDArray | float | int: ...

@overload
def get_value(obj: T) -> T: ...

def get_value(obj: Any):
    """
    Return the numeric value of an object,
    stripping units if present.

    If the object exposes a `value` attribute
    (e.g., an Astropy `u.Quantity`), that attribute
    is returned. Otherwise, the object itself is
    returned unchanged.

    Parameters
    ----------
    obj : any
        Object that may expose a `value` attribute.

    Returns
    -------
    any :
        The underlying numeric value with units removed,
        if applicable.
    """
    return obj.value if hasattr(obj, 'value') else obj


@overload
def to_array(obj: Any, keep_unit: Literal[False] = False) -> NDArray: ...

@overload
def to_array(obj: Any, keep_unit: Literal[True]) -> NDArray | u.Quantity: ...

@overload
def to_array(obj: Any, keep_unit: bool) -> NDArray | u.Quantity: ...

def to_array(obj: Any, keep_unit: bool = False) -> NDArray | u.Quantity:
    """
    Return input object as either a np.ndarray or u.Quantity.

    Parameters
    ----------
    obj : array-like, np.ndarray, u.Quantity or SpectralCube
        Any array-like object, or an object that exposes
        a `data` or `value` attribute.
    keep_unit : bool, optional, default=False
        If True, keep astropy units attached if present.

    Returns
    -------
    array : np.ndarray
        u.Quantity array if `keep_unit` is True, else a NumPy array.

    Raises
    ------
    TypeError :
        If obj is None.
    """
    if obj is None:
        raise TypeError('None cannot be converted to an array')

    if isinstance(obj, u.Quantity):
        return obj if keep_unit else np.asarray(obj.value)

    elif _HAS_SPECTRAL_CUBE and isinstance(obj, SpectralCube):
        q = obj.filled_data[:]
        if not isinstance(q, u.Quantity):
            q = u.Quantity(np.asarray(q), unit=obj.unit)
        return q if keep_unit else np.asarray(q.value)

    elif isinstance(obj, np.ndarray):
        return obj

    # check if obj had data or value attributes
    # with priority to data
    for attr in ('data', 'value'):
        if hasattr(obj, attr):
            inner = getattr(obj, attr)
            if inner is not obj:
                result = to_array(inner, keep_unit=keep_unit)

                # check for unit in either obj or obj attribute
                if keep_unit and not isinstance(result, u.Quantity):
                    unit = getattr(obj, 'unit', None) or getattr(inner, 'unit', None)
                    if unit is not None:
                        return u.Quantity(result, unit=unit)

                return result

    try:
        return np.asarray(obj)
    except Exception:
        raise TypeError(
            f'Object of type {type(obj).__name__} cannot be converted to an array'
        )


def to_list(obj: T | list[T] | tuple[T, ...]) -> list[T]:
    """
    Normalize input to a list. If input is a tuple,
    convert it to a list via `list(obj)`. To simply
    wrap in a list an object that isnt a list, use `as_list`.

    Thus `to_list((1,2,3))` returns `[1,2,3]`
    while `as_list((1,2,3))` returns `[(1,2,3)]`

    Parameters
    ----------
    obj : object or list/tuple of objects
        Input data.

    Returns
    -------
    list
        A list containing `obj` if a single object was provided,
        or `obj` converted to a list if it was already a list or tuple.
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    return [obj]


def as_list(obj: T | list[T]) -> list[T]:
    """
    Ensure the return value is a list.

    If `obj` is a list, it is returned unchanged. Any other object,
    including tuples, arrays and generators, is wrapped in a single-element
    list. To convert a tuple (or other iterable) into a list element-wise,
    use `to_list`.

    `as_list((1, 2, 3))` returns `[(1, 2, 3)]`, while
    `to_list((1, 2, 3))` returns `[1, 2, 3]`.

    Parameters
    ----------
    obj : T | list[T]
        Single object or list of objects. Tuples are treated as single objects.

    Returns
    -------
    list[T]
        `obj` itself if it is a list (aliased, not copied), otherwise `[obj]`.

    Notes
    -----
    - If `T` is itself a list, the input is ambiguous and is treated as a
      list of items.
    - `None` is wrapped as `[None]`. Handle `None` before calling.
    """
    return obj if isinstance(obj, list) else [obj]


def _ensure_iterable(obj: T | Sequence[T] | NDArray) -> Sequence[T] | NDArray:
    """
    Ensure `obj` can be iterated over, wrapping single items in a list.

    Lists, tuples and arrays (ndarray, Quantity) with ndim >= 1 are treated
    as collections and returned unchanged (no copy, no element-wise
    conversion). Everything else is treated as a single item.

    Parameters
    ----------
    obj : T | Sequence[T] | numpy.ndarray | astropy.units.Quantity
        Single item or collection of items. Strings, 0-d arrays and
        scalar Quantity are single items. Other iterables
        (generators, sets) are materialized into a list.

    Returns
    -------
    Sequence[T] | numpy.ndarray
        `obj` itself if it is a list, tuple or array with ndim >= 1
        (aliased, not copied), otherwise a new list.

    Notes
    -----
    - Arrays with ndim > 1 iterate over the first axis.
    - `None` is wrapped as `[None]`. Handle `None` before calling.
    - Do not use for parameters where a tuple is a single value
      (e.g., `xlim`, `figsize`).
    """
    if isinstance(obj, str):
        return [obj]

    if getattr(obj, 'ndim', None) == 0:
        return [obj]

    if isinstance(obj, (list, tuple, np.ndarray)):
        return obj

    if isinstance(obj, Iterable):
        return list(obj)

    return [obj]


def _array_length(
    obj: float | u.Quantity | NDArray | Sequence[float | u.Quantity]
) -> int:
    """
    Return the length of the first axis, treating scalars as length 1.

    Parameters
    ----------
    obj : float | u.Quantity | NDArray | Sequence[float | u.Quantity]
        Input data. Not converted, not copied.

    Returns
    -------
    n : int
        First-axis length, or `1` for scalar and 0D input.
    """
    if _is_scalar(obj):
        return 1
    shape = getattr(obj, 'shape', None)
    if shape is not None:
        return shape[0] if shape else 1
    return len(obj)


def _ndim(
    obj: float | u.Quantity | NDArray | Sequence[float | u.Quantity],
) -> int:
    """
    Return the number of dimensions, treating scalars as 0D.

    Parameters
    ----------
    obj : float | astropy.units.Quantity | numpy.ndarray | Sequence[float | astropy.units.Quantity]
        Input data. Never converted to an array. For `list` and `tuple`,
        the depth is inferred from the first element.

    Returns
    -------
    ndim : int
        Number of dimensions. `0` for scalar, `str` and 0D input;
        `1` for an empty `list` or `tuple`.

    Notes
    -----
    - Nesting is assumed regular.
    """
    if _is_scalar(obj):
        return 0
    ndim = getattr(obj, 'ndim', None)
    if ndim is not None:
        return int(ndim)
    if isinstance(obj, (str, bytes)):
        return 0
    if isinstance(obj, (list, tuple)):
        return 1 + (_ndim(obj[0]) if obj else 0)
    return int(np.ndim(obj))


def _to_xy(xy: ArrayLike) -> NDArray | None:
    """
    Convert an array-like into a float (N, 2) array, preserving NaNs.

    Masked values are replaced with NaN. Non-finite points are kept.

    Parameters
    ----------
    xy : ArrayLike
        Array-like of shape (N, 2).

    Returns
    -------
    NDArray | None
        Float array of shape (N, 2), or `None` if the input is empty
        or malformed.
    """
    xy = np.ma.filled(np.ma.asarray(xy, dtype=float), np.nan)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0:
        return None
    return xy
