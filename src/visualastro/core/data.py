"""
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2026-10-06
Description:
    Public data / array related functions.
"""

from __future__ import annotations
from typing import Any, Literal, TypeVar, overload

from astropy import units as u
import numpy as np
from numpy.typing import NDArray

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
    convert it to a list via `tuple(obj)`. To simply
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
    Ensure return value is always a list.
    If `obj` is not a list, wrap it in a list.
    Otherwise, return `obj`. To simply
    convert a tuple into a list use `to_list`.

    Thus `to_list((1,2,3))` returns `[1,2,3]`
    while `as_list((1,2,3))` returns `[(1,2,3)]`

    Parameters
    ----------
    obj : object or list/tuple of objects
        Input data.

    Returns
    -------
    list
        A list containing `obj` if `obj` is not a list,
        or `obj` itself it is already a list.
    """
    return obj if isinstance(obj, list) else [obj]
