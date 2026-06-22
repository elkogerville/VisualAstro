"""
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2026-03-11
Description:
    Numerical utility functions.
"""

from collections.abc import Sequence
from typing import Any, Literal, TypeVar, overload

from astropy import units as u
import numpy as np
from numpy.typing import ArrayLike, NDArray
from spectral_cube import SpectralCube

from visualastro.core.config import (
    config,
    _Unset,
    _UNSET,
    _resolve_default
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

    elif isinstance(obj, SpectralCube):
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


# Numerical Operation Functions
# -----------------------------
def flatten(data: ArrayLike) -> NDArray | None:
    """
    Flatten a dataset or a list of datasets into
    a single 1D array.

    Parameters
    ----------
    data : array-like or list of array-like
        Dataset(s) to flatten.

    Returns
    -------
    flat_array : np.ndarray
        Flattened array.
    """
    if data is None:
        return None

    if isinstance(data, (list, tuple)):
        arrays = [
            np.asarray(d).ravel()
            for d in data
            if d is not None and np.size(d) > 0
        ]
        return np.concatenate(arrays) if arrays else None

    array = np.asarray(data).ravel()
    return array if array.size > 0 else None


def finite(
    obj: ArrayLike,
    *,
    keep_unit: bool = True,
    keep_inf: bool = False
) -> NDArray | u.Quantity:
    """
    Filter NaN and optionally infinite values from
    array-like input. The output is always 1D.

    Parameters
    ----------
    obj : ArrayLike
        Input data. May be a `np.ndarray`, `list`, `DataCube`,
        `FitsFile`, `u.Quantity`, or any object compatible with `to_array`.
    keep_unit : bool, optional, default=True
        If `True`, preserve astropy units if present on the input.
    keep_inf : bool, optional, default=False
        If `True`, keep ±inf values and remove only NaNs.
        If `False`, remove NaN and ±inf values.

    Returns
    -------
    np.ndarray or u.Quantity
        A 1-D array containing the filtered values. Units are preserved
        if `keep_unit=True` and the input carries units.

    Notes
    -----
    - Filtering is performed using `np.isfinite` when `keep_inf=False`,
        and `~np.isnan` when `keep_inf=True`.
    """
    data = to_array(obj, keep_unit)
    mask = mask_finite(data, keep_inf=keep_inf)

    return data[mask]


def mask_finite(
    obj: ArrayLike,
    *,
    keep_inf: bool = False
) -> NDArray[np.bool_]:
    """
    Return a boolean mask identifying finite values in array-like input.

    Parameters
    ----------
    obj : ArrayLike
        Input data. May be a `np.ndarray`, `list`, `DataCube`,
        `FitsFile`, `u.Quantity`, or any object compatible with `to_array`.
    keep_inf : bool, default=False
        If `False`, mask excludes NaN and ±inf values.
        If `True`, mask excludes only NaNs and retains ±inf values.

    Returns
    -------
    np.ndarray[bool]
        Boolean mask with the same shape as the input data.
        `True` indicates values that are kept.

    Notes
    -----
    - Uses `np.isfinite` when `keep_inf=False`.
    - Uses `~np.isnan` when `keep_inf=True`.
    """
    data = to_array(obj)
    return ~np.isnan(data) if keep_inf else np.isfinite(data)


def mask_within_range(
    x: ArrayLike,
    xlim: tuple[float, float] | None = None
) -> NDArray[np.bool_]:
    """
    Return a boolean mask for values of x within the given limits.

    Parameters
    ----------
    x : array-like
        Data array (e.g., wavelength or flux values).
    xlim : tuple[float, float] or None, optional, default=None
        (xmin, xmax) range. If None, uses the min/max of `x`.

    Returns
    -------
    mask : ndarray of bool
        True where x is within the limits.
    """
    x = np.asarray(get_value(x), dtype=float)

    xmin = get_value(xlim[0]) if xlim is not None else np.nanmin(x)
    xmax = get_value(xlim[1]) if xlim is not None else np.nanmax(x)

    mask = (x >= xmin) & (x <= xmax)

    return mask


def _is_scalar_quantity(obj):
    """Check if `obj` is a scalar Quantity (0-dimensional)."""
    return isinstance(obj, u.Quantity) and obj.ndim == 0


def _is_scalar(obj):
    """Check if `obj` is a scalar or scalar Quantity."""
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


def _is_ndarray(obj) -> bool:
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


def _extract_xyz(
    *data: NDArray | u.Quantity | Sequence[NDArray | u.Quantity | float] | float,
    order: Literal['c', 'fortran'] | _Unset = _UNSET,
    index_spec: tuple[int, int, int] | _Unset = _UNSET
) -> list[tuple]:
    """
    Extract X, Y, Z coordinates from a variety of supported input formats.

    Supported inputs:

    * 2D array with at least 3 columns/rows, or a list of such arrays.
    * Three 1D array-like or a list of such: `(X, Y, Z)`.
    * Three sequences of 1D array-like: `([x1,x2,], [y1,y2,], [z1,z2,])`.
    * Three scalars: `(x, y, z)`.

    Parameters
    ----------
    *data : NDArray | u.Quantity | Sequence[NDArray | u.Quantity | float] | float
        Input(s) to extract X, Y, Z values from.
        ndarray | Sequence[ndarray] | tuple[array-like, array-like, array-like] | tuple[scalar, scalar, scalar]

        Supported calling conventions:

        * NDArray | Sequence[NDArray]:

            * Each array should be 2D and have at least 3 axes. The extracted
            axes are set by `index_spec`.

        * Three 1D array-like:

            * Three 1D arrays of the same shape.

        * Three Sequences[1D array-like]

            * Three Sequences each containing an array-like to plot.
            Corresponding elements across sequences must share the same shape,
            i.e. `shape(xi) == shape(yi) == shape(zi)`.

    order : {'c', 'fortran'}, optional
        Array traversal order used when extracting from 2D arrays.
        If `_UNSET` uses `config.array_order`.
    index_spec : tuple[int, int, int], optional
        Axis indices `(i_x, i_y, i_z)` identifying X, Y, Z columns/rows
        in the 2D input array. If `_UNSET`, uses `config.index_specification_3D`.

    Returns
    -------
    list[tuple] :
        List of `(x, y, z)` tuples. Each tuple represents a collection to plot,
        and can either be a tuple of scalars or a tuple of 1D array-like.

    Raises
    ------
    ValueError :
        If `len(data) == 3` and the three array-likes are not uniformly
        1D or flattenable to 1D (shapes `(N,1)` or `(1,N)`).
    ValueError
        If the input signature matches none of the supported conventions.
    """
    array_order = _resolve_default(order, config.array_order)
    index_spec = _resolve_default(index_spec, config.index_specification_3D)

    # input is either 2D NDArray or Sequence[2D NDArray]
    if len(data) == 1:
        obj = to_list(data[0])
        result = []
        for o in obj:
            result.append(_extract_xyz_from_ndarray(o, array_order, index_spec))
        return result

    # input is tuple[Any, Any, Any]
    elif len(data) == 3:
        if all(_is_array_like(d) for d in data):
            # input is tuple[array, array, array] where array is either
            # NDArray or u.Quantity array
            if (all(_is_ndarray(d) for d in data)):
                return [tuple(data)]
            # input is tuple[Sequence[array-like], Sequence[array-like], Sequence[array-like]]
            # where array-like is all 1D
            if (
                all(isinstance(d, (list, tuple)) for d in data) and
                all(_is_1d(sublist) for d in data for sublist in d)
            ):
                return [tuple(d) for d in zip(*data)]

            # input is tuple[array-like, array-like, array-like]
            # where array-like is either all 1D or all (N,1) or (1,N)
            if (
                all(_is_wrapped_1d(d) for d in data) or
                all(_is_1d(d) for d in data)
            ):
                return [tuple(d) for d in zip(*data)]

            raise ValueError(
                'inputs must either be all 1D or must all be 2D '
                ' arrays with shapes: (N,1) or (1,N)!'
            )

        # input is tuple[scalar, scalar, scalar]
        if all(_is_scalar(d) for d in data):
            return as_list((data[0], data[1], data[2]))

    raise ValueError(
        'inputs must either be a 2D array, list[2D arrays] or X, Y, Z inputs! '
        'X, Y, Z must all be array-like or list of such.'
    )


def _extract_xyz_from_ndarray(
    obj: NDArray | u.Quantity,
    order: Literal['c', 'fortran'],
    index_spec: tuple[int, int, int]
) -> tuple[NDArray | u.Quantity, NDArray | u.Quantity, NDArray | u.Quantity]:
    """
    Given a 2D NDArray with shape (N,3) or (3,N), extract the x,y,z values.

    Parameters
    ----------
    obj : np.ndarray | u.Quantity
        2D array with shape (N,3) or (3,N), depending on `order`.
    order : {'c', 'fortran'}
        Array order. If `'c'`, `obj` should have shape (N,3).
        If `'fortran'`, `obj` should have shape (3,N).
    index_spec : tuple[int, int, int]
        Specifies which columns (`order='c'`) or rows (`order='fortran'`)
        should be used for extraction.

    Returns
    -------
    tuple[NDArray | u.Quantity, NDArray | u.Quantity, NDArray | u.Quantity] :
        X, Y, and Z values extracted from `obj`. Are all 1D arrays.
        Units are preserved.

    Examples
    --------
    >>> a = np.random.rand(10,3)
    >>> _extract_xyz_from_ndarray(a, order='c', index_spec=[0,1,2])
    (a[:,0], a[:,1], a[:,2])

    >>> _extract_xyz_from_ndarray(a.T, order='fortran', index_spec=[0,1,2])
    (a[0,:], a[1,:], a[2,:])

    >>> b = np.random.rand(10,8)
    >>> _extract_xyz_from_ndarray(a, order='c', index_spec=[0,4,7])
    (a[0,:], a[4,:], a[7,:])
    """
    if (
        isinstance(obj, (np.ndarray, u.Quantity))
        and _is_array_like(obj) and obj.ndim == 2
    ):
        if len(index_spec) != 3:
            raise ValueError(
                'index_spec must be a tuple[int, int, int]!'
            )
        ax0, ax1, ax2 = index_spec
        if order.lower() == 'c':
            return obj[:,ax0], obj[:,ax1], obj[:,ax2]
        else:
            return obj[ax0,:], obj[ax1,:], obj[ax2,:]
    else:
        raise ValueError(
            'input arrays must be 2D with at least 3 axes! '
            'ie. np.random.rand(10,3) or a list of such.'
        )


@overload
def _cycle(data: list[T], i: int) -> T: ...

@overload
def _cycle(data: tuple[T, ...], i: int) -> T: ...

@overload
def _cycle(data: NDArray, i: int) -> Any: ...

@overload
def _cycle(data: Sequence[T], i: int) -> T: ...

def _cycle(data, i):
    """
    Cycle through a list continuously. When
    the bounds are reached, the index is reset
    to zero.

    This function is meant to be called inside
    of a loop, when lists of different lengths
    need to be iterated upon concurently.

    Parameters
    ----------
    data : list[T]
        Input data list.
    i : int
        Loop index.
    j : int
        Offset to add onto `i`. For internal
        cycling uses.

    Returns
    -------
    T :
        `data` element.
    """
    if not isinstance(data, (Sequence, np.ndarray)):
        raise ValueError(
            'data must be a Sequence or NDArray! '
            f'got {type(data).__name__}'
        )
    return data[int(i + j) % len(data)]


@overload
def _unwrap_if_single(
    array: list[T]
) -> T | list[T]: ...

@overload
def _unwrap_if_single(
    array: tuple[T, ...]
) -> T | tuple[T, ...]: ...

@overload
def _unwrap_if_single(
    array: Any
) -> Any: ...

def _unwrap_if_single(
    array: Sequence[T] | NDArray[Any]
) -> T | Sequence[T] | NDArray[Any]:
    """
    Unwrap an array-like object if it contains exactly one element.

    If the input has length 1, the sole element is returned.
    Otherwise, the input is returned unchanged. This is primarily
    intended for user-facing APIs that return either a single object
    or a collection depending on the number of results.

    Parameters
    ----------
    array : Sequence[T]
        A sequence-like object supporting `len()` and indexing.
        Must have at least one element.

    Returns
    -------
    T or Sequence[T]
        The sole element if `len(array) == 1`, otherwise the original
        input sequence.
    """
    if isinstance(array, (Sequence, np.ndarray)):
        return array[0] if len(array) == 1 else array
    return array
