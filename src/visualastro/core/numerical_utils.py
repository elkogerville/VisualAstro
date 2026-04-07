"""
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2026-03-11
Description:
    Numerical utility functions.
Dependencies:
    - astropy
    - numpy
    - scipy
    - spectral-cube
Module Structure:
    - Type Checking Arrays and Objects
        Utility functions for type checking.
    - Science Operation Functions
        Utility functions related to scientific operations.
    - Numerical Operation Functions
        Utility functions related to numerical computations.
"""

from typing import Any, Literal, Sequence, TypeVar, overload
from astropy import units as u
from astropy.units import Quantity
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.interpolate import interp1d, CubicSpline
from spectral_cube import SpectralCube
from visualastro.core.validation import _type_name


T = TypeVar('T')

# Type Checking Arrays and Objects
# --------------------------------
def get_data(obj):
    """
    Return the `data` attribute of an object if present;
    otherwise return the object unchanged.

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
    if isinstance(obj, (np.ndarray, Quantity)):
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

    If the object exposes a ``value`` attribute
    (e.g., an Astropy ``Quantity``), that attribute
    is returned. Otherwise, the object itself is
    returned unchanged.

    Parameters
    ----------
    obj : any
        Object that may expose a ``value`` attribute.

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
def to_array(obj: Any, keep_unit: Literal[True]) -> NDArray | Quantity: ...

@overload
def to_array(obj: Any, keep_unit: bool) -> NDArray | Quantity: ...

def to_array(obj: Any, keep_unit: bool = False) -> NDArray | Quantity:
    """
    Return input object as either a np.ndarray or Quantity.

    Parameters
    ----------
    obj : array-like, np.ndarray, Quantity or SpectralCube
        Any array-like object, or an object that exposes
        a ``data`` or ``value`` attribute.
    keep_unit : bool, optional, default=False
        If True, keep astropy units attached if present.

    Returns
    -------
    array : np.ndarray
        Quantity array if `keep_unit` is True, else a NumPy array.

    Raises
    ------
    TypeError :
        If obj is None.
    """
    if obj is None:
        raise TypeError('None cannot be converted to an array')

    if isinstance(obj, Quantity):
        return obj if keep_unit else np.asarray(obj.value)

    elif isinstance(obj, SpectralCube):
        q = obj.filled_data[:]
        if not isinstance(q, Quantity):
            q = Quantity(np.asarray(q), unit=obj.unit)
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
                if keep_unit and not isinstance(result, Quantity):
                    unit = getattr(obj, 'unit', None) or getattr(inner, 'unit', None)
                    if unit is not None:
                        return Quantity(result, unit=unit)

                return result

    try:
        return np.asarray(obj)
    except Exception:
        raise TypeError(
            f'Object of type {_type_name(obj)} cannot be converted to an array'
        )

@overload
def to_list(obj: list[T]) -> list[T]: ...

@overload
def to_list(obj: tuple[T, ...]) -> list[T]: ...

@overload
def to_list(obj: T) -> list[T]: ...

def to_list(obj: Any) -> list:
    """
    Normalize input to a list.

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
    return obj if isinstance(obj, list) else (
        list(obj) if isinstance(obj, tuple) else [obj]
    )


# Science Operation Functions
# ---------------------------
def kde2d(x, y, bw_method='scott', gridsize=200, padding=0.2):
    """
    Estimate the 2D density of a set of particles using a Gaussian KDE.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-coordinates of shape (N,).
    y : np.ndarray
        1D array of y-coordinates of shape (N,).
    bw_method : {'scott', 'silverman'}, scalar or callable, optional, default='scott'
        The method used to calculate the bandwidth factor for the Gaussian KDE.
        Can be one of:
        - 'scott' or 'silverman': use standard rules of thumb.
        - a scalar constant: directly used as the bandwidth factor.
        - a callable: should take a `scipy.stats.gaussian_kde` instance as its
            sole argument and return a scalar bandwidth factor.
    gridsize : int, optional, default=200
        Grid resolution for the KDE.
    padding : float, optional, default=0.2
        Fractional padding applied to the data range when generating
        the evaluation grid, expressed as a fraction of the total span
        along each axis. For example, a value of `0.2` expands the grid
        limits by 20% beyond the minimum and maximum of the data in both
        `x` and `y` directions. This helps capture the tails of the
        Gaussian kernel near the plot boundaries.

    Returns
    -------
    xgrid : np.ndarray
        2D array of x-coordinates for the evaluation grid (shape res×res).
    ygrid : np.ndarray
        2D array of y-coordinates for the evaluation grid (shape res×res).
    Z : np.ndarray
        2D array of estimated density values on the grid (shape res×res).
    """
    # compute bounds with % padding
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    padding_x = (xmax - xmin) * padding
    padding_y = (ymax - ymin) * padding

    # generate grid
    xgrid, ygrid = np.mgrid[
        (xmin - padding_x):(xmax + padding_x):complex(gridsize),
        (ymin - padding_y):(ymax + padding_y):complex(gridsize)
    ]
    # KDE evaluation
    values = np.vstack([x, y])
    grid = np.vstack([xgrid.ravel(), ygrid.ravel()])
    kernel = stats.gaussian_kde(values, bw_method=bw_method)
    Z = np.reshape(kernel(grid), xgrid.shape)

    return xgrid, ygrid, Z


# Numerical Operation Functions
# -----------------------------
def flatten(data: Any) -> NDArray | None:
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


@overload
def interpolate(
    xp: Quantity,
    yp: Quantity,
    x_range: Sequence | NDArray,
    N_samples: int,
    method: Literal['linear', 'cubic', 'cubic_spline'] = 'linear'
) -> tuple[Quantity, Quantity]: ...

@overload
def interpolate(
    xp: NDArray,
    yp: NDArray,
    x_range: Sequence | NDArray,
    N_samples: int,
    method: Literal['linear', 'cubic', 'cubic_spline'] = 'linear'
) -> tuple[NDArray, NDArray]: ...

def interpolate(
    xp: NDArray | Quantity,
    yp: NDArray | Quantity,
    x_range: Sequence | NDArray,
    N_samples: int,
    method: Literal['linear', 'cubic', 'cubic_spline'] = 'linear'
) -> tuple[NDArray | Quantity, NDArray | Quantity]:
    """
    Interpolate a 1D array over a specified range.

    Parameters
    ----------
    xp : array-like
        The x-coordinates of the data points. Must be 1D.
    yp : array-like
        The y-coordinates of the data points. Must be 1D.
    x_range : tuple of float
        The (min, max) range over which to interpolate.
    N_samples : int
        Number of points in the interpolated output.
    method : {'linear', 'cubic', 'cubic_spline'}, default='linear'
        Interpolation method. Options:
        - ``'linear'`` : linear interpolation
        - ``'cubic'`` : cubic interpolation using ``interp1d``
        - ``'cubic_spline'`` : cubic spline interpolation using ``CubicSpline``

    Returns
    -------
    x_interp : np.ndarray
        The evenly spaced x-coordinates over the specified range.
    y_interp : np.ndarray
        The interpolated y-values corresponding to ``x_interp``.
    """
    x_unit = xp.unit if isinstance(xp, Quantity) else None
    y_unit = yp.unit if isinstance(yp, Quantity) else None
    xp = np.asarray(xp)
    yp = np.asarray(yp)

    if xp.ndim != 1 or yp.ndim != 1:
        raise ValueError("'xp' and 'yp' must be 1D arrays")

    if len(xp) != len(yp):
        raise ValueError(
            f"'xp' and 'yp' must have the same length. "
            f"Got xp: {len(xp)}, yp: {len(yp)}"
        )
    if len(xp) < 2:
        raise ValueError(f'need at least 2 points for interpolation, got{len(xp)}')

    if not isinstance(N_samples, (int, np.integer)) or N_samples < 2:
        raise ValueError(f"'N_samples' must be an integer >= 2, got {N_samples}")

    if not isinstance(x_range, (Sequence, np.ndarray)) or len(x_range) != 2:
        raise ValueError(f"'x_range' must be a tuple of (min, max), got {x_range}")
    if x_range[0] >= x_range[1]:
        raise ValueError(
            f"'x_range' must be (min, max) with min < max, "
            f'got ({x_range[0]}, {x_range[1]})'
        )

    valid_methods = {'linear', 'cubic', 'cubic_spline'}
    if method not in valid_methods:
        raise ValueError(
            f"'method' must be one of {valid_methods}, got '{method}'"
        )

    # generate new interpolation samples
    x_interp = np.linspace(x_range[0], x_range[1], N_samples)

    # get interpolation method
    if method == 'cubic_spline':
        f_interp = CubicSpline(xp, yp)
    else:
        kind = method if method in ['linear', 'cubic'] else 'linear'
        f_interp = interp1d(xp, yp, kind=kind)

    # interpolate over new samples
    y_interp = f_interp(x_interp)

    if x_unit is not None:
        x_interp *= x_unit
    if y_unit is not None:
        y_interp *= y_unit

    return x_interp, y_interp


def percent_difference(a, b):
    """
    Compute the percent difference between two arrays.

    The percent difference is defined as the absolute difference between
    ``a`` and ``b`` divided by their mean, expressed as a percentage:

        percent_difference = |a - b| / ((a + b) / 2) * 100

    Parameters
    ----------
    a : array-like
        First input array.
    b : array-like
        Second input array. Must be broadcastable with ``a``.

    Returns
    -------
    numpy.ndarray
        Percent difference between ``a`` and ``b``, element-wise.
        Returns ``nan`` where both ``a`` and ``b`` are zero.

    Notes
    -----
    Uses ``numpy.errstate`` to suppress division by zero and invalid
    value warnings. Elements where the mean of ``a`` and ``b`` is zero
    will produce ``nan`` in the output.

    Examples
    --------
    >>> percent_difference(1.0, 2.0)
    66.666...
    >>> percent_difference(np.array([1, 2, 3]), np.array([2, 2, 4]))
    array([66.666...,  0.    , 28.571...])
    >>> percent_difference(0.0, 0.0)
    nan
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = (a + b) / 2
    with np.errstate(invalid='ignore', divide='ignore'):
        result = (np.abs(a - b) / denominator) * 100
    return result


def finite(obj, *, keep_unit=True, keep_inf=False):
    """
    Filter NaN and optionally infinite values from
    array-like input. The output is always 1D.

    Parameters
    ----------
    obj : array_like
        Input data. May be a NumPy array, list, DataCube, FitsFile,
        Quantity, or any object compatible with `to_array`.
    keep_unit : bool, optional, default=True
        If True, preserve astropy units if present on the input.
    keep_inf : bool, optional, default=False
        If True, keep ±inf values and remove only NaNs.
        If False, remove NaN and ±inf values.

    Returns
    -------
    ndarray or Quantity
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


def mask_finite(obj, *, keep_inf=False):
    """
    Return a boolean mask identifying finite values in array-like input.

    Parameters
    ----------
    obj : array_like
        Input data. May be a NumPy array, list, DataCube, FitsFile,
        Quantity, or any object compatible with `to_array`.
    keep_inf : bool, optional, default=False
        If False, mask excludes NaN and ±inf values.
        If True, mask excludes only NaNs and retains ±inf values.

    Returns
    -------
    ndarray of bool
        Boolean mask with the same shape as the input data.
        True indicates values that are kept.

    Notes
    -----
    - Uses `np.isfinite` when `keep_inf=False`.
    - Uses `~np.isnan` when `keep_inf=True`.
    """
    data = to_array(obj)
    return ~np.isnan(data) if keep_inf else np.isfinite(data)


def mask_within_range(x, xlim=None):
    """
    Return a boolean mask for values of x within the given limits.

    Parameters
    ----------
    x : array-like
        Data array (e.g., wavelength or flux values)
    xlim : tuple or list, optional
        (xmin, xmax) range. If None, uses the min/max of x.

    Returns
    -------
    mask : ndarray of bool
        True where x is within the limits.
    """
    x = get_value(x)

    xmin = get_value(xlim[0]) if xlim is not None else np.nanmin(x)
    xmax = get_value(xlim[1]) if xlim is not None else np.nanmax(x)

    mask = (x >= xmin) & (x <= xmax)

    return mask


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
