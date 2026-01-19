'''
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2025-10-17
Description:
    Numerical utility functions.
Dependencies:
    - astropy
    - numpy
    - scipy
    - spectral_cube
Module Structure:
    - Type Checking Arrays and Objects
        Utility functions for type checking.
    - Science Operation Functions
        Utility functions related to scientific operations.
    - Numerical Operation Functions
        Utility functions related to numerical computations.
'''

from typing import Any
from astropy import units as u
from astropy.units import Quantity
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.interpolate import interp1d, CubicSpline
from spectral_cube import SpectralCube


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
    return obj.data if hasattr(obj, 'data') else obj


def get_value(obj: Any) -> NDArray:
    """
    Return the underlying NumPy array from
    objects exposing a `value` attribute.

    If the object does not expose `value`,
    it is treated as the value itself. The
    result is returned using np.asarray().

    Parameters
    ----------
    obj : any
        An object that may expose a `value` attribute
        (e.g. an Astropy `Quantity`) or is array-like.

    Returns
    -------
    numpy.ndarray
        The extracted data as a NumPy array.
    """
    value = obj.value if hasattr(obj, 'value') else obj
    return np.asarray(value)


def finite(obj, *, keep_units=True, keep_inf=False):
    """
    Filter NaN and optionally infinite values from array-like input.

    Parameters
    ----------
    obj : array_like
        Input data. May be a NumPy array, list, DataCube, FitsFile,
        Quantity, or any object compatible with `to_array`.
    keep_units : bool, optional, default=True
        If True, preserve astropy units if present on the input.
    keep_inf : bool, optional, default=False
        If True, keep ±inf values and remove only NaNs.
        If False, remove NaN and ±inf values.

    Returns
    -------
    ndarray or Quantity
        A 1-D array containing the filtered values. Units are preserved
        if `keep_units=True` and the input carries units.

    Notes
    -----
    - Filtering is performed using `np.isfinite` when `keep_inf=False`,
        and `~np.isnan` when `keep_inf=True`.
    """
    data = to_array(obj, keep_units)
    mask = mask_finite(data, keep_inf=keep_inf)

    return data[mask]


def to_array(obj, keep_units=False):
    """
    Return input object as either a np.ndarray or Quantity.

    Parameters
    ----------
    obj : array-like or Quantity
        Array or DataCube object.
    keep_units : bool, optional, default=False
        If True, keep astropy units attached if present.

    Returns
    -------
    array : np.ndarray
        Quantity array if `keep_units` is True, else a NumPy array.
    """
    if isinstance(obj, Quantity):
        return obj if keep_units else obj.value

    elif isinstance(obj, SpectralCube):
        q = obj.filled_data[:]
        return q if keep_units else q.value

    if hasattr(obj, 'value'):
        value = obj.value
        unit = getattr(obj, 'unit', None)
        if keep_units and unit is not None:
            if isinstance(value, Quantity):
                return value
            return value * unit

        return np.asarray(value)

    if hasattr(obj, 'data'):
        return to_array(obj.data, keep_units=keep_units)

    try:
        return np.asarray(obj)
    except Exception:
        raise TypeError(
            f'Object of type {type(obj).__name__} cannot be converted to an array'
        )


# Science Operation Functions
# ---------------------------
def compute_density_kde(x, y, bw_method='scott', resolution=200, padding=0.2):
    '''
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
    resolution : int, optional, default=200
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
    '''
    # compute bounds with % padding
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    padding_x = (xmax - xmin) * padding
    padding_y = (ymax - ymin) * padding

    # generate grid
    xgrid, ygrid = np.mgrid[
        (xmin - padding_x):(xmax + padding_x):complex(resolution),
        (ymin - padding_y):(ymax + padding_y):complex(resolution)
    ]
    # KDE evaluation
    values = np.vstack([x, y])
    grid = np.vstack([xgrid.ravel(), ygrid.ravel()])
    kernel = stats.gaussian_kde(values, bw_method=bw_method)
    Z = np.reshape(kernel(grid), xgrid.shape)

    return xgrid, ygrid, Z


def shift_by_radial_vel(spectral_axis, radial_vel):
    '''
    Shift spectral axis to rest frame using a radial velocity.

    Parameters
    ----------
    spectral_axis : astropy.units.Quantity
        The spectral axis to shift. Can be in frequency or wavelength units.
    radial_vel : float, astropy.units.Quantity or None
        Radial velocity in km/s (astropy units are optional). Positive values
        correspond to a redshift (moving away). If None, no shift is applied.

    Returns
    -------
    shifted_axis : astropy.units.Quantity
        The spectral axis shifted to the rest frame according to the given
        radial velocity. If the input is in frequency units, the classical
        Doppler formula for frequency is applied; otherwise, the classical
        formula for wavelength is applied.
    '''
    # speed of light in km/s in vacuum
    c = 299792.458 # [km/s]
    if radial_vel is not None:
        if isinstance(radial_vel, Quantity):
            radial_vel = radial_vel.to(u.km/u.s).value # type: ignore
        # if spectral axis in units of frequency
        if spectral_axis.unit.is_equivalent(u.Unit('Hz')):
            spectral_axis /= (1 - radial_vel / c)
        # if spectral axis in units of wavelength
        else:
            spectral_axis /= (1 + radial_vel / c)

    return spectral_axis


# Numerical Operation Functions
# -----------------------------
def interpolate(xp, yp, x_range, N_samples, method='linear'):
    '''
    Interpolate a 1D array over a specified range.

    Parameters
    ----------
    xp : array-like
        The x-coordinates of the data points.
    yp : array-like
        The y-coordinates of the data points.
    x_range : tuple of float
        The (min, max) range over which to interpolate.
    N_samples : int
        Number of points in the interpolated output.
    method : str, default='linear'
        Interpolation method. Options:
        - 'linear' : linear interpolation
        - 'cubic' : cubic interpolation using 'interp1d'
        - 'cubic_spline' : cubic spline interpolation using 'CubicSpline'

    Returns
    -------
    x_interp : np.ndarray
        The evenly spaced x-coordinates over the specified range.
    y_interp : np.ndarray
        The interpolated y-values corresponding to 'x_interp'.
    '''
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

    return x_interp, y_interp


def mask_finite(obj, *, keep_inf=False):
    """
    Return a boolean mask identifying finite values in array-like input.

    Parameters
    ----------
    obj : array_like
        Input data. May be a NumPy array, list, DataCube, FitsFile,
        Quantity, or any object compatible with `to_array`.
    keep_units : bool, optional, default=True
        Passed through to `to_array`. Units do not affect the mask.
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
    '''
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
    '''
    x = get_value(x)
    xlim = get_value(xlim)

    xmin = xlim[0] if xlim is not None else np.nanmin(x)
    xmax = xlim[1] if xlim is not None else np.nanmax(x)

    mask = (x >= xmin) & (x <= xmax)

    return mask
