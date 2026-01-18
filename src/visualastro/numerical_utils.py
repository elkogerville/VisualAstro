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

from astropy import units as u
from astropy.units import Quantity
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d, CubicSpline
from spectral_cube import SpectralCube


# Type Checking Arrays and Objects
# --------------------------------
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


def get_data(obj):
    '''
    Extract the underlying data attribute from a DataCube or FitsFile object.
    Parameters
    ----------
    obj : DataCube or FitsFile or np.ndarray
        The object from which to extract the data. If a raw array is provided,
        it is returned unchanged.
    Returns
    -------
    np.ndarray, or data extension
        The data attribute contained in the object, or the input array itself
        if it is not a DataCube or FitsFile.
    '''
    if hasattr(obj, 'data'):
        return obj.data

    return obj


def return_array_values(array):
    '''
    Extract the numerical values from an 'astropy.units.Quantity'
    or return the array as-is.

    Parameters
    ----------
    array : astropy.units.Quantity or array-like
        The input array. If it is a Quantity, the numerical values are extracted.
        Otherwise, the input is returned unchanged.

    Returns
    -------
    np.ndarray or array-like
        The numerical values of the array, without units if input was a Quantity,
        or the original array if it was not a Quantity.
    '''
    array = array.value if isinstance(array, Quantity) else array

    return array


def non_nan(obj, keep_units=False):
    '''
    Return the input data with all NaN values removed.

    Parameters
    ----------
    obj : array_like
        Input array or array-like object. This may be a NumPy
        array, list, DataCube, FitsFile, or Quantity.
    keep_units : bool, optional, default=False
        If True, keep astropy units attached if present.

    Returns
    -------
    ndarray
        A 1-D array containing only the non-NaN elements from `obj`.

    Notes
    -----
    This function converts the input to a NumPy array using
    `to_array(obj)`, then removes entries where the value is NaN.
    If the input contains units (e.g., an `astropy.units.Quantity`),
    the returned object will retain the original units.
    '''
    data = to_array(obj, keep_units)
    non_nans = ~np.isnan(data)

    return data[non_nans]


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
    # compute bounds with 20% padding
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
def interpolate_arrays(xp, yp, x_range, N_samples, method='linear'):
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
        # fallback to linear if method is unknown
        kind = method if method in ['linear', 'cubic'] else 'linear'
        f_interp = interp1d(xp, yp, kind=kind)
    # interpolate over new samples
    y_interp = f_interp(x_interp)

    return x_interp, y_interp


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
    x = return_array_values(x)
    xlim = return_array_values(xlim)

    xmin = xlim[0] if xlim is not None else np.nanmin(x)
    xmax = xlim[1] if xlim is not None else np.nanmax(x)

    mask = (x >= xmin) & (x <= xmax)

    return mask
