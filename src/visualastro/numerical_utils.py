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

import warnings
from astropy import units as u
from astropy.io import fits
from astropy.io.fits import Header
from astropy.units import (
    physical, Quantity, spectral, Unit, UnitBase, UnitConversionError
)
import numpy as np
from reproject import reproject_interp, reproject_exact
from scipy import stats
from scipy.interpolate import interp1d, CubicSpline
from spectral_cube import SpectralCube
from .visual_classes import DataCube, ExtractedSpectrum, FitsFile, va_config
from .va_config import get_config_value, _default_flag


# Type Checking Arrays and Objects
# ––––––––––––––––––––––––––––––––
def check_is_array(data, keep_units=False):
    '''
    Ensure array input is np.ndarray.
    Parameters
    ––––––––––
    data : np.ndarray, DataCube, FitsFile, or Quantity
        Array or DataCube object.
    keep_units : bool, optional, default=False
        If True, keep astropy units attached if present.
    Returns
    –––––––
    data : np.ndarray
        Array or 'data' component of DataCube.
    '''
    if isinstance(data, DataCube):
        if keep_units:
            return data.value * data.unit
        else:
            data = data.value
    elif isinstance(data, FitsFile):
        if keep_units:
            return data.data * data.unit
        else:
            data = data.data
    if isinstance(data, Quantity):
        if keep_units:
            return data
        else:
            data = data.value

    return np.asarray(data)


def check_units_consistency(datas):
    '''
    Check that all input objects have the same units and warn if they differ.
    Additionally ensure that the input is iterable by wrapping in a list.
    Parameters
    ––––––––––
    datas : object or list/tuple of objects
        Objects to check. Can be Quantity, SpectralCube, DataCube, etc.

    Returns
    –––––––
    datas : list
        The input objects as a list.
    '''
    datas = datas if isinstance(datas, (list, tuple)) else [datas]

    first_unit = get_units(datas[0])

    for i, obj in enumerate(datas[1:], start=1):
        unit = get_units(obj)
        if unit != first_unit:
            warnings.warn(
                f"\nInput at index {i} has unit `{unit}`, which differs from unit `{first_unit}`."
                f"at index 0. Values may be plotted incorrectly..."
            )


    return datas


def get_data(obj):
    '''
    Extract the underlying data attribute from a DataCube or FitsFile object.
    Parameters
    ––––––––––
    obj : DataCube or FitsFile or np.ndarray
        The object from which to extract the data. If a raw array is provided,
        it is returned unchanged.
    Returns
    –––––––
    np.ndarray, or data extension
        The data attribute contained in the object, or the input array itself
        if it is not a DataCube or FitsFile.
    '''
    if isinstance(obj, DataCube):
        obj = obj.data
    elif isinstance(obj, FitsFile):
        obj = obj.data

    return obj


def get_physical_type(obj):
    '''
    Extract the physical_type attribute from an object with
    a unit attribute. Returns None if no units.
    Parameters
    ––––––––––
    obj : Quantity or Unit
        Object with a .unit attribute. Custom data types
        are permitted as long as the .unit is a Astropy Unit.

    Returns
    –––––––
    physical_type : astropy.units.physical.PhysicalType or None
        Physical type of the unit or None if no units are found.
    '''

    if isinstance(obj, Quantity):
        return obj.unit.physical_type # type: ignore
    elif isinstance(obj, UnitBase):
        return obj.physical_type
    elif hasattr(obj, 'unit'):
        return obj.unit.physical_type
    else:
        return None


def get_units(obj):
    '''
    Extract the unit from an object, if it exists.
    Parameters
    ––––––––––
    obj : Quantity, SpectralCube, FITS-like object, or any
        The input object from which to extract a unit. This can be:
        - an astropy.units.Quantity
        - a SpectralCube
        - a DataCube or FitsFile
        - a FITS-like object with a header containing a 'BUNIT' keyword
        - any other object (returns None if no unit is found)
    Returns
    –––––––
    astropy.units.Unit or None
        The unit associated with the input object, if it exists.
        Returns None if the object has no unit or if the unit cannot be parsed.
    '''
    # check if spectral unit
    spectral_unit = getattr(obj, 'spectral_unit', None)
    if spectral_unit is not None:
        try:
            if spectral_unit.physical_type in {
                physical.frequency, physical.length, physical.speed, physical.energy
            }:
                return spectral_unit
        except Exception:
            pass

    # check if object has unit attribute
    unit = getattr(obj, 'unit', None)
    if unit is not None:
        return unit

    # try obj.data.unit
    data = getattr(obj, 'data', None)
    if data is not None:
        unit = getattr(data, 'unit', None)
        if unit is not None:
            return unit

    # check if ExtractedSpectrum
    if isinstance(obj, ExtractedSpectrum):
        for attr in ('spectrum1d', 'flux'):
            unit = getattr(getattr(obj, attr, None), 'unit', None)
            if unit is not None:
                return unit
        spectrum1d = getattr(obj, 'spectrum1d', None)
        if spectrum1d is not None:
            unit = getattr(getattr(spectrum1d, 'flux', None), 'unit', None)
            if unit is not None:
                return unit

    # try to extract unit from header
    # use either header extension or obj if obj is a header
    header = getattr(obj, 'header', obj if isinstance(obj, Header) else None)
    if isinstance(header, Header) and 'BUNIT' in header:
        try:
            return Unit(header['BUNIT'])
        except Exception:
            return None

    return None


def quantities_2_array(values):
    '''
    Convert a list of scalar Quantity objects into a single 1D Quantity array.
    Guaranteed to return shape (N,) even if inputs have shape (1,).

    Parameters
    ----------
    values : array-like of Quantity
        An array-like object where each element is a Quantity.

    Returns
    -------
    flattened_quantity : Quantity
        A 1D Quantity array with shape (N,).
    '''

    flat = []
    for v in values:
        if not isinstance(v, Quantity):
            raise TypeError(
                'All elements must be astropy Quantity objects.'
            )

        arr = np.asarray(v)

        # flatten if shape is (1,)
        if arr.shape == (1,):
            flat.append(arr[0] * v.unit)
        else:
            flat.append(v)

    return u.Quantity(flat)


def return_array_values(array):
    '''
    Extract the numerical values from an 'astropy.units.Quantity'
    or return the array as-is.
    Parameters
    ––––––––––
    array : astropy.units.Quantity or array-like
        The input array. If it is a Quantity, the numerical values are extracted.
        Otherwise, the input is returned unchanged.
    Returns
    –––––––
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
    ––––––––––
    obj : array_like
        Input array or array-like object. This may be a NumPy
        array, list, DataCube, FitsFile, or Quantity.
    keep_units : bool, optional, default=False
        If True, keep astropy units attached if present.
    Returns
    –––––––
    ndarray
        A 1-D array containing only the non-NaN elements from `obj`.

    Notes
    –––––
    This function converts the input to a NumPy array using
    `check_is_array(obj)`, then removes entries where the value is NaN.
    If the input contains units (e.g., an `astropy.units.Quantity`),
    the returned object will retain the original units.
    '''
    data = check_is_array(obj, keep_units)
    non_nans = ~np.isnan(data)

    return data[non_nans]

# Science Operation Functions
# –––––––––––––––––––––––––––
def compute_density_kde(x, y, bw_method='scott', resolution=200, padding=0.2):
    '''
    Estimate the 2D density of a set of particles using a Gaussian KDE.
    Parameters
    ––––––––––
    x : np.ndarray
        1D array of x-coordinates of shape (N,).
    y : np.ndarray
        1D array of y-coordinates of shape (N,).
    bw_method : str, scalar or callable, optional, default='scott'
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
    –––––––
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


def convert_units(quantity, unit):
    '''
    Convert an Astropy Quantity to a specified unit, with a fallback if conversion fails.
    Parameters
    ––––––––––
    quantity : astropy.units.Quantity
        The input quantity to convert.
    unit : str, astropy.units.Unit, or None
        The unit to convert to. If None, no conversion is performed.
    Returns
    –––––––
    astropy.units.Quantity
        The quantity converted to the requested unit if possible; otherwise,
        the original quantity with its existing unit.
    Notes
    –––––
    - Uses 'spectral()' equivalencies to allow conversions between
        wavelength, frequency, and velocity units.
    - If conversion fails, prints a warning and returns the original quantity.
    '''
    if unit is None:
        return quantity
    try:
        # convert string unit to Unit if necessary
        target_unit = Unit(unit) if isinstance(unit, str) else unit
        return quantity.to(target_unit, equivalencies=spectral())
    except UnitConversionError:
        print(
            f'Could not convert to unit: {unit}.'
            f'Defaulting to unit: {quantity.unit}.'
            )
        return quantity


def reproject_wcs(
    input_data,
    reference_wcs,
    method=None,
    return_footprint=None,
    parallel=None,
    block_size=_default_flag
):
    '''
    Reproject data arrays or DataCubes/FitsFile objects onto a reference WCS.

    Parameters
    ––––––––––
    input_data : array-like, DataCube, FitsFile, list, or tuple
        The input data to be reprojected. May be:
        - A HDUList object
        - A `(np.ndarray, WCS)` or `(np.ndarray, Header)` tuple
        - A `DataCube` or `FitsFile` object containing `.value` and either `.wcs` or `.header`
        - A list containing any of the above inputs
        Note:
            - [np.ndarray, WCS/Header] is not allowed! Ensure they are all tuples.
            - [(np.ndarray, WCS), DataCube, FitsFile] is a valid input.
    reference_wcs : astropy.wcs.WCS or astropy.io.fits.Header
        The target WCS or FITS header to which `input_data` will be reprojected.
    method : {'interp', 'exact'} or None, default=None
        Reprojection method:
        - 'interp' : use `reproject_interp`
        - 'exact' : use `reproject_exact`
        If None, uses the default value
        set by `va_config.reproject_method`.
    return_footprint : bool or None, optional, default=None
        If True, return both reprojected data and reprojection
        footprints. If False, return only the reprojected data.
        If None, uses the default value set by `va_config.return_footprint`.
    parallel : bool, int, str, or None, optional, default=None
        If True, the reprojection is carried out in parallel,
        and if a positive integer, this specifies the number
        of threads to use. The reprojection will be parallelized
        over output array blocks specified by `block_size` (if the
        block size is not set, it will be determined automatically).
        If None, uses the default value set by `va_config.reproject_parallel`.
    block_size : tuple, ‘auto’, or None, optional, default=`_default_flag`
        The size of blocks in terms of output array pixels that each block
        will handle reprojecting. Extending out from (0,0) coords positively,
        block sizes are clamped to output space edges when a block would extend
        past edge. Specifying 'auto' means that reprojection will be done in
        blocks with the block size automatically determined. If `block_size` is
        not specified or set to None, the reprojection will not be carried out in blocks.
        If `_default_flag`, uses the default value set by `va_config.reproject_block_size`.

    Returns
    –––––––
    reprojected : ndarray or list of ndarray
        The reprojected data array(s). If `input_data` contains
        multiple items, the output is a list; otherwise a single array.
    footprint : ndarray or list of ndarray
        The reprojection footprint(s). Only returned if `return_footprint=True`.
        If `input_data` contains multiple items, the output is a list;
        otherwise a single array.

    Raises
    ––––––
    ValueError
        If a `DataCube` or `FitsFile` object has neither `.wcs` nor `.header`
        attribute available.

    Examples
    ––––––––
    Reproject a single array:

    >>> reproject_wcs((data, wcs), reference_wcs)

    Reproject a list of DataCube objects:

    >>> reproject_wcs([cube1, cube2], reference_wcs, method='exact')
    '''
    method = get_config_value(method, 'reproject_method')
    return_footprint = get_config_value(return_footprint, 'return_footprint')
    parallel = get_config_value(parallel, 'reproject_parallel')
    block_size = va_config.reproject_block_size if block_size is _default_flag else block_size

    # normalize input into a list
    if isinstance(input_data, list):
        input_list = [_normalize_reproject_input(item) for item in input_data]
    else:
        input_list = [_normalize_reproject_input(input_data)]

    # Select reproject function
    reproject_method = {
        'interp': reproject_interp,
        'exact': reproject_exact
    }.get(method, reproject_interp)

    reprojected_data = []
    footprints = []

    for i, data in enumerate(input_list):

        # Extract (value, wcs/header) from custom objects
        if isinstance(data, (DataCube, FitsFile)):

            # priority: wcs > header
            for attr in ('wcs', 'header'):
                wcs = getattr(data, attr, None)
                if wcs is not None:
                    break
            else:
                wcs = None

            if wcs is None:
                if len(input_list) == 1:
                    raise ValueError("input_data has no wcs/header information!")
                else:
                    raise ValueError(f"input_data[{i}] has no wcs/header information!")

            data = (data.value, wcs)

        # Run reprojection
        reprojected, footprint = reproject_method(
                                            data,
                                            reference_wcs,
                                            parallel=parallel,
                                            block_size=block_size)

        reprojected_data.append(reprojected)
        footprints.append(footprint)

    # Unwrap single-element lists
    if len(reprojected_data) == 1:
        reprojected_data = reprojected_data[0]
        footprints = footprints[0]

    return (reprojected_data, footprints) if return_footprint else reprojected_data


def _normalize_reproject_input(input_data):
    '''
    Ensures that the reprojection inputs are one of the accepted types.
    Parameters
    ––––––––––
    input_data : fits.HDUList, tuple, or object
        Input data to be reprojected. Must follow one of the formats:
            - fits.HDUList : hdul object containing data and header
            - tuple : tuple containing (data, header/WCS)
            - object : object containing a `.value` and `.header`/.`wcs` attribute

    Returns
    –––––––
    input_data : fits.HDUList, tuple, or object
        The exact same input_data object

    Raises
    ––––––
    TypeError
        If `input_data` is not an accepted type/format.
    '''
    if isinstance(input_data, fits.HDUList):
        return input_data
    elif isinstance(input_data, tuple) and len(input_data) == 2:
        return input_data
    elif hasattr(input_data, 'value') and (hasattr(input_data, 'wcs') or hasattr(input_data, 'header')):
        return input_data
    else:
        raise TypeError(
            'Each input must be an HDUList, a (data, header/WCS) tuple, '
            'or an object with `.value` and `.wcs` or `.header`.'
        )


def shift_by_radial_vel(spectral_axis, radial_vel):
    '''
    Shift spectral axis to rest frame using a radial velocity.
    Parameters
    ––––––––––
    spectral_axis : astropy.units.Quantity
        The spectral axis to shift. Can be in frequency or wavelength units.
    radial_vel : float, astropy.units.Quantity or None
        Radial velocity in km/s (astropy units are optional). Positive values
        correspond to a redshift (moving away). If None, no shift is applied.
    Returns
    –––––––
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
# –––––––––––––––––––––––––––––
def interpolate_arrays(xp, yp, x_range, N_samples, method='linear'):
    '''
    Interpolate a 1D array over a specified range.
    Parameters
    ––––––––––
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
    –––––––
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
    ––––––––––
    x : array-like
        Data array (e.g., wavelength or flux values)
    xlim : tuple or list, optional
        (xmin, xmax) range. If None, uses the min/max of x.
    Returns
    –––––––
    mask : ndarray of bool
        True where x is within the limits.
    '''
    x = return_array_values(x)
    xlim = return_array_values(xlim)

    xmin = xlim[0] if xlim is not None else np.nanmin(x)
    xmax = xlim[1] if xlim is not None else np.nanmax(x)

    mask = (x >= xmin) & (x <= xmax)

    return mask
