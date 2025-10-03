import warnings
from astropy.io.fits import Header
from astropy import units as u
from astropy.units import Quantity, spectral, Unit, UnitConversionError
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from spectral_cube import SpectralCube
from .visual_classes import DataCube, ExtractedSpectrum, FitsFile


# Type Checking Arrays and Objects
# ––––––––––––––––––––––––––––––––
def check_is_array(data, keep_units=False):
    '''
    Ensure array input is np.ndarray.
    Parameters
    ––––––––––
    data : np.ndarray, DataCube, FitsFile, or Quantity
        Array or DataCube object.
    keep_inits : bool, optional, default=False
        If True, keep astropy units attached if present.
    Returns
    –––––––
    data : np.ndarray
        Array or 'data' component of DataCube.
    '''
    if isinstance(data, DataCube):
        data = data.value
    elif isinstance(data, FitsFile):
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
    ----------
    datas : object or list/tuple of objects
        Objects to check. Can be Quantity, SpectralCube, DataCube, etc.
    Returns
    -------
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
                f"at index 0."
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
    # check if object is DataCube or FitsFile
    data = get_data(obj)
    # check if unit extension exists
    if isinstance(data, (DataCube, FitsFile, Quantity, SpectralCube)):
        return data.unit
    if isinstance(obj, ExtractedSpectrum):
        return obj.spectrum1d.unit

    header = getattr(obj, 'header', None)
    if isinstance(header, Header) and 'BUNIT' in header:
        try:
            return Unit(obj.header['BUNIT'])
        except Exception:
            return None

    return None


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


# Science Operation Functions
# –––––––––––––––––––––––––––
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
    mask = (x > xmin) & (x < xmax)

    return mask
