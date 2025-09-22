from astropy import units as u
from astropy.units import Quantity
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from .visual_classes import DataCube

def check_is_array(data):
    '''
    Ensure array input is np.ndarray.
    Parameters
    ––––––––––
    data : np.ndarray or DataCube
        Array or DataCube object.
    Returns
    –––––––
    data : np.ndarray
        Array or 'data' component of DataCube.
    '''
    if isinstance(data, DataCube):
        data = data.value

    return np.asarray(data)

def return_array_values(array):
    array = array.value if isinstance(array, Quantity) else array

    return array

def return_cube_data(cube):
    if isinstance(cube, DataCube):
        cube = cube.data
    return cube


def shift_by_radial_vel(spectral_axis, radial_vel):
    '''
    Shift spectral axis by a radial velocity.
    Parameters
    ––––––––––
    spectral_axis : astropy.units.Quantity
        The spectral axis to shift. Can be in frequency or wavelength units.
    radial_vel : float or None
        Radial velocity in km/s (astropy units are not needed). Positive values
        correspond to a redshift (moving away). If None, no shift is applied.
    Returns
    –––––––
    shifted_axis : astropy.units.Quantity
        The spectral axis shifted according to the given radial velocity.
        If the input is in frequency units, the relativistic Doppler
        formula for frequency is applied; otherwise, the formula for
        wavelength is applied.
    '''
    # speed of light in km/s in vacuum
    c = 299792.458 # [km/s]
    if radial_vel is not None:
        # if spectral axis in units of frequency
        if spectral_axis.unit.is_equivalent(u.Hz):
            spectral_axis /= (1 - radial_vel / c)
        # if spectral axis in units of wavelength
        else:
            spectral_axis /= (1 + radial_vel / c)

    return spectral_axis

def interpolate_arrays(xp, yp, x_range, N_samples, method='linear'):
    interpolation_map = {
        'linear': interp1d,
        'cubic': interp1d,
        'cubic_spline': CubicSpline
    }
    interp = interpolation_map.get(method, interp1d)
    x_interp = np.linspace(x_range[0], x_range[1], N_samples)
    if method == 'cubic_spline':
        f_interp = interp(xp, yp)
    else:
        f_interp = interp(xp, yp, kind=method)
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

    xmin = xlim[0] if xlim is not None else np.nanmin(x)
    xmax = xlim[1] if xlim is not None else np.nanmax(x)
    mask = (x > xmin) & (x < xmax)

    return mask
