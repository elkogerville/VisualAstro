import astropy.units as u
from astropy.units import spectral
import numpy as np
from .numerical_utils import get_data


def header_2_array(cube, key):
    headers = cube.header
    array = []
    for i in range(len(headers)):
        array.append(headers[i][key])

    return np.asarray(array)

# Cube Manipulation Functions
# ___________________________
def extract_spectral_axis(cube, unit=None):
    '''
    Extract the spectral axis from a data cube and optionally
    convert it to a specified unit.
    Parameters
    ––––––––––
    cube : SpectralCube
        The input spectral data cube.
    unit : astropy.units.Unit, optional
        Desired unit for the spectral axis. If None, the axis
        is returned in its native units.
    Returns
    –––––––
    spectral_axis : Quantity
        The spectral axis of the cube, optionally converted
        to the specified unit.
    '''
    axis = cube.spectral_axis
    # return axis if unit is None
    if unit is None:
        return axis
    # if a unit is specified, attempt to
    # convert axis to those units
    try:
        return axis.to(unit, equivalencies=spectral())
    except u.UnitConversionError:
        raise ValueError(f"Cannot convert spectral axis from {axis.unit} to {unit}")

def return_cube_slice(cube, idx):
    '''
    Return a slice of a data cube along the first axis.
    Parameters
    ––––––––––
    cube : np.ndarray
        Input data cube, typically with shape (T, N, ...) where T is the first axis.
    idx : int or list of int
        Index or indices specifying the slice along the first axis:
        - i -> returns 'cube[i]'
        - [i] -> returns 'cube[i]'
        - [i, j] -> returns 'cube[i:j+1].sum(axis=0)'
    Returns
    –––––––
    cube : np.ndarray
        Sliced cube with shape (N, ...).
    '''
    cube = get_data(cube)
    # if index is integer
    if isinstance(idx, int):
        return cube[idx]
    # if index is list of integers
    elif isinstance(idx, list):
        # list of len 1
        if len(idx) == 1:
            return cube[idx[0]]
        # list of len 2
        elif len(idx) == 2:
            start, end = idx
            return cube[start:end+1].sum(axis=0)
    raise ValueError("'idx' must be an int or a list of one or two integers")

def get_spectral_slice_value(spectral_axis, idx):
    '''
    Return a representative value from a spectral axis
    given an index or index range.
    Parameters
    ––––––––––
    spectral_axis : Quantity
        The spectral axis (e.g., wavelength, frequency, or
        velocity) as an 'astropy.units.Quantity' array.
    idx : int or list of int
        Index or indices specifying the slice along the first axis:
        - i -> returns 'spectral_axis[i]'
        - [i] -> returns 'spectral_axis[i]'
        - [i, j] -> returns '(spectral_axis[i] + spectral_axis[j+1])/2'
    Returns
    –––––––
    spectral_value : float
        The spectral value at the specified index or index
        range, in the units of 'spectral_axis'.
    '''
    if isinstance(idx, int):
        return spectral_axis[idx].value
    elif isinstance(idx, list):
        if len(idx) == 1:
            return spectral_axis[idx[0]].value
        elif len(idx) == 2:
            return (spectral_axis[idx[0]].value + spectral_axis[idx[1]+1].value)/2
    raise ValueError("'idx' must be an int or a list of one or two integers")

# Cube Masking Functions
# ––––––––––––––––––––––
def compute_line(points):
    '''
    Compute the slope and intercept of a line passing through two points.
    Parameters
    ––––––––––
    points : list or tuple of tuples
        A sequence containing exactly two points, each as (x, y), e.g.,
        [(x0, y0), (x1, y1)].
    Returns
    –––––––
    m : float
        Slope of the line.
    b : float
        Intercept of the line (y = m*x + b).
    Notes
    –––––
    - The function assumes the two points have different x-coordinates.
    - If the x-coordinates are equal, a ZeroDivisionError will be raised.
    '''
    m = (points[0][1] - points[1][1]) / (points[0][0] - points[1][0])
    b = points[0][1] - m*points[0][0]

    return m, b
