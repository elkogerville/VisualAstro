'''
Author: Elko Gerville-Reache
Date Created: 2025-12-10
Date Modified: 2025-12-10
Description:
    Utility functions for validating inputs and type checks.
Dependencies:
    - astropy
'''
from astropy.units import Quantity
import numpy as np


def _check_shapes_match(a, b, name_a='a', name_b='b'):
    '''
    Check that two input arrays have the same shape.

    Parameters
    ----------
    a : array-like
    b : array-like
    name_a : str, optional, default='a'
        Name to use for array 'a' in error messages.
    name_b : str, optional, default='b'
        Name to use for array 'b' in error messages.

    Returns
    -------
    None :
        Returns None if the shapes match,
        otherwise raises a ValueError.

    Raises
    ------
    ValueError :
        Arrays have differing shapes.
    '''

    A = np.asarray(a)
    B = np.asarray(b)
    if A.shape != B.shape:
        raise ValueError(
            f'Shape mismatch: {name_a}.shape={A.shape}, {name_b}.shape={B.shape}'
        )


def _validate_type(
    data, types, default=None, allow_none=True, name='data'
):
    '''
    Validate that `data` is an instance of one of the allowed types.

    Parameters
    ----------
    data : object
        The object to validate.
    types : type or tuple of types
        A type or tuple of types that `data` is allowed to be.
        Ex: Quantity, (Quantity), or (Quantity, SpectralCube)
    default : object, optional, default=None
        Value to return if `data` is None. Use this to provide
        a default instance when None is passed.
    allow_none : bool, default=True
        If True, None is a valid input. If False, None will raise TypeError.
    name : str
        Name of object for error message.

    Raises
    ------
    TypeError
        If `data` is not an instance of any of the types in `types`.
    '''
    if data is None and allow_none:
        return default

    # make iterable
    if not isinstance(types, tuple):
        types = (types,)

    if not isinstance(data, types):
        allowed = ', '.join(t.__name__ for t in types)
        raise TypeError(
            f"'{name}' must be one of: {allowed}; got {type(data).__name__}."
        )

    return data


def _allclose(a, b):
    '''
    Determine whether two array-like objects are equal within a tolerance,
    with additional handling for `astropy.units.Quantity` and None.
    This function behaves like `numpy.allclose`, but adds logic to safely
    compare Quantities (ensuring matching units) and to treat None as
    a valid sentinel value.

    Parameters
    ----------
    a, b : array-like, `~astropy.units.Quantity`, scalar, or None
        The inputs to compare. Inputs may be numerical arrays, scalars, or
        `Quantity` objects with units. If one argument is None, the result is
        False unless both are None.

    Returns
    -------
    bool
        True if the inputs are considered equal, False otherwise.
        Equality rules:
        - Both None → True
        - One None → False
        - Quantities with mismatched units → False
        - Quantities with identical units → value arrays compared via
            `numpy.allclose`
        - Non-Quantity arrays/scalars → compared via `numpy.allclose`

    Notes
    -----
    - This function does **not** attempt unit conversion.
        Quantities must already share identical units.
    - This function exists to support `.update()` logic where user-supplied
        wavelength/flux arrays should only trigger recomputation if they
        differ from stored values.
    '''
    # case 1: both are None → equal
    if a is None and b is None:
        return True

    # case 2: only one is None → different
    if a is None or b is None:
        return False

    # case 3: one is Quantity, one is not
    if isinstance(a, Quantity) != isinstance(b, Quantity):
        return False

    # case 4: both Quantities
    if isinstance(a, Quantity) and isinstance(b, Quantity):
        if a.unit != b.unit:
            return False
        return np.allclose(a.value, b.value)

    # case 5: both unitless arrays/scalars
    return np.allclose(a, b)
