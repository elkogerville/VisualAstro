'''
Author: Elko Gerville-Reache
Date Created: 2025-12-10
Date Modified: 2026-01-17
Description:
    Utility functions for validating inputs and type checks.
Dependencies:
    - astropy
'''
from astropy.units import Quantity
import numpy as np


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=True):
    '''
    Determine whether two array-like objects are equal within a tolerance,
    with additional handling for `astropy.units.Quantity` and None. This
    function behaves like `numpy.allclose`, but adds logic to safely
    compare Quantities (ensuring matching units).

    If the following equation is element-wise True, then allclose returns True:
        absolute(a - b) <= (atol + rtol * absolute(b))

    Parameters
    ----------
    a, b : array-like, `~astropy.units.Quantity`, scalar, or None
        The inputs to compare. Inputs may be numerical arrays, scalars, or
        `Quantity` objects with units. If one argument is None, the result is
        False unless both are None.
    rtol : array_like
        Relative tolerance. Sets how close two values must be as a fraction
        of the reference value `b`. This allows larger absolute differences
        when comparing larger numbers. Increase this when you expect small
        percentage-level differences due to numerical error or algorithmic
        approximations.
    atol : array_like
        Absolute tolerance. Sets the minimum absolute difference below which
        two values are considered equal, regardless of their magnitude. This
        is especially important when comparing values near zero. Increase
        this when small nonzero values should be treated as effectively zero.
    equal_nan : bool, optional, default=True
        Whether to compare NaN’s as equal. If True, NaN’s in a will be
        considered equal to NaN’s in b in the output array.

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
    - allclose(a, b) != allclose(b, a) in some rare cases.

    - The default value of atol is not appropriate when
      the reference value b has magnitude smaller than one.
      i.e. it is unlikely that a = 1e-9 and b = 2e-9 should
      be considered “close”, yet allclose(1e-9, 2e-9) is True
      with default settings. Be sure to select atol for the
      use case at hand, especially for defining the threshold
      below which a non-zero value in a will be considered “close”
      to a very small or zero value in b.
    '''
    # case 1: both are None → equal
    if a is None and b is None:
        return True

    # case 2: only one is None → different
    elif a is None or b is None:
        return False

    # case 3: one is Quantity, one is not
    elif isinstance(a, Quantity) != isinstance(b, Quantity):
        return False

    # case 4: both Quantities
    elif isinstance(a, Quantity) and isinstance(b, Quantity):
        if a.unit != b.unit:
            return False
        return np.allclose(
            a.value, b.value, rtol=rtol,
            atol=atol, equal_nan=equal_nan
        )

    # case 5: both unitless arrays/scalars
    return np.allclose(
        a, b, rtol=rtol, atol=atol, equal_nan=equal_nan
    )


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


def _validate_iterable_type(obj, types, name='object'):
    '''
    Validate that obj is an iterable whose elements are
    all instances of the given type(s).

    Raises a TypeError if the type(s) do not match.

    Parameters
    ----------
    obj : iterable
        Iterable to validate.
    types : type or tuple of types
        Required type(s) for each element.
    name : str
        Name used in error messages.

    Raises
    ------
    ValueError
        If obj is not iterable or contains invalid elements.
    TypeError
        If an element of obj is not the correct type.
    '''
    if not isinstance(types, tuple):
        types = (types,)

    try:
        iterator = iter(obj)
    except TypeError:
        raise ValueError(f'{name} must be an iterable.')

    for i, item in enumerate(iterator):
        if not isinstance(item, types):
            raise TypeError(
                f'Element {i} of {name} must be instance of {types}, '
                f'got {type(item)}.'
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
