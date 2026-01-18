'''
Author: Elko Gerville-Reache
Date Created: 2025-12-10
Date Modified: 2026-01-17
Description:
    Utility functions for validating inputs and type checks.
Dependencies:
    - astropy
'''
from collections.abc import Iterable, Iterator, Sequence
from typing import Tuple, Type, TypeVar
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

T = TypeVar('T')
C = TypeVar('C', bound=Sequence)

def _validate_type(
    data: T | None,
    types: Type[T] | Tuple[Type[T], ...],
    default: T | None = None,
    allow_none: bool = True,
    name: str = 'data'
) -> T | None:
    """
    Validate that `data` is an instance of one of the allowed types.

    Parameters
    ----------
    data : T or None
        The object to validate. Can be any type.
    types : type[T] or tuple of type[T]
        A type or tuple of types that `data` is allowed to be.
        For example: `Quantity` or `(Quantity, SpectralCube)`.
    default : T or None, optional, default=None
        Value to return if `data` is None and `allow_none` is True.
        The default value must be an instance of one of the allowed types.
    allow_none : bool, default=True
        If True, None inputs return `default`.
        If False, passing None raises a TypeError.
    name : str
        Name of the object for use in error messages.

    Returns
    -------
    data
        The validated object. Guaranteed to be an instance of one of the
        types specified by `types` if no exception is raised.

    Raises
    ------
    TypeError
        If `data` is not an instance of any of the types in `types`.
    """
    if not isinstance(types, tuple):
        types = (types,)

    if data is None:
        if not allow_none:
            raise TypeError(f"'{name}' cannot be None.")
        if default is None:
            return None
        if not isinstance(default, types):
            allowed = ', '.join(t.__name__ for t in types)
            raise TypeError(
                f"'default' must be one of: {allowed}; got {type(default).__name__}."
            )
        return default

    if not isinstance(data, types):
        allowed = ', '.join(t.__name__ for t in types)
        raise TypeError(
            f"'{name}' must be one of: {allowed}; got {type(data).__name__}."
        )

    return data


def _validate_iterable_type(
    obj: C,
    types: Type[T] | Tuple[Type[T], ...],
    name: str = 'object',
) -> C:
    """
    Validate that `obj` is a non-iterator iterable whose elements are all
    instances of the given type(s).

    This function is intended for collections such as lists or tuples.
    Iterator-like objects and string-like objects are rejected. If
    validation succeeds, the input object is returned unchanged.

    Parameters
    ----------
    obj : iterable
        A non-iterator iterable (e.g. list or tuple) to validate.
    types : type or tuple of types
        Required type(s) for each element of `obj`.
    name : str, optional
        Name used in error messages.

    Returns
    -------
    obj : iterable
        The validated iterable. Guaranteed to be the same object passed in
        and to contain only instances of `types`.

    Raises
    ------
    TypeError
        If `obj` is not iterable, is an iterator, is a string or bytes
        object, or if any element of `obj` is not an instance of `types`.
    """
    if not isinstance(types, tuple):
        types = (types,)

    if (
        not isinstance(obj, Iterable) or
        isinstance(obj, (Iterator, str, bytes))
    ):
        raise TypeError(
            f'{name} must be a non-iterator iterable (e.g. list or tuple), '
            f'got {type(obj).__name__}.'
        )

    allowed = ', '.join(t.__name__ for t in types)
    for i, item in enumerate(obj):
        if not isinstance(item, types):
            raise TypeError(
                f'Element {i} of {name} must be instance of {allowed}, '
                f'got {type(item).__name__}.'
            )

    return obj
