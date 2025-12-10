'''
Author: Elko Gerville-Reache
Date Created: 2025-12-10
Date Modified: 2025-12-10
Description:
    Utility functions for astropy units.
Dependencies:
    - astropy
    - numpy
'''


from astropy.io.fits import Header
from astropy.units import Quantity, Unit, UnitsError
import numpy as np


def _check_unit_equality(unit1, unit2, name1="unit1", name2="unit2"):
    '''
    Validate that two units are exactly equal.

    Parameters
    ––––––––––
    unit1, unit2 : str or astropy.units.Unit or None
        Units to compare. None means 'unitless'.
    name1, name2 : str
        Labels used in error messages.

    Raises
    ––––––
    UnitsError
        If units differ (either convertible or incompatible).
    '''
    # case 1: either of the units are None
    if unit1 is None or unit2 is None:
        return

    try:
        u1 = Unit(unit1)
        u2 = Unit(unit2)
    except Exception:
        raise UnitsError('Invalid unit(s) supplied')

    # case 1: units are exactly equal
    if u1 == u2:
        return

    # case 2: equivalent but not equal
    if u1.is_equivalent(u2):
        raise UnitsError(
            f'{name1} and {name2} units are equivalent but not equal '
            f'({u1} vs {u2}). Convert one to match.'
        )

    # case 3: mismatch
    raise UnitsError(
        f'{name1} and {name2} have incompatible units: '
        f'{u1} vs {u2}.'
    )


def get_common_units(objs):
    '''
    Extract units of each object in objs
    and validate that units match.

    Parameters
    ––––––––––
    obj : array-like
        A single object or list/array of objects with unit data.
        Can be Quantities, Headers with 'BUNIT', or a mix of both.

    Returns
    –––––––
    None
        If no units are present.
    astropy.units.Unit
        If units are present and are consistent.

    Raises
    ––––––
    UnitsError
        If units exist and do not match, or if BUNIT is invalid.
    '''
    if not np.iterable(objs) or isinstance(objs, (Header, Quantity)):
        objs = [objs]

    # create unique set of each unit
    units = set()
    for i, obj in enumerate(objs):
        if isinstance(obj, Quantity):
            units.add(obj.unit)
        elif isinstance(obj, Header) and 'BUNIT' in obj:
            try:
                units.add(Unit(obj['BUNIT'])) # type: ignore
            except Exception as e:
                raise UnitsError(
                    f'Invalid BUNIT in header at index {i}: '
                    f"'{obj['BUNIT']}' ({e})"
                )
    # raise error if more than one unit found
    if len(units) > 1:
        raise UnitsError(
            f'Inconsistent units found: {units}'
        )

    # return either single unit or None
    return next(iter(units), None)
