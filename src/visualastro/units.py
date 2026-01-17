'''
Author: Elko Gerville-Reache
Date Created: 2025-12-10
Date Modified: 2026-01-17
Description:
    Utility functions for astropy units.
Dependencies:
    - astropy
    - numpy
'''


from astropy.io.fits import Header
from astropy.units import (
    dimensionless_unscaled, Quantity, Unit, UnitBase, UnitsError
)
import numpy as np


def get_units(obj):
    """
    Extract the unit from an object, if it exists.

    This function checks if the object has a .unit attribute,
    or if 'BUNIT' exists in `obj`.

    Parameters
    ----------
    obj : Object
        The input object from which to extract a unit. This can be:
        - an astropy.units.Quantity
        - a fits.Header with a 'BUNIT' key
        - any object with a .data attribute
        - any object with a .header attribute
    Returns
    -------
    astropy.units.Unit or None
        The unit associated with the input object, if it exists.
        Returns None if the object has no unit or if the unit cannot be parsed.
    """
    if isinstance(obj, Quantity):
        return obj.unit

    if isinstance(obj, Header):
        bunit = obj.get('BUNIT')
        if isinstance(bunit, str):
            try:
                return Unit(bunit)
            except Exception:
                return None
        return None

    unit = getattr(obj, 'unit', None)
    if unit is not None:
        try:
            return unit if isinstance(unit, UnitBase) else Unit(unit)
        except Exception:
            pass

    if hasattr(obj, 'data'):
        data = obj.data
        if data is not obj:
            unit = get_units(data)
            if unit is not None:
                return unit

    if hasattr(obj, 'header'):
        header = obj.header
        if header is not obj:
            unit = get_units(header)
            if unit is not None:
                return unit

    return None


def _is_unitless(obj):
    """
    Validate that an object has no unit.

    u.dimensionless_unscaled is treated
    as no units.

    Parameters
    ----------
    obj : object
        Object to check if unitless.

    Returns
    -------
    bool :
        If object has a unit.
    """
    if isinstance(obj, Quantity):
        return obj.unit == dimensionless_unscaled

    unit = getattr(obj, 'unit', None)
    if isinstance(unit, UnitBase):
        return unit == dimensionless_unscaled

    return True


def _check_unit_equality(unit1, unit2, name1='unit1', name2='unit2'):
    """
    Validate that two units are exactly equal.

    Parameters
    ----------
    unit1, unit2 : str or astropy.units.Unit or None
        Units to compare. None means 'unitless'.
    name1, name2 : str
        Labels used in error messages.

    Raises
    ------
    UnitsError
        If units differ (either convertible or incompatible).
    """
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


def _get_physical_type(obj):
    '''
    Extract the physical_type attribute from an object with
    a unit attribute. Returns None if no units.

    Parameters
    ----------
    obj : Quantity or Unit
        Object with a .unit attribute. Custom data types
        are permitted as long as the .unit is a Astropy Unit.

    Returns
    -------
    physical_type : astropy.units.physical.PhysicalType or None
        Physical type of the unit or None if no units are found.
    '''

    if isinstance(obj, Quantity):
        unit = obj.unit
        if isinstance(unit, UnitBase):
            return unit.physical_type
        return None

    elif isinstance(obj, UnitBase):
        return obj.physical_type

    elif hasattr(obj, 'unit'):
        return obj.unit.physical_type

    return None


def _validate_units_consistency(objs, *, label=None):
    '''
    Extract units of each object in objs
    and validate that units match.

    Parameters
    ----------
    obj : array-like
        A single object or list/array of objects with unit data.
        Can be Quantities, Headers with 'BUNIT', or a mix of both.

    Returns
    -------
    None
        If no units are present.
    astropy.units.Unit
        If units are present and are consistent.

    Raises
    ------
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
        elif hasattr(obj, 'unit'):
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
        prefix = f'{label} ' if label is not None else ''
        raise UnitsError(
            f'Inconsistent {prefix}units found: {units}'
        )

    # return either single unit or None
    return next(iter(units), None)
