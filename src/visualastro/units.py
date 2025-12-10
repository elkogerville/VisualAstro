'''
Author: Elko Gerville-Reache
Date Created: 2025-12-10
Date Modified: 2025-12-10
Description:
    Utility functions for astropy units.
Dependencies:
    - astropy
'''


from astropy.io.fits import Header
from astropy.units import Quantity, Unit, UnitsError


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
