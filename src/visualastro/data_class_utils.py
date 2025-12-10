'''
Author: Elko Gerville-Reache
Date Created: 2025-12-10
Date Modified: 2025-12-10
Description:
    Utility functions for DataCube and FitsFile.
Dependencies:
    - astropy
    - numpy
'''

from astropy.io.fits import Header
from astropy.time import Time
from astropy.units import Quantity, Unit, UnitsError
import numpy as np


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


def log_history(header, message):
    '''
    Add `HISTORY` entry to header.

    Parameters
    ––––––––––
    header : astropy.Header
    message : str
    '''
    timestamp = Time.now().isot
    log = f'{timestamp} {message}'

    header.add_history(log)


def update_BUNIT(unit, header, primary_header):
    '''
    Update BUNIT in header(s) and log the conversion.

    Parameters
    ––––––––––
    unit : astropy.units.Unit or str
        New unit to set in BUNIT.
    header : Header or list[Header]
        The header(s) to update.
    primary_header : Header
        The primary header (for logging original unit).

    Returns
    –––––––
    fits.Header or list of fits.Header or None
        Updated header(s) with new BUNIT and history entry.
        '''
    if unit is not None:
        unit = Unit(unit)

    old_unit = primary_header.get('BUNIT', 'unknown')

    # case 1: single Header
    if isinstance(header, Header):
        new_hdr = header.copy()
        new_hdr['BUNIT'] = unit.to_string()
        log_history(new_hdr, f'Updated BUNIT: {old_unit} -> {unit}')

    # case 2: header is list of Headers
    elif isinstance(header, (list, np.ndarray, tuple)):
        new_hdr = [h.copy() for h in header]
        for hdr in new_hdr:
            hdr['BUNIT'] = unit.to_string()
        log_history(
            new_hdr[0], f'Updated BUNIT across all slices: {old_unit} -> {unit}'
        )

    # case 3: no valid Header
    else:
        new_hdr = None

    return new_hdr


def validate_type(
    data, types, default=None, allow_none=True, name='data'
):
    '''
    Validate that `data` is an instance of one of the allowed types.

    Parameters
    ––––––––––
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
    ––––––
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
