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
from astropy.units import Unit
import numpy as np


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
