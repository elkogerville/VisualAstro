'''
Author: Elko Gerville-Reache
Date Created: 2025-12-10
Date Modified: 2025-12-10
Description:
    Utility functions for Astropy Fits files.
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


def with_updated_header_key(key, value, header, primary_header):
    '''
    Returns a copy of header(s) with updated key.

    Parameters
    ––––––––––
    key : str
        FITS header keyword to update (e.g., 'BUNIT', 'CTYPE1').
    value : str or Unit or any FITS-serializable value
        New value for the keyword.
    header : Header or list[Header]
        The header(s) to update.
    primary_header : Header
        The primary header (for logging original unit).

    Returns
    –––––––
    Header or list[Header] or None
        A copy of the input header(s) with the updated keyword.
        '''
    try:
        value_str = value.to_string()
    except AttributeError:
        value_str = str(value)

    old_value = 'unknown'
    if isinstance(primary_header, Header):
        old_value = primary_header.get(key, 'unknown')

    # case 1: single Header
    if isinstance(header, Header):
        new_hdr = header.copy()
        new_hdr[key] = value_str

        log_history(
            new_hdr, f'Updated {key}: {old_value} -> {value_str}'
        )

    # case 2: header is list of Headers
    elif isinstance(header, (list, np.ndarray, tuple)):
        new_hdr = [h.copy() for h in header]

        for hdr in new_hdr:
            hdr[key] = value_str

        log_history(
            new_hdr[0],
            f'Updated {key} across all slices: {old_value} -> {value_str}'
        )

    # case 3: no valid Header
    else:
        new_hdr = None

    return new_hdr
