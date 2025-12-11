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


def update_header_key(key, value, header, primary_header):
    '''
    Update header(s) in place with a new key-value pair.

    Parameters
    ––––––––––
    key : str
        FITS header keyword to update (e.g., 'BUNIT', 'CTYPE1').
    value : str or Unit or any FITS-serializable value
        New value for the keyword.
    header : Header or list[Header]
        The header(s) to update in place.
    primary_header : Header
        The primary header (for logging original unit).

    Returns
    –––––––
    None
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
        header[key] = value_str
        _log_history(
            primary_header, f'Updated {key}: {old_value} -> {value_str}'
        )

    # case 2: header is list of Headers
    elif isinstance(header, (list, np.ndarray, tuple)):
        for hdr in header:
            hdr[key] = value_str
        _log_history(
            header[0],
            f'Updated {key} across all slices: {old_value} -> {value_str}'
        )


def with_updated_header_key(key, value, header, primary_header):
    '''
    Returns a copy of header(s) with a new key-value pair.

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

        _log_history(
            new_hdr, f'Updated {key}: {old_value} -> {value_str}'
        )
        return new_hdr

    # case 2: header is list of Headers
    elif isinstance(header, (list, np.ndarray, tuple)):
        new_hdr = [h.copy() for h in header]

        for hdr in new_hdr:
            hdr[key] = value_str

        _log_history(
            new_hdr[0],
            f'Updated {key} across all slices: {old_value} -> {value_str}'
        )
        return new_hdr

    # case 3: no valid Header
    else:
        return None


def _get_history(header):
    '''
    Get `HISTORY` cards from a Header as a list.

    Parameters
    ––––––––––
    header : Header
        Fits Header with `HISTORY` cards.

    Returns
    –––––––
    list or None :
        all `HISTORY` cards or None if no entries.
    '''
    if not isinstance(header, Header) or "HISTORY" not in header:
        return None

    history = header["HISTORY"]

    if isinstance(history, str):
        return [history]

    return list(history) # type: ignore


def _log_history(header, message):
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


def _transfer_history(header1, header2):
    '''
    Trasfer `HISTORY` cards from one
    header to another. This is not a
    destructive action.

    Parameters
    ––––––––––
    header1 : Header
        Fits Header with `HISTORY` cards to send.
    header2 : Header
        Fits Header to copy `HISTORY` cards to.

    Returns
    –––––––
    header2 : Header
        Fits Header with updated `HISTORY`.
    '''
    # get logs from header 1
    hdr1_history = _get_history(header1)

    # add logs to header 2
    if hdr1_history is not None:
        for history in hdr1_history:
            header2.add_history(history)

    return header2
