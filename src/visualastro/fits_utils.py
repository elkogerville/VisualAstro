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


def _update_header_key(key, value, header, primary_header=None):
    '''
    Update header(s) in place with a new key-value pair.

    Parameters
    ----------
    key : str
        FITS header keyword to update (e.g., 'BUNIT', 'CTYPE1').
    value : str or Unit or any FITS-serializable value
        New value for the keyword.
    header : Header or list[Header]
        The header(s) to update in place.
    primary_header : Header
        Header used for logging the key update.
        If None, uses `header` for logging.

    Returns
    -------
    None
    '''
    try:
        value_str = value.to_string()
    except AttributeError:
        value_str = str(value)

    if isinstance(header, Header):
        headers = [header]
    elif isinstance(header, (list, np.ndarray, tuple)):
        if len(header) == 0:
            raise ValueError('Header list is empty.')
        headers = list(header)
    else:
        raise TypeError(
            'header must be a Header or an array-like of Headers.'
        )

    if primary_header is None:
        primary_header = headers[0]

    old_value = primary_header.get(key, 'DNE')

    for hdr in headers:
        hdr[key] = value_str

    if len(headers) == 1:
        msg = f'Updated {key}: {old_value} -> {value_str}'
    else:
        msg = (
            f'Updated {key} across all slices: {old_value} -> {value_str}'
        )

    _log_history(primary_header, msg)


def _get_history(header):
    '''
    Get `HISTORY` cards from a Header as a list.

    Parameters
    ----------
    header : Header
        Fits Header with `HISTORY` cards.

    Returns
    -------
    list or None :
        all `HISTORY` cards or None if no entries.
    '''
    if not isinstance(header, Header) or 'HISTORY' not in header:
        return None

    history = header['HISTORY']

    if isinstance(history, str):
        return [history]

    return list(history) # type: ignore


def _log_history(header, description):
    '''
    Add a `HISTORY` entry to header
    in place. A timestamp is included.

    Parameters
    ----------
    header : astropy.Header
        Header for logging a `HISTORY` card.
    description : str
        Description of log.
    '''
    timestamp = Time.now().isot
    log = f'{timestamp} {description}'

    header.add_history(log)


def _transfer_history(header1, header2):
    '''
    Transfer `HISTORY` cards from one
    header to another. This is not a
    destructive action.

    Parameters
    ----------
    header1 : Header
        Fits Header with `HISTORY` cards to send.
    header2 : Header
        Fits Header to copy `HISTORY` cards to.

    Returns
    -------
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


def _copy_headers(headers):
    '''
    copy a single or list of fits.Header.

    Parameters
    ----------
    headers : fits.Header or array-like of fits.Header
        Header(s) to be copied.

    Returns
    -------
    fits.Header or list of fits.Header
    '''

    if isinstance(headers, Header):
        return headers.copy()

    elif (
        isinstance(headers, (list, np.ndarray, tuple))
        and isinstance(headers[0], Header)
    ):
        return [hdu.copy() for hdu in headers]
    else:
        raise ValueError(
            'Invalid header(s) inputs!'
        )
