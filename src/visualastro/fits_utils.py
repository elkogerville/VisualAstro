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
import numpy as np
from specutils import SpectralRegion
from .units import require_spectral_region


def _copy_headers(headers):
    '''
    Copy a single or list of fits.Header.

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
    If `header` is a list of Headers,
    the `HISTORY` is only added to the
    first header.

    Parameters
    ----------
    header : astropy.Header
        Header for logging a `HISTORY` card.
    description : str
        Description of log.
    '''
    timestamp = Time.now().isot
    log = f'{timestamp} {description}'

    if isinstance(header, Header):
        header.add_history(log)

    elif isinstance(header, (list, np.ndarray, tuple)):
        header[0].add_history(log)

    else:
        raise ValueError(
            'header must be a Header or list of Headers!'
        )


def _remove_history(header):
    '''
    Remove any `HISTORY` cards from a header in place.

    Parameters
    ----------
    header : fits.Header or list of fits.Header
        Header(s) with `HISTORY` cards to remove.
    '''
    if isinstance(header, (list, tuple, np.ndarray)):
        for h in header:
            _remove_history(h)
        return

    while 'HISTORY' in header:
        header.remove('HISTORY')


def _transfer_history(header1, header2):
    '''
    Transfer `HISTORY` cards from one header (header1)
    to another (header2). If header2 is a list of headers,
    the HISTORY is written to header2[0].

    This is not a destructive action.

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

    if isinstance(header2, list):
        target_header = header2[0]
    else:
        target_header = header2

    # add logs to header 2
    if hdr1_history is not None:
        for history in hdr1_history:
            target_header.add_history(history)

    return header2


def _region_to_history(region: SpectralRegion) -> str:
    """
    Format a SpectralRegion for inclusion in a FITS HISTORY card.

    Parameters
    ----------
    region : specutils.SpectralRegion
        Spectral region or set of sub-regions to encode.

    Returns
    -------
    history_str : str
        String representation of the spectral region bounds of the form:
            region[unit]: (lo1,hi1), (lo2,hi2), ...
    """
    region = require_spectral_region(region)
    unit = region.lower.unit.to_string('fits')

    subregions = region.subregions
    if subregions is None:
        raise ValueError('SpectralRegion.subregions is None!')

    parts = [f'({lo.value:.2f}, {hi.value:.2f})' for lo, hi in subregions]

    return f'region[{unit}]: ' + ', '.join(parts)


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
