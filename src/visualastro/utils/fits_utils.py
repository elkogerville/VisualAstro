"""
Author: Elko Gerville-Reache
Date Created: 2025-12-10
Date Modified: 2026-06-26
"""

from collections.abc import Sequence
from typing import overload

from astropy.io.fits import Header
from astropy.time import Time
import astropy.units as u
import numpy as np
from specutils import SpectralRegion

from visualastro.core.units import _require_spectral_region


@overload
def _copy_headers(headers: Header) -> Header: ...

@overload
def _copy_headers(headers: Sequence[Header]) -> list[Header]: ...

def _copy_headers(headers: Header | Sequence[Header]) -> Header | list[Header]:
    """
    Copy a single or `Sequence` of `fits.Header`.

    Parameters
    ----------
    headers : fits.Header | Sequence[fits.Header]
        Header(s) to be copied.

    Returns
    -------
    fits.Header | list[fits.Header]
    """

    if isinstance(headers, Header):
        return headers.copy()

    elif (
        isinstance(headers, (list, np.ndarray, tuple))
        and all(isinstance(header, Header) for header in headers)
    ):
        return [hdu.copy() for hdu in headers]

    else:
        raise ValueError(
            'Invalid header(s) inputs!'
        )


def _get_history(header: Header | Sequence[Header]) -> list | None:
    """
    Get `HISTORY` cards from a Header as a list.
    If `header` is a `Sequence` of `Header`, only
    the `HISTORY` cards in `header[0]` are transfered.

    Parameters
    ----------
    header : Header | Sequence[Header]
        Fits Header(s) with `HISTORY` cards. If a `Sequence`,
        only the first header is queried.

    Returns
    -------
    list | None :
        all `HISTORY` cards or `None` if no entries.
    """
    if _is_sequence_of_headers(header):
        header = header[0]

    if not isinstance(header, Header) or 'HISTORY' not in header:
        return None

    history = header['HISTORY']
    if history is None:
        return None

    if isinstance(history, str):
        return [history]

    return list(history)


def _log_history(
    header: Header | Sequence[Header],
    description: str
) -> None:
    """
    Add a `HISTORY` entry to header
    in place. A timestamp is included.
    If `header` is a list of Headers,
    the `HISTORY` is only added to the
    first header.

    Parameters
    ----------
    header : fits.Header | Sequence[fits.Header]
        Header for logging a `HISTORY` card.
    description : str
        Description of log.
    """
    timestamp = Time.now().isot
    log = f'{timestamp} {description}'

    if isinstance(header, Header):
        header.add_history(log)

    elif (
        isinstance(header, (list, np.ndarray, tuple)) and
        isinstance(header[0], Header)
    ):
        header[0].add_history(log)

    else:
        raise ValueError(
            'header must be a Header or list of Headers!'
        )


def _remove_history(header: Header | Sequence[Header]) -> None:
    """
    Remove all `HISTORY` cards from a header in place.

    Parameters
    ----------
    header : fits.Header | Sequence[fits.Header]
        Header(s) with `HISTORY` cards to remove.
    """
    if isinstance(header, (list, tuple, np.ndarray)):
        for h in header:
            _remove_history(h)
        return

    if isinstance(header, Header):
        while 'HISTORY' in header:
            header.remove('HISTORY')


def _transfer_history(
    sender_header: Header | Sequence[Header],
    reciever_header: Header | Sequence[Header]
) -> Header | list[Header]:
    """
    Transfer `HISTORY` cards from one `sender_header`
    to `reciever_header`.

    If `sender_header` is a `Sequence`, only the first
    header is queried. If `reciever_header` is a `Sequence`,
    the HISTORY is written to `reciever_header[0]`.

    This is not a destructive action.

    Parameters
    ----------
    sender_header : Header | Sequence[Header]
        Fits Header with `HISTORY` cards to send.
    reciever_header : Header | Sequence[Header]
        Fits Header to copy `HISTORY` cards to.

    Returns
    -------
    header2 : Header
        Fits Header with updated `HISTORY`.
    """
    sender_history = _get_history(sender_header)

    if isinstance(reciever_header, Sequence):
        if not _is_sequence_of_headers(reciever_header):
            raise ValueError(
                'Non Header objects detected in reciever_header!'
            )
        target_header = reciever_header[0]
    else:
        target_header = reciever_header

    if sender_history is not None:
        for history in sender_history:
            target_header.add_history(history)

    return list(reciever_header)


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
    region = _require_spectral_region(region)
    unit = region.lower.unit.to_string('fits')

    subregions = region.subregions
    if subregions is None:
        raise ValueError('SpectralRegion.subregions is None!')

    parts = [f'({lo.value:.2f}, {hi.value:.2f})' for lo, hi in subregions]

    return f'region[{unit}]: ' + ', '.join(parts)


def _update_header_key(
    key: str,
    value: str | u.UnitBase | u.StructuredUnit,
    header: Header | Sequence[Header],
    primary_header: Header | None = None
) -> None:
    """
    Update header(s) in place with a new key-value pair.

    Parameters
    ----------
    key : str
        FITS header keyword to update (e.g., 'BUNIT', 'CTYPE1').
    value : str | u.UnitBase | u.StructuredUnit | any FITS-serializable value
        New value for the keyword.
    header : Header | Sequence[Header]
        The header(s) to update in place.
    primary_header : Header | None
        Header used for logging the key update.
        If None, uses `header` for logging.

    Returns
    -------
    None
    """
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


def _is_sequence_of_headers(headers: Header | Sequence[Header]) -> bool:
    """Check that a list contains only `Header` objects."""
    if (
        isinstance(headers, (list, tuple)) and
        all(isinstance(h, Header) for h in headers)
    ):
        return True
    return False
