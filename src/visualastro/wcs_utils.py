"""
Author: Elko Gerville-Reache
Date Created: 2025-12-06
Date Modified: 2025-12-06
Description:
    WCS utility functions.
Dependencies:
    - astropy
    - numpy
Module Structure:
    - Data Transformations
        Lightweight data class for fits files.
"""

import copy
from typing import Any, cast
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import Header
from astropy.nddata import Cutout2D
import astropy.units as u
from astropy.units import Quantity
from astropy.wcs import WCS
import numpy as np
from reproject import reproject_interp, reproject_exact
from spectral_cube import SpectralCube
from spectral_cube.wcs_utils import strip_wcs_from_header
from tqdm import tqdm
from .fits_utils import _log_history
from .utils import _unwrap_if_single
from .config import get_config_value, config, _default_flag


def get_wcs(header: Any) -> WCS | list[WCS] | None:
    """
    Extract WCS from header(s).

    Parameters
    ----------
    header : Header or array-like of Header
        Single Header or array-like of Headers.

    Returns
    -------
    WCS, list of WCS, or None
        - Single ``WCS`` if ``header`` is a ``Header`` and wcs extraction succeeds.
        - List of ``WCS`` if ``header`` is a sequence and *all* headers yield
            valid WCS objects.
        - None if ``header`` is a single ``Header`` and WCS extraction fails,
            or if a sequence is provided and *no* headers yield valid WCS.

    Raises
    ------
    ValueError
        If a sequence of headers is provided and only a subset yield valid WCS.
    """
    if isinstance(header, WCS):
        return header

    if isinstance(header, Header):
        try:
            return WCS(header)
        except Exception:
            return None

    # if a list of headers extract a list of wcs
    if isinstance(header, (list, np.ndarray, tuple)):
        wcs_list: list[WCS | None] = []
        for h in header:
            if not isinstance(h, Header):
                wcs_list.append(None)
                continue
            try:
                wcs_list.append(WCS(h))
            except Exception:
                wcs_list.append(None)

        if all(w is None for w in wcs_list):
            return None
        if any(w is None for w in wcs_list):
            raise ValueError(
                'Inconsistent WCS: some headers have WCS and some do not.'
            )

        return cast(list[WCS], [w for w in wcs_list if w is not None])

    return None


def _is_valid_wcs_slice(key):
    """
    Check if a key is a valid slice for a WCS object.

    Parameters
    ----------
    key : slice or tuple
        key to slice WCS with.

    Returns
    -------
    bool :
        If key is a valid WCS slice.
    """
    if isinstance(key, slice):
        return True

    if isinstance(key, tuple):
        return all(isinstance(k, slice) for k in key)

    return False


# Data Transformations
# --------------------
def crop2D(data, size, position=None, wcs=None, mode='trim', frame='icrs', origin_idx=0):
    """
    Create a Cutout2D from array-like data using WCS and a world/pixel position.

    Parameters
    ----------
    data : array-like or Quantity
        The image to crop. Must be 2D.
    size : Quantity, float, int, or tuple
        Size of the cutout. Interpreted as pixels if unitless.
        Ex:
            - 6 * u.arcsec
            - (6*u.deg, 4*u.deg)
            - (7, 8)
    position : tuple, Quantity tuple, or SkyCoord, optional, default=None
        The center of the cutout region. Accepted formats are:
        - `(x, y)` : pixel coordinates (integers or floats)
        - `(ra, dec)` : sky coordinates as `~astropy.units.Quantity` in angular units
        - `~astropy.coordinates.SkyCoord` : directly specify a coordinate object
        - If None, defaults to the center of the image.
    wcs : astropy.wcs.WCS
        WCS corresponding to `data`. If `data` has an attribute
        `.wcs`, it will be used automatically.
    mode : {'trim', 'partial', 'strict'}, default='trim'
        Defines how the function handles edges that fall outside the image:
        - 'trim': Trim the cutout to fit within the image bounds.
        - 'partial': Include all pixels that overlap the image, padded with NaNs.
        - 'strict': Raise an error if any part of the cutout is outside the image.
    frame : str, default='icrs'
        Coordinate frame for interpreting RA/Dec values when creating the `SkyCoord`.
    origin_idx : int, default=0
        Origin index for pixel-to-world conversion (0 for 0-based, 1 for 1-based).

    Returns
    -------
    Quantity or ndarray
        The cropped region with units if the original data had units.
    WCS
        The WCS of the cropped region

    Raises
    ------
    ValueError
        If the WCS is missing (None) or cutout creation fails.
    TypeError
        If `position` is not a supported type.

    Notes
    -----
    - If the data were transposed and the WCS was swapped via `wcs.swapaxes(0, 1)`,
        the method will automatically attempt to correct for inverted RA/Dec axes.

    """
    if isinstance(wcs, WCS):
        wcs = wcs.celestial
    elif hasattr(data, 'wcs'):
        wcs = data.wcs.celestial

    if wcs is None:
        raise ValueError (
            'WCS is None. Provide a WCS or crop manually with slicing.'
        )

    # default to center of image
    if position is None:
        ny, nx = data.shape
        position = [nx / 2, ny / 2]

    # assume floats and ints are pixel coordinates
    if (
        isinstance(position, (list, np.ndarray, tuple))
        and len(position) == 2
        and isinstance(position[0], (float, int))
        and isinstance(position[1], (float, int))
        ):
        ra, dec = wcs.wcs_pix2world(position[0], position[1], origin_idx)
        center = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame=frame) # type: ignore

    # convert Quantities to degrees
    elif (
        isinstance(position, (list, np.ndarray, tuple))
        and len(position) == 2
        and all(isinstance(p, Quantity) for p in position)
        ):
        p0 = position[0].to(u.deg)
        p1 = position[1].to(u.deg)

        test1 = SkyCoord(ra=p0, dec=p1, frame=frame)
        x1, y1 = wcs.world_to_pixel(test1)

        if np.isfinite(x1) and np.isfinite(y1):
            ra, dec = p0, p1

        else:
            test2 = SkyCoord(ra=p1, dec=p0, frame=frame)
            x2, y2 = wcs.world_to_pixel(test2)

            if np.isfinite(x2) and np.isfinite(y2):
                ra, dec = p1, p0
            else:
                raise ValueError('Could not interpret input as RA/Dec.')

        center = SkyCoord(ra=ra, dec=dec, frame=frame)

    # if position passed in as SkyCoord, use that
    elif isinstance(position, SkyCoord):
        center = position
    else:
        raise TypeError(
            'Position must be a (x, y) pixel tuple, (ra, dec) in degrees, or a SkyCoord.'
        )

    # crop image
    try:
        cutout = Cutout2D(data, position=center, size=size, wcs=wcs, mode=mode)
    except ValueError:
        # fallback if WCS RA/Dec swapped
        center_swapped = SkyCoord(ra=center.dec, dec=center.ra, frame=frame)

        cutout = Cutout2D(data, position=center_swapped, size=size, wcs=wcs, mode=mode)

    crop_data = cutout.data
    crop_wcs = cutout.wcs

    return crop_data, crop_wcs


def _reproject_wcs(
    input_data,
    reference_wcs,
    method=None,
    return_footprint=None,
    parallel=None,
    block_size=_default_flag,
    log_file=None,
    **kwargs
):
    """
    Reproject data or Quantity arrays onto a reference WCS.

    Units are conserved if present in `input_data`.

    Parameters
    ----------
    input_data : tuple
        A `(np.ndarray, WCS)` or `(np.ndarray, Header)` tuple
        Note:
            - [np.ndarray, WCS/Header] is not allowed! Ensure they are all tuples.
    reference_wcs : astropy.wcs.WCS or astropy.io.fits.Header
        The target WCS or FITS header to which `input_data` will be reprojected.
        Dimensional handling:
        Input WCS → Reference WCS
            - 2D → 2D: Direct reprojection
            - 2D → 3D: Uses celestial WCS from 3D target (ignores spectral)
            - 3D → 2D: Reprojects each spectral slice onto 2D target (preserves spectral axis)
            - 3D → 3D: Direct reprojection (spectral axes must be compatible)
    method : {'interp', 'exact'} or None, default=None
        Reprojection method:
            - 'interp' : use `reproject_interp`
            - 'exact' : use `reproject_exact`
        If None, uses the default value
        set by `config.reproject_method`.
    return_footprint : bool or None, optional, default=None
        If True, return both reprojected data and reprojection
        footprints. If False, return only the reprojected data.
        If None, uses the default value set by `config.return_footprint`.
    parallel : bool, int, str, or None, optional, default=None
        If True, the reprojection is carried out in parallel,
        and if a positive integer, this specifies the number
        of threads to use. The reprojection will be parallelized
        over output array blocks specified by `block_size` (if the
        block size is not set, it will be determined automatically).
        If None, uses the default value set by `config.reproject_parallel`.
    block_size : tuple, ‘auto’, or None, optional, default=`_default_flag`
        The size of blocks in terms of output array pixels that each block
        will handle reprojecting. Extending out from (0,0) coords positively,
        block sizes are clamped to output space edges when a block would extend
        past edge. Specifying 'auto' means that reprojection will be done in
        blocks with the block size automatically determined. If `block_size` is
        not specified or set to None, the reprojection will not be carried out in blocks.
        If `_default_flag`, uses the default value set by `config.reproject_block_size`.
    log_file : fits.Header or None, optional, default=None
        If provided, reprojection details are logged to this header's
        HISTORY. Intended for internal use within VisualAstro.
    description : str or None, optional, default=None
        Description message for the progress bar. If None, a default message
        is used. Intended for internal use within VisualAstro.

    Returns
    -------
    reprojected : ndarray
        The reprojected data array.
    footprint : ndarray, optional
        The reprojection footprint. Only returned if `return_footprint=True`.

    Raises
    ------
    ValueError
        If the inputs are not able to be reprojected.
    """
    description = kwargs.get('description', None)

    method = get_config_value(method, 'reproject_method')
    return_footprint = get_config_value(return_footprint, 'return_footprint')
    parallel = get_config_value(parallel, 'reproject_parallel')
    block_size = config.reproject_block_size if block_size is _default_flag else block_size

    if isinstance(reference_wcs, (list, tuple)):
        raise ValueError('reference_wcs must be a single WCS or Header')

    if isinstance(reference_wcs, fits.Header):
        reference_wcs = WCS(reference_wcs)

    if log_file is not None:
        if not isinstance(log_file, Header):
            raise TypeError(
                'log_file must be a Header!'
            )

    obj, unit = _normalize_reproject_input(input_data)

    reproject_func = {
        'interp': reproject_interp,
        'exact': reproject_exact
    }.get(method, reproject_interp)

    if log_file is not None:
        _log_history(
            log_file, f'Reprojection algorithm: reproject_{method}'
        )

    data, wcs = obj
    data_ndim = np.asarray(data).ndim
    ref_ndim = reference_wcs.naxis

    # 3D data -> 2D reference: reproject each slice
    if data_ndim == 3 and ref_ndim == 2:
        wcs_celestial = wcs.celestial if wcs.naxis > 2 else wcs

        reprojects = []
        footprints_slice = []
        desc = 'Looping over each slice' if description is None else description

        for i in tqdm(range(data.shape[0]), desc=desc):
            repr_i, foot_i = reproject_func(
                (data[i], wcs_celestial), reference_wcs,
                parallel=parallel, block_size=block_size
            )
            reprojects.append(repr_i)
            footprints_slice.append(foot_i)

        reprojected = np.stack(reprojects, axis=0)
        footprint = np.stack(footprints_slice, axis=0)
        if log_file is not None:
            _log_history(
                log_file, 'Reprojection mode: 3D data -> 2D reference (slice-by-slice)'
            )

    # single reprojection
    else:
        # determine which WCS to use based on dimensions
        if data_ndim == 2 and ref_ndim > 2:
            # 2D data -> 3+D reference: use celestial slice
            ref_wcs = reference_wcs.celestial
            mode = '2D data -> 3D reference (celestial WCS only)'

        elif data_ndim == ref_ndim:
            # matching dimensions
            ref_wcs = reference_wcs
            mode = f'{data_ndim}D data -> {ref_ndim}D reference'

        else:
            raise ValueError(
                f'Unsupported dimension combination: data_ndim={data_ndim}, '
                f'ref_ndim={ref_ndim}. Supported: (2,2), (2,3), (3,2), (3,3)'
            )

        if log_file is not None:
            _log_history(log_file, f'Reprojection mode: {mode}')

        reprojected, footprint = reproject_func(
            (data, wcs), ref_wcs,
            parallel=parallel, block_size=block_size
        )

    if unit is not None:
        reprojected *= unit

    return (reprojected, footprint) if return_footprint else reprojected


def reproject_wcs(
    input_data_list,
    reference_wcs,
    method=None,
    return_footprint=None,
    parallel=None,
    block_size=_default_flag,
    log_file=None,
    **kwargs
):
    """
    Parameters
    ----------
    input_data_list : tuple or list of tuple
        A single `(np.ndarray, WCS/Header)` tuple or a list of such tuples.
        Note:
            - [np.ndarray, WCS/Header] is not allowed!
              Ensure they follow the format:
                - [(np.ndarray, WCS/Header), ...]
    reference_wcs : astropy.wcs.WCS or astropy.io.fits.Header
        The target WCS or FITS header to which `input_data` will be reprojected.
        Dimensional handling:
        Input WCS → Reference WCS
            - 2D → 2D: Direct reprojection
            - 2D → 3D: Uses celestial WCS from 3D target (ignores spectral)
            - 3D → 2D: Reprojects each spectral slice onto 2D target (preserves spectral axis)
            - 3D → 3D: Direct reprojection (spectral axes must be compatible)
    method : {'interp', 'exact'} or None, default=None
        Reprojection method:
            - 'interp' : use `reproject_interp`
            - 'exact' : use `reproject_exact`
        If None, uses the default value
        set by `config.reproject_method`.
    return_footprint : bool or None, optional, default=None
        If True, return both reprojected data and reprojection
        footprints. If False, return only the reprojected data.
        If None, uses the default value set by `config.return_footprint`.
    parallel : bool, int, str, or None, optional, default=None
        If True, the reprojection is carried out in parallel,
        and if a positive integer, this specifies the number
        of threads to use. The reprojection will be parallelized
        over output array blocks specified by `block_size` (if the
        block size is not set, it will be determined automatically).
        If None, uses the default value set by `config.reproject_parallel`.
    block_size : tuple, ‘auto’, or None, optional, default=`_default_flag`
        The size of blocks in terms of output array pixels that each block
        will handle reprojecting. Extending out from (0,0) coords positively,
        block sizes are clamped to output space edges when a block would extend
        past edge. Specifying 'auto' means that reprojection will be done in
        blocks with the block size automatically determined. If `block_size` is
        not specified or set to None, the reprojection will not be carried out in blocks.
        If `_default_flag`, uses the default value set by `config.reproject_block_size`.
    log_file : fits.Header or None, optional, default=None
        If provided, reprojection details are logged to this header's
        HISTORY. Intended for internal use within VisualAstro.
    description : str or None, optional, default=None
        Description message for the progress bar. If None, a default message
        is used. Intended for internal use within VisualAstro.

    Returns
    -------
    reprojected_data : ndarray or list of ndarray
        Reprojected data. A single array is returned if a single input
        was provided; otherwise a list of arrays.
    footprint : ndarray or list of ndarray, optional
        Reprojection footprint(s), returned only if `return_footprint=True`.

    Raises
    ------
    ValueError
        If the inputs are not able to be reprojected.
    """
    if not isinstance(input_data_list, list):
        input_data_list = [input_data_list]

    if log_file is not None and not isinstance(log_file, Header):
        raise TypeError(
            'log_file must be a fits.Header or None!'
        )
    if log_file is not None and len(input_data_list) > 1:
        _log_history(
            log_file,
            f'Reprojected batch of {len(input_data_list)} datasets'
        )

    reprojected_data = []
    footprints = []

    for i, item in enumerate(input_data_list):
        log_once = log_file if i == 0 else None
        reprojected, footprint = _reproject_wcs(
            item, reference_wcs,
            method=method,
            return_footprint=True,
            parallel=parallel,
            block_size=block_size,
            log_file=log_once,
            **kwargs
        )
        reprojected_data.append(reprojected)
        footprints.append(footprint)

    reprojected_data = _unwrap_if_single(reprojected_data)
    footprints = _unwrap_if_single(footprints)

    return (reprojected_data, footprints) if return_footprint else reprojected_data


def _copy_wcs(wcs):
    """
    Copy a single or list of WCS.

    Parameters
    ----------
    wcs : WCS or array-like of WCS
        WCS(s) to be copied.

    Returns
    -------
    WCS or list of WCS
    """
    if wcs is None:
        return None

    elif isinstance(wcs, WCS):
        return copy.deepcopy(wcs)

    elif (
        isinstance(wcs, (list, np.ndarray, tuple))
        and isinstance(wcs[0], WCS)
    ):
        return [copy.deepcopy(w) for w in wcs]

    else:
        raise ValueError(
            'Invalid wcs(s) inputs!'
        )


def _normalize_reproject_input(input_data):
    """
    Ensures that the reprojection input is a valid (data, WCS) tuple.

    Parameters
    ----------
    input_data : tuple
        Input data as (data, header/WCS) tuple.

    Returns
    -------
    normalized : tuple
        A `(ndarray, WCS)` tuple suitable for reprojection.
    unit : astropy.units.Unit or None
        The unit of the input data, if present.

    Raises
    ------
    TypeError
        If `input_data` is not a valid tuple format.
    TypeError
        If WCS/Header is not a recognized type.
    ValueError
        If `data` in input_data is neither 2D or 3D.
    """
    if not (isinstance(input_data, tuple) and len(input_data) == 2):
        raise TypeError(
            'Input must be a (data, header/WCS) tuple.'
        )

    data, wcs_or_header = input_data

    if isinstance(wcs_or_header, fits.Header):
        wcs = WCS(wcs_or_header)
    elif isinstance(wcs_or_header, WCS):
        wcs = wcs_or_header
    else:
        raise TypeError(
            f'WCS must be a Header or WCS object, got {type(wcs_or_header).__name__}'
        )

    if isinstance(data, SpectralCube):
        value = data.filled_data[:].value
        unit  = data.unit
    else:
        value = np.asarray(data)
        unit = getattr(data, 'unit', None)

    if value.ndim not in (2, 3):
        raise ValueError(
            f'Data must be 2D or 3D for reprojection, got ndim={value.ndim}'
        )

    return (value, wcs), unit


def _strip_wcs_from_header(
    header: Header | list[Header] | tuple[Header]
) -> Header | list[Header]:
    """
    Strip all WCS information from a Header.

    Uses `spectral_cube.wcs_utils.strip_wcs_from_header` under the hood.

    Parameters
    ----------
    header : Header or array-like of Headers
        Header(s) to strip WCS related entries from.

    Returns
    -------
    nowcs_header : Header or array-like of Headers
        Header(s) with no WCS information.
    """
    if isinstance(header, (list, tuple)):
        return [strip_wcs_from_header(h) for h in header]

    if not isinstance(header, Header):
        raise TypeError('header must be an astropy.io.fits.Header or sequence thereof')

    return strip_wcs_from_header(header)


def _update_header_from_wcs(header, wcs):
    """
    Update Header key-value pairs in place using a WCS object.

    The WCS is converted to a Header, then each key-value
    pair is iterated over to update the original header.
    This should only update WCS related keys. It is ideal
    to call `_strip_wcs_from_header` before calling this function,
    to avoid stale WCS keywords.

    Parameters
    ----------
    header : Header
        Astropy Header object to update.
    wcs : WCS
        WCS object to update header with.
    """
    if not isinstance(header, Header):
        raise ValueError(
            'header must be a Fits Header!'
        )
    if isinstance(wcs, WCS):
        wcs_header = wcs.to_header()
    else:
        raise ValueError(
            'wcs must be a WCS object!'
        )

    for key in wcs_header:
        header[key] = wcs_header[key]
