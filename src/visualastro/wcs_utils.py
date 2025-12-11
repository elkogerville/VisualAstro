'''
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
'''

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import Header
from astropy.nddata import Cutout2D
import astropy.units as u
from astropy.units import Quantity, Unit
from astropy.wcs import WCS
import numpy as np
from reproject import reproject_interp, reproject_exact
from spectral_cube import SpectralCube
from tqdm import tqdm
from .va_config import get_config_value, va_config, _default_flag


def get_wcs(header):
    '''
    Extract WCS from header(s).

    Parameters
    ––––––––––
    header : Header or list of Header
        Single header or list of headers.

    Returns
    –––––––
    WCS, list of WCS/None, or None
        - Single WCS if header is a Header and WCS extraction succeeds
        - List of WCS/None if header is a list (None for failed extractions)
        - None if header is a single Header and WCS extraction fails
    '''
    if isinstance(header, Header):
        try:
            return WCS(header)
        except Exception:
            return None

    # if a list of headers extract a list of wcs
    elif isinstance(header, (list, np.ndarray, tuple)):
        wcs_list = []
        for h in header:
            if not isinstance(h, Header):
                wcs_list.append(None)
                continue
            try:
                wcs_list.append(WCS(h))
            except Exception:
                wcs_list.append(None)
        return wcs_list

    else:
        return None


# Data Transformations
# ––––––––––––––––––––
def crop2D(data, size, position=None, wcs=None, mode='trim', frame='icrs', origin_idx=0):
    '''
    Create a Cutout2D from array-like data using WCS and a world/pixel position.

    Parameters
    ––––––––––
    data : array-like
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
    –––––––
    Quantity or ndarray
        The cropped region with units if the original data had units.
    WCS
        The WCS of the cropped region

    Raises
    ––––––
    ValueError
        If the WCS is missing (None) or cutout creation fails.
    TypeError
        If `position` is not a supported type.

    Notes
    –––––
    - If the data were transposed and the WCS was swapped via `wcs.swapaxes(0, 1)`,
        the method will automatically attempt to correct for inverted RA/Dec axes.

    '''
    if hasattr(data, 'wcs'):
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
        ra = position[0].to(u.deg) # type: ignore
        dec = position[1].to(u.deg) # type: ignore
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

    # re-attach units
    crop_data = cutout.data
    if hasattr(data, 'unit'):
        unit = Unit(data.unit)
        crop_data *= unit

    crop_wcs = cutout.wcs

    return crop_data, crop_wcs


def reproject_wcs(
    input_data,
    reference_wcs,
    method=None,
    return_footprint=None,
    parallel=None,
    block_size=_default_flag,
    description=None
):
    '''
    Reproject data or Quantity arrays onto a reference WCS.

    Parameters
    ––––––––––
    input_data : array-like, list, or tuple
        The input data to be reprojected. May be:
        - A HDUList object
        - A `(np.ndarray, WCS)` or `(np.ndarray, Header)` tuple
        - A list containing any of the above inputs
        Note:
            - [np.ndarray, WCS/Header] is not allowed! Ensure they are all tuples.
            - [(np.ndarray, WCS), HDUList] is a valid input.
    reference_wcs : astropy.wcs.WCS or astropy.io.fits.Header
        The target WCS or FITS header to which `input_data` will be reprojected.
        Dimensional handling:
        - 2D → 2D: Direct reprojection
        - 2D → 3D: Uses celestial WCS from 3D target (ignores spectral)
        - 3D → 2D: Reprojects each spectral slice onto 2D target (preserves spectral axis)
        - 3D → 3D: Direct reprojection (spectral axes must be compatible)
    method : {'interp', 'exact'} or None, default=None
        Reprojection method:
        - 'interp' : use `reproject_interp`
        - 'exact' : use `reproject_exact`
        If None, uses the default value
        set by `va_config.reproject_method`.
    return_footprint : bool or None, optional, default=None
        If True, return both reprojected data and reprojection
        footprints. If False, return only the reprojected data.
        If None, uses the default value set by `va_config.return_footprint`.
    parallel : bool, int, str, or None, optional, default=None
        If True, the reprojection is carried out in parallel,
        and if a positive integer, this specifies the number
        of threads to use. The reprojection will be parallelized
        over output array blocks specified by `block_size` (if the
        block size is not set, it will be determined automatically).
        If None, uses the default value set by `va_config.reproject_parallel`.
    block_size : tuple, ‘auto’, or None, optional, default=`_default_flag`
        The size of blocks in terms of output array pixels that each block
        will handle reprojecting. Extending out from (0,0) coords positively,
        block sizes are clamped to output space edges when a block would extend
        past edge. Specifying 'auto' means that reprojection will be done in
        blocks with the block size automatically determined. If `block_size` is
        not specified or set to None, the reprojection will not be carried out in blocks.
        If `_default_flag`, uses the default value set by `va_config.reproject_block_size`.
    description : str or None, optional, default=None
        Description message for loading bar. If None, uses default message.

    Returns
    –––––––
    reprojected : ndarray or list of ndarray
        The reprojected data array(s). If `input_data` contains
        multiple items, the output is a list; otherwise a single array.
    footprint : ndarray or list of ndarray
        The reprojection footprint(s). Only returned if `return_footprint=True`.
        If `input_data` contains multiple items, the output is a list;
        otherwise a single array.

    Raises
    ––––––
    ValueError
        If a the inputs are not able to be reprojected.

    '''
    method = get_config_value(method, 'reproject_method')
    return_footprint = get_config_value(return_footprint, 'return_footprint')
    parallel = get_config_value(parallel, 'reproject_parallel')
    block_size = va_config.reproject_block_size if block_size is _default_flag else block_size

    if isinstance(reference_wcs, fits.Header):
        reference_wcs = WCS(reference_wcs)

    # normalize input into a list
    if isinstance(input_data, list):
        normalized = [_normalize_reproject_input(item) for item in input_data]
    else:
        normalized = [_normalize_reproject_input(input_data)]

    # separate inputs and units
    input_list  = [item[0] for item in normalized]
    units = [item[1] for item in normalized]

    # select reproject function
    reproject_method = {
        'interp': reproject_interp,
        'exact': reproject_exact
    }.get(method, reproject_interp)

    reprojected_data = []
    footprints = []

    for (obj, unit) in zip(input_list, units):

        # case 1: HDUList
        if isinstance(obj, fits.HDUList):
            reprojected, footprint = reproject_method(
                obj, reference_wcs, parallel=parallel, block_size=block_size
            )
            if unit is not None:
                reprojected *= unit
            reprojected_data.append(reprojected)
            footprints.append(footprint)
            continue

        # get data/wcs dimensions
        data, wcs = obj
        data_ndim = np.asarray(data).ndim
        wcs_ndim = wcs.naxis
        ref_ndim = reference_wcs.naxis

        # case 2: data ndim == reference wcs ndim
        if (
            (data_ndim == 2 and ref_ndim == 2)
            or (data_ndim == 3 and ref_ndim == 3)
        ):
            reprojected, footprint = reproject_method(
                (data, wcs), reference_wcs,
                parallel=parallel, block_size=block_size
            )

        # case 3: reference wcs ndim > data ndim
        elif data_ndim == 2 and ref_ndim > 2:
            ref_celestial = reference_wcs.celestial
            reprojected, footprint = reproject_method(
                (data, wcs), ref_celestial,
                parallel=parallel, block_size=block_size
            )

        # case 4: data ndim > reference wcs ndim
        # for each data slice, reproject
        elif data_ndim == 3 and ref_ndim == 2:
            wcs_celestial = wcs.celestial if wcs_ndim > 2 else wcs

            reprojects = []
            footprints_slice = []
            desc = 'Looping over each slice' if description is None else description
            for i in tqdm(range(data.shape[0]), desc=desc):
                repr_i, foot_i = reproject_method(
                    (data[i], wcs_celestial), reference_wcs,
                    parallel=parallel, block_size=block_size
                )
                reprojects.append(repr_i)
                footprints_slice.append(foot_i)

            reprojected = np.stack(reprojects, axis=0)
            footprint = np.stack(footprints_slice, axis=0)

        else:
            raise ValueError(
                f'Unsupported dimension combination: data_ndim={data_ndim}, '
                f'ref_ndim={ref_ndim}. Supported: (2,2), (2,3), (3,2), (3,3)'
            )

        # re-attach units
        if unit is not None:
            reprojected *= unit

        reprojected_data.append(reprojected)
        footprints.append(footprint)

    # unwrap single-element lists
    if len(reprojected_data) == 1:
        reprojected_data = reprojected_data[0]
        footprints = footprints[0]

    return (reprojected_data, footprints) if return_footprint else reprojected_data


def _normalize_reproject_input(input_data):
    '''
    Ensures that the reprojection inputs are one of the accepted types.

    Parameters
    ––––––––––
    input_data : fits.HDUList or tuple
        Input data to be reprojected. Must follow one of the formats:
            - fits.HDUList : hdul object containing data and header
            - tuple : (data, header/WCS) containing array and WCS/Header

    Returns
    –––––––
    input_data : fits.HDUList or tuple
        Normalized input (HDUList unchanged, tuple converted to (array, WCS))
    unit : Astropy.Unit or None
        The unit of the input data (None for HDUList)

    Raises
    ––––––
    TypeError
        If `input_data` is not an accepted type/format.
    '''
    no_unit = None
    # case 1: HDUList
    if isinstance(input_data, fits.HDUList):
        return input_data, no_unit
    # case 2: (data, header/wcs) tuple
    elif isinstance(input_data, tuple) and len(input_data) == 2:
        data, wcs_or_header = input_data
        if isinstance(wcs_or_header, fits.Header):
            wcs = WCS(wcs_or_header)
        else:
            wcs = wcs_or_header

        if isinstance(data, SpectralCube):
            value = data.unmasked_data[:].value
            unit  = data.unit
        else:
            value = np.asarray(data)
            unit = getattr(data, 'unit', None)

        return (value, wcs), unit

    else:
        raise TypeError(
            'Each input must be an HDUList, or a (data, header/WCS) tuple.'
        )
