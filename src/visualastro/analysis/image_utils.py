"""
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2026-03-11
Description:
    Utility functions for image manipulations.
Dependencies:
    - astropy
    - numpy
    - regions
    - spectral-cube
Module Structure:
    - Cube Manipulation Functions
        Utility functions used when manipulating datacubes numerically.
    - Cube/Image Masking Functions
        Utility functions used when masking datacubes.
"""

import glob
import warnings
from astropy.io import fits
from astropy.io.fits import Header
import astropy.units as u
from astropy.units import Quantity
from astropy.wcs import WCS
import numpy as np
from numpy.typing import DTypeLike
from regions import PixCoord, EllipsePixelRegion, EllipseAnnulusPixelRegion
from spectral_cube import SpectralCube
from tqdm import tqdm
from visualastro.core.config import config, get_config_value, resolve_default, _Unset, _UNSET
from visualastro.core.io import get_errors, _get_dtype
from visualastro.core.numerical_utils import get_data
from visualastro.core.units import get_unit
from visualastro.core.validation import _type_name
from visualastro.dataclasses.datacube import DataCube
from visualastro.dataclasses.fitsfile import FitsFile
from visualastro.utils.wcs_utils import _reproject_wcs


# I/O Functions
# -------------
def load_data_cube(
    filepath: str,
    error: bool = True,
    hdu: int | _Unset = _UNSET,
    dtype: DTypeLike | str | _Unset = _UNSET,
    print_info: bool | _Unset =_UNSET,
    transpose: bool | _Unset = _UNSET,
    invert_wcs: bool | _Unset = _UNSET
) -> DataCube:
    """
    Load a sequence of FITS files into a 3D data cube.

    This function searches for all FITS files matching a
    given path pattern, loads them into a NumPy array of shape
    (T, M, N), and bundles the data, headers, errors, and WCS
    into a ``DataCube`` object.

    Parameters
    ----------
    filepath : str
        Path pattern to FITS files. Wildcards are supported.
        ie. ``'Spectro-Module/raw/HARPS*.fits'``
    error : bool, optional, default=True
        If ``True``, try to extract the error extension.
        Converts variance to errors using ``get_errors``.
    hdu : int | _Unset, optional, default=_UNSET
        Hdu extension to use. If ``_UNSET``, uses the
        default value set by ``config.hdu_idx``.
    dtype : np.dtype | str | _Unset, optional, default=_UNSET
        Data type for the loaded FITS data. If ``_UNSET``, will use
        the dtype of the provided data, promoting integer or
        unsigned to ``config.default_dtype``.
    print_info : bool | _Unset, optional, default=_UNSET
        If True, print summary information about the loaded cube.
        If ``_UNSET``, uses the default value set by ``config.print_info``.
    transpose : bool | _Unset, optional, default=_UNSET
        If True, transpose each 2D image before stacking into the cube.
        This will also transpose each error array if available and
        swap the WCS axes for consistency. The swapping of the WCS
        can be disabled by ``config.invert_wcs_if_transpose``.
        If ``_UNSET``, uses the default value set by `config.transpose`.
    invert_wcs : bool | _Unset, optional, default=_UNSET
        If True, will perform a swapaxes(0,1) on the wcs if ``transpose=True``.
        If ``_UNSET``, uses the default value set by ``config.invert_wcs_if_transpose``.

    Returns
    -------
    cube : DataCube
        A DataCube object containing:
        - `cube.data` : np.ndarray of shape (T, M, N)
        - `cube.header` : list of astropy.io.fits.Header objects
        - `cube.error` : np.ndarray of shape (T, M, N)
        - `cube.wcs` : list of `astropy.wcs.wcs.WCS`

    Examples
    --------
    Search for all fits files starting with 'HARPS' with .fits extention and load them:
        >>> filepath = 'Spectro-Module/raw/HARPS.*.fits'
    """
    hdu = resolve_default(hdu, config.hdu_idx)
    dtype = resolve_default(dtype, config.default_dtype)
    print_info = resolve_default(print_info, config.print_info)
    transpose = resolve_default(transpose, config.transpose)
    invert_wcs = resolve_default(invert_wcs, config.invert_wcs_if_transpose)

    # searches for all files within a directory
    fits_files = sorted(glob.glob(filepath))
    if not fits_files:
        raise FileNotFoundError(f'No FITS files found for pattern: {filepath}')
    # allocate ixMxN data cube array and header array
    n_files = len(fits_files)

    # load first file to determine shape, dtype, and check for errors
    with fits.open(fits_files[0]) as hdul:
        if print_info:
            hdul.info()

        data = hdul[hdu].data
        header = hdul[hdu].header
        err = get_errors(hdul, dtype)

    dt = _get_dtype(data, dtype)

    try:
        wcs = WCS(header)
    except ValueError:
        wcs = None

    if transpose:
        data = data.T
        if wcs is not None and invert_wcs:
            wcs = wcs.swapaxes(0,1)
        if err is not None:
            err = err.T

    # preallocate data cube and headers
    datacube = np.zeros((n_files, data.shape[0], data.shape[1]), dtype=dt)
    datacube[0] = data.astype(dt)
    headers = []
    headers.append(header)
    wcs_list = []
    wcs_list.append(wcs)
    # preallocate error array if needed and error exists
    error_array = None
    if error and err is not None:
        error_array = np.zeros_like(datacube, dtype=dt)
        error_array[0] = err.astype(dt)

    # loop through remaining files
    for i, file in enumerate(tqdm(fits_files[1:], desc='Loading FITS')):
        with fits.open(file) as hdul:
            data = hdul[hdu].data
            headers.append(hdul[hdu].header)
            err = get_errors(hdul, dt)
            try:
                wcs = WCS(headers[i+1])
            except ValueError:
                wcs = None

        if transpose:
            data = data.T
            if wcs is not None and invert_wcs:
                wcs = wcs.swapaxes(0,1)
            if err is not None:
                err = err.T
        datacube[i+1] = data.astype(dt)
        if error_array is not None and err is not None:
            error_array[i+1] = err.astype(dt)
        wcs_list.append(wcs)

    if all(w is None for w in wcs_list):
        wcs_list = None
    elif any(w is None for w in wcs_list):
        missing_indices = [i for i, w in enumerate(wcs_list) if w is None]
        raise ValueError(
            f'Inconsistent WCS: files at indices {missing_indices} have no WCS, '
            f'but other files do. Either all files must have WCS or none should.'
        )

    return DataCube(datacube, headers, error_array, wcs_list)


def load_fits(filepath, header=True, error=True,
              print_info=None, transpose=None,
              dtype=None, target_wcs=_UNSET,
              invert_wcs=None, **kwargs):
    '''
    Load a FITS file and return its data, header, and errors.
    The WCS is also extracted if possible. Optionally, the
    data and errors can be reprojected onto a target wcs.
    Parameters
    ----------
    filepath : str
        Path to the FITS file to load.
    header : bool, default=True
        If True, return the FITS header along with the data
        as a FitsFile object.
        If False, only the data is returned.
    error : bool, default=True
        If True, return the 'ERR' extention of the fits file.
    print_info : bool or None, default=None
        If True, print HDU information using 'hdul.info()'.
        If None, uses the default value set by `config.print_info`.
    transpose : bool or None, default=None
        If True, transpose the data array before returning.
        This will also transpose the error array and swap
        the WCS axes for consistency. The swapping of the WCS
        can be disabled by `config.invert_wcs_if_transpose`.
        If None, uses the default value set by `config.transpose`.
    dtype : np.dtype, default=None
        Data type to convert the FITS data to. If None,
        determines the dtype from the data. Will convert to
        np.float64 if not floating.
    target_wcs : Header, WCS or None, optional, default=`_UNSET`
        Reproject the input data onto the WCS of another
        data set. Input data must have a valid header
        to extract WCS from. If None, will not reproject
        the input data. If `_UNSET`, uses the default
        value set by `config.target_wcs`.
    invert_wcs : bool or None, optional, default=None
        If True, will perform a swapaxes(0,1) on the wcs if `transpose=True`.
        If None, uses the default value set by `config.invert_wcs_if_transpose`.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `reproject_method` : {'interp', 'exact'} or None, default=`config.reproject_method`
            Reprojection method:
            - 'interp' : use `reproject_interp`
            - 'exact' : use `reproject_exact`
        - `return_footprint` : bool or None, optional, default=`config.return_footprint`
            If True, return both reprojected data and reprojection
            footprints. If False, return only the reprojected data.
        - `parallel` : bool, int, str, or None, optional, default=`config.reproject_parallel`
            If True, the reprojection is carried out in parallel,
            and if a positive integer, this specifies the number
            of threads to use. The reprojection will be parallelized
            over output array blocks specified by `block_size` (if the
            block size is not set, it will be determined automatically).
        - `block_size` : tuple, ‘auto’, or None, optional, default=`config.reproject_block_size`
            The size of blocks in terms of output array pixels that each block
            will handle reprojecting. Extending out from (0,0) coords positively,
            block sizes are clamped to output space edges when a block would extend
            past edge. Specifying 'auto' means that reprojection will be done in
            blocks with the block size automatically determined. If `block_size` is
            not specified or set to None, the reprojection will not be carried out in blocks.

    Returns
    -------
    FitsFile
        If header or error is True, returns an object containing:
        - data: `np.ndarray` of the FITS data
        - header: `astropy.io.fits.Header` if `header=True` else None
        - error: `np.ndarray` of the FITS error if `error=True` else None
        - wcs: `astropy.wcs.wcs.WCS` if `header=True` else None
            By default, is extracted from the header.
            If a `target_wcs` is passed in, will override the default header.
    data : np.ndarray
        If header is False, returns just the data component.
    '''
    # ---- KWARGS ----
    reproject_method = kwargs.get('reproject_method', config.reproject_method)
    return_footprint = kwargs.get('return_footprint', config.return_footprint)
    parallel = kwargs.get('parallel', config.reproject_parallel)
    block_size = kwargs.get('block_size', config.reproject_block_size)

    # get default config values
    print_info = get_config_value(print_info, 'print_info')
    transpose = get_config_value(transpose, 'transpose')
    target_wcs = config.target_wcs if target_wcs is _UNSET else target_wcs
    invert_wcs = get_config_value(invert_wcs, 'invert_wcs_if_transpose')

    # disable transpose if reprojecting
    if target_wcs is not None and transpose:
        warnings.warn('`transpose=True` ignored because `target_wcs` was provided.')
        transpose = False

    data = None
    fits_header = None
    errors = None
    wcs = None
    footprint = None

    # print fits file info
    with fits.open(filepath) as hdul:
        if print_info:
            hdul.info()

        # extract data and optionally the header from the file
        # if header is not requested, return None
        for hdu in hdul:
            if hdu.data is not None: # type: ignore
                data = hdu.data # type: ignore
                fits_header = hdu.header if header else None # type: ignore
                break
        if data is None:
            raise ValueError(
                f'No image HDU with data found in file: {filepath}!'
            )

        dt = _get_dtype(data, dtype)
        data = data.astype(dt, copy=False)
        if error:
            errors = get_errors(hdul, dt, transpose)

        # reproject wcs if user inputs a reference wcs or header
        # otherwise try to extract wcs from fits header
        if target_wcs is not None:
            # ensure target_wcs has wcs information
            if isinstance(target_wcs, Header):
                wcs = WCS(target_wcs)
            elif isinstance(target_wcs, WCS):
                wcs = target_wcs
            else:
                raise ValueError(
                    f'target_wcs must be Header or WCS, got {_type_name(target_wcs)}'
                )
            input_wcs = WCS(fits_header).celestial
            data, footprint = _reproject_wcs((data, input_wcs), wcs,
                                             method=reproject_method,
                                             return_footprint=True,
                                             parallel=parallel,
                                             block_size=block_size)
            if errors is not None:
                errors = _reproject_wcs((errors, input_wcs), wcs,
                                        method=reproject_method,
                                        return_footprint=False,
                                        parallel=parallel,
                                        block_size=block_size)
        else:
            # try extracting wcs from header
            if fits_header is not None:
                try:
                    wcs = WCS(fits_header)
                except Exception:
                    wcs = None

        if transpose:
            data = data.T
            if wcs is not None and invert_wcs:
                wcs = wcs.swapaxes(0, 1)

    if header or error:
        fitsfile = FitsFile(data, fits_header, errors, wcs)
        fitsfile.footprint = footprint if return_footprint else None
        return fitsfile

    else:
        return data


def load_spectral_cube(
    filepath, hdu, error=True, header=True, dtype=None, print_info=None
):
    """
    Load a spectral cube from a FITS file,
    optionally including errors and header.

    Parameters
    ----------
    filepath : str
        Path to the FITS file to read.
    hdu : int or str
        HDU index or name to read from the FITS file.
    error : bool, optional, default=True
        If True, load the associated error array using `get_errors`.
    header : bool, optional, default=True
        If True, load the HDU header.
    dtype : data-type, optional, default=None
        Desired NumPy dtype for the error array. If None, inferred
        from FITS data, promoting integer and unsigned to `np.float64`.
    print_info : bool or None, optional, default=None
        If True, print FITS file info to the console.
        If None, uses default value set by `config.print_info`.

    Returns
    -------
    DataCube
        A `DataCube` object containing:
        - data : SpectralCube
            Fits file data loaded as SpectralCube object.
        - header : astropy.io.fits.Header
            Fits file header.
        - error : np.ndarray
            Fits file error array.
        - value : np.ndarray
            Fits file data as np.ndarray.
        Ex:
        data = cube.data
    """
    print_info = get_config_value(print_info, 'print_info')

    # load SpectralCube from filepath
    spectral_cube = SpectralCube.read(filepath, hdu=hdu)
    # initialize error and header objects
    error_array = None
    hdr = None
    # open fits file
    with fits.open(filepath) as hdul:
        # print fits info
        if print_info:
            hdul.info()
        # load error array
        if error:
            error_array = get_errors(hdul, dtype)
        # load header
        if header:
            hdr = hdul[hdu].header # type: ignore

    return DataCube(spectral_cube, header=hdr, error=error_array)


# Cube Manipulation Functions
# ---------------------------

_STACK_METHODS = {
    'mean': np.nanmean,
    'median': np.nanmedian,
    'sum': np.nansum,
    'max': np.nanmax,
    'min': np.nanmin,
    'std': np.nanstd,
}

def stack_cube(cube, *, idx=None, method=None, axis=0):
    """
    Stack or extract slices from a data cube.

    Parameters
    ----------
    cube : ndarray, Quantity, SpectralCube, or DataCube
        3D data cube to stack.
    idx : int, list of int, or None, optional, default=None
        Index specification:
        - int: extract single slice
        - [start, end]: extract range (inclusive)
        - None: use entire cube
    method : {'mean', 'median', 'sum', 'max', 'min', 'std'}, default=None
        Stacking method. If None, uses the default value set
        by ``config.stack_cube_method``.
    axis : int, optional, default=0
        Axis along which to stack.

    Returns
    -------
    ndarray, Quantity, or SpectralCube
        Stacked result (2D if axis=0) or extracted slice.
    """
    method = get_config_value(method, 'stack_cube_method')

    if method not in _STACK_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Options: {', '.join(_STACK_METHODS.keys())}"
        )

    cube = get_data(cube)

    if idx is not None:
        if isinstance(idx, int):
            return cube[idx]

        if isinstance(idx, list):
            if len(idx) == 1:
                return cube[idx[0]]
            if len(idx) != 2:
                raise ValueError('idx must be an int, [start, end], or None')

            start, end = idx
            cube = cube[start:end+1]
        else:
            raise TypeError(f'idx must be int, list, or None; got {_type_name(idx)}')

    if isinstance(cube, SpectralCube):
        stack_func = getattr(cube, method)
        return stack_func(axis=axis)

    return _STACK_METHODS[method](cube, axis=axis)


# Cube/Image Masking Functions
# ----------------------------
def mask_image(
    image, ellipse_region=None, region=None,
    line_points=None, invert_region=False,
    above_line=True, preserve_shape=True,
    existing_mask=None, combine_method='union', **kwargs):
    '''
    Mask an image with modular filters.
    Supports applying an elliptical or annular region mask, an optional
    line cut (upper or lower half-plane), and combining with an existing mask.

    Parameters
    ----------
    image : array-like, DataCube, FitsFile, or SpectralCube
        Input image or cube. If higher-dimensional, the mask is applied
        to the last two axes.
    ellipse_region : `EllipsePixelRegion` or `EllipseAnnulusPixelRegion`, optional, default=None
        Region object specifying an ellipse or annulus.
    region : str {'annulus', 'ellipse'}, optional, default=None
        Type of region to apply. Ignored if `ellipse_region` is provided.
    line_points : array-like, shape (2, 2), optional, default=None
        Two (x, y) points defining a line for masking above/below.
        Ex: [[0,2], [20,10]]
    invert_region : bool, default=False
        If True, invert the region mask.
    above_line : bool, default=True
        If True, keep the region above the line. If False, keep below.
    preserve_shape : bool, default=True
        If True, return an array of the same shape with masked values set to NaN.
        If False, return only the unmasked pixels.
    existing_mask : ndarray of bool, optional, default=None
        An existing mask to combine (union) with the new mask.
    combine_method : {'union', 'intersect'}, optional, default=None
        If 'union', combine masks with `|`. If 'intersect', use `&`.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - center : tuple of float, optional, default=None
            Center coordinates (x, y).
        - w : float, optional, default=None
            Width of ellipse.
        - h : float, optional, default=None
            Height of ellipse.
        - angle : float, optional, default=0
            Rotation angle in degrees.
        - tolerance : float, list, or tuple, optional, default=2
            ±Tolerance (distance from radius) for annulus inner/outer radii.
            If array-like, uses the first element as the minus bound,
            and the second element as the positive.

    Returns
    -------
    masked_image : ndarray or SpectralCube
        Image with mask applied. Type matches input.
    masks : ndarray of bool or list
        If multiple masks are combined, returns a list containing the
        master mask followed by individual masks. Otherwise returns a single mask.
    '''
    # ---- Kwargs ----
    center = kwargs.get('center', None)
    w = kwargs.get('w', None)
    h = kwargs.get('h', None)
    angle = kwargs.get('angle', 0)
    tolerance = kwargs.get('tolerance', 2)
    if (isinstance(tolerance, (list, tuple))
        and len(tolerance) == 2):
        mtol = tolerance[0]
        ptol = tolerance[1]
    else:
        mtol = tolerance
        ptol = tolerance

    # extract units
    unit = get_unit(image)

    # ensure working with array
    if isinstance(image, (DataCube, FitsFile)):
        image = image.data
    else:
        image = np.asarray(image)

    # determine image shape
    N, M = image.shape[-2:]
    y, x = np.indices((N, M))
    # empty list to hold all masks
    masks = []

    # early return if just applying an existing mask
    if (ellipse_region is None and region is None
        and line_points is None and existing_mask is not None):
        if existing_mask.shape != image.shape[-2:]:
            raise ValueError('existing_mask must have same shape as image')

        if isinstance(image, np.ndarray):
            if preserve_shape:
                masked_image = np.full_like(image, np.nan, dtype=float)
                masked_image[..., existing_mask] = image[..., existing_mask]
            else:
                masked_image = image[..., existing_mask]

            if isinstance(unit, u.UnitBase) and not isinstance(image, Quantity):
                masked_image *= unit
        else:
            # if spectral cube or similar object
            masked_image = image.with_mask(existing_mask)

        return masked_image

    # ---- Region Mask ----
    # if ellipse region is passed in use those values
    if ellipse_region is not None:
        center = ellipse_region.center
        a = ellipse_region.width / 2
        b = ellipse_region.height / 2
        angle = ellipse_region.angle if ellipse_region.angle is not None else 0
    # accept user defined center, w, and h values if used
    elif None not in (center, w, h):
        a = w / 2
        b = h / 2
    # stop program if attempting to plot a region without necessary data
    elif region is not None:
        raise ValueError("Either 'ellipse_region' or 'center', 'w', 'h' must be provided.")

    # construct region
    if region is not None:
        if region.lower() == 'annulus':
            region_obj = EllipseAnnulusPixelRegion(
                center=PixCoord(center[0], center[1]), # type: ignore
                inner_width=2*(a - mtol),
                inner_height=2*(b - mtol),
                outer_width=2*(a + ptol),
                outer_height=2*(b + ptol),
                angle=angle * u.deg
            )
        elif region.lower() == 'ellipse':
            region_obj = EllipsePixelRegion(
                center=PixCoord(center[0], center[1]), # type: ignore
                width=2*a,
                height=2*b,
                angle=angle * u.deg
            )
        else:
            raise ValueError("region must be 'annulus' or 'ellipse'")

        # filter by region mask
        region_mask = region_obj.to_mask(mode='center').to_image((N, M)).astype(bool)
        if invert_region:
            region_mask = ~region_mask
        masks.append(region_mask.copy())
    else:
        # empty mask if no region
        region_mask = np.ones((N, M), dtype=bool)

    # ---- Line Mask ----
    if line_points is not None:
        # start from previous mask
        line_mask = region_mask.copy()
        # compute slope and intercept of line
        m, b_line = compute_line(line_points)
        # filter out points above/below line
        line_mask &= (y >= m*x + b_line) if above_line else (y <= m*x + b_line)
        # add line region to mask array
        masks.append(line_mask.copy())
    else:
        # empty mask if no region
        line_mask = region_mask.copy()

    # ---- Combine Masks ----
    # start master mask with line_mask (or region if no line)
    mask = line_mask.copy()

    # union with existing mask if provided
    if existing_mask is not None:
        if existing_mask.shape != mask.shape:
            raise ValueError('existing_mask must have the same shape as the image')
        if combine_method == 'union':
            mask |= existing_mask
        elif combine_method == 'intersect':
            mask &= existing_mask
        else:
            raise ValueError(
                f"`combine_method` has to be 'union' or 'intersect'! "
                f'Got {combine_method}.'
            )

    # ---- Apply Mask ----
    # if numpy array:
    if isinstance(image, np.ndarray):
        if preserve_shape:
            masked_image = np.full_like(image, np.nan, dtype=float)
            masked_image[..., mask] = image[..., mask]
        else:
            masked_image = image[..., mask]
        if isinstance(unit, u.UnitBase) and not isinstance(image, Quantity):
            masked_image *= unit
    # if spectral cube object
    else:
        masked_image = image.with_mask(mask)

    # ---- Final Mask List ----
    # Return master mask as first element
    masks = [mask] + masks if len(masks) > 1 else mask

    return masked_image, masks


def compute_line(points):
    '''
    Compute the slope and intercept of a line passing through two points.
    Parameters
    ----------
    points : list or tuple of tuples
        A sequence containing exactly two points, each as (x, y), e.g.,
        [(x0, y0), (x1, y1)].
    Returns
    -------
    m : float
        Slope of the line.
    b : float
        Intercept of the line (y = m*x + b).
    Notes
    -----
    - The function assumes the two points have different x-coordinates.
    - If the x-coordinates are equal, a ZeroDivisionError will be raised.
    '''
    m = (points[0][1] - points[1][1]) / (points[0][0] - points[1][0])
    b = points[0][1] - m*points[0][0]

    return m, b
