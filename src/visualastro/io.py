'''
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2025-10-20
Description:
    Functions for I/O operations within visualastro.
Dependencies:
    - astropy
    - matplotlib
    - numpy
    - tqdm
Module Structure:
    - Fits File I/O Operations
        Functions to handle Fits files I/O operations.
    - Figure I/O Operations
        Functions to handle matplotlib figure I/O operations.
'''

import warnings
from astropy.io import fits
from astropy.io.fits import Header
import astropy.units as u
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from .numerical_utils import check_is_array
from .FitsFile import FitsFile
from .va_config import get_config_value, va_config, _default_flag
from .wcs_utils import reproject_wcs


# Fits File I/O Operations
# ––––––––––––––––––––––––
def load_fits(filepath, header=True, error=True,
              print_info=None, transpose=None,
              dtype=None, target_wcs=_default_flag,
              invert_wcs=None, **kwargs):
    '''
    Load a FITS file and return its data, header, and errors.
    The WCS is also extracted if possible. Optionally, the
    data and errors can be reprojected onto a target wcs.
    Parameters
    ––––––––––
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
        If None, uses the default value set by `va_config.print_info`.
    transpose : bool or None, default=None
        If True, transpose the data array before returning.
        This will also transpose the error array and swap
        the WCS axes for consistency. The swapping of the WCS
        can be disabled by `va_config.invert_wcs_if_transpose`.
        If None, uses the default value set by `va_config.transpose`.
    dtype : np.dtype, default=None
        Data type to convert the FITS data to. If None,
        determines the dtype from the data. Will convert to
        np.float64 if not floating.
    target_wcs : Header, WCS or None, optional, default=`_default_flag`
        Reproject the input data onto the WCS of another
        data set. Input data must have a valid header
        to extract WCS from. If None, will not reproject
        the input data. If `_default_flag`, uses the default
        value set by `va_config.target_wcs`.
    invert_wcs : bool or None, optional, default=None
        If True, will perform a swapaxes(0,1) on the wcs if `transpose=True`.
        If None, uses the default value set by `va_config.invert_wcs_if_transpose`.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `reproject_method` : {'interp', 'exact'} or None, default=`va_config.reproject_method`
            Reprojection method:
            - 'interp' : use `reproject_interp`
            - 'exact' : use `reproject_exact`
        - `return_footprint` : bool or None, optional, default=`va_config.return_footprint`
            If True, return both reprojected data and reprojection
            footprints. If False, return only the reprojected data.
        - `parallel` : bool, int, str, or None, optional, default=`va_config.reproject_parallel`
            If True, the reprojection is carried out in parallel,
            and if a positive integer, this specifies the number
            of threads to use. The reprojection will be parallelized
            over output array blocks specified by `block_size` (if the
            block size is not set, it will be determined automatically).
        - `block_size` : tuple, ‘auto’, or None, optional, default=`va_config.reproject_block_size`
            The size of blocks in terms of output array pixels that each block
            will handle reprojecting. Extending out from (0,0) coords positively,
            block sizes are clamped to output space edges when a block would extend
            past edge. Specifying 'auto' means that reprojection will be done in
            blocks with the block size automatically determined. If `block_size` is
            not specified or set to None, the reprojection will not be carried out in blocks.

    Returns
    –––––––
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
    # –––– KWARGS ––––
    reproject_method = kwargs.get('reproject_method', va_config.reproject_method)
    return_footprint = kwargs.get('return_footprint', va_config.return_footprint)
    parallel = kwargs.get('parallel', va_config.reproject_parallel)
    block_size = kwargs.get('block_size', va_config.reproject_block_size)

    # get default va_config values
    print_info = get_config_value(print_info, 'print_info')
    transpose = get_config_value(transpose, 'transpose')
    target_wcs = va_config.target_wcs if target_wcs is _default_flag else target_wcs
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

        dt = get_dtype(data, dtype)
        data = data.astype(dt, copy=False) # type: ignore
        # get errors
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
                    f'target_wcs must be Header or WCS, got {type(target_wcs)}'
                )
            input_wcs = WCS(fits_header).celestial
            data, footprint = reproject_wcs((data, input_wcs), wcs,
                                             method=reproject_method,
                                             return_footprint=True,
                                             parallel=parallel,
                                             block_size=block_size)
            if errors is not None:
                errors = reproject_wcs((errors, input_wcs), wcs,
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
            data = data.T # type: ignore
            if wcs is not None and invert_wcs:
                wcs = wcs.swapaxes(0, 1)

    if header or error:
        fitsfile = FitsFile(data, fits_header, errors, wcs)
        fitsfile.footprint = footprint if return_footprint else None
        return fitsfile

    else:
        return data


def get_dtype(data, dtype=None, default_dtype=None):
    '''
    Returns the dtype from the provided data. Promotes
    integers to floats if needed.
    Parameters
    ––––––––––
    data : array-like
        Input array whose dtype will be checked.
    dtype : data-type, optional, default=None
        If provided, this dtype is returned directly.
        If None, returns `data.dtype` if floating or
        `np.float64` if integer or unsigned.
    default_dtype : data-type, optional, default=None
        Float type to use if `data` is integer or unsigned.
        If None, uses the default unit set in `va_config.default_dtype`.
    Returns
    –––––––
    dtype : np.dtype
        NumPy dtype object: user dtype if given, otherwise the array's
        float dtype or `default_dtype` if array is integer/unsigned.
    '''
    # get default va_config values
    default_dtype = get_config_value(default_dtype, 'default_unit')

    # return user dtype if passed in
    if dtype is not None:
        return np.dtype(dtype)

    data = check_is_array(data)
    # by default use data dtype if floating
    # if unsigned or int use default_dtype
    if np.issubdtype(data.dtype, np.floating):
        return np.dtype(data.dtype)
    else:
        return np.dtype(default_dtype)


def get_errors(hdul, dtype=None, transpose=False):
    '''
    Return the error array from an HDUList, falling back to square root
    of variance if needed. If a unit is found from the header, return
    the error array as a Quantity object instead.
    Parameters
    ––––––––––
    hdul : astropy.io.fits.HDUList
        The HDUList object containing FITS extensions to search for errors or variance.
    dtype : data-type, optional, default=np.float64
        The desired NumPy dtype of the returned error array.
    Returns
    –––––––
    errors : np.ndarray or None
        The error array if found, or None if no suitable extension is present.
    '''
    errors = None
    error_unit = None

    for hdu in hdul[1:]:
        extname = hdu.header.get('EXTNAME', '').upper()

        if extname in {'ERR', 'ERROR', 'UNCERT'}:
            dt = get_dtype(hdu.data, dtype)
            errors = hdu.data.astype(dt, copy=False)

            try:
                error_unit = u.Unit(hdu.header.get('BUNIT'))
                errors *= error_unit
            except Exception:
                warnings.warn(
                    'Error extension has invalid BUNIT; returning errors without units.'
                )

            break

    # fallback to variance if no explicit errors
    if errors is None:
        for hdu in hdul[1:]:
            extname = hdu.header.get('EXTNAME', '').upper()

            if extname in {'VAR', 'VARIANCE', 'VAR_POISSON', 'VAR_RNOISE'}:
                dt = get_dtype(hdu.data, dtype)
                variance = hdu.data.astype(dt, copy=False)
                errors = np.sqrt(variance)

                try:
                    var_unit = u.Unit(hdu.header.get('BUNIT'))
                    error_unit = var_unit**0.5
                    errors *= error_unit
                except Exception:
                    warnings.warn(
                        'Variance extension has invalid BUNIT; returning errors without units.'
                    )

                break

    if transpose and errors is not None:
        errors = errors.T

    return errors


def write_cube_2_fits(cube, filename, overwrite=False):
    '''
    Write a 3D data cube to a series of FITS files.

    Parameters
    ––––––––––
    cube : ndarray (N_frames, N, M)
        Data cube containing N_frames images of shape (N, M).
    filename : str
        Base filename (without extension). Each
        output file will be saved as "{filename}_i.fits".
    overwrite : bool, optional, default=False
        If True, existing files with the same name
        will be overwritten.

    Notes
    –––––
    Prints a message indicating the number of
    frames and the base filename.
    '''
    N_frames, N, M = cube.shape
    print(f"Writing {N_frames} fits files to {filename}_i.fits")
    for i in tqdm(range(N_frames)):
        output_name = filename + f"_{i}.fits"
        fits.writeto(output_name, cube[i], overwrite=overwrite)


# Figure I/O Operations
# –––––––––––––––––––––
def get_kwargs(kwargs, *names, default=None):
    '''
    Return the first matching kwarg value from a list of possible names.

    Parameters
    ––––––––––
    kwargs : dict
            Dictionary of keyword arguments, typically taken from ``**kwargs``.
    *names : str
        One or more possible keyword names to search for. The first name found
        in ``kwargs`` with a non-None value is returned.
    default : any, optional, default=None
        Value to return if none of the provided names are found in ``kwargs``.
        Default is None.

    Returns
    –––––––
    value : any
        The value of the first matching keyword argument, or `default` if
        none are found.
    '''
    for name in names:
        if name in kwargs and kwargs[name] is not None:
            return kwargs[name]

    return default


def save_figure_2_disk(
    dpi=None,
    pdf_compression=None,
    transparent=False,
    bbox_inches=_default_flag,
    **kwargs
):
    '''
    Saves current figure to disk as a
    eps, pdf, png, or svg, and prompts
    user for a filename and format.

    Parameters
    ––––––––––
    dpi : float, int, or None, optional, default=None
        Resolution in dots per inch. If None, uses
        the default value set by `va_config.dpi`.
    pdf_compression : int or None, optional, default=None
        'Pdf.compression' value for matplotlib.rcParams.
        Accepts integers from 0-9, with 0 meaning no
        compression. If None, uses the default value
        set by `va_config.pdf_compression`.
    transparent : bool, optional, default=False
        If True, the Axes patches will all be transparent;
        the Figure patch will also be transparent unless
        facecolor and/or edgecolor are specified via kwargs.
    bbox_inches : str, Bbox, or None, default=`_default_flag`
        Bounding box in inches: only the given portion of the
        figure is saved. If 'tight', try to figure out the
        tight bbox of the figure. If `_default_flag`, uses
        the default value set by `va_config.bbox_inches`.

    **kwargs : dict, optional
        Additional parameters.

        Supported keyword arguments include:

        - `facecolorcolor` : str, default='auto'
            The facecolor of the figure. If 'auto',
            use the current figure facecolor.
        - `edgecolorcolor` : str, default='auto'
            The edgecolor of the figure. If 'auto',
            use the current figure edgecolor.
    '''
    # –––– KWARGS ––––
    facecolor = get_kwargs(kwargs, 'facecolor', 'fc', default='auto')
    edgecolor = get_kwargs(kwargs, 'edgecolor', 'ec', default='auto')

    # get default va_config values
    dpi = get_config_value(dpi, 'dpi')
    pdf_compression = get_config_value(pdf_compression, 'pdf_compression')
    bbox_inches = va_config.bbox_inches if bbox_inches is _default_flag else bbox_inches

    allowed_formats = va_config.allowed_formats
    # prompt user for filename, and extract extension
    filename = input("Input filename for image (ex: myimage.pdf): ").strip()
    basename, *extension = filename.rsplit(".", 1)
    # if extension exists, and is allowed, extract extension from list
    if extension and extension[0].lower() in allowed_formats:
        extension = extension[0]
    # else prompt user to input a valid extension
    else:
        extension = ""
        while extension not in allowed_formats:
            extension = (
                input(f"Please choose a format from ({', '.join(allowed_formats)}): ")
                .strip()
                .lower()
            )
    # construct complete filename
    filename = f"{basename}.{extension}"

    with plt.rc_context(rc={'pdf.compression': int(pdf_compression)} if extension == 'pdf' else {}):
        # save figure
        plt.savefig(
            fname=filename,
            format=extension,
            transparent=transparent,
            bbox_inches=bbox_inches,
            facecolor=facecolor,
            edgecolor=edgecolor,
            dpi=dpi
        )
