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

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from .numerical_utils import check_is_array
from .visual_classes import FitsFile
from .va_config import va_config


# Fits File I/O Operations
# ––––––––––––––––––––––––
def load_fits(filepath, header=True, error=True,
              print_info=True, transpose=False, dtype=None):
    '''
    Load a FITS file and return its data and optional header.
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
    print_info : bool, default=True
        If True, print HDU information using 'hdul.info()'.
    transpose : bool, default=False
        If True, transpose the data array before returning.
    dtype : np.dtype, default=None
        Data type to convert the FITS data to. If None,
        determines the dtype from the data. Will convert to
        np.float64 if not floating.
    Returns
    –––––––
    FitsFile
        If header is True, returns an object containing:
        - data: 'np.ndarray' of the FITS data
        - header: 'astropy.io.fits.Header' if 'header=True', else 'None'
    data : np.ndarray
        If header is False, returns just the data component.
    '''
    # print fits file info
    with fits.open(filepath) as hdul:
        if print_info:
            hdul.info()
        # extract data and optionally the header from the file
        # if header is not requested, return None
        result = fits.getdata(filepath, header=header)
        data, fits_header = result if isinstance(result, tuple) else (result, None)

        dt = get_dtype(data, dtype)
        data = data.astype(dt, copy=False)
        if transpose:
            data = data.T

        errors = get_errors(hdul, dt, transpose)

    if header or error:
        return FitsFile(data, fits_header, errors)
    else:
        return data


def get_dtype(data, dtype=None, default_dtype=va_config.default_unit):
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
    default_dtype : data-type, optional, default=np.float64
        Float type to use if `data` is integer or unsigned.
    Returns
    –––––––
    dtype : np.dtype
        NumPy dtype object: user dtype if given, otherwise the array's
        float dtype or `default_dtype` if array is integer/unsigned.
    '''
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
    of variance if needed.
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
    for hdu in hdul[1:]:
        extname = hdu.header.get('EXTNAME', '').upper()
        if extname in {'ERR', 'ERROR', 'UNCERT'}:
            dt = get_dtype(hdu.data, dtype)
            errors = hdu.data.astype(dt, copy=False)
            break
    # fallback to variance if no explicit errors
    if errors is None:
        for hdu in hdul[1:]:
            extname = hdu.header.get('EXTNAME', '').upper()
            if extname in {'VAR', 'VARIANCE', 'VAR_POISSON', 'VAR_RNOISE'}:
                dt = get_dtype(hdu.data, dtype)
                errors = np.sqrt(hdu.data.astype(dtype, copy=False))
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


def save_figure_2_disk(dpi=va_config.dpi, pdf_compression=va_config.pdf_compression,
                       transparent=False, bbox_inches=va_config.bbox_inches, **kwargs):
    '''
    Saves current figure to disk as a
    eps, pdf, png, or svg, and prompts
    user for a filename and format.
    Parameters
    ––––––––––
    dpi : float or int, optional, default=600
        Resolution in dots per inch.
    pdf_compression : int, optional, default=False
        'Pdf.compression' value for matplotlib.rcParams.
        Accepts integers from 0-9, with 0 meaning no
        compression.
    transparent : bool, optional, default=False
        If True, the Axes patches will all be transparent;
        the Figure patch will also be transparent unless
        facecolor and/or edgecolor are specified via kwargs.
    bbox_inches : str or Bbox, default='tight'
        Bounding box in inches: only the given portion of the
        figure is saved. If 'tight', try to figure out the
        tight bbox of the figure.

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
