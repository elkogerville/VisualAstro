"""
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2026-03-11
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
"""

import os
import warnings
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from visualastro.core.config import get_config_value, config, _UNSET
from visualastro.core.numerical_utils import to_array, to_list
from visualastro.core.units import get_units


def get_dtype(data, dtype=None, default_dtype=None):
    '''
    Returns the dtype from the provided data. Promotes
    integers to floats if needed.
    Parameters
    ----------
    data : array-like
        Input array whose dtype will be checked.
    dtype : data-type, optional, default=None
        If provided, this dtype is returned directly.
        If None, returns `data.dtype` if floating or
        `np.float64` if integer or unsigned.
    default_dtype : data-type, optional, default=None
        Float type to use if `data` is integer or unsigned.
        If None, uses the default unit set in `config.default_dtype`.
    Returns
    -------
    dtype : np.dtype
        NumPy dtype object: user dtype if given, otherwise the array's
        float dtype or `default_dtype` if array is integer/unsigned.
    '''
    # get default config values
    default_dtype = get_config_value(default_dtype, 'default_unit')

    # return user dtype if passed in
    if dtype is not None:
        return np.dtype(dtype)

    data = to_array(data)
    # by default use data dtype if floating
    # if unsigned or int use default_dtype
    if np.issubdtype(data.dtype, np.floating):
        return np.dtype(data.dtype)
    else:
        return np.dtype(default_dtype)


def get_errors(
    hdul: fits.HDUList,
    dtype=None,
    transpose=False
):
    """
    Return the error array from an HDUList, falling back to square root
    of variance if needed. If a unit is found from the header, return
    the error array as a Quantity object instead.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        The HDUList object containing FITS extensions to search for errors or variance.
    dtype : data-type, optional, default=np.float64
        The desired NumPy dtype of the returned error array.

    Returns
    -------
    errors : np.ndarray or None
        The error array if found, or None if no suitable extension is present.
    """
    errors = None
    error_unit = None

    for hdu in hdul[1:]:
        if not isinstance(hdu, (fits.ImageHDU, fits.BinTableHDU, fits.CompImageHDU)) or hdu.data is None:
            continue

        extname = str(hdu.header.get('EXTNAME', '')).upper()

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
            if not isinstance(hdu, (fits.ImageHDU, fits.BinTableHDU, fits.CompImageHDU)) or hdu.data is None:
                continue

            extname = str(hdu.header.get('EXTNAME', '')).upper()

            if extname in {'VAR', 'VARIANCE', 'VAR_POISSON', 'VAR_RNOISE'}:
                dt = get_dtype(hdu.data, dtype)
                variance = hdu.data.astype(dt, copy=False)

                try:
                    var_unit = u.Unit(hdu.header.get('BUNIT'))
                    errors = np.sqrt(variance * var_unit)
                except (ValueError, TypeError, KeyError):
                    warnings.warn(
                        f'Variance extension {extname} has invalid or missing BUNIT; '
                        'returning errors without units.',
                        UserWarning
                    )
                    errors = np.sqrt(variance)
                break

    if transpose and errors is not None:
        errors = errors.T

    return errors


def write_arrays_2_file(
    arrays: NDArray | u.Quantity | list[NDArray | u.Quantity],
    filename: str,
    headers: list[str] | None = None,
    delimiter: str = '\t',
    precision: int = 16,
) -> None:
    """
    Save multiple 1D arrays as columns to a text file with optional headers and units.

    Parameters
    ----------
    arrays : NDArray, Quantity, or list of (NDArray or Quantity)
        One or more 1D arrays to be written as columns.
    filename : str
        Output filename.
    headers : list of str, optional
        Column header labels. If fewer headers than arrays are given,
        remaining headers are left blank.
    delimiter : str, optional
        Column delimiter written between values.
    precision : int, optional
        Number of digits in scientific notation for numeric values.

    Notes
    -----
    If arrays contain quantities, their units are written on a second header
    line using the format '[unit]'. Numeric values are written in scientific
    notation.

    Examples
    --------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> arr1 = np.array([1.23e-5, 4.56e-3, 7.89e2]) * u.m
    >>> arr2 = np.array([9.87e6, 6.54e4, 3.21e-1]) * u.s
    >>> save_arrays_to_file([arr1, arr2], 'data.txt', headers=['Distance', 'Time'])
    """

    arrays = to_list(arrays)

    lengths = [len(a) for a in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f'All arrays must have the same length. Got lengths: {lengths}')

    n_cols = len(arrays)

    if headers is None:
        headers = [''] * n_cols
    else:
        headers = list(headers) + [''] * (n_cols - len(headers))

    units = get_units(arrays)
    unit_strs = [f'[{u.to_string()}]' if u is not None else '' for u in units]

    data_arrays = [getattr(a, 'value', np.asarray(a)) for a in arrays]

    data = np.column_stack(data_arrays)

    fmt = f'%.{precision}e'
    sample = fmt % data.flat[0]
    col_width = max(len(sample), *(len(h) for h in headers), *(len(u) for u in unit_strs))

    header_line = delimiter.join(h.center(col_width) for h in headers)
    unit_line = delimiter.join(u.center(col_width) for u in unit_strs)

    np.savetxt(
        filename,
        data,
        fmt=f'%{col_width}.{precision}e',
        delimiter=delimiter,
        header=header_line + '\n' + unit_line,
        comments=''
    )


def write_cube_2_fits(cube, filename, overwrite=False):
    """
    Write a 3D data cube to a series of FITS files.

    Parameters
    ----------
    cube : ndarray (N_frames, N, M)
        Data cube containing N_frames images of shape (N, M).
    filename : str
        Base filename (without extension). Each
        output file will be saved as ``'{filename}_i.fits'``.
    overwrite : bool, optional, default=False
        If True, existing files with the same name
        will be overwritten.

    Notes
    -----
    Prints a message indicating the number of
    frames and the base filename.
    """
    N_frames, N, M = cube.shape
    print(f'Writing {N_frames} fits files to {filename}_i.fits')
    for i in tqdm(range(N_frames)):
        output_name = filename + f'_{i}.fits'
        fits.writeto(output_name, cube[i], overwrite=overwrite)


# Figure I/O Operations
# ---------------------
def get_kwargs(kwargs, *names, default=None):
    '''
    Return the first matching kwarg value from a list of possible names.

    Parameters
    ----------
    kwargs : dict
            Dictionary of keyword arguments, typically taken from ``**kwargs``.
    *names : str
        One or more possible keyword names to search for. The first name found
        in ``kwargs`` with a non-None value is returned.
    default : any, optional, default=None
        Value to return if none of the provided names are found in ``kwargs``.
        Default is None.

    Returns
    -------
    value : any
        The value of the first matching keyword argument, or `default` if
        none are found.
    '''
    for name in names:
        if name in kwargs and kwargs[name] is not None:
            return kwargs[name]

    return default


def save_array(arr, filename, fmt=None):
    '''
    Save a NumPy array to disk.

    Parameters
    ----------
    arr : np.ndarray
        Array to be saved.
    filename : str
        Output filename. Should include the extension, otherwise
        the array will be saved according to `fmt`.
    fmt : {'.dat', '.csv', '.npy', '.txt'} or None, optional, default=None
        Default output extension. Only used if filename has
        no extension. If None, uses the default value set by
        `config.save_format`. The format is correctly recognized
        even if the '.' is omitted.

    Raises
    ------
    ValueError :
        If provided `fmt` is invalid.
    '''
    VALID_EXTS = {'.dat', '.csv', '.npy', '.txt'}

    fmt = get_config_value(fmt, 'save_format')
    fmt = f".{fmt.lstrip('.')}"

    if fmt not in VALID_EXTS:
        raise ValueError(f"Invalid fmt '{fmt}'. Must be one of {VALID_EXTS}")

    if not isinstance(arr, np.ndarray):
        raise TypeError('arr must be a NumPy ndarray!')

    root, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext == '':
        filename = root + fmt

    if ext == '.npy':
        np.save(filename, arr)
    elif ext in {'.txt', '.dat', '.csv'}:
        np.savetxt(filename, arr)
    else:
        raise ValueError(f'Unsupported file extension: {ext}')


def save_quantity(quantity, filename):
    '''
    Save an astropy Quantity to disk.

    Parameters
    ----------
    arr : Quantity
        Array to be saved.
    filename : str
        Output filename with ot without `.npz` extension.

    Raises
    ------
    TypeError :
        If `quantity` is not a astropy.units.Quantity.
    '''
    if not isinstance(quantity, u.Quantity):
        raise TypeError("quantity must be an astropy.units.Quantity")

    np.savez(
        filename,
        data=quantity.value,
        unit=str(quantity.unit)
    )


def load_quantity(filename):
    '''
    Load a saved quantity array.

    Parameters
    ----------
    filename : str
        Path to quantity array. Should be a `.npz`.

    Returns
    -------
    Quantity : astropy.units.Quantity
        Numpy array with units.
    '''
    with np.load(filename) as f:
        return f['data'] * u.Unit(f['unit'].item())


def savefig(
    dpi=None,
    pdf_compression=None,
    transparent=False,
    bbox_inches=_UNSET,
    **kwargs
):
    '''
    Saves current figure to disk as a
    eps, pdf, png, or svg, and prompts
    user for a filename and format.

    Parameters
    ----------
    dpi : float, int, or None, optional, default=None
        Resolution in dots per inch. If None, uses
        the default value set by `config.dpi`.
    pdf_compression : int or None, optional, default=None
        'Pdf.compression' value for matplotlib.rcParams.
        Accepts integers from 0-9, with 0 meaning no
        compression. If None, uses the default value
        set by `config.pdf_compression`.
    transparent : bool, optional, default=False
        If True, the Axes patches will all be transparent;
        the Figure patch will also be transparent unless
        facecolor and/or edgecolor are specified via kwargs.
    bbox_inches : str, Bbox, or None, default=`_UNSET`
        Bounding box in inches: only the given portion of the
        figure is saved. If 'tight', try to figure out the
        tight bbox of the figure. If `_UNSET`, uses
        the default value set by `config.bbox_inches`.

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
    # ---- KWARGS ----
    facecolor = get_kwargs(kwargs, 'facecolor', 'fc', default='auto')
    edgecolor = get_kwargs(kwargs, 'edgecolor', 'ec', default='auto')

    # get default config values
    dpi = get_config_value(dpi, 'dpi')
    pdf_compression = get_config_value(pdf_compression, 'pdf_compression')
    bbox_inches = config.bbox_inches if bbox_inches is _UNSET else bbox_inches

    allowed_formats = config.allowed_formats
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
