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
from types import SimpleNamespace
from typing import Any
import warnings
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from tqdm import tqdm

from visualastro.core.config import (
    get_config_value,
    config,
    _UNSET,
    _resolve_default
)
from visualastro.core.numerical_utils import to_array, to_list, _type_name
from visualastro.core.units import get_units


# KWARGS
# ------
KWARG_ALIASES: dict['str', tuple[str, ...]] = {
    'alpha': ('alphas', 'a'),
    'array_order': ('order',),
    'color': ('colors',),
    'edgecolor': ('edgecolors', 'ec'),
    'facecolor': ('facecolors', 'fc'),
    'marker': ('markers', 'm'),
    'markeredgecolor': ('markeredgecolors', 'mec'),
    'label': ('labels', 'l'),
    'linestyle': ('linestyles', 'ls'),
    'linewidth': ('linewidths', 'lw'),
    'size': ('sizes', 's'),
}


def _param(name: str, value: Any, default: Any) -> tuple[str, Any, Any]:
    """
    Helper function for defining a parameter in `_resolve_kwargs`.

    Parameters
    ----------
    name : str
        Name of the parameter.
    value : Any
        Value of the parameter.
    default : Any
        Fallback value if `value` is `_UNSET`.

    Returns
    -------
    tuple :
        Returns a tuple of name, value, and default, unchanged.
    """
    if not isinstance(name, str):
        raise ValueError(
            f'name must be a str! got: {_type_name(name)}'
        )
    return (name, value, default)


def _kwarg(name: str, default: Any) -> tuple[str, Any]:
    if not isinstance(name, str):
        raise ValueError(
            f'name must be a str! got: {_type_name(name)}'
        )
    return (name, default)


def _resolve_kwargs(
    params: list[tuple[str, Any, Any]],
    kwargs: dict,
    additional_kwargs: list[tuple[str, Any]] | None = None
) -> SimpleNamespace:
    out = {}

    for name, value, default in params:
        value = _pop_kwargs(kwargs, name, value)
        out[name] = _resolve_default(value, default)

    if additional_kwargs is not None:
        for name, default in additional_kwargs:
            out[name] = kwargs.pop(name, default)

    return SimpleNamespace(**out)


def _pop_kwargs(
    kwargs: dict[str, Any],
    name: str,
    default: Any = None
) -> Any:
    """
    Pop a keyword argument by canonical name or registered alias.

    Searches `kwargs` for the canonical `name` or any of its registered
    aliases (from `KWARG_ALIASES` in `visualastro.core.io`), removes the
    first match found, and returns its value.

    Parameters
    ----------
    kwargs : dict[str, Any]
        Dictionary of keyword arguments to mutate.
    name : str
        Canonical name corresponding to a key in `KWARG_ALIASES`.
    default : Any, optional
        Value returned if `name` and all aliases absent from `kwargs`.
        Default is None.

    Returns
    -------
    Any
        Value associated with `name` or its first matched alias.
        If no match found, returns `default`.

    Notes
    -----
    Mutates `kwargs` by removing the matched key. Search order is:
    canonical name first, then aliases in order defined in `KWARG_ALIASES`.

    Examples
    --------
    >>> from visualastro.core.io import KWARG_ALIASES
    >>> KWARG_ALIASES['edgecolor'] = ('edgecolors', 'ec')
    >>> kwargs = {'ec': 'red', 'lw': 2}
    >>> value = _pop_kwargs(kwargs, 'edgecolor', default='black')
    >>> value
    'red'
    >>> kwargs
    {'lw': 2}
    """
    for key in (name, *KWARG_ALIASES.get(name, ())):
        if (value := kwargs.get(key, _UNSET)) is not _UNSET:
            kwargs.pop(key)
            return value

    return default


def get_sci_from_hdul(
    hdul: fits.HDUList,
) -> tuple[NDArray, list[fits.Header]]:

    sci = [h for h in hdul if h.name == 'SCI' or h.name == 'DATA'] # type: ignore

    if len(sci) == 0:
        raise ValueError('No SCI HDUs found')

    data = np.stack([h.data for h in sci], axis=0) # type: ignore

    headers = [h.header for h in sci] # type: ignore

    return data, headers


def get_errors(
    hdul: fits.HDUList,
    dtype: DTypeLike | str | None = None,
    transpose: bool = False
) -> u.Quantity | NDArray | None:
    """
    Return the error array from an HDUList, falling back to square root
    of variance if needed. If a unit is found from the header, return
    the error array as a Quantity object instead.

    Parameters
    ----------
    hdul : fits.HDUList
        The HDUList object containing FITS extensions to search for errors or variance.
    dtype : np.dtype | str | None, optional, default=None
        The desired NumPy dtype of the returned error array.
        If None, uses the default unit set in ``config.default_dtype``.

    Returns
    -------
    errors : u.Quantity | np.ndarray | None
        The error array if found, or None if no suitable extension is present.
    """
    errors: NDArray | None = None
    error_unit = None

    for hdu in hdul[1:]:
        if (
            not isinstance(hdu, (fits.ImageHDU, fits.BinTableHDU, fits.CompImageHDU))
            or hdu.data is None
        ):
            continue

        extname = str(hdu.header.get('EXTNAME', '')).upper()

        if extname in config.hdu.error_extensions:
            dt = _get_dtype(hdu.data, dtype)
            errors = hdu.data.astype(dt, copy=False)

            if hdu.header is not None:
                try:
                    error_unit = u.Unit(str(hdu.header.get('BUNIT')))
                    errors *= error_unit
                except Exception:
                    warnings.warn(
                        'Error extension has invalid BUNIT; returning errors without units.'
                    )

            break

    # fallback to variance if no explicit errors
    if errors is None:
        for hdu in hdul[1:]:
            if (
                not isinstance(hdu, (fits.ImageHDU, fits.BinTableHDU, fits.CompImageHDU))
                or hdu.data is None
            ):
                continue

            extname = str(hdu.header.get('EXTNAME', '')).upper()

            if extname in config.hdu.variance_extensions:
                dt = _get_dtype(hdu.data, dtype)
                variance = hdu.data.astype(dt, copy=False)

                if hdu.header is not None:
                    try:
                        var_unit = u.Unit(str(hdu.header.get('BUNIT')))
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
    facecolor = _pop_kwargs(kwargs, 'facecolor', 'fc', default='auto')
    edgecolor = _pop_kwargs(kwargs, 'edgecolor', 'ec', default='auto')

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


def _get_dtype(
    data: ArrayLike | u.Quantity,
    dtype=None,
    default_dtype=_UNSET
) -> DTypeLike:
    """
    Returns the dtype from the provided data. Promotes
    integers and unsigned to floats.

    Used internally by visualastro data I/O functions.

    Parameters
    ----------
    data : ArrayLike | u.Quantity
        Input array whose dtype will be checked.
        Can be anything convertible to an NDArray
        by ``to_array``.
    dtype : data-type, optional, default=None
        If provided, this dtype is returned directly.
        If None, returns ``data.dtype`` if floating or
        ``default_dtype`` if integer or unsigned.
    default_dtype : data-type, optional, default=None
        Float type to use if ``data`` is integer or unsigned.
        If None, uses the default unit set in ``config.default_dtype``.

    Returns
    -------
    dtype : np.dtype
        User dtype if given, otherwise the array's float dtype
        or ``default_dtype`` if array is integer/unsigned.
    """
    default_dtype = _resolve_default(default_dtype, config.default_dtype)

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
