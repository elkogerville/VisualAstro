"""
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2026-06-01
Description:
    Functions for I/O operations within visualastro.
"""

import os
from types import SimpleNamespace
from typing import Any, Literal
import warnings

from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from tqdm import tqdm

from visualastro.core.config import (
    get_config_value,
    config,
    _Unset,
    _UNSET,
    _resolve_default
)
from visualastro.core.numerical_utils import to_array, to_list
from visualastro.core.units import get_units


# KWARGS
# ------
# define aliases here! each key represents a parameter used in visualastro functions,
# and their values are aliases of that parameter. this way any each set can be used
# in a function, provided that the function uses the `_resolve_kwargs` interface.
# ensure that the values are tuples even for single items, meaning: 'key': (value,),
KWARG_ALIASES: dict['str', tuple[str, ...]] = {
    'color': ('colors',),
    'edgecolor': ('edgecolors', 'ec'),
    'facecolor': ('facecolors', 'fc'),
    'marker': ('markers', 'm'),
    'size': ('sizes', 's'),
    'alpha': ('alphas', 'a'),
    'markeredgecolor': ('markeredgecolors', 'mec'),
    'label': ('labels', 'l'),
    'legend_handles': ('legend_handle',),
    'legend_labels': ('legend_label',),
    'linecolor': ('linecolors', 'lc'),
    'linestyle': ('linestyles', 'ls'),
    'linewidth': ('linewidths', 'lw'),
    'linealpha': ('linealphas', 'la'),
    'array_order': ('order',),
    'colorbar': ('colorbars', 'cbar', 'cbars'),
    'cbar_width': ('colorbar_width',),
    'cbar_pad': ('colorbar_pad',),
    'cbar_label': ('colorbar_label', 'colorbar_labels', 'cbar_labels'),
    'cbar_tick_which': ('colorbar_tick_which',),
    'cbar_tick_dir': ('colorbar_tick_dir', 'colorbar_tick_direction', 'cbar_tick_direction'),
    'gridlines': ('gridline', 'grid_line', 'grid_lines'),
    'ellipses': ('ellipse',),
    'text_color': ('textcolor',),
    'unit_fmt': ('unit_format', 'unit_label_fmt', 'unit_label_format'),
    'axis_style': ('axes_style',),
}


ParamSpec = tuple[str, Any, Any]
KwargSpec = tuple[str, Any]


def _resolve_kwargs(
    kwargs: dict,
    params: list[ParamSpec] | None = None,
    additional_kwargs: list[KwargSpec] | None = None,
    copy_kwargs: list[KwargSpec] | None = None
) -> SimpleNamespace:
    """
    Resolve keyword arguments into a namespace of normalized parameters.

    `params` follow the form: [_param('name', var, default), ...].
    `additional_kwargs` follow the form: [_kwarg('name', default), ...].

    Parameters defined in `params` are intended for arguments that also
    exist in the parent function signature. Their values are first processed
    through `_pop_kwargs` to handle aliases, then passed through
    `_resolve_default` so that `_UNSET` values are replaced with their
    configured defaults.

    Parameters defined in `additional_kwargs` are intended for optional
    keyword-only passthrough arguments that are not part of the parent
    function signature. These are resolved using `_pop_kwargs`.

    Both `params` and `additional_kwargs` are aware to aliases defined
    in `visualastro.core.io.KWARG_ALIASES`.

    The resolved values are returned as attributes on a
    :class:`types.SimpleNamespace`.

    Parameters
    ----------
    kwargs : dict
        Dictionary of keyword arguments to resolve. Resolved parameters
        are popped in place.
    params : list[ParamSpec] | None, optional, default=None
        Sequence of `(name, value, default)` tuples describing parameters
        defined in the parent function signature.

        Each parameter is resolved as:

            1. Retrieve the value from `kwargs` using `_pop_kwargs`
            2. Replace unset sentinel values using `_resolve_default`

    additional_kwargs : list[KwargSpec] | None, optional, default=None
        Sequence of `(name, default)` tuples describing optional keyword
        arguments that should be retrieved directly from `kwargs` using
        fallback defaults.
    copy_kwargs : list[KwargSpec] | None, optional, default=None
        Sequence of `(name, default)` tuples describing keyword arguments
        to preserve in `kwargs` without popping.

        Values are retrieved from `kwargs` using `_get_kwargs` (non-mutating),
        allowing the original key-value pairs to remain available after
        resolution. Useful when downstream functions need access to
        arguments that are also resolved into the returned namespace.

        Aliases defined in `KWARG_ALIASES` are respected during lookup.

    Returns
    -------
    types.SimpleNamespace
        Namespace containing all resolved parameters as attributes.

    Raises
    ------
    ValueError :
        If `params`, `additional_kwargs`, and `copy_kwargs` are None.

    Examples
    --------
    >>> params = _resolve_kwargs(
    ...     kwargs,
    ...     [
    ...         _param('alpha', alpha, config.alpha),
    ...         _param('color', color, config.color),
    ...     ],
    ...     [
    ...         _kwarg('label', None),
    ...         _kwarg('cmap', config.cmap),
    ...     ]
    ... )
    >>>
    >>> params.alpha
    0.8
    >>> params.cmap
    'viridis'

    Notes
    -----
    `params` should be used for arguments originating from the function
    signature, especially when `_UNSET` or aliases must be handled.

    `params`, `additional_kwargs`, and `copy_kwargs` do not need aliases defined.

    `additional_kwargs` should be used for optional passthrough keyword
    arguments that behave like standard `kwargs.pop` retrievals.

    See Also
    --------
    visualastro.core.io._pop_kwargs
    visualastro.core.config._resolve_default
    visualastro.core.io._param
    visualastro.core.io._kwarg
    """
    if params is None and additional_kwargs is None and copy_kwargs is None:
        raise ValueError(
            'params, additional_kwargs, and copy_kwargs cannot all be None!'
        )

    out = {}

    if copy_kwargs is not None:
        for name, default in copy_kwargs:
            out[name] = _get_kwargs(kwargs, name, default)

    if params is not None:
        for name, value, default in params:
            value = _pop_kwargs(kwargs, name, value)
            out[name] = _resolve_default(value, default)

    if additional_kwargs is not None:
        for name, default in additional_kwargs:
            out[name] = _pop_kwargs(kwargs, name, default)

    return SimpleNamespace(**out)


def _get_kwargs(
    kwargs: dict[str, Any],
    name: str,
    default: Any = None
) -> Any:
    """
    Retrieve a keyword argument by canonical name or registered alias.

    Identical to `_pop_kwargs` but does not mutate `kwargs`.

    Parameters
    ----------
    kwargs : dict[str, Any]
        Dictionary of keyword arguments to query.
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
    """
    for key in (name, *KWARG_ALIASES.get(name, ())):
        if (value := kwargs.get(key, _UNSET)) is not _UNSET:
            return value
    return default


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

    Identical to `_get_kwargs` but does mutate `kwargs`.

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


def _extract_kwargs(
    kwargs: dict,
    params: list[ParamSpec] | None = None,
    additional_kwargs: list[KwargSpec] | None = None,
    copy_kwargs: list[KwargSpec] | None = None
) -> dict:
    """
    Helper function for to return the output of _resolve_kwargs
    as a `dict` instead of `SimpleNamespace`.

    See `visualastro.core.io._resolve_kwargs` for documentation.
    """
    return vars(_resolve_kwargs(kwargs, params, additional_kwargs, copy_kwargs))


def _param(name: str, value: Any, default: Any) -> ParamSpec:
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
            f'name must be a str! got: {type(name).__name__}'
        )
    return (name, value, default)


def _kwarg(name: str, default: Any) -> KwargSpec:
    if not isinstance(name, str):
        raise ValueError(
            f'name must be a str! got: {type(name).__name__}'
        )
    return (name, default)


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
        If None, uses the default unit set in `config.default_dtype`.

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
    delimiter: str = r'\t',
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
    >>> distance = np.array([1.23e-5, 4.56e-3, 7.89e2]) * u.m
    >>> time = np.array([9.87e6, 6.54e4, 3.21e-1]) * u.s
    >>> save_arrays_to_file([distance, time], 'data.txt', headers=['Distance', 'Time'])
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
        output file will be saved as `'{filename}_i.fits'`.
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
        no extension. If None, uses
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
        raise TypeError('quantity must be an astropy.units.Quantity')

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
    filename: str | None = None,
    dpi: float | Literal['figure'] | _Unset = _UNSET,
    pdf_compression: int | _Unset = _UNSET,
    transparent: bool = False,
    bbox_inches: str | Bbox | None | _Unset = _UNSET,
    **kwargs
) -> None:
    """
    Saves current figure to disk as a `.eps`, `.pdf`, `.png`,
    or `.svg`, and prompts user for a filename and format.

    Parameters
    ----------
    filename : str | None, optional, default=None
        If `None`, prompts user for a filename and format.
        If `str`, uses `filename` as the filename directly.
    dpi : float | {'figure'} | _Unset, optional, default=_UNSET
        Resolution in dots per inch. If `'figure'`, uses the
        figure dpi. If `_UNSET`, uses `config.savefig.dpi`.
    pdf_compression : int | _Unset, optional, default=_UNSET
        Pdf compression value for matplotlib.rcParams.
        Accepts integers from 0-9, with 0 meaning no
        compression. If `_UNSET`, uses `config.savefig.pdf_compression`.
    transparent : bool, optional, default=False
        If `True`, the Axes patches will all be transparent;
        the Figure patch will also be transparent unless
        `facecolor` and/or `edgecolor` are specified via kwargs.
    bbox_inches : str | Bbox | None | _Unset, default=`_UNSET`
        Bounding box in inches: only the given portion of the
        figure is saved. If `'tight'`, try to figure out the
        tight bbox of the figure. If `_UNSET`, uses
        `config.savefig.bbox_inches`.
    facecolorcolor : {'auto'} | str, optional, default='auto'
        The facecolor of the figure. If `'auto'`,
        use the current figure facecolor.
    edgecolorcolor : {'auto'} | str, default='auto'
        The edgecolor of the figure. If 'auto',
        use the current figure edgecolor.
    """
    params = _resolve_kwargs(
        kwargs,
        [
            _param('dpi', dpi, config.savefig.dpi),
            _param('pdf_compression', pdf_compression, config.savefig.pdf_compression),
            _param('transparent', transparent, config.savefig.transparent),
            _param('bbox_inches', bbox_inches, config.savefig.bbox_inches),
        ],
        [
            _kwarg('facecolor', 'auto'),
            _kwarg('edgecolor', 'auto'),
        ]
    )

    allowed_formats = config.savefig.allowed_formats
    # prompt user for filename, and extract extension
    if filename is None:
        filename = input('Input filename for image (ex: myimage.pdf): ').strip()
    elif not isinstance(filename, str):
        raise TypeError(f'filename must be str or None, got {type(filename).__name__}')

    if not filename:
        raise ValueError('filename cannot be empty')

    basename, *extension = filename.rsplit('.', 1)
    # if extension exists, and is allowed, extract extension from list
    if extension and extension[0].lower() in allowed_formats:
        extension = extension[0]
    # else prompt user to input a valid extension
    else:
        extension = ''
        while extension not in allowed_formats:
            extension = (
                input(f"Please choose a format from ({', '.join(allowed_formats)}): ")
                .strip()
                .lower()
            )
    filename = f'{basename}.{extension}'

    with plt.rc_context(rc={'pdf.compression': int(params.pdf_compression)} if extension == 'pdf' else {}):
        plt.savefig(
            fname=filename,
            format=extension,
            transparent=params.transparent,
            bbox_inches=params.bbox_inches,
            facecolor=params.facecolor,
            edgecolor=params.edgecolor,
            dpi=params.dpi
        )


def _find_root_path(marker='pyproject.toml') -> Path:
    """Find path to VisualAstro root directory."""
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f'Could not find project root via {marker}')

def _get_src_path() -> Path:
    """Find path to VisualAstro source directory."""
    rootdir = _find_root_path()
    srcpath = rootdir / 'src' / 'visualastro'
    if srcpath.exists():
        return srcpath
    raise FileNotFoundError(
        'Fatal error! Could not find src path! This is not supposed to happen >.<'
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
        by `to_array`.
    dtype : data-type, optional, default=None
        If provided, this dtype is returned directly.
        If None, returns `data.dtype` if floating or
        `default_dtype` if integer or unsigned.
    default_dtype : data-type, optional, default=None
        Float type to use if `data` is integer or unsigned.
        If None, uses the default unit set in `config.default_dtype`.

    Returns
    -------
    dtype : np.dtype
        User dtype if given, otherwise the array's float dtype
        or `default_dtype` if array is integer/unsigned.
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
