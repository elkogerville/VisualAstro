"""
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2026-10-25
Description:
    Plotting input normalization functions.
"""

from collections.abc import Sequence
from typing import Any, Literal

from astropy import units as u
from matplotlib.typing import ColorType
import numpy as np
from numpy.typing import NDArray

from visualastro.core.config import (
    config,
    _Unset,
    _UNSET,
    _resolve_default
)
from visualastro.core.array_types import (
    _is_1d,
    _is_array_like,
    _is_ndarray_or_quantity_array,
    _is_scalar,
    _is_wrapped_1d,
)
from visualastro.core.data import as_list, get_value, to_list, _array_length, _ndim
from visualastro.core.kwargs import (
    _extract_kwargs, _params_from_explicit, _resolve_kwargs
)
from visualastro.plotting.core.colormaps import get_cmap, get_norm
from visualastro.plotting.core.colors.colors import get_color, get_colors
from visualastro.plotting.core.colors.utils import _is_color_sequence


def _normalize_order(order: str) -> Literal['c', 'fortran']:
    """Normalize array order specification.

    Parameters
    ----------
    order : str
        Array order. Accepted (case-insensitive): 'c', 'f', 'fortran'.

    Returns
    -------
    {'c', 'fortran'}
        Canonical order string.

    Raises
    ------
    ValueError
        If `order` is not recognized.
    """
    _ORDER_ALIASES = {'c': 'c', 'f': 'fortran', 'fortran': 'fortran'}
    try:
        return _ORDER_ALIASES[order.lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f'order must be one of {sorted(_ORDER_ALIASES)}, got {order!r}.'
        ) from None


def _get_vector(
    obj: NDArray | u.Quantity,
    idx: int,
    order: Literal['c', 'fortran'],
) -> NDArray | u.Quantity:
    """
    Return the `idx`-th variable of a 2D array.

    Parameters
    ----------
    obj : NDArray | u.Quantity
        2D array.
    idx : int
        Variable index.
    order : {'c', 'fortran'}
        'c': variables along columns, ``obj[:, idx]``.
        'fortran': variables along rows, ``obj[idx, :]``.

    Returns
    -------
    NDArray | u.Quantity
        1D view of the selected variable.
    """
    return obj[:, idx] if order == 'c' else obj[idx, :]


def _extract_xy_from_ndarray2D(
    obj: NDArray | u.Quantity,
    order: Literal['c', 'fortran'],
    xy_indices: tuple[int, int],
) -> tuple[NDArray | u.Quantity, NDArray | u.Quantity]:
    """Extract x and y vectors from a 2D array.

    Parameters
    ----------
    obj : NDArray | u.Quantity
        2D array.
    order : {'c', 'fortran'}
        Variable layout; see `_get_vector`.
    xy_indices : tuple[int, int]
        Indices of the (x, y) variables.

    Returns
    -------
    tuple[NDArray | u.Quantity, NDArray | u.Quantity]
        x and y vectors (views, not copies).

    Raises
    ------
    ValueError
        If `obj` is not 2D or `xy_indices` does not have length 2.
    """
    if not isinstance(obj, np.ndarray) or obj.ndim != 2:
        raise ValueError(
            'Input must be a 2D array, i.e. np.random.rand(10, 2).'
        )
    if len(xy_indices) != 2:
        raise ValueError('xy_indices must be a tuple[int, int].')

    ix, iy = xy_indices
    return _get_vector(obj, ix, order), _get_vector(obj, iy, order)


def _extract_xy(
    *data: float | u.Quantity | NDArray | Sequence[float | u.Quantity],
    kwargs: dict[str, Any],
    **explicit
):
    """
    Resolve positional plot inputs into an `(x, y)` pair.

    Parameters
    ----------
    *data : float | u.Quantity | NDArray | Sequence[float | u.Quantity]
        One or two positional inputs. Accepted signatures:

        - `(y)`: scalar, 0D or 1D. Returned as `(None, y)`.
        - `(y)`: 2D `ndarray`. Split into `(x, y)` via `order` and
          `xy_indices`.
        - `(x, y)`: returned unchanged.

    kwargs : dict[str, Any]
        Keyword arguments forwarded to matplotlib.
    **explicit
        Any of the following keyword arguments:

        - order : {'c', 'fortran'} | _Unset, optional, default=_UNSET
            Array order of 2D input. Single-input only. Defines what a
            column is for `xy_indices`.

            - 'c': row-major, shape (N, M)
            - 'fortran': column-major, shape (M, N)

            Accepted case-insensitively, with 'f' as alias of 'fortran'.
            If `_UNSET`, uses `config.array_order`.
        - xy_indices : tuple[int, int] | None | _Unset, optional, default=_UNSET
            Variable selection for 2D input. Single-input only.

            - tuple `(i, j)`: `x` = variable `i`, `y` = variable `j`
              (negative indices allowed).
            - `None`: no split. Returns `(None, obj)`, with `obj` transposed
              if `order` is 'fortran'.

            If `_UNSET`, uses `config.xy_indices`.

    Returns
    -------
    x : float | u.Quantity | NDArray | Sequence[float | u.Quantity] | None
        X plotting data. `None` if only `y` was given (implicit x).
    y : float | u.Quantity | NDArray | Sequence[float | u.Quantity]
        Y plotting data.

    Raises
    ------
    ValueError
        - Number of positional inputs is not 1 or 2.
        - Two inputs and `order` or `xy_indices` is given.
        - Two inputs and `y` is `None`.
        - Single input is `None`.
        - Single input has ndim > 2.
        - `order` is not recognized.
        - `xy_indices` does not have length 2.
    IndexError
        If an entry of `xy_indices` is out of bounds for the array.

    Notes
    -----
    - 2D splitting returns views, not copies.
    - Units of `u.Quantity` inputs are preserved.
    - Scalar, 0D and 1D single inputs ignore `order` and `xy_indices`.
    - 2D input must be an `ndarray` (or subclass). Nested sequences are
      not supported.
    - With two inputs, `x=None` is accepted and returned unchanged.
    """
    n = len(data)
    if n not in (1, 2):
        raise ValueError(f'Expected 1 or 2 positional inputs, got {n}.')

    if n == 2:
        if (
            explicit.get('order', _UNSET) is not _UNSET
            or explicit.get('xy_indices', _UNSET) is not _UNSET
        ):
            raise ValueError(
                '`order` and `xy_indices` apply only to a single input.'
            )
        x, y = data
        if y is None:
            raise ValueError('`y` must not be None.')
        return x, y

    defaults = {
        'order': config.array_order,
        'xy_indices': config.xy_indices,
    }
    params = _resolve_kwargs(
        kwargs,
        params=_params_from_explicit(explicit, defaults),
    )

    obj = data[0]
    if obj is None:
        raise ValueError('Input must not be None.')

    if _is_scalar(obj):
        return None, obj

    ndim = _ndim(obj)
    if ndim <= 1:
        return None, obj
    if ndim != 2:
        raise ValueError(f'Input must be 1D or 2D, got {ndim}D.')

    order = _normalize_order(params.order)
    indices = params.xy_indices

    if indices is None:
        return None, (obj.T if order == 'fortran' else obj)

    indices = tuple(indices)
    if len(indices) != 2:
        raise ValueError('xy_indices must be a tuple[int, int].')
    return _extract_xy_from_ndarray2D(obj, order, indices)


def _resolve_implicit_x(
    x: float | u.Quantity | NDArray | Sequence[float | u.Quantity] | None,
    y: float | u.Quantity | NDArray | Sequence[float | u.Quantity],
    *,
    mode: Literal['plot', 'scatter'],
) -> tuple:
    """
    Resolve the plotting data for a matplotlib call when `x` may be implicit.

    Parameters
    ----------
    x : float | u.Quantity | NDArray | Sequence[float | u.Quantity] | None
        X data. `None` means implicit x.
    y : float | u.Quantity | NDArray | Sequence[float | u.Quantity]
        Y data.
    mode : {'plot', 'scatter'}
        Target matplotlib method.

        - 'plot': implicit x is delegated to matplotlib, returns `(y,)`.
        - 'scatter': implicit x is resolved as `arange(N)`, returns `(x, y)`.

    Returns
    -------
    args : tuple
        Positional arguments to unpack into the matplotlib call.

    Raises
    ------
    ValueError
        - `mode` is not recognized.
        - `mode` is 'scatter', `x` is `None` and `y` is not scalar or 1D.

    Notes
    -----
    - Scalar `y` is promoted to 1D, so it yields `x = [0]`.
    - Explicit `x` is passed through unchanged in both modes.
    - 2D `y` with implicit x is rejected for 'scatter' because `scatter`
      flattens its inputs and the x/y pairing would be ambiguous.
    """
    if mode not in ('plot', 'scatter'):
        raise ValueError(f"`mode` must be 'plot' or 'scatter', got {mode!r}.")

    if mode == 'scatter' and _ndim(y) > 1:
        raise ValueError(
            f'scatter requires scalar or 1D y, got {_ndim(y)}D. '
            'Select two variables with `xy_indices=(i, j)` or plot columns '
            'separately.'
        )

    if x is not None:
        return x, y

    if mode == 'plot':
        return (y,)

    return np.arange(_array_length(y)), y


def _resolve_xy_plot_inputs(
    *data: float | u.Quantity | NDArray | Sequence[float | u.Quantity],
    kwargs: dict[str, Any],
    mode:  Literal['plot', 'scatter'],
    **explicit
):
    """
    Resolve positional plot inputs into arguments for a matplotlib call.

    Combines `_extract_xy` and `_resolve_implicit_x`: first splits the
    positional inputs into an `(x, y)` pair, then fills in or defers an
    implicit `x` according to `mode`.

    Parameters
    ----------
    *data : float | u.Quantity | NDArray | Sequence[float | u.Quantity]
        One or two positional inputs. Accepted signatures:

        - `(y)`: scalar, 0D or 1D. `x` is implicit.
        - `(y)`: 2D `ndarray`. If `xy_indices` is a tuple, split into
          `(x, y)` via `order` and `xy_indices`. If `xy_indices` is `None`,
          returned as `(None, y)` with no split (transposed if `order` is
          'fortran').
        - `(x, y)`: used as given.

    kwargs : dict[str, Any]
        Keyword arguments forwarded to matplotlib.
    mode : {'plot', 'scatter'}
        Target matplotlib method.

        - 'plot': implicit x is delegated to matplotlib, returns `(y,)`.
        - 'scatter': implicit x is resolved as `arange(N)`, returns `(x, y)`.

    **explicit
        Any of the following keyword arguments, see `_extract_xy`:

        - order : {'c', 'fortran'} | _Unset, optional, default=_UNSET
            Array order of 2D input. Single-input only. If `_UNSET`,
            uses `config.array_order`.
        - xy_indices : tuple[int, int] | None | _Unset, optional, default=_UNSET
            Variable selection for 2D input. Single-input only. If
            `_UNSET`, uses `config.xy_indices`.

    Returns
    -------
    args : tuple
        Positional arguments to unpack into the matplotlib call, i.e.
        `ax.plot(*args)` or `ax.scatter(*args)`.

    Raises
    ------
    ValueError
        - Number of positional inputs is not 1 or 2.
        - Two inputs and `order` or `xy_indices` is given.
        - Two inputs and `y` is `None`.
        - Single input is `None`.
        - Single input has ndim > 2.
        - `order` is not recognized.
        - `xy_indices` does not have length 2.
        - `mode` is not recognized.
        - `mode` is 'scatter' and `y` is 2D or higher after splitting.

    IndexError
        If an entry of `xy_indices` is out of bounds for the array.

    See Also
    --------
    _extract_xy : Resolve positional inputs into `(x, y)`.
    _resolve_implicit_x : Resolve an implicit `x` for a given `mode`.
    """
    x, y = _extract_xy(*data, kwargs=kwargs, **explicit)
    return _resolve_implicit_x(x, y, mode=mode)


def _extract_xyz(
    *data: NDArray | u.Quantity | Sequence[NDArray | u.Quantity | float] | float,
    order: Literal['c', 'fortran'] | _Unset = _UNSET,
    index_spec: tuple[int, int, int] | _Unset = _UNSET
) -> list[tuple]:
    """
    Extract X, Y, Z coordinates from a variety of supported input formats.

    Supported inputs:

    * 2D array with at least 3 columns/rows, or a list of such arrays.
    * Three 1D array-like or a list of such: `(X, Y, Z)`.
    * Three sequences of 1D array-like: `([x1,x2,], [y1,y2,], [z1,z2,])`.
    * Three scalars: `(x, y, z)`.

    Parameters
    ----------
    *data : NDArray | u.Quantity | Sequence[NDArray | u.Quantity | float] | float
        Input(s) to extract X, Y, Z values from.
        ndarray | Sequence[ndarray] | tuple[array-like, array-like, array-like] | tuple[scalar, scalar, scalar]

        Supported calling conventions:

        * NDArray | Sequence[NDArray]:

            * Each array should be 2D and have at least 3 axes. The extracted
            axes are set by `index_spec`.

        * Three 1D array-like:

            * Three 1D arrays of the same shape.

        * Three Sequences[1D array-like]

            * Three Sequences each containing an array-like to plot.
            Corresponding elements across sequences must share the same shape,
            i.e. `shape(xi) == shape(yi) == shape(zi)`.

    order : {'c', 'fortran'}, optional
        Array traversal order used when extracting from 2D arrays.
        If `_UNSET` uses `config.array_order`.
    index_spec : tuple[int, int, int], optional
        Axis indices `(i_x, i_y, i_z)` identifying X, Y, Z columns/rows
        in the 2D input array. If `_UNSET`, uses `config.index_specification_3D`.

    Returns
    -------
    list[tuple] :
        List of `(x, y, z)` tuples. Each tuple represents a collection to plot,
        and can either be a tuple of scalars or a tuple of 1D array-like.

    Raises
    ------
    ValueError :
        If `len(data) == 3` and the three array-likes are not uniformly
        1D or flattenable to 1D (shapes `(N,1)` or `(1,N)`).
    ValueError
        If the input signature matches none of the supported conventions.
    """
    array_order = _resolve_default(order, config.array_order)
    index_spec = _resolve_default(index_spec, config.index_specification_3D)

    # input is either 2D NDArray or Sequence[2D NDArray]
    if len(data) == 1:
        obj = to_list(data[0])
        result = []
        for o in obj:
            result.append(_extract_xyz_from_ndarray(o, array_order, index_spec))
        return result

    # input is tuple[Any, Any, Any]
    elif len(data) == 3:
        if all(_is_array_like(d) for d in data):
            # input is tuple[array, array, array] where array is either
            # NDArray or u.Quantity array
            if (all(_is_ndarray_or_quantity_array(d) for d in data)):
                return [tuple(data)]
            # input is tuple[Sequence[array-like], Sequence[array-like], Sequence[array-like]]
            # where array-like is all 1D
            if (
                all(isinstance(d, (list, tuple)) for d in data) and
                all(_is_1d(sublist) for d in data for sublist in d)
            ):
                return [tuple(d) for d in zip(*data)]

            # input is tuple[array-like, array-like, array-like]
            # where array-like is either all 1D or all (N,1) or (1,N)
            if (
                all(_is_wrapped_1d(d) for d in data) or
                all(_is_1d(d) for d in data)
            ):
                return [tuple(d) for d in zip(*data)]

            raise ValueError(
                'inputs must either be all 1D or must all be 2D '
                ' arrays with shapes: (N,1) or (1,N)!'
            )

        # input is tuple[scalar, scalar, scalar]
        if all(_is_scalar(d) for d in data):
            return as_list((data[0], data[1], data[2]))

    raise ValueError(
        'inputs must either be a 2D array, list[2D arrays] or X, Y, Z inputs! '
        'X, Y, Z must all be array-like or list of such.'
    )


def _extract_xyz_from_ndarray(
    obj: NDArray | u.Quantity,
    order: Literal['c', 'fortran'],
    index_spec: tuple[int, int, int]
) -> tuple[NDArray | u.Quantity, NDArray | u.Quantity, NDArray | u.Quantity]:
    """
    Given a 2D NDArray with shape (N,3) or (3,N), extract the x,y,z values.

    Parameters
    ----------
    obj : np.ndarray | u.Quantity
        2D array with shape (N,3) or (3,N), depending on `order`.
    order : {'c', 'fortran'}
        Array order. If `'c'`, `obj` should have shape (N,3).
        If `'fortran'`, `obj` should have shape (3,N).
    index_spec : tuple[int, int, int]
        Specifies which columns (`order='c'`) or rows (`order='fortran'`)
        should be used for extraction.

    Returns
    -------
    tuple[NDArray | u.Quantity, NDArray | u.Quantity, NDArray | u.Quantity] :
        X, Y, and Z values extracted from `obj`. Are all 1D arrays.
        Units are preserved.

    Examples
    --------
    >>> a = np.random.rand(10,3)
    >>> _extract_xyz_from_ndarray(a, order='c', index_spec=[0,1,2])
    (a[:,0], a[:,1], a[:,2])

    >>> _extract_xyz_from_ndarray(a.T, order='fortran', index_spec=[0,1,2])
    (a[0,:], a[1,:], a[2,:])

    >>> b = np.random.rand(10,8)
    >>> _extract_xyz_from_ndarray(a, order='c', index_spec=[0,4,7])
    (a[0,:], a[4,:], a[7,:])
    """
    if (
        isinstance(obj, (np.ndarray, u.Quantity))
        and _is_array_like(obj) and obj.ndim == 2
    ):
        if len(index_spec) != 3:
            raise ValueError(
                'index_spec must be a tuple[int, int, int]!'
            )
        ax0, ax1, ax2 = index_spec
        if order.lower() == 'c':
            return obj[:,ax0], obj[:,ax1], obj[:,ax2]
        else:
            return obj[ax0,:], obj[ax1,:], obj[ax2,:]
    else:
        raise ValueError(
            'input arrays must be 2D with at least 3 axes! '
            'i.e. np.random.rand(10,3) or a list of such.'
        )


def _resolve_color_kwargs(
    kwargs: dict[str, Any],
    *,
    allow_c: bool = True,
    allow_color_sequences: bool = False,
    color_keys: dict[str, str | None] | None = None,
    **explicit: Any,
) -> dict[str, Any]:
    """
    Resolve color kwargs for a matplotlib call.

    Resolves `'c'` and `'color'`, edge/face colors and the color modifiers
    (color transform, color vision deficiency simulation).

    Updates kwargs.

    Parameters
    ----------
    kwargs : dict[str, Any]
        Keyword arguments forwarded to matplotlib.
    allow_c : bool, optional, default=True
        Whether the target function accepts `c` (e.g. `scatter`).
    allow_color_sequences : bool, optional, default=False
        If `True`, color input is allowed to be `Sequence[ColorType]`.
    color_keys : dict[str, str | None] | None, optional, default=None
        Maps the logical names `'edgecolor'` and `'facecolor'` to the names
        used by the target function (e.g. `'markeredgecolor'` for `plot`).
        A value of `None` disables that entry. Unspecified entries keep
        their logical name.
    **explicit
        Any of the following keyword arguments. Edge/face names follow
        `color_keys`.

        Color:

        - color : ColorType | None | _Unset, optional, default=_UNSET
            Color. Mutually exclusive with `c`.
            If `_UNSET` and `c` is not given, uses `config.color`.
        - c : ColorType | NDArray | Quantity | None, optional, default=None
            Color or values to be color-mapped. Mutually exclusive with
            `color`. Passed through `get_value`. Only if `allow_c`.
        - edgecolor : Literal['face', 'none'] | ColorType | _Unset, optional, default=_UNSET
            Edge color. If `_UNSET`, uses `config.edgecolor`.
        - facecolor : Literal['none'] | ColorType | _Unset, optional, default=_UNSET
            Face color. If `_UNSET`, uses `config.facecolor`.

        Color modifiers:

        - transform : str | None | _Unset, optional, default=_UNSET
            Color transform: `'lighten'`, `'darken'`, `'saturate'`,
            `'desaturate'` or `None`. If `_UNSET`, uses
            `config.color_transform`.
        - factor : float | _Unset, optional, default=_UNSET
            Strength of `transform`. If `_UNSET`, uses
            `config.color_transform_factor`.
        - cvd_type : str | None, optional, default=None
            Color vision deficiency type to simulate: `'deuteranomaly'`,
            `'protanomaly'`, `'tritanomaly'` or `None`.
        - severity : float, optional, default=100
            Severity of the simulated deficiency.

    Returns
    -------
    dict[str, Any]
        The same `kwargs` object, updated in place.

    Raises
    ------
    ValueError
        If both `c` and `color` are given.
    TypeError
        If `c` is in `kwargs` and `allow_c` is False, or if `explicit`
        contains a name that is not accepted.

    Notes
    -----
    - `c` is passed through `get_value`. `color` and the edge/face colors
      are passed through `get_color` with the color modifiers.
    - Only keys owned by this resolver are written to `kwargs`.
    """
    keys = {'edgecolor': 'edgecolor', 'facecolor': 'facecolor'}
    if color_keys:
        keys.update(color_keys)
    edge_key, face_key = keys['edgecolor'], keys['facecolor']

    if not allow_c and 'c' in kwargs:
        raise TypeError("'c' is not supported by this function.")

    color_defaults: dict[str, Any] = {'color': None}
    if allow_c:
        color_defaults['c'] = None
    if edge_key is not None:
        color_defaults[edge_key] = config.edgecolor
    if face_key is not None:
        color_defaults[face_key] = config.facecolor
    modifier_defaults = {
        'transform': config.color_transform,
        'factor': config.color_transform_factor,
        'cvd_type': None,
        'severity': 100,
    }

    unknown = (
        explicit.keys() - color_defaults.keys() - modifier_defaults.keys()
    )
    if unknown:
        raise TypeError(f'Unexpected explicit arguments: {sorted(unknown)}')

    color_kw = _extract_kwargs(
        kwargs, params=_params_from_explicit(explicit, color_defaults)
    )
    modifiers = _extract_kwargs(
        kwargs, params=_params_from_explicit(explicit, modifier_defaults)
    )

    c = color_kw.pop('c', None)
    color = color_kw.pop('color')
    if c is not None and color is not None:
        raise ValueError("'c' and 'color' are mutually exclusive.")
    if c is None and color is None:
        color = config.color

    colors = {'color': color, **color_kw}

    if c is not None:
        kwargs['c'] = get_value(c)

    def _convert(value: Any) -> Any:
        if allow_color_sequences and _is_color_sequence(value):
            return get_colors(value, **modifiers)
        return get_color(value, **modifiers)

    kwargs.update(
        {name: _convert(value) for name, value in colors.items() if value is not None}
    )

    return kwargs


def _resolve_cmap_kwargs(
    kwargs: dict[str, Any],
    *,
    active: bool = True,
    **explicit: Any,
) -> dict[str, Any]:
    """
    Resolve colormap and normalization kwargs for a matplotlib call.

    Updates kwargs.

    Parameters
    ----------
    kwargs : dict[str, Any]
        Keyword arguments forwarded to matplotlib.
    active : bool, optional, default=True
        Whether `cmap` and `norm` apply to this call (e.g. for `scatter`:
        `c` is given). If False, all owned keys are popped from `kwargs`
        and discarded, so they do not leak into matplotlib.
    **explicit
        Any of the following keyword arguments.

        Colormap:

        - cmap : str | Colormap | None | _Unset, optional, default=_UNSET
            Colormap. If `_UNSET`, uses `config.cmap`.
        - cmap_range : tuple[float, float], optional, default=(0, 1)
            Normalized sub-range to extract from `cmap`.
        - bad_color : ColorType | None, optional, default=None
            Color for invalid values (NaN).
        - under_color : ColorType | None, optional, default=None
            Color for values below the norm range.
        - over_color : ColorType | None, optional, default=None
            Color for values above the norm range.

        Normalization:

        - norm : str | Normalize | None, optional, default=None
            Possible norms are `'asinh'`, `'asinhnorm'`, `'log'`, `'power'`,
            `'twoslope'` or `None`.
        - vmin : float | None, optional, default=None
            Lower bound of the norm.
        - vmax : float | None, optional, default=None
            Upper bound of the norm.
        - vcenter : float | None, optional, default=None
            Center of the norm.
        - gamma : float | _Unset, optional, default=_UNSET
            Exponent for power-law norms. If `_UNSET`, uses `config.gamma`.
        - linear_width : float | _Unset, optional, default=_UNSET
            Linear region width for symmetric-log norms.
            If `_UNSET`, uses `config.linear_width`.

    Returns
    -------
    dict[str, Any]
        The same `kwargs` object, updated in place.

    Raises
    ------
    TypeError
        If `explicit` contains a name that is not accepted.
    """
    cmap_defaults = {
        'cmap': config.cmap,
        'cmap_range': (0, 1),
        'bad_color': None,
        'under_color': None,
        'over_color': None,
    }
    norm_defaults = {
        'norm': None,
        'vmin': None,
        'vmax': None,
        'vcenter': None,
        'gamma': config.gamma,
        'linear_width': config.linear_width,
    }

    unknown = explicit.keys() - cmap_defaults.keys() - norm_defaults.keys()
    if unknown:
        raise TypeError(f'Unexpected explicit arguments: {sorted(unknown)}')

    cmap_kw = _extract_kwargs(
        kwargs, params=_params_from_explicit(explicit, cmap_defaults)
    )
    norm_kw = _extract_kwargs(
        kwargs, params=_params_from_explicit(explicit, norm_defaults)
    )
    if not active:
        return kwargs

    norm = norm_kw.pop('norm')
    cmap = get_cmap(cmap_kw.pop('cmap'), **cmap_kw)
    if cmap is not None:
        kwargs['cmap'] = cmap
    if norm is not None:
        kwargs['norm'] = get_norm(norm, **norm_kw)

    return kwargs


def _resolve_props_kwargs(
    kwargs: dict[str, Any],
    defaults: dict[str, Any],
    *,
    size_target: Literal['s', 'markersize'] | None,
    **explicit: Any,
) -> dict[str, Any]:
    """
    Resolve artist property kwargs (marker, size, line style, alpha...).

    Only names in `defaults` are read or written. Priority per name:
    `explicit` > existing `kwargs` entry > `defaults`.

    Updates kwargs.

    Parameters
    ----------
    kwargs : dict[str, Any]
        Keyword arguments forwarded to matplotlib.
    defaults : dict[str, Any]
        Maps each property name accepted by the target function to its
        default value. Build it at call time, e.g. with `_scatter_props()`,
        so config changes made after import are respected.
    size_target : {'s', 'markersize'} | None
        Controls which parameter is passed on to matplotlib.
        If `None`, does not perform any `size` or `markersize` conversion.
        Only pass `None` for functions that do not have `size` or `markersize`.
    **explicit
        Property overrides. Each name must be in `defaults`.
        Use `_UNSET` to fall back to the default.

    Returns
    -------
    dict[str, Any]
        The same `kwargs` object, updated in place.

    Raises
    ------
    TypeError
        If `explicit` contains a name that is not in `defaults`.

    Notes
    -----
    - Resolved values that are `None` are not written to `kwargs`.
    """
    unknown = explicit.keys() - defaults.keys()
    if unknown:
        raise TypeError(f'Unexpected explicit arguments: {sorted(unknown)}')

    props_kw = _extract_kwargs(
        kwargs, params=_params_from_explicit(explicit, defaults)
    )

    if size_target is not None:
        _resolve_marker_size(props_kw, size_target)

    kwargs.update({k: v for k, v in props_kw.items() if v is not None})

    return kwargs


def _plot_color_keys() -> dict[str, str | None]:
    """Return the `color_keys` mapping for `plot`."""
    return {
        'edgecolor': 'markeredgecolor',
        'facecolor': 'markerfacecolor',
    }


def _scatter_props() -> dict[str, Any]:
    """Return the property defaults for `scatter`."""
    return {
        'marker': config.marker,
        's': _UNSET,
        'markersize': _UNSET,
        'alpha': config.alpha
    }


def _plot_props() -> dict[str, Any]:
    """Return the property defaults for `plot`."""
    return {
        'marker': config.plot_marker,
        's': _UNSET,
        'markersize': _UNSET,
        'linestyle': config.linestyle,
        'linewidth': config.linewidth,
        'alpha': config.alpha,
        'markeredgecolor': None,
        'markeredgewidth': None,
        'markerfacecolor': None,
        'fillstyle': 'full',
        'drawstyle': 'default'
    }

def _hist_props() -> dict[str, Any]:
    """Return the property defaults for `hist`."""
    return {
        'linestyle': config.linestyle,
        'linewidth': config.linewidth,
        'alpha': config.alpha,
    }

def _resolve_marker_size(
    props_kw: dict[str, Any],
    target: Literal['s', 'markersize'],
) -> None:
    """
    Resolve `s` and `markersize` in `props_kw` into the target convention.

    Parameters
    ----------
    props_kw : dict[str, Any]
        Extracted property values. `s` and `markersize` are updated in place.
        Both must be `_UNSET` if not supplied by the user.
    target : {'s', 'markersize'}
        `'s'` is area in points^2 (scatter).
        `'markersize'` is size in points (plot).

    Raises
    ------
    TypeError
        If both `s` and `markersize`/`ms` are given.
    ValueError
        If `target='markersize'` and the size is not scalar, or `s` is negative.
    """
    s = props_kw.pop('s', _UNSET)
    ms = props_kw.pop('markersize', _UNSET)

    if s is not _UNSET and ms is not _UNSET:
        raise TypeError("`s` and `markersize`/`ms` are mutually exclusive.")

    if s is _UNSET and ms is _UNSET:
        markersize = float(config.markersize)
        props_kw[target] = (
            markersize if target == 'markersize' else markersize**2
        )
        return

    s_given = s is not _UNSET
    value = np.asarray(get_value(s if s_given else ms), dtype=float)

    if target == 'markersize':
        if value.ndim != 0:
            raise ValueError("`plot` does not support per-point sizes; use `scatter`.")
        if s_given and value < 0:
            raise ValueError("`s` must be non-negative.")
        props_kw[target] = float(np.sqrt(value) if s_given else value)
    else:
        props_kw[target] = value if s_given else np.square(value)


def _resolve_color_alias(
    kwargs: dict[str, Any],
    color: ColorType | Sequence[ColorType] | _Unset,
) -> ColorType | Sequence[ColorType] | _Unset:
    """
    Resolve the `c` alias of `color`.

    Used internally by `visualastro.plotting.base.plot` and
    `visualastro.plotting.base.hist`.

    Parameters
    ----------
    kwargs : dict[str, Any]
        Keyword arguments. `c` is removed in place if present.
    color : ColorType | Sequence[ColorType] | _Unset
        Value of the `color` argument.

    Returns
    -------
    ColorType | Sequence[ColorType] | _Unset
        Value of `c` if given, otherwise `color`.

    Raises
    ------
    ValueError
        If both `c` and `color` are given.

    Notes
    -----
    - `c=None` is treated as not given.
    """
    if (c := kwargs.pop('c', None)) is not None:
        if color is not _UNSET:
            raise ValueError("'c' and 'color' are mutually exclusive.")
        return c
    return color
