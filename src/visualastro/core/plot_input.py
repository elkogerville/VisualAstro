"""
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2026-10-06
Description:
    Plotting input normalization functions.
"""

from collections.abc import Sequence
from typing import Literal

from astropy import units as u
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
from visualastro.core.data import as_list, to_list


def _normalize_order(order: str) -> Literal['c', 'fortran']:
    """Normalize array order specification.

    Parameters
    ----------
    order : str
        Array order. Accepted (case-insensitive): 'c', 'f', 'fortran'.

    Returns
    -------
    Literal['c', 'fortran']
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


def _2d_to_1d_array(
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
            'Input must be a 2D array, ie. np.random.rand(10, 2).'
        )
    if len(xy_indices) != 2:
        raise ValueError('xy_indices must be a tuple[int, int].')

    ix, iy = xy_indices
    return _2d_to_1d_array(obj, ix, order), _2d_to_1d_array(obj, iy, order)


def _extract_xy2(
    *data,
    order: Literal['c', 'fortran'] | _Unset = _UNSET,
    xy_indices: tuple[int, int] | _Unset = _UNSET,
):
    """
    Extract x and y data from positional inputs.

    Parameters
    ----------
    *data : float | u.Quantity | NDArray | Sequence[float | u.Quantity | NDArray]
        Either a single object (y only, or 2D array) or two objects (x, y).
    order : {'c', 'fortran'} | _Unset, optional, default=config.array_order
        Variable layout for 2D arrays.
    xy_indices : tuple[int, int] | _Unset, optional, default=config.xy_indices
        - tuple: (x index, y index).

    Returns
    -------
    x : float | u.Quantity | NDArray | Sequence[...] | None
        None if not provided or implicit.
    y : float | u.Quantity | NDArray | Sequence[...]

    Raises
    ------
    ValueError
        On invalid `order`, `xy_indices`, argument count, or array ndim.
    """
    order = _normalize_order(_resolve_default(order, config.array_order))
    indices = _resolve_default(xy_indices, config.xy_indices)

    if len(data) == 2:
        return data[0], data[1]

    if len(data) != 1:
        raise ValueError(f'Expected 1 or 2 positional inputs, got {len(data)}.')

    obj = data[0]
    if _is_scalar(obj):
        return None, obj
    if not isinstance(obj, np.ndarray):
        return None, obj
    if obj.ndim == 1:
        return None, obj
    if obj.ndim != 2:
        raise ValueError(f'Input must be 1D or 2D, got {obj.ndim}D.')

    return _extract_xy_from_ndarray2D(obj, order, tuple(indices))


def _extract_xy(
    *data: float | u.Quantity | NDArray | Sequence[float | u.Quantity | NDArray],
    order: Literal['c', 'fortran'] | _Unset = _UNSET,
    index_spec: Literal['implicit', 'explicit'] | tuple[int, int] | _Unset = _UNSET
) -> tuple[
    float | u.Quantity | NDArray | Sequence[float | u.Quantity | NDArray] | None,
    float | u.Quantity | NDArray | Sequence[float | u.Quantity | NDArray],
]:
    """
    Extract X and Y coordinates from flexible numeric inputs.

    Interprets input based on dimensionality and structure. For 2D arrays,
    extracts columns according to `order` and `index_spec`. Returns X as None
    if only Y values are detected.

    This function is used by `_normalize_plotting_input(s)`.

    Parameters
    ----------
    *data : float | u.Quantity | NDArray | Sequence[float | u.Quantity | NDArray]
        Input data. Supported forms:

        * Single argument:
            * 1D array-like or Quantity: Y values, X = None
            * 2D array or Quantity: extract X, Y according to `order` and `index_spec`
            * list/tuple of scalars: Y values, X = None
            * scalar or scalar Quantity: single Y value, X = None

        * Two arguments:
            * (X, Y) pairs passed through unchanged

    order : {'c', 'fortran'} | _Unset, optional, default=_UNSET
        Memory layout for 2D input interpretation. Defines what a
        column is for `index_spec`.

        * 'c': row-major, shape (N, 2)
        * 'fortran': column-major, shape (2, N)

        If `_UNSET`, uses `config.array_order`.
    index_spec : {'implicit', 'explicit'} | tuple[int, int], optional
        Column extraction mode for 2D inputs.

        * 'implicit': return (None, [col_0, col_1, ...])
        * 'explicit': return (col_0, col_1)
        * tuple (i, j): return (col_i, col_j)

    Returns
    -------
    X : ndarray | Quantity | scalar | list | None
        X coordinates. None indicates implicit indexing (caller should generate).
        Type preserves input semantics.
    Y : ndarray | Quantity | scalar | list
        Y coordinates. Preserves input type.

    Raises
    ------
    ValueError
        If input has unsupported dimensionality, structure, or count.

    Notes
    -----
    For 2D inputs with 'implicit' mode, X is returned as None and Y as a list
    of column arrays, delegating index generation to the caller.
    """
    array_order = _resolve_default(order, config.array_order)
    index_spec = _resolve_default(index_spec, config.index_specification)

    if len(data) == 1:
        obj = data[0]
        if isinstance(obj, (np.ndarray, u.Quantity)):
            if obj.ndim == 1:
                return None, obj
            if obj.ndim == 2:
                if array_order.lower() == 'c':
                    axis = 0
                    get_col = lambda i: obj[:, i]
                else:
                    axis = 1
                    get_col = lambda i: obj[i, :]

                if isinstance(index_spec, (list, tuple)):
                    ix, iy = index_spec
                    return get_col(ix), get_col(iy)
                elif index_spec == 'explicit':
                    return get_col(0), get_col(1)
                elif index_spec == 'implicit':
                    y = [
                        get_col(i) for i in range(
                            obj.shape[1] if axis == 0 else obj.shape[0]
                        )
                    ]
                    return None, y

        if _is_scalar(obj):
            return None, obj

        if isinstance(obj, (list, tuple)):
            if all(_is_scalar(x) for x in obj):
                return None, obj

            if all(isinstance(x, (np.ndarray, u.Quantity)) for x in obj):
                xlist, ylist = [], []
                for o in obj:
                    x, y = _extract_xy(o, order=array_order, index_spec=index_spec)
                    xlist.append(x)
                    ylist.append(y)

                if all(x is None for x in xlist):
                    xlist = None
                # flatten the ylist if list[list[NDArray]]
                if all(isinstance(y, (list, tuple)) for y in ylist):
                    if all(_is_ndarray_or_quantity_array(item) for y in ylist for item in y):
                        ylist = [item for sublist in ylist for item in sublist]

                return xlist, ylist

        raise ValueError(f'Unsupported input type {type(obj).__name__}')

    if len(data) == 2:
        return data[0], data[1]

    raise ValueError(f'Expected 1 or 2 arguments, got {len(data)}')


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
            'ie. np.random.rand(10,3) or a list of such.'
        )
