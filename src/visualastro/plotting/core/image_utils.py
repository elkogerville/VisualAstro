"""
Author: Elko Gerville-Reache
Date Created: 2026-06-07
Date Modified: 2026-10-06
Description:
    Image utility functions for plotting.
"""

from __future__ import annotations
from typing import Any, Literal
from importlib.resources import files

import astropy.units as u
import matplotlib.axes as maxes
from matplotlib.colors import Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from visualastro.analysis.image_utils import stack_cube
from visualastro.core.config import (
    config,
    _resolve_default,
    _Unset,
    _UNSET,
)
from visualastro.core.data import (
    get_data, get_value, to_array
)
from visualastro.datamodels.datacube import DataCube
from visualastro.datamodels.fitsfile import FitsFile
from visualastro.optional_dependencies._spectralcube import SpectralCube
from visualastro.plotting.core.colormaps import get_norm


def get_imshow_norm(
    norm: str | Normalize | None,
    vmin: float | np.floating | None,
    vmax: float | np.floating | None,
    **kwargs: Any,
) -> Normalize | None:
    """
    Return a normalization object for image display.

    Thin wrapper around `get_norm` that adds the `imshow`-specific
    boolean-array shortcut.

    Parameters
    ----------
    norm : {'asinh', 'asinhnorm', 'log', 'power', 'twoslope'} | Normalize | None
        Normalization algorithm. See `get_norm`.
    vmin, vmax : float | np.floating | None
        Normalization limits. If `vmin == 0` and `vmax == 1`, the data is
        assumed to be boolean and `None` (linear stretch) is returned.
    **kwargs : Any
        Forwarded to `get_norm`. Accepted keys: `linear_width`, `gamma`,
        `vcenter`.

    Returns
    -------
    Normalize | None
        Normalization object to pass to `imshow`. `None` if `norm` is `None`
        or the boolean shortcut applies.

    Raises
    ------
    ValueError
        If `norm` is set but `vmin` or `vmax` is `None`.

    See Also
    --------
    get_norm
    """
    if norm is None or (vmin == 0 and vmax == 1):
        return None
    vmin = None if vmin is None else float(vmin)
    vmax = None if vmax is None else float(vmax)

    return get_norm(norm, vmin, vmax, **kwargs)


def get_vmin_vmax(
    data: NDArray | u.Quantity | DataCube | FitsFile | SpectralCube,
    percentile: tuple[float, float] | _Unset = _UNSET,
    vmin: float | np.floating | None = None,
    vmax: float | np.floating | None = None
) -> tuple[float | np.floating | int, float | np.floating | int]:
    """
    Compute vmin and vmax for image display. By default uses the
    data nanpercentile set by `percentile`, but optionally `vmin`
    and/or `vmax` can be set by the user.

    Passing in a boolean array returns `vmin=0`, `vmax=1`.
    This function is used internally by  `compute_imshow_scale`.

    Parameters
    ----------
    data : NDArray | u.Quantity | DataCube | FitsFile | SpectralCube
        Input data array (e.g., 2D image) for which to compute vmin and vmax.
        Must be convertable to an array via `to_array`.
    percentile : tuple[float, float] |  _Unset, optional, default=_UNSET
        Percentile range `[pmin, pmax]` to compute vmin and vmax.
        If None, sets vmin and vmax to None. If `_UNSET`, uses
        default value from `config.percentile`.
    vmin, vmax : float | None, optional, default=None
        If provided, overrides the computed vmin and vmax.

    Returns
    -------
    vmin, vmax : float | int | None
        Minimum and maximum values for image scaling.
    """
    percentile_range = config.percentile if percentile is _UNSET else percentile
    if percentile_range is None:
        raise ValueError(
            'get_vmin_vmax requires a valid percentile range. '
            'Received None. This function should only be called '
            'when percentile-based scaling is enabled.'
        )

    # check if data is an array
    data = to_array(data, keep_unit=False)
    # check if data is boolean
    if data.dtype == bool:
        return 0, 1

    if vmin is None:
        vmin = np.nanpercentile(data, percentile_range[0])
    if vmax is None:
        vmax = np.nanpercentile(data, percentile_range[1])

    return vmin, vmax


def compute_imshow_scale(
    data: NDArray | u.Quantity | DataCube | FitsFile | SpectralCube,
    norm: str | Normalize | None,
    vmin: float | np.floating | None,
    vmax: float | np.floating | None,
    percentile: tuple[float, float] | None,
    **kwargs
) -> tuple[
    Normalize | None,
    float | np.floating | None,
    float | np.floating | None
]:
    """
    Compute normalization and intensity scaling parameters for `plt.imshow`.

    Resolves interaction between `norm`, `vmin`, `vmax`, and
    `percentile` into consistent display parameters.

    Modes
    -----
    Resolution proceeds by the first matching case:

    1. Boolean data (dtype is `bool`):

       - Forces `vmin=0`, `vmax=1`.
       - Returns `(None, 0, 1)`.

    2. Normalized scaling (`norm` is not `None` and not `'linear'`):

       - `vmin`/`vmax` from explicit values, else from `percentile`.
       - Returns `(norm_obj, vmin, vmax)`.

    3. Linear with explicit limits (`norm` is `None` or `'linear'`, `vmin` or `vmax` set):

       - Missing bounds from `percentile` if provided, else `None`.
       - Returns `(None, vmin, vmax)`.

    4. Linear with percentile (`norm` is `None` or `'linear'`, only `percentile` set):

       - `vmin`/`vmax` from `percentile`.
       - Returns `(None, vmin, vmax)`.

    5. Matplotlib autoscaling (`norm`, `vmin`, `vmax`, `percentile` all `None`):

       - Returns `(None, None, None)`.

    Parameters
    ----------
    data : ndarray | Quantity | DataCube | FitsFile | SpectralCube
        Input image data. Must be convertible to a NumPy array via `to_array`.
    norm: str | mcolors.Normalize | None,
        Normalization algorithm for colormap scaling.
        If `None`, linear scaling is used.
    vmin, vmax : float | None
        Lower and upper bounds for intensity scaling. If `None`, may be computed
        from `percentile` or left unset depending on mode.
    percentile : tuple[float, float] | None
        Percentile range `(pmin, pmax)` used to compute `vmin` and
        `vmax`. If `None`, no automatic clipping is applied.
    linear_width : float, optional, default=config.linear_width
        The effective width of the linear region, beyond which the
        transformation becomes asymptotically logarithmic.
        Used for `norm='asinhnorm'`.
    gamma : float, optional, default=config.gamma
        Power law exponent. Used for `norm='power'`.
    vcenter : float, optional, default=None
        Center point of normalization. Must be in between `vmin` and `vmax`.
        If `None`, is the midpoint between `vmin` and `vmax`.

    Returns
    -------
    norm_obj : Normalize | None
        Normalization object for `imshow`. None indicates linear scaling.
    vmin, vmax : float | None
        Lower and upper intensity bounds for `imshow`.
    """
    data = to_array(data)

    if data.dtype == bool:
        return None, 0, 1

    if norm == 'linear':
        return None, None, None

    if norm is None:
        if percentile is not None:
            vmin, vmax = get_vmin_vmax(data, percentile, vmin, vmax)

        return None, vmin, vmax

    if percentile is not None:
        vmin, vmax = get_vmin_vmax(data, percentile, vmin, vmax)

    elif vmin is None or vmax is None:
        raise ValueError(
            'vmin and vmax must not be None if norm is not None! '
            f'got: norm: {norm}, vmin: {vmin}, vmax: {vmax}'
        )
    img_norm = get_imshow_norm(norm, vmin, vmax, **kwargs)

    return img_norm, vmin, vmax


def nanpercentile_limits(
    data,
    vmin,
    vmax,
    *,
    slice_idx=None,
    stack_method=_UNSET,
    axis=0,
) -> tuple[np.floating, np.floating]:
    """
    Compute NaN-safe percentile-based color limits.

    If input is 3D, it is reduced to 2D before computing limits.

    Parameters
    ----------
    data : ArrayLike | SpectralCube | DataCube
        Input image or cube.
    vmin, vmax : float
        Lower and upper percentiles (0–100).
    slice_idx : int | tuple[int, int] | None, optional, default=None
        Index specification:

        * int: extract single slice
        * tuple[start, end]: extract range (inclusive)
        * None: use entire cube

    stack_method : {'mean', 'median', 'sum', 'max', 'min', 'std'} | _Unset, default=_UNSET
        Reduction method if stacking is required. If `_UNSET`,
        uses `config.stack_cube_method`.
    axis : int, default=0
        Axis to flatten if data.ndim > 2 and no slice_idx is given.

    Returns
    -------
    vmin_val, vmax_val : float
        Lower and upper intensity bounds.
    """
    stack_method = _resolve_default(stack_method, config.stack_cube_method)

    data = get_data(data)

    if slice_idx is not None:
        data = stack_cube(
            data, idx=slice_idx, method=stack_method, axis=axis
        )

    arr = to_array(get_value(data))

    if arr.ndim > 2:
        arr = getattr(np, f'nan{stack_method}')(arr, axis=axis)

    return (
        np.nanpercentile(arr, vmin),
        np.nanpercentile(arr, vmax),
    )


def thorlabs_logo(
    ax: maxes.Axes,
    loc: Literal['best', 'lower left', 'lower right', 'upper left', 'upper right'] | None | _Unset = _UNSET,
    transparent: bool | _Unset = _UNSET,
    zoom: float = 0.015,
    darkmode: bool | _Unset = _UNSET
) -> None:
    """
    Add a thorlabs logo to a plot.

    Parameters
    ----------
    ax : maxes.Axes
        Matplotlib axes to plot on.
    loc : {'best', 'lower left', 'lower right', 'upper left', 'upper right'} | None | _Unset, optional, default=_UNSET
        Figure location to plot image on. If `'best'`, finds the most optimal location for the image.
        The algorithm chooses the corner farthest away from any data point or legend on the plot.
        If `None`, does nothing. If `_UNSET`, uses `config.thorlabs_loc`.
    transparent : bool | _Unset, optional, default=_UNSET
        If `True`, logo will have a transparent background. If `_UNSET`,
        uses `config.thorlabs_transparent`.
    zoom : float, optional, default=0.015
        Logo size scaling factor.
    darkmode : bool | _Unset, optional, default=_UNSET
        If `True`, inverts the color of the logo. If `_UNSET`,
        uses `config.thorlabs_darkmode`.
    """
    if loc is None:
        return None

    loc = _resolve_default(loc, config.thorlabs_loc)
    transparent = _resolve_default(transparent, config.thorlabs_transparent)
    darkmode = _resolve_default(darkmode, config.thorlabs_darkmode)

    logo_name = 'thorlabs_dark' if darkmode else 'thorlabs'
    logo = logo_name+'.png' if transparent else logo_name+'.jpg'

    img_path = files('visualastro') / 'data' / logo

    thorlabs = plt.imread(img_path)
    if plt.rcParams['image.origin'] == 'lower':
        thorlabs = thorlabs[::-1]

    imagebox = OffsetImage(thorlabs, zoom=zoom)

    if loc == 'best':
        segments = []
        for line in ax.lines:
            segments.append(line.get_xydata())
        for col in ax.collections:
            offsets = col.get_offsets().data
            if len(offsets) > 0:
                segments.append(offsets)
        legend = ax.get_legend()
        if legend is not None:
            bb = legend.get_window_extent().transformed(ax.transData.inverted())
            segments.append([[bb.x0, bb.y0], [bb.x1, bb.y1]])

        xy = np.vstack(segments) if segments else None
        if xy is not None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            corners = {
                'lower left':  (xmin, ymin),
                'lower right': (xmax, ymin),
                'upper left':  (xmin, ymax),
                'upper right': (xmax, ymax),
            }
            loc = max(
                corners,
                key=lambda k: np.nanmin(cdist(xy, [corners[k]], 'sqeuclidean'))
            )

        else:
            loc = 'upper right'

    img_locations = {
        'lower left': (0.01, 0.01, 0, 0),
        'lower right': (0.99, 0.01, 1, 0),
        'upper left': (0.01, 0.99, 0, 1),
        'upper right': (0.99, 0.99, 1, 1),
    }

    img_loc = img_locations.get(loc, (0.99, 0.99, 1, 1))
    ab = AnnotationBbox(
        imagebox,
        xy=img_loc[:2],
        xycoords='axes fraction',
        box_alignment=img_loc[2:],
        frameon=False
    )
    ax.add_artist(ab)
