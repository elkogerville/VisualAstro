"""
Author: Elko Gerville-Reache
Date Created: 2026-06-02
Date Modified: 2026-10-28
Description:
    Functions related to Matplotlib axes.
"""

from typing import Any, Literal

from astropy.io.fits import Header
import astropy.units as u
from astropy.visualization.wcsaxes.core import WCSAxes
from astropy.wcs import WCS
import matplotlib.axes as maxes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure, SubFigure
import matplotlib.gridspec as _gridspec
from matplotlib.lines import Line2D
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.text import Text
import matplotlib.ticker as ticker
from matplotlib.typing import ColorType
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.typing import ArrayLike, NDArray

from visualastro.core.config import (
    config,
    _Unset,
    _UNSET,
    _resolve_default
)
from visualastro.core.data import get_value, to_list, _to_xy
from visualastro.core.kwargs import _resolve_kwargs, _params_from_explicit
from visualastro.core.units import (
    get_physical_type,
    get_unit_label,
    _infer_physical_type_label,
)
from visualastro.plotting.core.colors import as_color, get_colors
from visualastro.utils.wcs_utils import get_wcs_celestial


def get_ax(
    ax: maxes.Axes | None,
    figsize: tuple[float, float] | _Unset = _UNSET
) -> maxes.Axes:
    """
    Get either the instance passed in or the current `Axes`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes | None
        If an `Axes`, returns unchanged. If `None`, returns `plt.gca()`

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if isinstance(ax, maxes.Axes):
        return ax

    figsize = _resolve_default(figsize, config.figsize)

    fig = plt.gcf()
    has_axes = bool(fig.axes)

    current_ax = plt.gca()

    if not has_axes:
        fig.set_size_inches(figsize)

    return current_ax


def get_ax3d(
    ax: Axes3D | None,
    figsize: tuple[float, float] | _Unset = _UNSET,
    **kwargs
) -> Axes3D:
    """
    Get either the instance passed in or the current `Axes3d`.

    If the current axis is an `Axes`, its figure is closed if no data
    is found via `has_data()`.

    Parameters
    ----------
    ax : Axes3D | None
        Returns `ax` if is an `Axes3D`. Otherwise returns
        `plt.gca()` if the current axis is an `Axis3D`.
        If `plt.gca()` is an `Axes`, returns a new
        `Axes3D` instance.
    figsize : tuple[float | float] | _Unset, optional, default=_UNSET
        Figsize if `ax` has to be created. If `_UNSET`, uses
        `config.figsize`.
    **kwargs :
        Additional keyword arguments passed to `plt.figure` if
        `Axes3D` must be created.

    Returns
    -------
    ax : Axes3D
    """
    if isinstance(ax, Axes3D):
        return ax

    current_ax = plt.gca()
    if isinstance(current_ax, Axes3D):
        return current_ax

    if not current_ax.has_data():
        plt.close()

    figsize = _resolve_default(figsize, config.figsize3D)
    fig, ax = ax3d(figsize=figsize, **kwargs)
    return ax


def get_wcsax(
    ax: WCSAxes | None,
    wcs: WCS | Header | None = None,
    figsize: tuple[float, float] | _Unset = _UNSET,
    **kwargs
) -> WCSAxes:
    """
    Get either the instance passed in or the current `WCSAxes`.

    If the current axis is an `Axes`, its figure is closed if no data
    is found via `has_data()`.

    Parameters
    ----------
    ax : WCSAxes | None
        Returns `ax` if `ax` is a `WCSAxes`. Otherwise returns
        `plt.gca()` if the current axis is an `WCSAxes`.
        If `plt.gca()` is an `Axes`, returns a new
        `WCSAxes` instance.
    wcs : WCS | Header | None, optional, default=None
        WCS information required to create `ax` if no valid
        `WCSAxes` could be found. If `None`, will only raise
        a `ValueError` if a valid `ax` cannot be returned.
        See docstring for ``get_wcs_celestial`` for full list
        of acceptable types.
    figsize : tuple[float | float] | _Unset, optional, default=_UNSET
        Figsize if `ax` has to be created. If `_UNSET`, uses
        `config.figsize`.
    **kwargs :
        Additional keyword arguments passed to `plt.figure` if
        `WCSAxes` must be created.

    Returns
    -------
    ax : WCSAxes

    Raises
    ------
    ValueError :
        If `wcs` does not have valid WCS and `ax` could not be found.
    """
    if isinstance(ax, WCSAxes):
        return ax

    current_ax = plt.gca()
    if isinstance(current_ax, WCSAxes):
        return current_ax

    if not current_ax.has_data():
        plt.close()
    if wcs is None:
        raise ValueError('No existing WCSAxes found and `wcs` is None.')

    figsize = _resolve_default(figsize, config.figsizeWCS)
    fig, ax = wcsax(wcs, figsize=figsize, **kwargs)
    return ax


def subplot(
    *args : int | tuple[int, int, int],
    figsize : tuple[float, float] | _Unset = _UNSET,
    projection: str | None = None,
    **kwargs
) -> tuple[Figure, maxes.Axes]:
    """
    Create a 3D plot instance.

    Parameters
    ----------
    *args :  int | tuple[int, int, *index*], optional, default=(1, 1, 1)
        The position of the subplot described by one of:

        - three integers `(*nrows*, *ncols*, *index*)`. The subplot will
          take the *index* position on a grid with *nrows* rows and *ncols* columns.
          *index* starts at 1 in the upper left corner and increases to the right.
          *index* can also be a two-tuple specifying the (*first*, *last*) indices
          (1-based, and including *last*) of the subplot, i.e. `(3, 1, (1, 2))`
          makes a subplot that spans the upper 2/3 of the figure.
        - A 3-digit integer. The digits are interpreted as if given separately as
          three single-digit integers, i.e. `235` is the same as `(2, 3, 5)`.
          Note that this can only be used if there are no more than 9 subplots.

    figsize : tuple[float, float] | _Unset, optional, default=_UNSET
        Figure size. If `_UNSET`, uses `config.figsize`.
    projection : str | None, optional, default=None
        The projection type of the subplot. Can be one of the accepted values:
        `'aitoff'`, `'hammer'`, `'lambert'`, `'mollweide'`, `'polar'`, `'rectilinear'`.
        If `None`, uses `'rectilinear'` projection.
    **kwargs :
        Additional keyword arguments passed to `plt.figure`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure instance.
    ax : matplotlib.axes.Axes
        Axes instance.
    """
    figsize = _resolve_default(figsize, config.figsize)

    fig = plt.figure(figsize=figsize, **kwargs)
    ax = fig.add_subplot(*args, projection=projection)

    return fig, ax


def add_subplot(
    *args : int | tuple[int, int, int],
    ax: maxes.Axes | None = None,
    fig: Figure | SubFigure | None = None,
    projection: str | None = None,
    **kwargs
) -> maxes.Axes | tuple[Figure, maxes.Axes]:
    """
    Add a subplot to a figure, optionally creating a new figure.

    Parameters
    ----------
    *args :  int | tuple[int, int, *index*], optional, default=(1, 1, 1)
        The position of the subplot described by one of:

        - three integers `(*nrows*, *ncols*, *index*)`. The subplot will
          take the *index* position on a grid with *nrows* rows and *ncols* columns.
          *index* starts at 1 in the upper left corner and increases to the right.
          *index* can also be a two-tuple specifying the (*first*, *last*) indices
          (1-based, and including *last*) of the subplot, i.e. `(3, 1, (1, 2))`
          makes a subplot that spans the upper 2/3 of the figure.
        - A 3-digit integer. The digits are interpreted as if given separately as
          three single-digit integers, i.e. `235` is the same as `(2, 3, 5)`.
          Note that this can only be used if there are no more than 9 subplots.

    ax : Axes | None, optional, default=None
        Axes instance. Either `ax` or `fig` should be provided.
    fig : Figure | SubFigure | None, optional, default=None
        Figure instance.  Either `ax` or `fig` should be provided.
    projection : str | None, optional, default=None
        Projection type for the subplot. Examples include WCSAxes or
        {None, '3d', 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar',
        'rectilinear', str}. If None, defaults to 'rectilinear'.
    **kwargs :
        Additional keyword arguments passed to `fig.add_subplot`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Created subplot axes.

    Examples
    --------
    Create a new figure and subplot:
    >>> fig, ax = add_subplot(return_fig=True)

    Add a subplot to an existing figure:
    >>> fig = plt.figure()
    >>> ax = add_subplot(121, fig=fig)

    Create a 3D subplot:
    >>> fig, ax = add_subplot(projection='3d', return_fig=True)
    """
    fig = ax.figure if ax is not None else fig
    if fig is None:
        raise ValueError(
            'Provide at least fig or ax!'
        )

    ax = fig.add_subplot(*args, projection=projection, **kwargs)

    return ax


def ax3d(
    *args : int | tuple[int, int, int],
    figsize: tuple[float, float] | _Unset = _UNSET,
    **kwargs
) -> tuple[Figure, Axes3D]:
    """
    Create a 3D plot instance.

    Parameters
    ----------
    *args :  int | tuple[int, int, *index*], optional, default=(1, 1, 1)
        The position of the subplot described by one of:

        - three integers `(*nrows*, *ncols*, *index*)`. The subplot will
          take the *index* position on a grid with *nrows* rows and *ncols* columns.
          *index* starts at 1 in the upper left corner and increases to the right.
          *index* can also be a two-tuple specifying the (*first*, *last*) indices
          (1-based, and including *last*) of the subplot, i.e. `(3, 1, (1, 2))`
          makes a subplot that spans the upper 2/3 of the figure.
        - A 3-digit integer. The digits are interpreted as if given separately as
          three single-digit integers, i.e. `235` is the same as `(2, 3, 5)`.
          Note that this can only be used if there are no more than 9 subplots.

    figsize : tuple[float, float] | _Unset, optional, default=_UNSET
        Figure size. If `_UNSET`, uses `config.figsize3D`.
    **kwargs :
        Additional keyword arguments passed to `plt.figure`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure instance.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        3D axes instance.
    """
    figsize = _resolve_default(figsize, config.figsize3D)

    fig = plt.figure(figsize=figsize, **kwargs)
    ax = fig.add_subplot(*args, projection='3d')

    return fig, ax


def add_ax3d(
    *args : int | tuple[int, int, int],
    ax: maxes.Axes | Axes3D | None = None,
    fig: Figure | SubFigure | None = None,
    **kwargs
) -> Axes3D:
    """
    Add a 3D subplot to a figure.

    Parameters
    ----------
    *args :  int | tuple[int, int, *index*], optional, default=(1, 1, 1)
        The position of the subplot described by one of:

        - three integers `(*nrows*, *ncols*, *index*)`. The subplot will
          take the *index* position on a grid with *nrows* rows and *ncols* columns.
          *index* starts at 1 in the upper left corner and increases to the right.
          *index* can also be a two-tuple specifying the (*first*, *last*) indices
          (1-based, and including *last*) of the subplot, i.e. `(3, 1, (1, 2))`
        - makes a subplot that spans the upper 2/3 of the figure.
        - A 3-digit integer. The digits are interpreted as if given separately as
          three single-digit integers, i.e. `235` is the same as `(2, 3, 5)`.
          Note that this can only be used if there are no more than 9 subplots.

    ax : Axes | Axes3D | None, optional, default=None
        Axes instance. Either `ax` or `fig` should be provided.
    fig : Figure | SubFigure | None, optional, default=None
        Figure instance.  Either `ax` or `fig` should be provided.
    **kwargs :
        Additional keyword arguments passed to `fig.add_subplot`.

    Returns
    -------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        Created 3D axes instance.
    """
    fig = ax.figure if ax is not None else fig
    if fig is None:
        raise ValueError(
            'Provide at least fig or ax!'
        )
    ax = fig.add_subplot(*args, projection='3d', **kwargs)

    return ax


def wcsax(
    *args : int | tuple[int, int, int],
    wcs: WCS | Header,
    figsize: tuple[float, float] | _Unset = _UNSET,
    **kwargs
) -> tuple[Figure, WCSAxes]:
    """
    Create a `WCSAxes` plot instance.

    Parameters
    ----------
    *args :  int | tuple[int, int, *index*], optional, default=(1, 1, 1)
        The position of the subplot described by one of:

        - three integers `(*nrows*, *ncols*, *index*)`. The subplot will
          take the *index* position on a grid with *nrows* rows and *ncols* columns.
          *index* starts at 1 in the upper left corner and increases to the right.
          *index* can also be a two-tuple specifying the (*first*, *last*) indices
          (1-based, and including *last*) of the subplot, i.e. `(3, 1, (1, 2))`
          makes a subplot that spans the upper 2/3 of the figure.
        - A 3-digit integer. The digits are interpreted as if given separately as
          three single-digit integers, i.e. `235` is the same as `(2, 3, 5)`.
          Note that this can only be used if there are no more than 9 subplots.

    wcs : WCS | Header
        Object from which to extract WCS. See docstring for
        ``get_wcs_celestial`` for full list of acceptable types.

    figsize : tuple[float, float] | _Unset, optional, default=_UNSET
        Figure size. If `_UNSET`, uses `config.figsize`.
    **kwargs :
        Additional keyword arguments passed to `plt.figure`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure instance.
    ax : WCSAxes
        `WCSAxes` instance.
    """
    figsize = _resolve_default(figsize, config.figsize)

    fig = plt.figure(figsize=figsize, **kwargs)
    wcs2d = get_wcs_celestial(wcs)
    ax = fig.add_subplot(*args, projection=wcs2d)

    return fig, ax


def gridspec(
    nrows: int | _Unset = _UNSET,
    ncols: int | _Unset = _UNSET,
    figsize: tuple[float, float] | _Unset = _UNSET,
    sharex: bool | _Unset = _UNSET,
    sharey: bool | _Unset = _UNSET,
    hspace: float | None | _Unset = _UNSET,
    wspace: float | None | _Unset = _UNSET,
    width_ratios: ArrayLike | None = None,
    height_ratios: ArrayLike | None = None,
    fancy_axes: bool = False,
    Nticks: int | None | _Unset = _UNSET,
    aspect: float | None = None
):
    """
    Create a grid of Matplotlib axes panels with consistent sizing
    and optional fancy tick styling.

    Parameters
    ----------
    nrows : int | _Unset, optional, default=_UNSET
        Number of subplot rows. If `_UNSET`, uses `config.nrows`.
    ncols : int | _Unset, optional, default=_UNSET
        Number of subplot columns. If `_UNSET`, uses `config.ncols`.
    figsize : tuple[float, float] | _Unset, optional, default=_UNSET
        Figure size in inches as (width, height). If `_UNSET`,
        uses `config.figsize_gridspec`.
    sharex : bool | _Unset, optional, default=_UNSET
        If `True`, share the x-axis among all subplots. If `_UNSET`,
        uses `config.axes.sharex`.
    sharey : bool | _Unset, optional, default=_UNSET
        If `True`, share the y-axis among all subplots. If `_UNSET`,
        uses `config.axes.sharey`.
    hspace : float | None | _Unset, optional, default=`_UNSET`
        Height padding between subplots. If `None`,
        Matplotlib's default spacing is used. If `_UNSET`,
        uses `config.axes.hspace`.
    wspace : float | None | _Unset, optional, default=`_UNSET`
        Width padding between subplots. If `None`,
        Matplotlib's default spacing is used. If `_UNSET`,
        uses `config.axes.wspace`.
    width_ratios : ArrayLike | None, optional, default=None
        ArrayLike of length `ncols` defining the width padding between
        subplots. If `None`, Matplotlib's default spacing is used.
        Defines the relative widths of the columns. Each column gets a
        relative width of `width_ratios[i] / sum(width_ratios)`. If not
        given, all columns will have the same width.
    height_ratios : ArrayLike | None, optional, default=None
        ArrayLike of length `nrows` defining the relative heights of the rows.
        Each row gets a relative height of `height_ratios[i] / sum(height_ratios)`.
        If not given, all rows will have the same height.
    fancy_axes : bool, optional, default=False
        If True, enables 'fancy' axes styling:

        - minor ticks on
        - inward ticks on all sides
        - axes labels on outer grid axes
        - h/wspace = 0.0

    Nticks : int | None | _Unset, optional, default=`_UNSET`
        Maximum number of major ticks per axis. If `None`,
        uses the default Matplotlib settings. If `_UNSET`,
        uses `config.axes.Nticks`.
    aspect : float | None, optional, default=None
        Changes the physical dimensions of the Axes,
        such that the ratio of the Axes height to the
        Axes width in physical units is equal to aspect.
        None will disable a fixed box aspect so that height
        and width of the Axes are chosen independently.

    Returns
    -------
    fig : ~matplotlib.figure.Figure
        The created Matplotlib Figure instance.
    axs : ndarray[~matplotlib.axes.Axes]
        Flattened array of Axes objects, ordered row-wise.
    """
    nrows = _resolve_default(nrows, config.nrows)
    ncols = _resolve_default(ncols, config.ncols)
    figsize = _resolve_default(figsize, config.figsize_gridspec)
    sharex = _resolve_default(sharex, config.axes.sharex)
    sharey = _resolve_default(sharey, config.axes.sharey)
    hspace = _resolve_default(hspace, config.axes.hspace)
    wspace = _resolve_default(wspace, config.axes.wspace)
    Nticks = _resolve_default(Nticks, config.axes.Nticks)

    Nx = nrows
    Ny = ncols

    if fancy_axes:
        labeltop = [[True if i == 0 else False for j in range(Ny)] for i in range(Nx)]
        labelright = [[True if i == Ny-1 else False for i in range(Ny)] for j in range(Nx)]
        hspace = 0.0 if hspace is None else hspace
        wspace = 0.0 if wspace is None else wspace

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        Nx, Ny,
        hspace=hspace, wspace=wspace,
        width_ratios=width_ratios,
        height_ratios=height_ratios
    )
    axs = gs.subplots(sharex=sharex, sharey=sharey)
    axs = np.atleast_1d(np.asarray(axs)).ravel()

    for i in range(Nx):
        for j in range(Ny):
            ax = axs[j + Ny*i]

            if fancy_axes:
                ax.minorticks_on()
                ax.tick_params(axis='both', length=2, direction='in',
                               which='both', labeltop=labeltop[i][j],
                               labelright=labelright[i][j],
                               right=True, top=True)
            if Nticks is not None:
                ax.xaxis.set_major_locator(ticker.MaxNLocator(Nticks))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(Nticks))
            ax.set_box_aspect(aspect)

    return fig, axs


def tripanel_figure(
    height_ratios: ArrayLike = (0.05, 1, 0.25),
    width_ratios: ArrayLike = (1, 0.25),
    sharex: bool = True,
    sharey: bool = True,
    left: float = 0.05,
    right: float = 0.95,
    bottom: float = 0.08,
    top: float = 0.93,
    hspace: float = 0.07,
    wspace: float = 0.055,
    figsize: tuple[float, float] = (8, 8),
    colorbar: bool = False
) -> tuple[Figure, np.ndarray]:
    """
    Create a triple-panel figure with a main axes, vertical marginal,
    horizontal marginal, and an optional colorbar axes.

    Layout (GridSpec 3x2):
        [cbax ][ -  ]   row 0 — colorbar (optional)
        [ ax  ][axv ]   row 1 — main + vertical marginal
        [ axh ][ -  ]   row 2 — horizontal marginal

    Returns `Fig, NDArray[ax, axh, axv, cbax]`

    Parameters
    ----------
    height_ratios : ArrayLike, optional, default=(0.05, 1, 0.2)
        Relative heights of the 3 rows.
    width_ratios : ArrayLike, optional, default=(1, 0.2)
        Relative widths of the 2 columns.
    sharex, sharey : bool, optional, default=True
        If `True`, `axh` and `axv` will share the x and y axes of `ax`
        respectively.
    left : float, optional, default=0.05
        Left boundary of the GridSpec as a fraction of figure width.
    right : float, optional, default=0.95
        Right boundary of the GridSpec as a fraction of figure width.
    bottom : float, optional, default=0.08
        Bottom boundary of the GridSpec as a fraction of figure height.
    top : float, optional, default=0.93
        Top boundary of the GridSpec as a fraction of figure height.
    hspace : float, optional, default=0.03
        Height spacing between rows as a fraction of average row height.
    wspace : float, optional, default=0.02
        Width spacing between columns as a fraction of average column width.
    figsize : tuple[float, float], optional, default=(8, 8)
        Figure dimensions (width, height) in inches.
    colorbar : bool, optional, default=False
        If `True`, a colorbar axes is added at row 0, column 0.
        If `False`, the axis visibility is set to `False`. Regardless
        of `colorbar`, axis is returned.

    Returns
    -------
    fig : Figure
        Figure instance.
    axes : NDArray[Axes]
        `NDArray[ax, axh, axv, cbax]`.
    """
    fig = plt.figure(figsize=figsize)
    gs = _gridspec.GridSpec(
        3, 2,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        left=left, right=right,
        bottom=bottom, top=top,
        hspace=hspace, wspace=wspace,
    )

    ax = fig.add_subplot(gs[1, 0])
    ax.tick_params(axis='x', bottom=False, labelbottom=False)

    axh = fig.add_subplot(gs[2, 0], sharex=ax if sharex else None)
    axh.tick_params(axis='y', left=False, labelleft=True)

    axv = fig.add_subplot(gs[1, 1], sharey=ax if sharey else None)
    axv.tick_params(
        axis='both',
        bottom=False, left=False,
        labelbottom=True, labelleft=False,
    )

    cbax = fig.add_subplot(gs[0, 0])
    cbax.tick_params(
        axis='both',
        bottom=False, left=False,
        labelbottom=False, labelleft=False
    )
    if not colorbar:
        cbax.set_visible(False)

    axes = np.asarray([ax, axh, axv, cbax]).ravel()

    return fig, axes


def set_limits(
    ax: maxes.Axes | None = None,
    *,
    limits: float | tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xpad: float | None | _Unset = _UNSET,
    ypad: float | None | _Unset = _UNSET,
    recompute_other_axis: bool | _Unset = _UNSET
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Set the limits of a Matplotlib Axes.

    If only one of `xlim` or `ylim` is provided, the limits of the other
    axis are recomputed from the data visible within the provided range.
    Matplotlib does not do this by itself. Call after all plotting calls.

    Parameters
    ----------
    ax : matplotlib.axes.Axes | None, optional, default=None
        Axes to modify. If `None`, uses the current Axes.
    limits : float | tuple[float, float] | None, optional, default=None
        Limits applied to both axes. Cannot be combined with `xlim` or `ylim`.
        - If a scalar, sets `xlim = ylim = (-abs(limits), abs(limits))`.
        - If a `tuple[float, float]`, sets `xlim = ylim = limits` as is.

    xlim, ylim : tuple[float, float] | None, optional, default=None
        X and Y-axis limits, applied as given (inverted limits invert the axis).
        If only one is provided and `recompute_other_axis=True`, the limits of the other
        axis are computed from the data visible within the provided range.
        If both are provided, no computation is done.
    xpad, ypad : float | None, optional, default=None
        Fractional padding of the computed X and Y-axis limits. Must be finite
        and `> -0.5`. If `None`, uses `ax.margins()[0]` and `ax.margins()[1]`.
        - `xpad` is only used if `ylim` is provided without `xlim`.
        - `ypad` is only used if `xlim` is provided without `ylim`.
        - Both are ignored if `recompute_other_axis=False`, or if `limits` is given.
        - Padding is never applied to user-provided limits.

    recompute_other_axis : bool | _Unset, optional, default=_UNSET
        If `True` and only one of `xlim` or `ylim` is provided, the limits
        of the other axis are computed from the data visible within the
        provided range. If `False`, only the provided limits are set and
        the other axis is autoscaled by Matplotlib. Ignored if `limits` is
        given or if both `xlim` and `ylim` are provided. If `_UNSET`,
        uses `config.axes.recompute_other_axis`.

    Returns
    -------
    xlim, ylim : tuple[float, float]
        Limits on `ax` after the call.

    Raises
    ------
    ValueError
        If `limits` is combined with `xlim` or `ylim`.
        If `limits` is not a scalar or a `tuple[float, float]`.
        If `xpad` or `ypad` is not finite or is `<= -0.5`.

    Notes
    -----
    - If no limits are given, nothing is done.
    - Limits are interpreted in the units the data was plotted in.
    - Setting limits disables autoscaling on the affected axes.
    """
    xpad = _resolve_default(xpad, config.axes.xpad)
    ypad = _resolve_default(ypad, config.axes.ypad)
    recompute_other_axis = _resolve_default(
        recompute_other_axis, config.axes.recompute_other_axis
    )
    ax = get_ax(ax)

    for name, pad in (('xpad', xpad), ('ypad', ypad)):
        if pad is not None and not (np.isfinite(pad) and pad > -0.5):
            raise ValueError(f'{name} must be finite and > -0.5! got {pad}!')

    if limits is not None:
        if xlim is not None or ylim is not None:
            raise ValueError('limits cannot be combined with xlim or ylim!')

        if isinstance(limits, (list, tuple)) and len(limits) == 2:
            lim = (float(get_value(limits[0])), float(get_value(limits[1])))
        elif (
            isinstance(limits, (int, float, np.integer, np.floating))
            and not isinstance(limits, (bool, np.bool_))
        ):
            lim = (-abs(float(limits)), abs(float(limits)))
        else:
            raise ValueError(
                'limits must be either a scalar or a tuple[float, float]! '
                f'got {type(limits).__name__}!'
            )
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        return ax.get_xlim(), ax.get_ylim()

    mx, my = ax.margins()
    _apply_limits(
        ax,
        xlim,
        ylim,
        xpad=mx if xpad is None else xpad,
        ypad=my if ypad is None else ypad,
        recompute_other_axis=recompute_other_axis,

    )

    return ax.get_xlim(), ax.get_ylim()


def _apply_limits(
    ax: maxes.Axes,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    *,
    xpad: float,
    ypad: float,
    recompute_other_axis: bool = True,
) -> None:
    """
    Set axis limits, recomputing the unspecified axis from the data
    visible within the specified one.

    Helper function for `visualastro.plotting.core.axes.set_axis_limits`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify. Must be populated before calling.
    xlim, ylim : tuple[float, float] | None, optional, default=None
        X and Y-axis limits, applied as given (inverted limits invert the axis).
        If only one is provided and `recompute_other_axis=True`, the limits of
        the other axis are computed from the data visible within the provided range.
        If both are provided, no computation is done.
    xpad, ypad : float
        Fractional padding applied to the computed X/Y limits.
    recompute_other_axis : bool, optional, default=True
        If `True`, the limits of the unspecified axis are computed from the
        data visible within the specified one. If `False`, only the
        specified limits are set and the other axis is left to Matplotlib
        autoscaling.


    Returns
    -------
    None

    Notes
    -----
    - If neither `xlim` nor `ylim` is provided, nothing is done and
      Matplotlib autoscaling is left untouched.
    - If no data lies within the provided limits, the other axis is
      left untouched.
    - Only `Line2D` segments are clipped to the window. All other
      artists contribute their vertices only.
    - In-window vertices are always included. `_clip_segments` only
      adds the points where segments cross the window edges.
    - If the computed range is singular, the limits are given by the
      `nonsingular` method of the axis major locator. `xpad`/`ypad`
      are not applied.
    - `Axes.set_xlim`/`set_ylim` disable autoscaling on the affected
      axis.
    """
    if xlim is None and ylim is None:
        return

    if xlim is not None:
        xlim = (float(get_value(xlim[0])), float(get_value(xlim[1])))
        ax.set_xlim(xlim)
    if ylim is not None:
        ylim = (float(get_value(ylim[0])), float(get_value(ylim[1])))
        ax.set_ylim(ylim)

    if not recompute_other_axis or (xlim is not None and ylim is not None):
        return

    w_is_x = xlim is not None
    lo, hi = sorted(xlim if w_is_x else ylim)
    w_log = (ax.get_xscale() if w_is_x else ax.get_yscale()) == 'log'
    t_log = (ax.get_yscale() if w_is_x else ax.get_xscale()) == 'log'
    t_axis = ax.yaxis if w_is_x else ax.xaxis
    set_t = ax.set_ylim if w_is_x else ax.set_xlim
    pad = ypad if w_is_x else xpad

    chunks = []
    for x, y, connected in _get_xy_pairs_from_ax(ax):
        w, t = (x, y) if w_is_x else (y, x)
        inside = (w >= lo) & (w <= hi) & np.isfinite(t)
        chunks.append(t[inside])
        if connected and len(w) > 1:
            chunks.append(
                _clip_segments(w, t, lo, hi, w_log=w_log, t_log=t_log)
            )

    if not chunks:
        return
    vals = np.concatenate(chunks)
    vals = vals[np.isfinite(vals)]
    if t_log:
        vals = vals[vals > 0]
    if vals.size == 0:
        return

    vmin, vmax = vals.min(), vals.max()
    if vmin == vmax:
        lim = t_axis.get_major_locator().nonsingular(vmin, vmax)
    else:
        lim = _pad_limits(vmin, vmax, pad, log=t_log)

    lim = (float(lim[0]), float(lim[1]))
    set_t(lim[::-1] if t_axis.get_inverted() else lim)


def _get_xy_pairs_from_ax(
    ax: maxes.Axes,
) -> list[tuple[NDArray, NDArray, bool]]:
    """
    Get 1D (x, y) arrays in data coordinates from the artists on an Axes.

    Includes `Line2D`, `PathCollection` (scatter), other collections
    drawn in data coordinates (e.g. `fill_between`, `errorbar` bars)
    and patches. Invisible artists and artists not drawn with
    `ax.transData` (e.g. `axhline`, `axvspan`) are skipped.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to read artists from.

    Returns
    -------
    list[tuple[NDArray, NDArray, bool]]
        One `(x, y, connected)` per artist (per path for generic
        collections). If `connected` is `True`, consecutive points are
        joined by straight segments and NaNs are preserved as gaps.
        Otherwise non-finite points have been dropped.
        Empty if `ax` has no valid artists.
    """
    out = []

    def _add(xy, connected=False):
        xy = _to_xy(xy)
        if xy is None:
            return
        if not connected:
            xy = xy[np.isfinite(xy).all(axis=1)]
        elif not np.isfinite(xy).all(axis=1).any():
            return
        if len(xy) > 0:
            out.append((xy[:, 0], xy[:, 1], connected))

    for line in ax.lines:
        if line.get_visible() and line.get_transform() == ax.transData:
            _add(line.get_xydata(), connected=_is_connected_line(line))

    for col in ax.collections:
        if not col.get_visible():
            continue
        if isinstance(col, PathCollection):
            if col.get_offset_transform() == ax.transData:
                _add(col.get_offsets())
        elif col.get_transform() == ax.transData:
            for path in col.get_paths():
                _add(_path_vertices(path))

    for patch in ax.patches:
        if patch.get_visible() and patch.get_data_transform() == ax.transData:
            verts = _path_vertices(patch.get_path())
            _add(patch.get_patch_transform().transform(verts))

    return out


def _path_vertices(path: Path) -> NDArray:
    """
    Get the vertices of a `Path`, excluding `CLOSEPOLY` dummy vertices.

    Helper for `_get_xy_pairs_from_ax`.

    Parameters
    ----------
    path : matplotlib.path.Path
        Path to read vertices from.

    Returns
    -------
    NDArray
        Array of shape (N, 2).
    """
    verts = np.asarray(path.vertices)
    if path.codes is not None:
        verts = verts[path.codes != Path.CLOSEPOLY]
    return verts


def _is_connected_line(line: Line2D) -> bool:
    """
    Whether the vertices of a `Line2D` are drawn as straight segments.

    Parameters
    ----------
    line : matplotlib.lines.Line2D
        Line to check.

    Returns
    -------
    bool
        `False` for marker-only lines and non-default drawstyles
        (e.g. `steps-*`), where the drawn path differs from the vertices.
    """
    return (
        line.get_drawstyle() == 'default'
        and line.get_linestyle() not in ('None', 'none', ' ', '')
    )


def _clip_segments(
    w: NDArray,
    t: NDArray,
    lo: float,
    hi: float,
    *,
    w_log: bool = False,
    t_log: bool = False,
) -> NDArray:
    """
    Get the `t` coordinates at the endpoints of polyline segments
    clipped to the window `lo <= w <= hi`.

    1D implementation of the Liang–Barsky clipping algorithm.
    Helper for `_apply_limits`.

    Parameters
    ----------
    w : NDArray
        Coordinates along the windowed axis, shape (N,). May contain NaN.
    t : NDArray
        Coordinates along the target axis, shape (N,). May contain NaN.
    lo, hi : float
        Window bounds on the `w` axis. Sorted internally.
    w_log, t_log : bool, optional, default=False
        If `True`, the respective axis has a log scale. Segments are
        interpolated in log10 space, matching how they are drawn.
        Non-positive coordinates on logarithmic axes are excluded.

    Returns
    -------
    NDArray
        Values of `t` in data space at the endpoints of the clipped
        segments. Empty if no segment intersects the window.

    Notes
    -----
    - Segments with a non-finite endpoint are discarded, preserving
      the gaps Matplotlib draws at NaNs.
    - Segments with constant `w` coordinates are discarded. Their
      endpoints must be handled separately by the caller.
    """
    lo, hi = sorted((lo, hi))
    if w_log:
        if hi <= 0:
            return np.empty(0)
        lo = np.log10(lo) if lo > 0 else -np.inf
        hi = np.log10(hi)

    ws, ts = _to_scale(w, w_log), _to_scale(t, t_log)
    # w0 & d0 are the starting coordinates of each segment
    # dw & dt are the displacements from each point to the next
    w0, dw = ws[:-1], np.diff(ws)
    t0, dt = ts[:-1], np.diff(ts)

    # coordinate of a point on a segment is given by:
    # x(s) = x0 + sΔx
    # where x0 is the starting point of the segment,
    # Δx is the total distance between the start and end points
    # of the segment, and s is the fraction of the way along the segment
    # of the point (i.e. s=0.5 is the midpoint).
    # plugging lo and hi for x(s) and solving for s we get:
    # s = (lo - x0) / Δx
    # sa and sb represent the s parameter required for each segment
    # to reach the low and hi clipping boundaries.
    # s = 0 is the start of the segment
    # s = 1 is the end of the segment
    # s < 0 & s > 1 lies outside the segment
    with np.errstate(divide='ignore', invalid='ignore'):
        sa = (lo - w0) / dw
        sb = (hi - w0) / dw
    # inner np.minimum and np.maximum of (sa, sb)
    # is to account for a flip in direction of the line
    # sa in one direction == sb in the opposite direction.
    # s0 ensures that all values are >= 0 while s1 <= 1.
    s0 = np.maximum(0.0, np.minimum(sa, sb))
    s1 = np.minimum(1.0, np.maximum(sa, sb))
    # keep finite, non-parallel segments whose clipped parameter
    # interval overlaps the segment's [0, 1] range. this restricts
    # the data to inside the window
    keep = np.isfinite(dw) & np.isfinite(dt) & (dw != 0) & (s0 <= s1)
    t0, dt, s0, s1 = t0[keep], dt[keep], s0[keep], s1[keep]

    # t(s) = t0 + sΔt
    # t-coordinates at the start and end of each clipped segment
    t_clipped = np.concatenate([t0 + s0 * dt, t0 + s1 * dt])

    return 10.0**t_clipped if t_log else t_clipped


def _to_scale(v: NDArray, log: bool) -> NDArray:
    """
    Transform values into axis scale space.

    Helper for `_clip_segments`.

    Parameters
    ----------
    v : NDArray
        Values in data space.
    log : bool
        If `True`, return `log10(v)` with non-positive values set to NaN.
        Otherwise return `v` unchanged.

    Returns
    -------
    NDArray
        Values in scale space.
    """
    if not log:
        return v
    return np.log10(np.where(v > 0, v, np.nan))


def _pad_limits(
    vmin: float,
    vmax: float,
    frac: float,
    *,
    log: bool = False,
) -> tuple[float, float]:
    """
    Pad a data range by a fraction of its extent.

    Helper for `_apply_limits`.

    Parameters
    ----------
    vmin, vmax : float
        Minimum and maximum of the data range. Must satisfy `vmin < vmax`,
        and be positive if `log=True`.
    frac : float
        Fractional padding applied to each side.
    log : bool, optional, default=False
        If `True`, pad in log10 space (multiplicative padding).

    Returns
    -------
    tuple[float, float]
        Padded (min, max) limits.
    """
    if log:
        vmin, vmax = np.log10(vmin), np.log10(vmax)

    pad = frac * (vmax - vmin)
    lo, hi = vmin - pad, vmax + pad

    if log:
        lo, hi = 10.0**lo, 10.0**hi

    return float(lo), float(hi)


def set_title(
    title: str,
    loc: Literal['center', 'left', 'right'] = 'center',
    pad: float = 0.6,
    fontsize: float | _Unset = _UNSET,
    ax: maxes.Axes | WCSAxes | None = None,
    **kwargs
) -> Text:
    """
    Set the title for an `Axes` instance.

    Parameters
    ----------
    title : str
        Figure title.
    loc : {'center', 'left', 'right'} | optional, default='center'
        Title location.
    pad : float, optional, default=0.6
        Title offset from the top of the `Axes`, in points.
    fontsize : float | _Unset, optional, default=_UNSET
        Title fontsize in points. If `_UNSET`, uses
        `config.fontsizes.title`, expressed as a scaling
        factor relative to `config.fontsizes.size`.
    ax : Axes | WCSAxes | None, optional, default=None
        Sets title to either `plt.gca` if `ax=None`, or
        the instance passed in.
    **kwargs : dict, optional
        Additional keyword arguments passed on to `ax.set_title`.

    Returns
    -------
    Text :
        `Text` object returned by `ax.set_title`.
    """
    ax = get_ax(ax)
    fontsize = _resolve_default(fontsize, config.fontsizes.resolve('title'))
    return ax.set_title(title, loc=loc, pad=pad, fontsize=fontsize, **kwargs)


def _set_axis_labels(
    X: u.Quantity | ArrayLike | None,
    Y: u.Quantity | ArrayLike | None,
    ax: maxes.Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    fontsize: float | _Unset = _UNSET,
    unit_bracket_style: Literal['round', 'square'] | _Unset = _UNSET,
    show_physical_type: bool | _Unset = _UNSET,
    show_unit: bool | _Unset = _UNSET,
    fmt: str | _Unset = _UNSET,
    **kwargs
) -> None:
    """
    Automatically generate and set axis labels from objects with physical
    units.

    This function creates formatted axis labels by combining an inferred
    physical type (e.g., 'Wavelength', 'Flux Density') with a formatted unit
    string (e.g., 'μm', 'MJy/sr'). Each component can be enabled or disabled
    independently, and custom labels may be provided.

    Parameters
    ----------
    X : u.Quantity | ArrayLike | None
        Object with a unit exposable with `get_data`. If `None`,
        does not set any labels unless `xlabel` is set.
    Y : ~astropy.units.Quantity | object with a unit
        Object with a unit exposable with `get_data`. If `None`,
        does not set any labels unless `ylabel` is set.
    ax : matplotlib.axes.Axes
        Matplotlib axes object on which to set the labels.
    xlabel, ylabel : str | None, optional, default=None
        Custom label for the x-axis and y-axis. If `None`,
        the label is inferred from `X` / `Y`.
    fontsize : float | _Unset, optional, default=_UNSET
        Fontsize of both axes labels in points. If `_UNSET`, uses
        `config.fontsizes.axes_labels`, expressed as a scaling
        factor relative to `config.fontsizes.size`.
    unit_bracket_style : {'round', 'square'} | _Unset, optional, default=_UNSET
        If `'round`' displays the unit of `X` and `Y` as (unit). If `'square`' as [unit].
    show_physical_type : bool | _Unset, optional, default=_UNSET
        If `True`, include the inferred physical type in the axis label.
        If `_UNSET`, uses `config.show_type_label`.
    show_unit : bool | _Unset, optional, default=_UNSET
        If `True`, include the unit in the axis label. If `_UNSET`, uses
        `config.show_unit_label`.
    fmt : str | _Unset, optional, default=_UNSET
        Format for unit rendering. Passed to `to_latex_unit`.

        Accepted options are `'latex'`, `'latex_inline'`, `'fits'`,
        `'unicode'`, `'console'`, `'vounit'`, `'cds'`, `'ogip'`

        If `_UNSET`, uses `config.unit_label_format`.
    **kwargs : dict, optional
        Additional parameters passed on to `ax.set_xlabel`
        and `ax.set_ylabel`.

    Examples
    --------
    >>> import astropy.units as u
    >>> wavelength = np.linspace(1, 10, 100) * u.um
    >>> flux = np.random.random(100) * u.MJy / u.sr
    >>> fig, ax = plt.subplots()
    >>> ax.plot(wavelength, flux)
    >>> _set_axis_labels(wavelength, flux, ax)
    # xlabel: 'Wavelength [μm]'
    # ylabel: 'Surface Brightness [MJy/sr]'

    >>> _set_axis_labels(wavelength, flux, ax, show_physical_type=False)
    # xlabel: '[μm]'
    # ylabel: '[MJy/sr]'

    >>> _set_axis_labels(wavelength, flux, ax, ylabel='Custom Flux')
    # ylabel: 'Custom Flux [MJy/sr]'

    Notes
    -----

    - Units are formatted using `to_latex_unit`, which provides LaTeX-friendly
      representations.
    - If both `show_physical_type` and `show_unit` are False, the resulting
      axis label is an empty string.
    """
    fontsize = _resolve_default(
        fontsize, config.fontsizes.resolve('axes_labels')
    )
    unit_bracket_style = _resolve_default(unit_bracket_style, config.unit_bracket_style)
    show_physical_type = _resolve_default(show_physical_type, config.show_type_label)
    show_unit = _resolve_default(show_unit, config.show_unit_label)
    fmt = _resolve_default(fmt, config.unit_label_format)

    xlabel = _format_axis_label(
        X, xlabel, unit_bracket_style, show_physical_type, show_unit, fmt
    )
    ylabel = _format_axis_label(
        Y, ylabel, unit_bracket_style, show_physical_type, show_unit, fmt
    )

    ax.set_xlabel(xlabel, fontsize=fontsize, **kwargs)
    ax.set_ylabel(ylabel, fontsize=fontsize, **kwargs)


def _format_axis_label(
    obj: Any,
    label: str | None,
    bracket_style: Literal['round', 'square'],
    show_physical_type: bool,
    show_unit: bool,
    fmt: str | _Unset = _UNSET
) -> str:
    r"""
    Create a scientific axis label with physical type and unit information.

    This function generates axis labels in the format `<physical label> [<unit>]`
    or `<physical label> (<unit>)`, where the physical label describes what the axis
    represents (e.g., 'Wavelength', 'Flux') and the unit is formatted in LaTeX notation.
    Both components can be customized or disabled independently.

    Parameters
    ----------
    obj : any
        An object from which physical type and unit information can be extracted.
        This may be an Astropy `Quantity`, a Spectrum-like object, or any object
        compatible with `get_unit` and `get_physical_type`.
    label : str | None
        If a string is provided, use it as the physical label directly, overriding
        any auto-detected physical type. If None, the physical label is inferred
        from the object's physical type (when `show_physical_type=True`).
    bracket_style : {'round', 'square'}
        If `'round`' displays the unit of `obj` as (unit). If `'square`' as [unit].
    show_physical_type : bool
        If `True`, include the physical type label in the output. If `False`, omit
        the physical label entirely (useful for creating unit-only labels).
    show_unit : bool
        If `True`, include the unit in the output. If `False`, omit the unit
        (useful for creating label-only outputs).
    fmt : str | _Unset, optional, default=_UNSET
        Format for unit rendering. Passed to `to_latex_unit`.

        Accepted options are `'latex'`, `'latex_inline'`, `'fits'`,
        `'unicode'`, `'console'`, `'vounit'`, `'cds'`, `'ogip'`

        If `_UNSET`, uses `config.unit_label_format`.

    Returns
    -------
    str
        A formatted axis label string. The format depends on the parameters:

        - Label only: 'Wavelength'
        - Unit only: '[$\\mu$m]'
        - Both enabled: 'Wavelength ($\\mathrm{\\mu m}$)'
        - Neither: '' (empty string)

    Notes
    -----

    - If the object has no unit or the unit cannot be formatted, the unit
      portion is omitted regardless of `show_unit`.
    - The output is stripped of leading/trailing whitespace.
    """
    physical_type = get_physical_type(obj)

    # format physical label
    if isinstance(label, str):
        physical_label = label

    elif show_physical_type:
        inferred = _infer_physical_type_label(obj)
        if inferred is not None:
            physical_label = inferred
        elif physical_type is not None:
            physical_label = str(physical_type).replace('_', ' ').title()
        else:
            physical_label = ''

    else:
        physical_label = ''

    if show_unit:
        unit_label = get_unit_label(obj, bracket_style=bracket_style, fmt=fmt)
    else:
        unit_label = ''

    return fr'{physical_label} {unit_label}'.strip()


def ax3d_axis_style(
    ax: Axes3D,
    style: str | None = 'cube'
) -> None:
    """
    Set the spine style of a 3D Matplotlib axes by drawing explicit edge lines.

    Overrides Matplotlib axes rendering while keeping the ticks and tick labels.
    For this reason these styles are experimental and may break in interactive mode.

    The styles are defined with respect to the default Matplotlib viewing angles of
    `elev=30`, `azim=-60`, and `roll=0`.

    Parameters
    ----------
    ax : Axes3D
        Target 3D axes object.
    style : str | None, optional, default='cube'
        Spine layout to apply.

        - `'triad'`        : 3 edges from the front-right-bottom corner (Matplotlib default-like).
        - `'floor'`        : 4 bottom edges of the bounding box.
        - `'ceiling'`      : 4 top edges of the bounding box.
        - `'cube'`         : all 12 edges of the bounding box.
        - `'panel_back'`   : floor + back vertical edges + back-top edges (alias `'panel'`).
        - `'panel_front'`  : floor + front vertical edges + front-top edges (alias `'panel_r'`).
        - `'x_panel'`      : floor + back x-aligned top edge + right vertical edges.
        - `'x_panel_front'`: floor + front x-aligned top edge + left vertical edges (alias `'x_panel_r'`).
        - `'y_panel'`      : floor + back y-aligned top edge + left vertical edges.
        - `'y_panel_front'`: floor + front y-aligned top edge + right vertical edges (alias `'y_panel_r'`).

    If `None`, uses Matplotlib rendering engine.
    """
    if style is None: return

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    zmin, zmax = ax.get_zlim()

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    color = ax.spines['left'].get_edgecolor()
    lw = ax.spines['left'].get_linewidth()

    x_top_front = ([xmin, xmax], [ymin, ymin], [zmax, zmax])
    x_top_back = ([xmin, xmax], [ymax, ymax], [zmax, zmax])
    x_bottom_front = ([xmin, xmax], [ymin, ymin], [zmin, zmin])
    x_bottom_back = ([xmin, xmax], [ymax, ymax], [zmin, zmin])

    y_top_front = ([xmax, xmax], [ymin, ymax], [zmax, zmax])
    y_top_back = ([xmin, xmin], [ymin, ymax], [zmax, zmax])
    y_bottom_front = ([xmax, xmax], [ymin, ymax], [zmin, zmin])
    y_bottom_back = ([xmin, xmin], [ymin, ymax], [zmin, zmin])

    z_center_front = ([xmax, xmax], [ymin, ymin], [zmin, zmax])
    z_center_back = ([xmin, xmin], [ymax, ymax], [zmin, zmax])
    z_left = ([xmin, xmin], [ymin, ymin], [zmin, zmax])
    z_right = ([xmax, xmax], [ymax, ymax], [zmin, zmax])

    triad = [x_bottom_front, y_bottom_front, z_right]
    floor = [x_bottom_front, x_bottom_back, y_bottom_front, y_bottom_back]
    ceiling = [x_top_front, x_top_back, y_top_front, y_top_back]
    sides = [z_center_front, z_center_back, z_left, z_right]
    panel_back = [*floor, x_top_back, y_top_back, z_left, z_right, z_center_back]
    panel_front = [*floor, x_top_front, y_top_front, z_left, z_right, z_center_front]
    x_panel_back = [*floor, x_top_back, z_center_back, z_right]
    x_panel_front = [*floor, x_top_front, z_center_front, z_left]
    y_panel_back = [*floor, y_top_back, z_center_back, z_left]
    y_panel_front = [*floor, y_top_front, z_center_front, z_right]
    cube = [*floor, *ceiling, *sides]

    axis_modes = {
        'triad': triad,
        'floor': floor,
        'ceiling': ceiling,
        'panel': panel_back,
        'panel_r': panel_front,
        'panel_back': panel_back,
        'panel_front': panel_front,
        'x_panel': x_panel_back,
        'x_panel_r': x_panel_front,
        'x_panel_front': x_panel_front,
        'y_panel': y_panel_back,
        'y_panel_r': y_panel_front,
        'y_panel_front': y_panel_front,
        'cube': cube,
    }
    edges = axis_modes.get(style, None)
    if edges is None: return

    ax.xaxis.line.set_linewidth(0)
    ax.yaxis.line.set_linewidth(0)
    if not style == 'floor' and not style == 'ceiling':
        ax.zaxis.line.set_linewidth(0)

    for xs, ys, zs in edges:
        ax.plot(xs, ys, zs, lw=lw, color=color, zorder=config.zorder.axes)


def ax3d_pane_color(
    pane_color: ColorType | tuple[ColorType, ColorType, ColorType],
    ax: Axes3D
) -> None:
    """
    Set 3D plot pane color.

    Parameters
    ----------
    pane_color: ColorType | tuple[ColorType, ColorType, ColorType]
        Either color for all panels or a tuple of X, Y, Z panel colors.
    ax : Axes3D
        3D axes instance on which to set panel colors.

    Examples
    --------
    # set all pane colors to red
    >>> fig = plt.figure(figsize=figsize, **kwargs)
    >>> ax = fig.add_subplot(*args, projection='3d')
    >>> ax3d_border_color('r', ax)

    # set each x,y,z color separately
    >>> fig = plt.figure(figsize=figsize, **kwargs)
    >>> ax = fig.add_subplot(*args, projection='3d')
    >>> ax3d_border_color(('r', 'g', 'b'), ax)
    """
    colors = to_list(pane_color)
    if len(colors) >= 3:
        colors = [as_color(get_colors(c), fmt='rgba') for c in colors]
    else:
        colors = [colors[0]]*3

    ax.xaxis.set_pane_color(colors[0])
    ax.yaxis.set_pane_color(colors[1])
    ax.zaxis.set_pane_color(colors[2])


def _configure_axes_scaling(
    ax: maxes.Axes,
    kwargs: dict[str, Any],
    **explicit: Any
) -> None:
    """
    Configure the autoscale state and axis scales of an Axes.

    Must be called before the main plotting call/loop in a plotting
    function, and before `visualastro.plotting.core.axes.set_limits`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes instance to configure.
    kwargs : dict[str, Any]
        Keyword arguments forwarded to matplotlib. Used to resolve
        the parameters below.
    **explicit : Any
        Any of the following keyword arguments. Values not given
        explicitly are taken from `kwargs`, then from `config.axes`.

        - autoscale : bool, optional
            Native Matplotlib autoscaling state, applied with
            `ax.set_autoscale_on(autoscale)`.
        - xlog, ylog : bool, optional
            If `True`, sets the respective axis to log scale.

    Returns
    -------
    None

    Notes
    -----
    - `autoscale=False` before plotting prevents Matplotlib from
      rescaling the view, so limits must then be provided to
      `set_limits`.
    - Axis scales are only set to `'log'` when requested. A `False`
      value does not reset an existing log scale to linear.
    """
    limits_defaults = {
        'autoscale': config.axes.autoscale,
        'xlog': config.axes.xlog,
        'ylog': config.axes.ylog,
    }
    params = _resolve_kwargs(
        kwargs,
        params=_params_from_explicit(explicit, limits_defaults)
    )

    ax.set_autoscale_on(bool(params.autoscale))

    if params.xlog:
        ax.set_xscale('log')
    if params.ylog:
        ax.set_yscale('log')
