"""
Author: Elko Gerville-Reache
Date Created: 2026-06-02
Date Modified: 2026-06-09
Description:
    Functions related to matplotlib axes.
"""

from collections.abc import Sequence
from typing import Any, Literal

from astropy.io.fits import Header
import astropy.units as u
from astropy.visualization.wcsaxes.core import WCSAxes
from astropy.wcs import WCS
import matplotlib.axes as maxes
from matplotlib.figure import Figure, SubFigure
import matplotlib.gridspec as _gridspec
import matplotlib.pyplot as plt
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
from visualastro.core.numerical_utils import (
    as_list,
    flatten,
    get_value,
    to_list,
    _is_iterable,
    _is_ndarray_or_quantity_array,
)
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
    Get either the current `Axes` or the instance passed in.

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
    Get either the current `Axes3d` or the instance passed in.

    If The current axis is an `Axes`, it is closed if no data
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

    fig, ax = ax3d(figsize=figsize, **kwargs)
    return ax


def get_wcsax(
    ax: WCSAxes | None,
    wcs: WCS | Header | None = None,
    figsize: tuple[float, float] | _Unset = _UNSET,
    **kwargs
) -> WCSAxes:
    """
    Get either the current `WCSAxes` or the instance passed in.

    If The current axis is an `Axes`, it is closed if no data
    is found via `has_data()`.

    Parameters
    ----------
    ax : WCSAxes | None
        Returns `ax` if is an `WCSAxes`. Otherwise returns
        `plt.gca()` if the current axis is an `WCSAxis`.
        If `plt.gca()` is an `Axes`, returns a new
        `WCSAxes` instance.
    wcs : WCS | Header | None, optional, default=None
        WCS information required to create `ax` if no valid
        `WCSAxes` could be found. If `None`, will only raise
        a `ValueError` if a valid `ax` cannot be returned.
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
        raise ValueError(
            f'Cannot create WCSAxes from header, got: {type(wcs).__name__}'
        )

    fig, ax = wcsax(wcs, figsize=figsize)
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

        * three integers `(*nrows*, *ncols*, *index*)`. The subplot will
        take the *index* position on a grid with *nrows* rows and *ncols* columns.
        *index* starts at 1 in the upper left corner and increases to the right.
        *index* can also be a two-tuple specifying the (*first*, *last*) indices
        (1-based, and including *last*) of the subplot, ie. `(3, 1, (1, 2))`
        makes a subplot that spans the upper 2/3 of the figure.
        * A 3-digit integer. The digits are interpreted as if given separately as
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

        * three integers `(*nrows*, *ncols*, *index*)`. The subplot will
        take the *index* position on a grid with *nrows* rows and *ncols* columns.
        *index* starts at 1 in the upper left corner and increases to the right.
        *index* can also be a two-tuple specifying the (*first*, *last*) indices
        (1-based, and including *last*) of the subplot, ie. `(3, 1, (1, 2))`
        makes a subplot that spans the upper 2/3 of the figure.
        * A 3-digit integer. The digits are interpreted as if given separately as
        three single-digit integers, i.e. `235` is the same as `(2, 3, 5)`.
        Note that this can only be used if there are no more than 9 subplots.

    ax : Axes | None, optional, default=None
        Axes instance. Either `ax` or `fig` should be provided.
    fig : Figure | SubFigure | None, optional, default=None
        Figure instance.  Either `ax` or `fig` should be provided.
    projection : str or None, optional, default=None
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

        * three integers `(*nrows*, *ncols*, *index*)`. The subplot will
        take the *index* position on a grid with *nrows* rows and *ncols* columns.
        *index* starts at 1 in the upper left corner and increases to the right.
        *index* can also be a two-tuple specifying the (*first*, *last*) indices
        (1-based, and including *last*) of the subplot, ie. `(3, 1, (1, 2))`
        makes a subplot that spans the upper 2/3 of the figure.
        * A 3-digit integer. The digits are interpreted as if given separately as
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

        * three integers `(*nrows*, *ncols*, *index*)`. The subplot will
        take the *index* position on a grid with *nrows* rows and *ncols* columns.
        *index* starts at 1 in the upper left corner and increases to the right.
        *index* can also be a two-tuple specifying the (*first*, *last*) indices
        (1-based, and including *last*) of the subplot, ie. `(3, 1, (1, 2))`
        makes a subplot that spans the upper 2/3 of the figure.
        * A 3-digit integer. The digits are interpreted as if given separately as
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

        * three integers `(*nrows*, *ncols*, *index*)`. The subplot will
        take the *index* position on a grid with *nrows* rows and *ncols* columns.
        *index* starts at 1 in the upper left corner and increases to the right.
        *index* can also be a two-tuple specifying the (*first*, *last*) indices
        (1-based, and including *last*) of the subplot, ie. `(3, 1, (1, 2))`
        makes a subplot that spans the upper 2/3 of the figure.
        * A 3-digit integer. The digits are interpreted as if given separately as
        three single-digit integers, i.e. `235` is the same as `(2, 3, 5)`.
        Note that this can only be used if there are no more than 9 subplots.

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
    ax = fig.add_subplot(111, projection=wcs2d)

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

        * minor ticks on
        * inward ticks on all sides
        * axes labels on outer grid axes
        * h/wspace = 0.0

    Nticks : int | None | _Unset, optional, default=`_UNSET`
        Maximum number of major ticks per axis. If `None`,
        uses the default matplotlib settings. If `_UNSET`,
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


def set_axis_limits(
    xdata: ArrayLike | Sequence[ArrayLike] | None = None,
    ydata: ArrayLike | Sequence[ArrayLike] | None = None,
    *,
    ax: maxes.Axes | None = None,
    limits: float | tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xpad: float | _Unset = _UNSET,
    ypad: float | _Unset = _UNSET,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """
    Set axis limits on a Matplotlib Axes based on data range
    and optional user-specified limits.

    To disable visualastro limit computation, set `compute_limits=False`
    or `config.axes.compute_limits=False`. This is recommended for heavy
    plotting calls.

    Parameters
    ----------
    xdata : ArrayLike | Sequence[ArrayLike] | None, optional, default=None
        X-axis data. Can be a single array, a sequence of arrays, or None.
        If `None`, the X-axis will be autoscaled unless `xlim` is provided.
    ydata : array-like, list of array-like, or None, optional
        Y-axis data. Can be a single array, a list of arrays, or None.
        If `None`, the Y-axis will be autoscaled unless `ylim` is provided.
    ax : matplotlib.axes.Axes
        The Axes object on which to set the limits. Required.
    compute_limits : bool | _Unset, optional, default=_UNSET
        If `False`, does not compute any limits based on data,
        and lets matplotlib decide axes limits. If `_UNSET`,
        uses `config.axes.compute_limits`.
    limits : float | tuple[float, float] | None, optional, default=None
        Set symmetric axis limits. Overrides `xlim` and `ylim`.
        If a single float, sets `xlim=ylim=(-abs(limits), limits)`.
        If a `tuple[float, float]`, sets `xlim=ylim=(limits[0], limits[1])`.
        If `None`, delegates limits to `xlim` and `ylim`.
    xlim : tuple[float, float] | None, optional, default=None
        User-defined X-axis limits. If provided, only data within this range
        is considered when computing Y-axis limits automatically. If `None`,
        uses the whole data range.
    ylim : tuple[float, float] | None, optional, default=None
        User-defined Y-axis limits. If provided, only data within this range
        is considered when computing X-axis limits automatically. If `None`,
        uses the whole data range.
    xpad : float | _Unset, optional, default=_UNSET
        Fractional padding to apply to X-axis limits. If `_UNSET`,
        uses `config.axes.xpad`.
    ypad : float | _Unset, optional, default=_UNSET
        Fractional padding to apply to Y-axis limits. If `_UNSET`,
        uses `config.axes.ypad`.

    Returns
    -------
    xlim, ylim : tuple[float, float] | None
        X and Y axis limits. Is only `None` if `xdata/ydata` and `xlim/ylim`
        are also `None`.

    Raises
    ------
    ValueError :
        If `limits` is not `None` and isn't either a `float` or `tuple[float, float]`.
    """
    xpad_frac = _resolve_default(xpad, config.axes.xpad)
    ypad_frac = _resolve_default(ypad, config.axes.ypad)

    ax = get_ax(ax)

    if limits is not None:
        if isinstance(limits, (list, tuple)) and len(limits) == 2:
            lmin, lmax = limits[0], limits[1]
        elif isinstance(limits, (float, int, np.floating, np.integer)):
            limits = np.abs(limits)
            lmin, lmax = -1*limits, limits
        else:
            raise ValueError(
                'limits must be either a scalar or a tuple[float, float]! '
                f'got {type(limits).__name__}!'
            )
        xlim = (lmin, lmax)
        ylim = (lmin, lmax)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return xlim, ylim

    xvals, yvals = _extract_xy_within_bounds(xdata, ydata, xlim=xlim, ylim=ylim)
    xvals, yvals = _add_bounds_from_ax(xvals, yvals, ax, xlim=xlim, ylim=ylim)

    if xlim is None and xvals is not None:
        xmin: float = np.nanmin(xvals)
        xmax: float = np.nanmax(xvals)
        if xmin == xmax:
            xpad = (
                abs(xmin) * xpad_frac if xmin != 0 else
                (xpad_frac if xpad_frac > 0 else 0.1)
            )
        else:
            xpad = xpad_frac * (xmax - xmin)
        xlim = (xmin - xpad, xmax + xpad)

    if ylim is None and yvals is not None:
        ymin: float = np.nanmin(yvals)
        ymax: float = np.nanmax(yvals)
        if ymin == ymax:
            ypad = (
                abs(ymin) * ypad_frac if ymin != 0 else
                (ypad_frac if ypad_frac > 0 else 0.1)
            )
        else:
            ypad = ypad_frac * (ymax - ymin)
        ylim = (ymin - ypad, ymax + ypad)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return xlim, ylim


def _extract_xy_within_bounds(
    xdata,
    ydata,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None
) -> tuple[NDArray | None, NDArray | None]:
    """
    Extracts the xy values of `xdata` and `ydata` within the bounds `xlim`
    and `ylim`.

    Helper function for `visualastro.plotting.core.axes.set_axis_limits`.

    Parameters
    ----------
    xdata, ydata : np.ndarray | Sequence[np.ndarray] :
        X and Y values to extract.
    xlim : tuple[float, float] | None, optional, default=None
        User-defined X-axis limits. If provided, only data within this range
        is considered when computing Y-axis limits automatically. If `None`,
        uses the whole data range.
    ylim : tuple[float, float] | None, optional, default=None
        User-defined Y-axis limits. If provided, only data within this range
        is considered when computing X-axis limits automatically. If `None`,
        uses the whole data range.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None] :
        Array of X/Y values if `xdata`/`ydata` is provided, else `None`.
    """
    if xdata is not None and ydata is not None:
        if not _is_iterable(xdata):
            xdata = as_list(xdata)
        if not _is_iterable(ydata):
            ydata = as_list(ydata)

        if len(xdata) == 1:
            xdata = xdata * len(ydata)
        if len(ydata) == 1:
            ydata = ydata * len(xdata)
        if len(xdata) != len(ydata):
            if _is_ndarray_or_quantity_array(xdata):
                xdata = [xdata]
                xdata = xdata * len(ydata)
            if len(xdata) != len(ydata):
                raise ValueError('Cannot broadcast xdata and ydata')

        xs, ys = [], []
        for x, y in zip(xdata, ydata):
            if x is None or y is None:
                continue

            x = np.asarray(x).ravel()
            y = np.asarray(y).ravel()
            if x.shape != y.shape:
                raise ValueError(
                    'Each x/y pair must match in shape! '
                    f'Got: x.shape: {x.shape}, y.shape: {y.shape}'
                )

            mask = np.ones_like(x, bool)
            if xlim is not None:
                mask &= (x >= get_value(xlim[0])) & (x <= get_value(xlim[1]))
            if ylim is not None:
                mask &= (y >= get_value(ylim[0])) & (y <= get_value(ylim[1]))

            if np.any(mask):
                xs.append(x[mask])
                ys.append(y[mask])

        xvals = np.concatenate(xs) if xs else None
        yvals = np.concatenate(ys) if ys else None

    else:
        xvals = None
        if xdata is not None:
            xvals = flatten(xdata)
            if xvals is not None and xlim is not None:
                mask = (xvals >= get_value(xlim[0])) & (xvals <= get_value(xlim[1]))
                xvals = xvals[mask] if np.any(mask) else xvals

        yvals = None
        if ydata is not None:
            yvals = flatten(ydata)
            if yvals is not None and ylim is not None:
                mask = (yvals >= get_value(ylim[0])) & (yvals <= get_value(ylim[1]))
                yvals = yvals[mask] if np.any(mask) else yvals

    return xvals, yvals


def _add_bounds_from_ax(
    xvals: NDArray | None,
    yvals: NDArray | None,
    ax: maxes.Axes,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None
) -> tuple[NDArray | None, NDArray | None]:
    """
    Add current axes bounds from an ax instance to
    existing arrays containing the plotting values
    used to determine optimal plotting limits.

    Parameters
    ----------
    xvals, yvals : np.ndarray | None
        Array of X and Y plotting values, or `None` if not available.
    ax : maxes.Axes
        Axes instance to grab existing plotting data from.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None] :
        Array of X and Y values if `xvals` or `yvals` is
        not `None` or if `ax` has plotting data.
    """
    xy = _get_xydata_from_ax(ax)

    xvals = np.empty(0) if xvals is None else xvals
    yvals = np.empty(0) if yvals is None else yvals

    if xy is not None:
        mask = np.ones(len(xy), dtype=bool)
        if xlim is not None:
            mask &= (xy[:, 0] >= xlim[0]) & (xy[:, 0] <= xlim[1])
        if ylim is not None:
            mask &= (xy[:, 1] >= ylim[0]) & (xy[:, 1] <= ylim[1])
        xy_masked = xy[mask]
        if xy_masked.size > 0:
            xlim_data = np.nanmin(xy_masked[:, 0]), np.nanmax(xy_masked[:, 0])
            ylim_data = np.nanmin(xy_masked[:, 1]), np.nanmax(xy_masked[:, 1])
            xvals = np.append(xvals, xlim_data)
            yvals = np.append(yvals, ylim_data)

    if xvals.size == 0:
        xvals = None
    if yvals.size == 0:
        yvals = None

    return xvals, yvals


def _get_xydata_from_ax(ax: maxes.Axes) -> NDArray | None:
    """
    Get the xy plotting data from an Axes instance.

    Returns
    -------
    xy : np.ndarray | None
        Array of `ndim=2` of xy plotting data, or `None`
        if `ax` is empty.
    """
    segments = []
    for line in ax.lines:
        segments.append(line.get_xydata())
    for col in ax.collections:
        offsets = col.get_offsets().data
        if len(offsets) > 0:
            segments.append(offsets)
    for patch in ax.patches:
        segments.append(patch.get_xy())

    xy = np.vstack(segments) if segments else None

    return xy


def set_axis_labels(
    X: u.Quantity | ArrayLike | None,
    Y: u.Quantity | ArrayLike | None,
    ax: maxes.Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    unit_bracket_style: Literal['round', 'square'] | _Unset = _UNSET,
    show_physical_type: bool | _Unset = _UNSET,
    show_unit: bool | _Unset = _UNSET,
    fmt: Literal['latex', 'latex_inline', 'fits', 'unicode', 'console', 'vounit', 'cds', 'ogip'] | _Unset = _UNSET
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
    Y : '~astropy.units.Quantity' or object with a unit
        Object with a unit exposable with `get_data`. If `None`,
        does not set any labels unless `ylabel` is set.
    ax : matplotlib.axes.Axes
        Matplotlib axes object on which to set the labels.
    xlabel : str | None, optional, default=None
        Custom label for the x-axis. If None, the label is inferred from `X`.
    ylabel : str | None, optional, default=None
        Custom label for the y-axis. If None, the label is inferred from `Y`.
    unit_bracket_style : Literal['round', 'square'] | _Unset, optional, default=_UNSET
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

    Examples
    --------
    >>> import astropy.units as u
    >>> wavelength = np.linspace(1, 10, 100) * u.um
    >>> flux = np.random.random(100) * u.MJy / u.sr
    >>> fig, ax = plt.subplots()
    >>> ax.plot(wavelength, flux)
    >>> set_axis_labels(wavelength, flux, ax)
    # xlabel: 'Wavelength [μm]'
    # ylabel: 'Surface Brightness [MJy/sr]'

    >>> set_axis_labels(wavelength, flux, ax, show_physical_type=False)
    # xlabel: '[μm]'
    # ylabel: '[MJy/sr]'

    >>> set_axis_labels(wavelength, flux, ax, ylabel='Custom Flux')
    # ylabel: 'Custom Flux [MJy/sr]'

    Notes
    -----
    - Units are formatted using `to_latex_unit`, which provides LaTeX-friendly
      representations.
    - If both `show_physical_type` and `show_unit` are False, the resulting
      axis label is an empty string.
    """
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

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _format_axis_label(
    obj: Any,
    label: str | None,
    bracket_style: Literal['round', 'square'],
    show_physical_type: bool,
    show_unit: bool,
    fmt: Literal['latex', 'latex_inline', 'fits', 'unicode', 'console', 'vounit', 'cds', 'ogip']
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
    bracket_style : Literal['round', 'square']
        If `'round`' displays the unit of `obj` as (unit). If `'square`' as [unit].
    show_physical_type : bool
        If `True`, include the physical type label in the output. If `False`, omit
        the physical label entirely (useful for creating unit-only labels).
    show_unit : bool
        If `True`, include the unit in the output. If `False`, omit the unit
        (useful for creating label-only outputs).
    fmt : str, default=_UNSET
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
    Set the spine style of a 3D matplotlib axes by drawing explicit edge lines.

    Overrides matplotlib axes rendering while keeping the ticks and tick labels.
    For this reason these styles are experimental and may break in interactive mode.

    The styles are defined with respect to the default matplotlib viewing angles of
    `elev=30`, `azim=-60`, and `roll=0`.

    Parameters
    ----------
    ax : Axes3D
        Target 3D axes object.
    style : str | None, optional, default='cube'
        Spine layout to apply.

        * `'triad'`        : 3 edges from the front-right-bottom corner (matplotlib default-like).
        * `'floor'`        : 4 bottom edges of the bounding box.
        * `'ceiling'`      : 4 top edges of the bounding box.
        * `'cube'`         : all 12 edges of the bounding box.
        * `'panel_back'`   : floor + back vertical edges + back-top edges (alias `'panel'`).
        * `'panel_front'`  : floor + front vertical edges + front-top edges (alias `'panel_r'`).
        * `'x_panel'`      : floor + back x-aligned top edge + right vertical edges.
        * `'x_panel_front'`: floor + front x-aligned top edge + left vertical edges (alias `'x_panel_r'`).
        * `'y_panel'`      : floor + back y-aligned top edge + left vertical edges.
        * `'y_panel_front'`: floor + front y-aligned top edge + right vertical edges (alias `'y_panel_r'`).

    If `None`, uses matplotlib rendering engine.
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


def _set_axis_limits_scaling_mode(ax, autoscale, compute_limits):
    """
    Set axes limit scaling mode. Must be called before
    the main plotting call/loop in a plotting function.

    Parameters
    ----------
    ax : maxes.Axes
        Axes instance.
    autoscale : bool
        If `True`, uses native matplotlib autoscaling,
        ie. `ax.set_autoscale_on(True)`, and disables
        VisualAstro's manual limits computation
        (`compute_limits=False`).
    compute_limits : bool
        Flag to be passed on to one of the visualastro
        plotting interface functions. If `True`, will
        invoke `visualastro.plotting.core.axes.set_axis_limits`.

    Returns
    -------
    compute_limits : bool
        Set to `False` if `autoscale=True`. Otherwise is left
        unchanged.
    """
    if autoscale:
        ax.set_autoscale_on(True)
        compute_limits = False
    else:
        ax.set_autoscale_on(False)

    return compute_limits
