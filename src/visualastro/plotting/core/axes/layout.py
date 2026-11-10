from matplotlib import ticker
from matplotlib.figure import Figure
import matplotlib.gridspec as _gridspec
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from visualastro.core.config import config, _Unset, _UNSET, _resolve_default


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
