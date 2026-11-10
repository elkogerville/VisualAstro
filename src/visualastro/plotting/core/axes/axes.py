"""
Author: Elko Gerville-Reache
Date Created: 2026-06-02
Date Modified: 2026-10-28
Description:
    Functions related to Matplotlib axes.
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

from astropy.io.fits import Header
from astropy.visualization.wcsaxes.core import WCSAxes
from astropy.wcs import WCS
import matplotlib.axes as maxes
from matplotlib.figure import Figure, SubFigure
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType
import numpy as np

from visualastro.core.config import (
    config,
    _Unset,
    _UNSET,
    _resolve_default
)
from visualastro.core.kwargs import _extract_kwargs, _param
from visualastro.utils.wcs_utils import get_wcs_celestial


def figure(
    num: int | str | Figure | SubFigure | None = None,
    figsize: tuple[float, float] | None | _Unset = _UNSET,
    dpi: float | None = None,
    *,
    facecolor: ColorType | None = None,
    edgecolor: ColorType | None = None,
    layout: Literal['constrained', 'compressed', 'tight', 'none', None] | _Unset = _UNSET,
    frameon: bool = True,
    clear: bool = False,
    **kwargs
) -> Figure:
    """
    Create a new figure, or activate an existing one.
    Wrapper of `matplotlib.pyplot.figure`

    Parameters
    ----------
    num : int | str | matplotlib.figure.Figure | matplotlib.figure.SubFigure | None, optional, default=None
        Figure identifier.

        - None: a new figure with an auto-incremented number is created.
        - int | str: the figure with this label is activated if it exists
            (other arguments are then ignored); otherwise it is created.
        - Figure | SubFigure: activated, as in `matplotlib.pyplot.figure`.

    figsize : tuple[float, float] | None | _Unset, optional, default=config.figsize
        Figure size in inches, ``(width, height)``. None defers to ``rcParams``.
    dpi : float | None, optional, default=None
        Resolution in dots per inch. None defers to ``rcParams``.
    facecolor : ColorType | None, optional, default=None
        Background color. None defers to ``rcParams``.
    edgecolor : ColorType | None, optional, default=None
        Border color. None defers to ``rcParams``.
    layout : {'constrained', 'compressed', 'tight', 'none'} | None | _Unset, optional, default=config.layout
        Layout engine. ``'none'`` disables layout. None defers to ``rcParams``.
    frameon : bool, optional, default=True
        If False, suppress drawing the figure background patch.
    clear : bool, optional, default=False
        If True and `num` refers to an existing figure, clear it.
    **kwargs
        Forwarded to `matplotlib.pyplot.figure` and then `matplotlib.figure.Figure`.

    Returns
    -------
    matplotlib.figure.Figure
        The new or activated figure. It becomes the current pyplot figure.

    See Also
    --------
    matplotlib.pyplot.figure
    """
    fig_kw = _extract_kwargs(
        kwargs,
        params=[
            _param('num', num, None),
            _param('figsize', figsize, config.figsize),
            _param('dpi', dpi, None),
            _param('facecolor', facecolor, None),
            _param('edgecolor', edgecolor, None),
            _param('layout', layout, config.layout),
            _param('frameon', frameon, True),
            _param('clear', clear, False),
        ]
    )
    return plt.figure(**fig_kw, **kwargs)


def subplots(
    nrows: int = 1, ncols: int = 1, *,
    sharex: bool | Literal['none', 'all', 'row', 'col'] = False,
    sharey: bool | Literal['none', 'all', 'row', 'col'] = False,
    squeeze: bool = True,
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
    subplot_kw: dict[str, Any] | None = None,
    gridspec_kw: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None | _Unset = _UNSET,
    **fig_kw,
) -> tuple[Figure, maxes.Axes | np.ndarray]:
    """
    Create a figure and a grid of `Axes`.

    Parameters
    ----------
    nrows, ncols : int, optional, default=1
        Grid shape.
    sharex, sharey : bool | {'none', 'all', 'row', 'col'}, optional, default=False
        Controls sharing of x-axis (*sharex*) or y-axis (*sharey*),
        as in `matplotlib.pyplot.subplots`.
    squeeze : bool, optional, default=True
        If True, singleton dimensions are dropped from the returned array.
    width_ratios, height_ratios : Sequence[float] | None, optional, default=None
        Relative column widths / row heights.
    subplot_kw : dict[str, Any] | None, optional, default=None
        Forwarded to `Figure.add_subplot` for every axes.
    gridspec_kw : dict[str, Any] | None, optional, default=None
        Forwarded to `matplotlib.gridspec.GridSpec`.
    figsize : tuple[float, float] | None | _Unset, optional, default=config.figsize
        Figure size in inches. None defers to ``rcParams``.
    **fig_kw
        Forwarded to ``figure`` (e.g. `dpi`, `layout`).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes | numpy.ndarray
        A single `Axes` or an array of them, per `squeeze`.

    See Also
    --------
    matplotlib.pyplot.subplots : Full parameter semantics.

    Notes
    -----
    - Differs from `matplotlib.pyplot.subplots` only in that the figure is
      created with the package ``figure``, so config defaults apply.
    """
    fig = figure(figsize=figsize, **fig_kw)
    axs = fig.subplots(
        nrows=nrows, ncols=ncols,
        sharex=sharex, sharey=sharey, squeeze=squeeze,
        width_ratios=width_ratios, height_ratios=height_ratios,
        subplot_kw=subplot_kw, gridspec_kw=gridspec_kw,
    )
    return fig, axs


def subplots_wcs(
    nrows: int = 1,
    ncols: int = 1,
    *,
    wcs: WCS | Header,
    sharex: bool | Literal['none', 'all', 'row', 'col'] = False,
    sharey: bool | Literal['none', 'all', 'row', 'col'] = False,
    squeeze: bool = True,
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
    subplot_kw: dict[str, Any] | None = None,
    gridspec_kw: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None | _Unset = _UNSET,
    **fig_kw,
) -> tuple[Figure, maxes.Axes | np.ndarray]:
    """
    Create a figure and a grid of `WCSAxes`.

    Parameters
    ----------
    nrows, ncols : int, optional, default=1
        Grid shape.
    wcs : astropy.wcs.WCS | astropy.io.fits.Header
        World coordinate system used as the projection of every axes.
        The celestial 2D part is extracted (see Notes).
    sharex, sharey : bool | {'none', 'all', 'row', 'col'}, optional, default=False
        Controls sharing of x-axis (*sharex*) or y-axis (*sharey*),
        as in `matplotlib.pyplot.subplots`.
    squeeze : bool, optional, default=True
        If True, singleton dimensions are dropped from the returned array.
    width_ratios, height_ratios : Sequence[float] | None, optional, default=None
        Relative column widths / row heights.
    subplot_kw : dict[str, Any] | None, optional, default=None
        Forwarded to `Figure.add_subplot` for every axes. `projection`
        must not be given (see Notes).
    gridspec_kw : dict[str, Any] | None, optional, default=None
        Forwarded to `matplotlib.gridspec.GridSpec`.
    figsize : tuple[float, float] | None | _Unset, optional, default=config.figsize
        Figure size in inches. None defers to ``rcParams``.
    **fig_kw
        Forwarded to ``figure`` (e.g. `dpi`, `layout`).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : astropy.visualization.wcsaxes.WCSAxes | numpy.ndarray
        A single `WCSAxes` or an array of them, per `squeeze`.

    Raises
    ------
    ValueError
        If `wcs` is not a `WCS` or `Header` (including when unset).
    """
    wcs2d = get_wcs_celestial(wcs)
    if not isinstance(wcs2d, WCS):
        raise ValueError(
            "`wcs` must be a `Header` or a `WCS` instance! "
            f"got: {type(wcs).__name__}"
        )

    fig = figure(
        figsize=_resolve_default(figsize, config.figsizeWCS), **fig_kw
    )
    subplot_kw = {**(subplot_kw or {}), 'projection': wcs2d}
    axs = fig.subplots(
        nrows=nrows, ncols=ncols,
        sharex=sharex, sharey=sharey, squeeze=squeeze,
        width_ratios=width_ratios, height_ratios=height_ratios,
        subplot_kw=subplot_kw, gridspec_kw=gridspec_kw,
    )
    return fig, axs


def get_ax(
    ax: maxes.Axes | None = None,
    **fig_kw,
) -> maxes.Axes:
    """
    Resolve the target `Axes`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes | None, optional, default=None
        If an `Axes`, returned unchanged. If None, the current pyplot
        `Axes` is returned, or one is created (see Notes).
    **fig_kw
        Forwarded to ``figure`` (e.g. `figsize`, `dpi`, `layout`) if a
        new figure is created. Unset values use the package config.

    Returns
    -------
    matplotlib.axes.Axes
        Target axes.

    Raises
    ------
    TypeError
        If `ax` is neither None nor an `Axes`.

    Notes
    -----
    - No figures exist: a new figure and axes are created.
    - Current figure is empty: an `Axes` is added to it; `fig_kw` is ignored.
    - Current figure has axes: its current axes is returned; `fig_kw` is ignored.
    - Existing figures are never closed.
    """

    def create(fig: Figure | None) -> maxes.Axes:
        if fig is None:
            fig = figure(**fig_kw)
        return fig.add_subplot()

    return _resolve_ax(ax, maxes.Axes, create)



def get_wcsax(
    ax: WCSAxes | None = None,
    wcs: WCS | Header | None = None,
    **fig_kw,
) -> WCSAxes:
    """
    Resolve the target `WCSAxes`.

    Uses pyplot global state; not thread-safe.

    Parameters
    ----------
    ax : astropy.visualization.wcsaxes.WCSAxes | None, optional, default=None
        If a `WCSAxes`, returned unchanged. If None, the current pyplot
        `WCSAxes` is returned, or one is created.
    wcs : astropy.wcs.WCS | astropy.io.fits.Header | None, optional, default=None
        WCS used if a `WCSAxes` must be created. See ``get_wcs_celestial``
        for accepted types. Ignored if an existing `WCSAxes` is returned.
    **fig_kw
        Forwarded to ``wcsax`` if a new figure is created.

    Returns
    -------
    astropy.visualization.wcsaxes.WCSAxes
        Target axes.

    Raises
    ------
    TypeError
        If `ax` is neither None nor a `WCSAxes`.
    ValueError
        If creation is required and `wcs` is None, or if the current
        figure contains a non-WCS `Axes`.

    Notes
    -----
    Same figure-handling rules as ``get_ax``.
    """

    def create(fig: Figure | None) -> WCSAxes:
        if wcs is None:
            raise ValueError('`wcs` is required to create a WCSAxes.')
        if fig is None:
            figsize = fig_kw.get('figsize', config.figsizeWCS)
            fig = figure(figsize=figsize, **fig_kw)
        return _add_wcsax(fig, wcs=wcs)

    return _resolve_ax(ax, WCSAxes, create)


def _add_wcsax(fig: Figure, *args, wcs: WCS | Header, **kwargs) -> WCSAxes:
    """Add a `WCSAxes` to `fig`. Parses `wcs` via ``get_wcs_celestial``."""
    return cast(
        WCSAxes,
        fig.add_subplot(*args, projection=get_wcs_celestial(wcs), **kwargs),
    )


def _resolve_ax(
    ax: maxes.Axes | None,
    cls: type[maxes.Axes],
    create: Callable[[Figure | None], maxes.Axes],
) -> maxes.Axes:
    """
    Resolve target axes of a given class.

    Parameters
    ----------
    ax : matplotlib.axes.Axes | None
        Returned unchanged if an instance of `cls`.
    cls : type[matplotlib.axes.Axes]
        Required axes class.
    create : Callable[[matplotlib.figure.Figure | None], matplotlib.axes.Axes]
        Factory called with the current empty figure, or None if no figure
        exists. Responsible for validation (e.g. WCS parsing) before
        creating anything.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    TypeError
        If `ax` is neither None nor an instance of `cls`.
    ValueError
        If the current figure contains axes that are not instances of `cls`.
    """
    if isinstance(ax, cls):
        return ax
    if ax is not None:
        raise TypeError(f'`ax` must be {cls.__name__} or None, got {type(ax).__name__}.')

    if not plt.get_fignums():
        return create(None)

    fig = plt.gcf()
    if not fig.axes:
        return create(fig)

    current = fig.gca()
    if isinstance(current, cls):
        return current
    raise ValueError(f'Current figure contains a non-{cls.__name__} Axes; pass `ax` explicitly.')
