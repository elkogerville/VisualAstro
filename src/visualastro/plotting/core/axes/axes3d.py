"""
Author: Elko Gerville-Reache
Date Created: 2026-06-02
Date Modified: 2026-10-28
Description:
    Functions related to Matplotlib Axes3D.
"""

from collections.abc import Sequence
from typing import Any, Literal, cast

from matplotlib.figure import Figure
from matplotlib.typing import ColorType
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from visualastro.core.config import config, _Unset, _UNSET, _resolve_default
from visualastro.core.data import to_list
from visualastro.plotting.core.axes.axes import _resolve_ax, figure
from visualastro.plotting.core.colors.colors import get_colors
from visualastro.plotting.core.colors.utils import as_color


def get_ax3d(
    ax: Axes3D | None = None,
    **fig_kw,
) -> Axes3D:
    """
    Resolve the target `Axes3D`.

    Uses pyplot global state; not thread-safe.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.Axes3D | None, optional, default=None
        If an `Axes3D`, returned unchanged. If None, the current pyplot
        `Axes3D` is returned, or one is created.
    **fig_kw
        Forwarded to ``ax3d`` if a new figure is created.

    Returns
    -------
    mpl_toolkits.mplot3d.Axes3D
        Target axes.

    Raises
    ------
    TypeError
        If `ax` is neither None nor an `Axes3D`.
    ValueError
        If the current figure contains a non-3D `Axes`.

    Notes
    -----
    Same figure-handling rules as ``get_ax``.
    """

    def create(fig: Figure | None) -> Axes3D:
        if fig is None:
            figsize = fig_kw.get('figsize', config.figsize3D)
            fig = figure(figsize=figsize, **fig_kw)
        return _add_ax3d(fig)

    return _resolve_ax(ax, Axes3D, create)


def subplots_3d(
    nrows: int = 1,
    ncols: int = 1,
    *,
    sharex: bool | Literal['none', 'all', 'row', 'col'] = False,
    sharey: bool | Literal['none', 'all', 'row', 'col'] = False,
    squeeze: bool = True,
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
    subplot_kw: dict[str, Any] | None = None,
    gridspec_kw: dict[str, Any] | None = None,
    figsize: tuple[float, float] | None | _Unset = _UNSET,
    **fig_kw,
) -> tuple[Figure, Axes3D | np.ndarray]:
    """
    Create a figure and a grid of `Axes3D`.

    Parameters
    ----------
    nrows, ncols : int, optional, default=1
        Grid shape.
    sharex, sharey : bool | {'none', 'all', 'row', 'col'}, optional, default=False
        Axis sharing, as in `matplotlib.pyplot.subplots`.
    squeeze : bool, optional, default=True
        If True, singleton dimensions are dropped from the returned array.
    width_ratios, height_ratios : Sequence[float] | None, optional, default=None
        Relative column widths / row heights.
    subplot_kw : dict[str, Any] | None, optional, default=None
        Forwarded to `Figure.add_subplot` for every axes. Any `projection`
        entry is overwritten (see Notes).
    gridspec_kw : dict[str, Any] | None, optional, default=None
        Forwarded to `matplotlib.gridspec.GridSpec`.
    figsize : tuple[float, float] | None | _Unset, optional, default=config.figsize3D
        Figure size in inches. None defers to ``rcParams``.
    **fig_kw
        Forwarded to ``figure`` (e.g. `dpi`, `layout`).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure instance.
    ax : mpl_toolkits.mplot3d.Axes3D | numpy.ndarray
        A single `Axes3D` or an array of them, per `squeeze`.
    """
    fig = figure(figsize=_resolve_default(figsize, config.figsize3D), **fig_kw)
    subplot_kw = {**(subplot_kw or {}), 'projection': '3d'}
    axs = fig.subplots(
        nrows=nrows, ncols=ncols,
        sharex=sharex, sharey=sharey, squeeze=squeeze,
        width_ratios=width_ratios, height_ratios=height_ratios,
        subplot_kw=subplot_kw, gridspec_kw=gridspec_kw,
    )
    return fig, axs


def _add_ax3d(fig: Figure, *args, **kwargs) -> Axes3D:
    """Add an `Axes3D` to `fig`. Isolates the stub-mismatch cast."""
    return cast(Axes3D, fig.add_subplot(*args, projection='3d', **kwargs))


def ax3d_axis_style(
    style: str | None = 'cube',
    ax: Axes3D | None = None,
) -> None:
    """
    Set the spine style of a 3D Matplotlib axes by drawing explicit edge lines.

    Overrides Matplotlib axes rendering while keeping the ticks and tick labels.
    For this reason these styles are experimental and may break in interactive mode.

    The styles are defined with respect to the default Matplotlib viewing angles of
    `elev=30`, `azim=-60`, and `roll=0`.

    Parameters
    ----------
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
    ax : Axes3D
        Target 3D axes object. If ``None``, uses ``get_ax3d``.
    """
    if style is None: return
    ax = get_ax3d(ax)

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
    ax: Axes3D | None = None,
) -> None:
    """
    Set 3D plot pane color.

    Parameters
    ----------
    pane_color: ColorType | tuple[ColorType, ColorType, ColorType]
        Either color for all panels or a tuple of X, Y, Z panel colors.
    ax : Axes3D | None, optional, default=None
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
    ax = get_ax3d(ax)
    colors = to_list(pane_color)
    if len(colors) >= 3:
        colors = [as_color(get_colors(c), fmt='rgba') for c in colors]
    else:
        colors = [colors[0]]*3

    ax.xaxis.set_pane_color(colors[0])
    ax.yaxis.set_pane_color(colors[1])
    ax.zaxis.set_pane_color(colors[2])
