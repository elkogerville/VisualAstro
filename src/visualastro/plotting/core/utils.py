"""
Author: Elko Gerville-Reache
Date Created: 2025-05-24
Date Modified: 2026-10-25
Description:
    Plotting utility functions.
"""

from collections.abc import Sequence
from typing import Literal
from functools import partial

import astropy.units as u
import matplotlib.axes as maxes
from matplotlib.collections import Collection
from matplotlib.image import AxesImage
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Circle, Ellipse
from matplotlib.typing import ColorType
import numpy as np
from numpy.typing import NDArray

from visualastro.core.config import (
    get_config_value,
    _Unset, _UNSET,
)
from visualastro.core.data import get_value, to_list
from visualastro.core.sequences import _cycle
from visualastro.optional_dependencies.register import _require_dependency
from visualastro.optional_dependencies._regions import (
    PixCoord, EllipsePixelRegion
)
from visualastro.plotting.core.axes import get_ax
from visualastro.plotting.core.colors import get_colors, sample_cmap
from visualastro.plotting.core.plot_input import (
    _extract_xy_from_ndarray2D,
    _normalize_order
)


def _get_zorder(zorders: list[float] | None, i: int, fallback: float):
    """
    Get zorder value from a list of zorders with a fallback zorder.
    Increments the fallback value by i.

    Parameters
    ----------
    zorders : list[float] | None
        List of zorders to cycle through. If `None`, returns
        `fallback + 1`.
    fallback : float
        Fallback value if `zorders` or `zorders[i]` is `None`.
        Always incremented by 1.
    """
    if zorders is None:
        return fallback + 1
    return _cycle(zorders, i) if _cycle(zorders, i) is not None else fallback+i


# Plot Matplotlib Patches and Shapes
# ----------------------------------
def plot_circles(
    circles,
    ax: maxes.Axes,
    colors=None,
    linewidth=None,
    fill=None,
    cmap=None
):
    """
    Plot one or more circles on a Matplotlib axis with customizable style.

    Parameters
    ----------
    circles : ArrayLike | None
        Circle coordinates and radii. Can be a single circle `[x, y, r]`
        or a list/array of circles `[[x1, y1, r1], [x2, y2, r2], ...]`.
        If None, no circles are plotted.
    ax : matplotlib.axes.Axes
        The Matplotlib axis on which to plot the circles.
    colors : list[colors] | str | None, optional, default=None
        List of colors to cycle through for each circle. None defaults
        to ['r', 'mediumvioletred', 'magenta']. A single color can also
        be passed. If there are more circles than colors, colors are
        sampled from a colormap using sample_cmap(cmap=`cmap`).
    linewidth : float | None, optional, default=None
        Width of the circle edge lines. If None,
        uses `config.linewidth`.
    fill : bool | None, optional, default=None
        Whether the circles are filled. If None,
        uses `config.circle_fill`.
    cmap : str | None, optional, default=None
        matplolib cmap used to sample default circle colors.
        If None, uses `config.cmap`.
    """
    # get default config values
    linewidth = get_config_value(linewidth, 'linewidth')
    fill = get_config_value(fill, 'circle_fill')
    cmap = get_config_value(cmap, 'cmap')

    if circles is not None:
        # ensure circles is list [x,y,r] or list of list [[x,y,r],[x,y,r]...]
        circles = np.atleast_2d(circles)
        if circles.shape[1] != 3:
            raise ValueError(
                'Circles must be either [x, y, r] or [[x1, y1, r1], [x2, y2, r2], ...]'
            )
        # number of circles to plot
        N = circles.shape[0]
        # set circle colors
        if colors is None:
            colors = ['r', 'mediumvioletred', 'magenta'] if N<=3 else sample_cmap(N, cmap=cmap)
        if isinstance(colors, str):
            colors = [colors]

        # plot each cirlce
        for i, circle in enumerate(circles):
            x, y, r = circle
            color = colors[i%len(colors)]
            circle_patch = Circle((x, y), radius=r, fill=fill, linewidth=linewidth, color=color)
            ax.add_patch(circle_patch)


def plot_ellipses(
    ellipses: Ellipse | list[Ellipse], ax: maxes.Axes | None = None
) -> None:
    """
    Plots an ellipse or list of ellipses to an axes.

    Parameters
    ----------
    ellipses : matplotlib.patches.Ellipse | list
        The Ellipse or list of Ellipses to plot.
    ax : matplotlib.axes.Axes | None, optional, default=None
        Matplotlib axis on which to plot the ellipses(s).
        If `None`, uses `plt.gca()`.
    """
    ax = get_ax(ax)
    if ellipses is not None:
        ellipses = to_list(ellipses)

        for ellipse in ellipses:
            if not isinstance(ellipse, Ellipse):
                raise ValueError(
                    'ellipses must contain matplotlib.patches.Ellipse instances! '
                    f'got: {type(ellipse).__name__}'
                )
            ax.add_patch(_copy_ellipse(ellipse))


def _copy_ellipse(ellipse: Ellipse) -> Ellipse:
    """
    Returns a copy of an Ellipse object.

    This function is used to avoid running into
    a matplotlib error when plotting the same
    artist onto multiple figures.

    Parameters
    ----------
    ellipse : matplotlib.patches.Ellipse
        The Ellipse object to copy.

    Returns
    -------
    matplotlib.patches.Ellipse
        A new Ellipse object with the same properties as the input.
    """
    return Ellipse(
        xy=ellipse.center,
        width=ellipse.width,
        height=ellipse.height,
        angle=ellipse.angle,
        edgecolor=ellipse.get_edgecolor(),
        facecolor=ellipse.get_facecolor(),
        lw=ellipse.get_linewidth(),
        ls=ellipse.get_linestyle(),
        alpha=ellipse.get_alpha()
    )


def plot_interactive_ellipse(center, w, h, ax, text_loc=None,
                             text_color=None, highlight=None,
                             angle=0.0, rotation_step=5.0):
    """
    Create an interactive ellipse selector on an Axes
    along with an interactive text window displaying
    the current ellipse center, width, and height.

    Parameters
    ----------
    center : tuple of float
        (x, y) coordinates of the ellipse center in data units.
    w : float
        Width of the ellipse.
    h : float
        Height of the ellipse.
    ax : matplotlib.axes.Axes
        The Axes on which to draw the ellipse selector.
    text_loc : list[float] | None, optional, default=None
        Position of the text label in Axes coordinates, given as [x, y].
        If None, uses `config.text_loc`.
    text_color : str | None, optional, default=None
        Color of the annotation text. If None, uses `config.text_color`.
    highlight : bool | None, optional, default=None
        If True, adds a bbox to highlight the text. If None,
        uses `config.highlight`.
    angle : float, optional, default=0.0
        Initial rotation angle in degrees.
    rotation_step : float, optional, default=5.0
        Rotation increment in degrees per key press.

    Returns
    -------
    matplotlib.widgets.EllipseSelector
        Interactive selector. Also stored as `ax._ellipse_selector`.

    Notes
    -----
    Key bindings:

    - 'e': rotate by `-rotation_step`.
    - 'f': rotate by `+rotation_step`.
    - 't': reset angle to 0.
    - 'o': toggle the selector on/off.

    Notes
    -----
    Ensure an interactive backend is active. This can be
    activated with use_interactive().
    """
    _require_dependency('regions')
    text_loc = get_config_value(text_loc, 'ellipse_label_loc')
    text_color = get_config_value(text_color, 'text_color')
    highlight = get_config_value(highlight, 'highlight')

    facecolor = 'k' if text_color == 'w' else 'w'
    bbox = dict(facecolor=facecolor, alpha=0.6, edgecolor="none") if highlight else None

    text = ax.text(
        text_loc[0], text_loc[1], '',
        transform=ax.transAxes,
        size='small',
        color=text_color,
        bbox=bbox
    )

    # create and store region explicitly
    region = EllipsePixelRegion(
        center=PixCoord(x=center[0], y=center[1]),
        width=w,
        height=h,
        angle=angle * u.deg
    )

    selector = region.as_mpl_selector(
        ax,
        callback=partial(_update_ellipse_region, text=text)
    )

    ax._ellipse_selector = selector
    ax._ellipse_region = region

    artist = selector._selection_artist
    artist.angle = angle

    # initialize display
    _update_ellipse_region(region, text=text)

    def _rotate(event):
        """
        Rotate the ellipse region based on key press events.

        Parameters
        ----------
        event :
        """
        key = event.key

        if key not in ('e', 'f', 't', 'o'):
            return

        artist = selector._selection_artist
        region = ax._ellipse_region

        if key == 'e':
            artist.angle -= rotation_step

        elif key == 'f':
            artist.angle += rotation_step

        elif key == 't':
            artist.angle = 0.0

        elif key == 'o':
            selector.set_active(not selector.active)
            return

        # synchronize region <- artist
        region.angle = artist.angle * u.deg

        artist.figure.canvas.draw_idle()

        _update_ellipse_region(region, text=text)

    ax.figure.canvas.mpl_connect('key_press_event', _rotate)

    return selector


def _update_ellipse_region(region, text):
    """
    Update ellipse information text when the
    interactive region is modified.

    Parameters
    ----------
    region : regions.EllipsePixelRegion
        The ellipse region being updated.
    text : matplotlib.text.Text
        The text object used to display ellipse parameters.
    """
    x_center = region.center.x
    y_center = region.center.y
    width = region.width
    height = region.height
    angle = region.angle
    major = max(width, height)
    minor = min(width, height)

    # display properties
    text.set_text(
        f'Center: [{x_center:.1f}, {y_center:.1f}]\n'
        f'Major: {major:.1f}\n'
        f'Minor: {minor:.1f}\n'
        f'Angle: {angle:.1f}\n'
    )


def _default_ellipse_geometry(
    img: AxesImage | Collection,
) -> tuple[tuple[float, float], float, float] | None:
    """
    Default ellipse center and size, in data coordinates, for an artist.

    Parameters
    ----------
    img : matplotlib.image.AxesImage | matplotlib.collections.Collection
        Plotted artist. `AxesImage` uses its extent; `Collection`
        (e.g. `PathCollection` from `ax.scatter`) uses the bounding box
        of its finite offsets.

    Returns
    -------
    center : tuple[float, float]
        (x, y) center of the artist's bounding box.
    w, h : float
        One fifth of the bounding box width and height.

    None
        If the artist type is unsupported, has no finite points,
        or has zero extent along either axis.
    """
    if isinstance(img, AxesImage):
        x0, x1, y0, y1 = img.get_extent()

    elif isinstance(img, Collection):
        offsets = np.ma.filled(img.get_offsets(), np.nan).astype(float)
        offsets = offsets[np.isfinite(offsets).all(axis=1)]
        if offsets.size == 0:
            return None
        x0, y0 = offsets.min(axis=0)
        x1, y1 = offsets.max(axis=0)

    else:
        return None

    w = abs(x1 - x0) / 5
    h = abs(y1 - y0) / 5
    if not (w > 0 and h > 0):
        return None

    return ((x0 + x1) / 2, (y0 + y1) / 2), w, h


def ellipse_patch(
    center, w, h, angle=0, fill=False,
    edgecolor='k', facecolor=None, **kwargs
):
    """
    Create a matplotlib.patches.Ellipse object.

    Parameters
    ----------
    center : tuple of float
        (x, y) coordinates of the ellipse center.
    w : float
        Width of the ellipse (along x-axis before rotation).
    h : float
        Height of the ellipse (along y-axis before rotation).
    angle : float, optional, default=0
        Rotation angle of the ellipse in degrees (counterclockwise).
    fill : bool, optional, default=False
        Whether the ellipse should be filled (True) or only outlined (False).
    edgecolor : str, optional, default='k'
        Color of patch edges.
    facecolor : str, optional, default=None
        Color of patch face. If not `None`,
        sets `fill=True`.
    linestyle or ls : {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}, optional
        Linestyle of ellipse patch.
    linewidth or lw : float | None, optional
        Linewidth of ellipse patch.
    Any other kwargs accepted by matplotlib.patches.Ellipse

    Returns
    -------
    ellipse : matplotlib.patches.Ellipse
        An Ellipse patch that can be added to a matplotlib Axes.
    """
    if facecolor is not None:
        fill = True

    return Ellipse(
        xy=(center[0], center[1]), width=w, height=h,
        angle=angle, fill=fill, edgecolor=edgecolor,
        facecolor=facecolor, **kwargs
    )


def plot_points(
    *points: float | u.Quantity | NDArray | Sequence[float | u.Quantity | NDArray],
    ax: maxes.Axes,
    color: ColorType | list[ColorType] | int = 'astro',
    size: float | list[float] = 20,
    marker: MarkerStyle | list[MarkerStyle] = '*',
    order: Literal['C', 'c', 'F', 'f', 'fortran'] | _Unset = _UNSET,
    xy_indices: Literal['implicit', 'explicit'] | tuple[int, int] = 'implicit',
) -> None:
    """
    Plot points on a given Matplotlib axis.

    Parameters
    ----------
    points : float | u.Quantity | NDArray | Sequence[float | u.Quantity | NDArray]
        Input data. Supported forms:

        - Single argument:
            - 1D ArrayLike or Quantity: Y values, X = None
            - 2D array or Quantity: extract X, Y according to `order` and `xy_indices`
            - list/tuple of scalars: Y values, X = None
            - scalar or scalar Quantity: single Y value, X = None

        - Two arguments:
            - (X, Y) pairs passed through unchanged

    ax : matplotlib.axes.Axes
        The Matplotlib axis on which to plot the points.
    color : ColorType | list[ColorType] | int, optional, default='astro'
        Color of the points. If an integer, will draw colors using `config.cmap`.
    size : float | list[float], optional, default=20
        Marker size(s).
    marker : MarkerStyle | list[MarkerStyle], optional, default='*'
        Matplotlib marker style(s).
    order : {'c', 'fortran'} | _Unset, optional, default=_UNSET
        Memory layout for 2D input interpretation. Defines what a
        column is for `xy_indices`.

        * 'c': row-major, shape (N, 2)
        * 'fortran': column-major, shape (2, N)

        If `_UNSET`, uses `config.array_order`.
    xy_indices : {'implicit', 'explicit'} | tuple[int, int], optional
        Column extraction mode for 2D inputs.

        * 'implicit': return (None, [col_0, col_1, ...])
        * 'explicit': return (col_0, col_1)
        * tuple (i, j): return (col_i, col_j)
    """
    if points[0] is not None:
        xlist, ylist = _normalize_plotting_inputs(
            *points, order=order, xy_indices=xy_indices
        )

        colors = get_colors(color)
        sizes = to_list(size)
        markers = to_list(marker)
        for i in range(len(ylist)):
            x = get_value(_cycle(xlist, i))
            y = get_value(_cycle(ylist, i))

            ax.scatter(
                x, y,
                marker=_cycle(markers, i),
                s=_cycle(sizes, i),
                color=_cycle(colors, i)
            )
