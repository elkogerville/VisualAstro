"""
Author: Elko Gerville-Reache
Date Created: 2025-05-24
Date Modified: 2026-10-25
Description:
    Hlines and vlines utility functions.
"""

from collections.abc import Sequence
from typing import Literal

import astropy.units as u
from astropy.units import Equivalency, StructuredUnit, UnitBase
from astropy.visualization.wcsaxes.core import WCSAxes
import matplotlib.axes as maxes
from matplotlib.lines import Line2D
from matplotlib.typing import ColorType, LineStyleType

from visualastro.core.array_types import _is_1d, _is_iterable
from visualastro.core.config import config, _Unset, _UNSET
from visualastro.core.data import to_list
from visualastro.core.kwargs import _extract_kwargs, _kwarg, _param
from visualastro.plotting.core.axes import get_ax, get_wcsax


def axvlines(
    *vlines: float | u.Quantity | Sequence[float | u.Quantity] | None,
    ax: maxes.Axes | None = None,
    color: ColorType | _Unset = _UNSET,
    linestyle: LineStyleType | _Unset = _UNSET,
    linewidth: float | _Unset = _UNSET,
    alpha: float | _Unset = _UNSET,
    unit: UnitBase | StructuredUnit | None = None,
    equivalencies: Equivalency | None = None,
    **kwargs,
) -> list[Line2D] | None:
    """
    Plot one or more vertical reference lines on a Matplotlib axis.

    Parameters
    ----------
    *vlines : float | astropy.units.Quantity | Sequence[float | astropy.units.Quantity]
        X-axis coordinate(s) at which to draw vertical line(s).
        - Each positional argument may be a scalar or a 1D sequence.
        - `Quantity` values are converted to `unit` before plotting.
        - `None` entries are skipped.
        - If no arguments are given, nothing is drawn and `None` is returned.

    ax : matplotlib.axes.Axes | None, optional, default=None
        Axes on which to draw the line(s). If None, resolved via `get_ax`.
    color : ColorType, optional, default=config.axline.color
        Line color.
    linestyle : LineStyleType, optional, default=config.axline.linestyle
        Line style.
    linewidth : float, optional, default=config.axline.linewidth
        Line width in points.
    alpha : float, optional, default=config.axline.alpha
        Line opacity, in [0, 1].
    unit : astropy.units.UnitBase | str | None, optional, default=None
        Unit to which `Quantity` values in `vlines` are converted before
        plotting. If None, no conversion is applied and the bare `.value` of
        each `Quantity` is used.
    equivalencies : astropy.units.Equivalency | None, optional, default=None
        Equivalencies used in the unit conversion. Ignored if `unit` is None.
    **kwargs
        Additional keyword arguments passed to `matplotlib.axes.Axes.axvline`.
        `zorder` defaults to `config.zorder.vlines`.

    Returns
    -------
    list[matplotlib.lines.Line2D] | None
        Created line artists, in input order. None if `vlines` is empty.

    Raises
    ------
    ValueError
        If any element of `vlines` is not a 1D sequence of x coordinates.
    """
    return _axlines(
        *vlines,
        orientation='v',
        ax=ax,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        unit=unit,
        equivalencies=equivalencies,
        **kwargs,
    )


def axhlines(
    *hlines: float | u.Quantity | Sequence[float | u.Quantity] | None,
    ax: maxes.Axes | None = None,
    color: ColorType | _Unset = _UNSET,
    linestyle: LineStyleType | _Unset = _UNSET,
    linewidth: float | _Unset = _UNSET,
    alpha: float | _Unset = _UNSET,
    unit: UnitBase | StructuredUnit | None = None,
    equivalencies: Equivalency | None = None,
    **kwargs,
) -> list[Line2D] | None:
    """
    Plot one or more horizontal reference lines on a Matplotlib axis.

    Parameters
    ----------
    *hlines : float | astropy.units.Quantity | Sequence[float | astropy.units.Quantity]
        Y-axis coordinate(s) at which to draw horizontal line(s).
        - Each positional argument may be a scalar or a 1D sequence.
        - `Quantity` values are converted to `unit` before plotting.
        - `None` entries are skipped.
        - If no arguments are given, nothing is drawn and `None` is returned.

    ax : matplotlib.axes.Axes | None, optional, default=None
        Axes on which to draw the line(s). If None, resolved via `get_ax`.
    color : ColorType, optional, default=config.axline.color
        Line color.
    linestyle : LineStyleType, optional, default=config.axline.linestyle
        Line style.
    linewidth : float, optional, default=config.axline.linewidth
        Line width in points.
    alpha : float, optional, default=config.axline.alpha
        Line opacity, in [0, 1].
    unit : astropy.units.UnitBase | str | None, optional, default=None
        Unit to which `Quantity` values in `hlines` are converted before
        plotting. If None, no conversion is applied and the bare `.value` of
        each `Quantity` is used.
    equivalencies : astropy.units.Equivalency | None, optional, default=None
        Equivalencies used in the unit conversion. Ignored if `unit` is None.
    **kwargs
        Additional keyword arguments passed to `matplotlib.axes.Axes.axhline`.
        `zorder` defaults to `config.zorder.hlines`.

    Returns
    -------
    list[matplotlib.lines.Line2D] | None
        Created line artists, in input order. None if `hlines` is empty.

    Raises
    ------
    ValueError
        If any element of `hlines` is not a 1D sequence of y coordinates.
    """
    return _axlines(
        *hlines,
        orientation='h',
        ax=ax,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        unit=unit,
        equivalencies=equivalencies,
        **kwargs,
    )


def _axlines(
    *coords: float | u.Quantity | Sequence[float | u.Quantity] | None,
    orientation: Literal['v', 'h'],
    ax: maxes.Axes | None = None,
    color: ColorType | _Unset = _UNSET,
    linestyle: LineStyleType | _Unset = _UNSET,
    linewidth: float | _Unset = _UNSET,
    alpha: float | _Unset = _UNSET,
    unit: UnitBase | StructuredUnit | None = None,
    equivalencies: Equivalency | None = None,
    **kwargs,
) -> list[Line2D] | None:
    """
    Shared implementation of `axvlines` and `axhlines`.

    Parameters
    ----------
    *coords : float | astropy.units.Quantity | Sequence[float | astropy.units.Quantity]
        Coordinate(s) along the axis perpendicular to the lines. See `axvlines`.
    orientation : {'v', 'h'}
        Line orientation. 'v' draws vertical lines (x coordinates) with
        `Axes.axvline`; 'h' draws horizontal lines (y coordinates) with
        `Axes.axhline`.
    ax, color, linestyle, linewidth, alpha, unit, equivalencies, **kwargs
        See `axvlines`.

    Returns
    -------
    list[matplotlib.lines.Line2D] | None
        Created line artists, in input order. None if `coords` is empty.

    Raises
    ------
    ValueError
        If `orientation` is invalid, or any element of `coords` is not a 1D
        sequence of coordinates.
    """
    if orientation == 'v':
        method, axis, name, zorder = 'axvline', 'x', 'vlines', config.zorder.vlines
    elif orientation == 'h':
        method, axis, name, zorder = 'axhline', 'y', 'hlines', config.zorder.hlines
    else:
        raise ValueError(f"orientation must be 'v' or 'h', got {orientation!r}")

    if len(coords) == 0:
        return
    params = _extract_kwargs(
        kwargs,
        params=[
            _param('color', color, config.axline.color),
            _param('linestyle', linestyle, config.axline.linestyle),
            _param('linewidth', linewidth, config.axline.linewidth),
            _param('alpha', alpha, config.axline.alpha),
        ],
        additional_kwargs=[
            _kwarg('zorder', zorder),
        ],
    )
    line_list = []
    ax = get_ax(ax)
    draw = getattr(ax, method)

    for lines in coords:
        if lines is None:
            continue

        if not _is_iterable(lines):
            lines = to_list(lines)

        if not _is_1d(lines):
            raise ValueError(
                f'{name} must be a 1D sequence of {axis} coordinates! '
                f"got {type(lines).__name__}"
            )

        for line in lines:
            if isinstance(line, u.Quantity):
                if unit is not None:
                    line = line.to(unit, equivalencies=equivalencies)
                line = line.value

            line_list.append(draw(line, **params, **kwargs))

    return line_list


def grid(
    *,
    ax: maxes.Axes | None = None,
    visible: bool | None = True,
    which: Literal['major', 'minor', 'both'] | _Unset = _UNSET,
    axis: Literal['both', 'x', 'y'] | _Unset = _UNSET,
    color: ColorType | _Unset = _UNSET,
    linestyle: LineStyleType | _Unset = _UNSET,
    linewidth: float | _Unset = _UNSET,
    alpha: float | _Unset = _UNSET,
    zorder: float | _Unset = _UNSET,
    **kwargs,
) -> None:
    """
    Add gridlines on a Matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes | None, optional, default=None
        Target axes. If None, the current axes are used.
    visible : bool | None, optional, default=True
        Whether to show the gridlines. If None, the visibility is toggled
        unless `kwargs` are given (Matplotlib behavior).
    which : {'major', 'minor', 'both'} | _Unset, optional, default=_UNSET
        Tick set to which the gridlines apply. If `_UNSET`, uses
        `config.gridlines.which`.
    axis : {'both', 'x', 'y'} | _Unset, optional, default=_UNSET
        Axis to which the gridlines apply. If `_UNSET`, uses
        `config.gridlines.axis`.
    color : ColorType, optional, default=config.gridlines.color
        Gridline color.
    linestyle : LineStyleType, optional, default=config.gridlines.linestyle
        Gridline style.
    linewidth : float, optional, default=config.gridlines.linewidth
        Gridline width in points.
    alpha : float, optional, default=config.gridlines.alpha
        Gridline opacity, in the range [0, 1].
    zorder : float, optional, default=config.zorder.gridlines
        Drawing order of the gridlines.
    **kwargs
        Additional keyword arguments forwarded to `matplotlib.axes.Axes.grid`
        (i.e. `matplotlib.lines.Line2D` properties).

    Returns
    -------
    None

    See Also
    --------
    matplotlib.axes.Axes.grid : Underlying Matplotlib implementation.
    wcsgrid : Equivalent function for WCS axes.
    """
    params = _extract_kwargs(
        kwargs,
        params=[
            _param('which', which, config.gridlines.which),
            _param('axis', axis, config.gridlines.axis),
            _param('color', color, config.gridlines.color),
            _param('linestyle', linestyle, config.gridlines.linestyle),
            _param('linewidth', linewidth, config.gridlines.linewidth),
            _param('alpha', alpha, config.gridlines.alpha),
            _param('zorder', zorder, config.zorder.gridlines),
        ]
    )
    kwargs.update(params)
    ax = get_ax(ax)
    ax.grid(
        visible=visible,
        **kwargs
    )


def wcsgrid(
    *,
    ax: WCSAxes | None = None,
    draw_grid: bool | None = True,
    grid_type: Literal['lines', 'contours'] | None = None,
    color: ColorType | _Unset = _UNSET,
    linestyle: LineStyleType | _Unset = _UNSET,
    linewidth: float | _Unset = _UNSET,
    alpha: float | _Unset = _UNSET,
    zorder: float | _Unset = _UNSET,
    **kwargs,
) -> None:
    """
    Add coordinate gridlines on an Astropy WCS axes.

    Parameters
    ----------
    ax : astropy.visualization.wcsaxes.WCSAxes | None, optional, default=None
        Target WCS axes. If None, the current WCS axes are used.
    draw_grid : bool | None, optional, default=True
        Whether to draw the gridlines. If None, treated as False.
    grid_type : {'lines', 'contours'} | None, optional, default=None
        Rendering method. 'lines' is faster; 'contours' is required for
        grids that cannot be drawn as straight/simple lines. If None, the
        Astropy default is used.
    color : ColorType, optional, default=config.wcsgrid.color
        Gridline color.
    linestyle : LineStyleType, optional, default=config.wcsgrid.linestyle
        Gridline style.
    linewidth : float, optional, default=config.wcsgrid.linewidth
        Gridline width in points.
    alpha : float, optional, default=config.wcsgrid.alpha
        Gridline opacity, in the range [0, 1].
    zorder : float, optional, default=config.zorder.wcsgrid
        Drawing order of the gridlines.
    **kwargs
        Additional keyword arguments forwarded to
        `astropy.visualization.wcsaxes.CoordinatesMap.grid`.

    Returns
    -------
    None

    See Also
    --------
    astropy.visualization.wcsaxes.CoordinatesMap.grid : Underlying Astropy implementation.
    grid : Equivalent function for Matplotlib axes.
    """
    params = _extract_kwargs(
        kwargs,
        params=[
            _param('grid_type', grid_type, config.wcsgrid.grid_type),
            _param('color', color, config.wcsgrid.color),
            _param('linestyle', linestyle, config.wcsgrid.linestyle),
            _param('linewidth', linewidth, config.wcsgrid.linewidth),
            _param('alpha', alpha, config.wcsgrid.alpha),
            _param('zorder', zorder, config.zorder.wcsgrid),
        ]
    )
    kwargs.update(params)
    wcsax = get_wcsax(ax)
    draw_grid = False if draw_grid is None else draw_grid
    wcsax.coords.grid(
        draw_grid=draw_grid,
        **kwargs,
    )
