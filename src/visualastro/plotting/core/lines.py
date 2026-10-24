"""
Author: Elko Gerville-Reache
Date Created: 2025-05-24
Date Modified: 2026-10-06
Description:
    Hlines and vlines utility functions.
"""

from collections.abc import Sequence
from typing import Literal

import astropy.units as u
from astropy.units import Equivalency, StructuredUnit, UnitBase
import matplotlib.axes as maxes
from matplotlib.lines import Line2D
from matplotlib.typing import ColorType, LineStyleType

from visualastro.core.array_types import _is_1d, _is_iterable
from visualastro.core.config import config, _Unset, _UNSET
from visualastro.core.data import to_list
from visualastro.core.kwargs import _extract_kwargs, _kwarg, _param
from visualastro.plotting.core.axes import get_ax


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
