"""
Author: Elko Gerville-Reache
Date Created: 2026-10-28
Date Modified: 2026-11-07
Description:
    Module for inset axes.
"""

from typing import Literal

import matplotlib.axes as maxes
from matplotlib.transforms import Transform
from matplotlib.typing import ColorType, LineStyleType

from visualastro.core.config import _Unset, _UNSET
from visualastro.core.kwargs import _extract_kwargs, _param
from visualastro.plotting.core.axes.axes import get_ax
from visualastro.plotting.core.axes.limits import set_limits


def inset_axes(
    bounds: tuple[float, float, float, float],
    ax: maxes.Axes | None = None,
    *,
    transform: Literal['axes', 'data'] | Transform | None = None,
    projection: str | None = None,
    indicate_zoom: bool = False,
    connectors: tuple[bool, bool, bool, bool] = (True, False, True, False),
    polar: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    limits: float | tuple[float, float] | None = None,
    edgecolor: ColorType = 'k',
    linestyle: LineStyleType | _Unset = _UNSET,
    linewidth: float | _Unset = _UNSET,
    alpha: float | _Unset = _UNSET,
    zorder: float = 5,
    **kwargs
):
    """
    Add an inset axes to an existing axes.

    The inset position and displayed data limits can be specified independently.
    Optionally, indicate the region displayed by the inset on the parent axes.

    Parameters
    ----------
    bounds : tuple[float, float, float, float]
        Four values `(x0, y0, width, height)` defining the inset position.
        The coordinate system is determined by `transform`.
    ax : matplotlib.axes.Axes, optional
        Parent axes. If `None`, uses `get_ax`.
    transform : {'axes', 'data'} | CompositeGenericTransform | None, optional, default=None
        Transform defining the coordinate system of `bounds`. Can be a
        Matplotlib Transform or one of:

        - `'data'` : parent axes data coordinates.
        - `'axes'` : parent axes coordinates (0–1).

        `None` defaults to `ax.transAxes`.
    projection : str | None, optional, default=None
        Projection of the inset axes. Supported projections include
        `'aitoff'`, `'hammer'`, `'lambert'`, `'mollweide'`, `'polar'`, and
        `'rectilinear'`.
    indicate_zoom : bool, optional, default=False
        Whether to indicate the region displayed by the inset on the parent
        axes using a rectangle and connecting lines.
    connectors : tuple[bool, bool, bool, bool], optional, default=(True, False, True, False)
        Visibility of the four connectors joining the zoom rectangle to the inset
        axes. The values correspond to the lower-left, upper-left, lower-right,
        and upper-right connectors, respectively.
    polar : bool, optional, default=False
        If `True`, equivalent to `projection='polar'`.
    xlim, ylim : tuple[float, float] | None, optional, default=None
        X and Y-axis limits, applied as given (inverted limits invert the axis).
        If only one is provided and `recompute_other_axis=True`, the limits of the other
        axis are computed from the data visible within the provided range.
        If both are provided, no computation is done.
    limits : float | tuple[float, float] | None, optional, default=None
        Limits applied to both axes. Cannot be combined with `xlim` or `ylim`.

        - If a scalar, sets `xlim = ylim = (-abs(limits), abs(limits))`.
        - If a `tuple[float, float]`, sets `xlim = ylim = limits` as is.

    edgecolor : ColorType, optional, default='k'
        Edge color of the zoom indication.
    linestyle : LineStyleType | _Unset, optional, default=_UNSET
        Line style of the zoom indication. If `_UNSET`, uses `None`.
    linewidth : float | _Unset, optional, default=_UNSET
        Line width of the zoom indication. If `_UNSET`, uses the
        inset axes spine linewidth.
    alpha : float | _Unset, optional, default=_UNSET
        Transparency of the zoom indication. If `_UNSET`, uses `0.5`.
    zorder : float, optional, default=5
        Drawing order of the inset axes.
    **kwargs
        Additional keyword arguments passed to `Axes.inset_axes()`.

    Returns
    -------
    matplotlib.axes.Axes
        The newly created inset axes.
    """
    params = _extract_kwargs(
        kwargs,
        params=[
            _param('edgecolor', edgecolor, 'k'),
            _param('linestyle', linestyle, None),
            _param('linewidth', linewidth, None),
            _param('alpha', alpha, 0.5),
        ]
    )
    params = {k:v for k,v in params.items() if v is not None}

    ax = get_ax(ax)

    try:
        bounds = tuple(bounds)
    except TypeError:
        bounds = ()
    if len(bounds) != 4:
        raise ValueError(
            "`bounds` must be (x0, y0, width, height)! "
            f"Got: {bounds}"
        )

    if isinstance(transform, str):
        transforms = {
            'data': ax.transData,
            'axes': ax.transAxes,
        }
        try:
            transform = transforms[transform]
        except KeyError:
            raise ValueError(
                f"`transform` must be one of {list(transforms)} "
                f"or a matplotlib Transform, got: {transform!r}"
            ) from None

    inset = ax.inset_axes(
        bounds,
        transform=transform,
        projection=projection,
        polar=polar,
        zorder=zorder,
        **kwargs
    )

    if any(v is not None for v in (xlim, ylim, limits)):
        set_limits(
            ax=inset,
            xlim=xlim,
            ylim=ylim,
            limits=limits,
            recompute_other_axis=False,
        )

    if indicate_zoom:
        if 'linewidth' not in params:
            params['linewidth'] = next(
                iter(inset.spines.values())
            ).get_linewidth()

        indicator = ax.indicate_inset_zoom(inset, **params)
        for connector, visible in zip(indicator.connectors, connectors):
            connector.set_visible(visible)

    return inset
