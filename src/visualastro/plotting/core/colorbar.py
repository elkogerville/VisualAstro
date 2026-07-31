"""
Author: Elko Gerville-Reache
Date Created: 2025-07-31
Date Modified: 2026-07-31
Description:
    Colorbar plotting functions.
"""

import matplotlib.axes as maxes
from matplotlib.cm import ScalarMappable

from visualastro.core.config import (
    config, _Unset, _UNSET, _resolve_default
)


def add_colorbar(
    im: ScalarMappable,
    ax: maxes.Axes,
    cbar_width: float | _Unset = _UNSET,
    cbar_pad: float | _Unset = _UNSET,
    label: str | None = None,
    fontsize: float | _Unset = _UNSET,
    tick_which=_UNSET,
    tick_dir=_UNSET,
    rasterized=_UNSET
) -> None:
    """
    Add a colorbar next to an Axes.

    Parameters
    ----------
    im : matplotlib.cm.ScalarMappable
        The image, contour set, or mappable object returned by
        a plotting function (e.g., 'imshow', 'scatter', etc...).
    ax : matplotlib.axes.Axes
        The axes to which the colorbar will be attached.
    cbar_width : float | _Unset, optional, default=_UNSET
        Width of the colorbar in figure coordinates.
        If `_UNSET`, uses `config.colorbar.width`.
    cbar_pad : float | _Unset, optional, default=_UNSET
        Padding between the main axes and the colorbar
        in figure coordinates. If `_UNSET`, uses `config.colorbar.pad`.
    label : str, optional, default=None
        Label for the colorbar. If `None`, no label is set.
    fontsize : float | _Unset, optional, default=_UNSET
        Fontsize for colorbar label. If `_UNSET`, uses
        `config.fontsizes.colorbar_label`.
    tick_which :  {'major', 'minor', 'both'} | _Unset, optional, default=_UNSET
        The group of ticks to which the parameters are applied.
    tick_dir : {'in', 'out', 'inout'} | _Unset, optional, default=_UNSET
        Puts ticks inside the Axes, outside the Axes, or both.
    rasterized : bool | _Unset, default=_UNSET
        Whether to rasterize colorbar. Rasterization
        converts the artist to a bitmap when saving to
        vector formats (e.g., PDF, SVG), which can
        significantly reduce file size for complex plots.
        If `_UNSET`, uses `config.rasterized`
    """
    cbar_width = _resolve_default(cbar_width, config.colorbar.width)
    cbar_pad = _resolve_default(cbar_pad, config.colorbar.pad)
    fontsize = _resolve_default(
        fontsize, config.fontsizes.resolve('colorbar_label')
    )
    tick_which = _resolve_default(tick_which, config.colorbar.tick_which)
    tick_dir = _resolve_default(tick_dir, config.colorbar.tick_dir)
    rasterized = _resolve_default(rasterized, config.rasterized)

    fig = ax.figure
    cax = fig.add_axes(
        [
            ax.get_position().x1+cbar_pad, ax.get_position().y0,
            cbar_width, ax.get_position().height
        ]
    )

    cbar = fig.colorbar(im, cax=cax, pad=0.04)
    cbar.ax.tick_params(which=tick_which, direction=tick_dir)
    if label:
        cbar.set_label(fr'{label}', fontsize=fontsize)

    if rasterized:
        cbar.solids.set_rasterized(True)
