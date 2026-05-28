"""
Author: Elko Gerville-Reache
Date Created: 2026-05-26
Date Modified: 2026-05-28
Description:
    Interface for plotting functions. Handles kwargs and
    automatically adds functionality such as colorbar creation,
    plotting patches, etc...

    Any utility defined in `_apply_plot_utils` does not need to be
    manually added to a plotting function once the plotting interface
    is added to a visualastro plotting function.

Examples
--------
To add the plotting interface to a visualastro plotting function:

    def plotting_function(X, Y, ax, **kwargs) -> None:
        plot_params = _extract_plot_util_kwargs(kwargs) -> removes any kwarg defined here from kwargs

        ** plotting logic **
        ie : ax.plot(X, Y, **kwargs)

        _apply_plot_utils(plot_params, ax=ax) -> applies the plotting utils to the plot

The interface is designed to be flexible and can be applied to any generic plotting function.
"""

from dataclasses import dataclass
from typing import Any

import astropy.units as u
from astropy.visualization.wcsaxes.core import WCSAxes
import matplotlib.axes as maxes
from matplotlib.cm import ScalarMappable

from visualastro.core.config import config
from visualastro.core.io import _extract_kwargs, _kwarg
from visualastro.core.units import unit_2_string
from visualastro.plotting.core.plot_utils import (
    add_colorbar,
    plot_ellipses,
    plot_hlines,
    plot_interactive_ellipse,
    plot_vlines,
    set_axis_labels,
    set_axis_limits
)
from visualastro.core.numerical_utils import _cycle


@dataclass(slots=True)
class PlotUtilParams:
    """
    Each parameter defined here should also be defined in
    `visualastro.plotting.core.interface._extract_plot_util_kwargs`.
    """
    reference_idx: Any = None
    ellipses: Any = None
    plot_ellipse: Any = None
    highlight: Any = None
    text_loc: Any = None
    text_color: Any = None
    vlines: Any = None
    hlines: Any = None
    label: Any = None
    legend_handles: Any = None
    legend_labels: Any = None
    legend_loc: Any = None
    legend_ncols: Any = None
    legend_fontsize: Any = None
    legend_fancybox: Any = None
    legend_framealpha: Any = None
    legend_facecolor: Any = None
    legend_edgecolor: Any = None
    legend_title: Any = None
    legend_alignment: Any = None
    legend_columnspacing: Any = None
    legend_draggable: Any = None
    xlabel: Any = None
    ylabel: Any = None
    unit_bracket_style: Any = None
    show_physical_type: Any = None
    show_unit: Any = None
    unit_fmt: Any = None
    xlim: Any = None
    ylim: Any = None
    xpad: Any = None
    ypad: Any = None
    wcs_grid: Any = None
    wcs_grid_color: Any = None
    wcs_grid_linestyle: Any = None
    wcs_grid_linewidth: Any = None
    wcs_grid_alpha: Any = None
    gridlines: Any = None
    grid_which: Any = None
    grid_color: Any = None
    grid_linestyle: Any = None
    grid_linewidth: Any = None
    grid_alpha: Any = None
    colorbar: Any = None
    cbar_width: Any = None
    cbar_pad: Any = None
    cbar_label: Any = None
    cbar_tick_which: Any = None
    cbar_tick_dir: Any = None


def _extract_plot_util_kwargs(kwargs) -> PlotUtilParams:
    """
    Extracts any keyword argument from a function related to
    visualastro plotting utilities. This way, kwargs can then
    be passed into a matplotlib function without any contamination
    from visualastro specific keyword arguments.

    Any kwarg defined here should also be defined in
    `visualastro.plotting.core.interface.PlotUtilParams`.

    Notes
    -----
    Only `additional_kwargs` and potentially (but probably not) `copy_kwargs`
    should be defined in `_extract_kwargs`. `copy_kwargs` should only be
    defined if a parameter is required both by `_apply_plot_utils` and by the
    matplotlib function the kwargs are forwarded to. In practice this should
    almost never happen because this would require being True for all plotting
    function that use the plotting interface.
    """
    params = _extract_kwargs(
        kwargs,
        additional_kwargs=[
            _kwarg('reference_idx', config.reference_idx),

            _kwarg('ellipses', None),
            _kwarg('plot_ellipse', False),
            _kwarg('highlight', config.highlight),
            _kwarg('text_loc', config.text_loc),
            _kwarg('text_color', config.text_color),

            _kwarg('vlines', None),
            _kwarg('hlines', None),

            _kwarg('legend_handles', config.legend.handles),
            _kwarg('legend_labels', config.legend.labels),
            _kwarg('legend_loc', config.legend.loc),
            _kwarg('legend_ncols', config.legend.ncols),
            _kwarg('legend_fontsize', config.legend.fontsize),
            _kwarg('legend_fancybox', config.legend.fancybox),
            _kwarg('legend_framealpha', config.legend.framealpha),
            _kwarg('legend_facecolor', config.legend.facecolor),
            _kwarg('legend_edgecolor', config.legend.edgecolor),
            _kwarg('legend_title', config.legend.title),
            _kwarg('legend_alignment', config.legend.alignment),
            _kwarg('legend_columnspacing', config.legend.columnspacing),
            _kwarg('legend_draggable', config.legend.draggable),

            _kwarg('xlabel', None),
            _kwarg('ylabel', None),
            _kwarg('unit_bracket_style', config.unit_bracket_style),
            _kwarg('show_physical_type', config.show_type_label),
            _kwarg('show_unit', config.show_unit_label),
            _kwarg('unit_fmt', config.unit_label_format),

            _kwarg('xlim', None),
            _kwarg('ylim', None),
            _kwarg('xpad', config.axes.xpad),
            _kwarg('ypad', config.axes.ypad),

            _kwarg('wcs_grid', config.wcs_grid),
            _kwarg('wcs_grid_color', config.wcs_grid_color),
            _kwarg('wcs_grid_linestyle', config.wcs_grid_linestyle),
            _kwarg('wcs_grid_linewidth', config.wcs_grid_linewidth),
            _kwarg('wcs_grid_alpha', config.wcs_grid_alpha),

            _kwarg('gridlines', config.gridlines),
            _kwarg('grid_which', config.grid_which),
            _kwarg('grid_color', config.grid_color),
            _kwarg('grid_linestyle', config.grid_linestyle),
            _kwarg('grid_linewidth', config.grid_linewidth),
            _kwarg('grid_alpha', config.grid_alpha),

            _kwarg('colorbar', config.colorbar.enable),
            _kwarg('cbar_width', config.colorbar.width),
            _kwarg('cbar_pad', config.colorbar.pad),
            _kwarg('cbar_label', config.colorbar.label),
            _kwarg('cbar_tick_which', config.colorbar.tick_which),
            _kwarg('cbar_tick_dir', config.colorbar.tick_dir),
        ],
    )

    return PlotUtilParams(**params)


def _apply_plot_utils(
    params: PlotUtilParams,
    ax: maxes.Axes | WCSAxes,
    xlist: list | None = None,
    ylist: list | None = None,
    im_list: list | None = None,
    ref_unit: u.UnitBase | u.StructuredUnit | None = None,
    **kwargs
) -> None:
    """
    Plotting interface for adding figure annotations and artists to a figure.

    To add the interface to a plotting function, first call
    `visualastro.plotting.core.interface_extract_plot_util_kwargs`,
    which will return a `PlotUtilParams` instance. Then call
    `visualastro.plotting.core.interface_apply_plot_utils`
    after the core plotting has been completed.

    Parameters
    ----------
    params : PlotUtilParams
        Dataclass containing all the values necessary for the plotting utlity functions.
        The paramters are defined in `_extract_plot_util_kwargs` and `PlotUtilParams`.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    ref_unit : u.UnitBase | u.StructuredUnit | None, optional, default=None
        Reference unit for any required unit related logic. Should probably be either `None`
        or the unit of the data being plotted.

    Attributes
    ----------
    ellipses : matplotlib.patches.Ellipse | list[matplotlib.patches.Ellipse]
        The Ellipse or list of Ellipses to plot.
    vlines : float | Quantity | Sequence[float | Quantity] | None, optional, default=None
        X-axis coordinate(s) at which to draw vertical line(s). If a `Quantity`,
        each value is converted to `ref_unit` before plotting. If an iterable is
        provided, a horizontal line is drawn for each element. If `None`, no lines
        are drawn.
    hlines : float | Quantity | Sequence[float | Quantity] | None, optional, default=None
        Y-axis coordinate(s) at which to draw horizontal line(s). If a `Quantity`,
        each value is converted to `ref_unit` before plotting. If an iterable is
        provided, a horizontal line is drawn for each element. If `None`, no lines
        are drawn.
    """
    if 'labels' in kwargs:
        if _cycle(kwargs['labels'], params.reference_idx) is not None:
            legend_kwargs = {
                'loc': params.legend_loc,
                'ncols': params.legend_ncols,
                'fontsize': params.legend_fontsize,
                'fancybox': params.legend_fancybox,
                'framealpha': params.legend_framealpha,
                'facecolor': params.legend_facecolor,
                'edgecolor': params.legend_edgecolor,
                'title': params.legend_title,
                'alignment': params.legend_alignment,
                'columnspacing': params.legend_columnspacing,
                'draggable': params.legend_draggable,
            }

            if params.legend_handles is not None:
                legend_kwargs['handles'] = params.legend_handles
            if params.legend_labels is not None:
                legend_kwargs['labels'] = params.legend_labels

            ax.legend(**legend_kwargs)

    set_axis_limits(
        xlist,
        ylist,
        ax=ax,
        xlim=params.xlim,
        ylim=params.ylim,
        xpad=params.xpad,
        ypad=params.ypad
    )

    if isinstance(ax, WCSAxes):
        xlabel = params.xlabel if params.xlabel is not None else config.right_ascension
        ylabel = params.ylabel if params.ylabel is not None else config.declination
        ax.coords['ra'].set_axislabel(xlabel)
        ax.coords['dec'].set_axislabel(ylabel)
        ax.coords['dec'].set_ticklabel(rotation=90)

    else:
        set_axis_labels(
            _cycle(xlist, params.reference_idx) if xlist is not None else None,
            _cycle(ylist, params.reference_idx) if ylist is not None else None,
            ax=ax,
            xlabel=params.xlabel,
            ylabel=params.ylabel,
            unit_bracket_style=params.unit_bracket_style,
            show_physical_type=params.show_physical_type,
            show_unit=params.show_unit,
            fmt=params.unit_fmt
        )

    if params.wcs_grid and isinstance(ax, WCSAxes):
        ax.coords.grid(
            True,
            color=params.wcs_grid_color,
            ls=params.wcs_grid_linestyle,
            lw=params.wcs_grid_linewidth,
            alpha=params.wcs_grid_alpha,
            zorder=config.zorder.wcs_grid,
        )

    if params.gridlines:
        ax.grid(
            True,
            which=params.grid_which,
            color=params.grid_color,
            ls=params.grid_linestyle,
            lw=params.grid_linewidth,
            alpha=params.grid_alpha,
            zorder=config.zorder.gridlines,
        )

    if params.colorbar:
        if im_list is not None and _has_color_mapping(_cycle(im_list, params.reference_idx)):
            cbar_unit = unit_2_string(ref_unit, fmt=config.unit_label_format)
            clabel = (
                params.cbar_label if isinstance(params.cbar_label, str) else
                cbar_unit if params.cbar_label else None
            )
            add_colorbar(
                _cycle(im_list, params.reference_idx),
                ax=ax,
                cbar_width=params.cbar_width,
                cbar_pad=params.cbar_pad,
                label=clabel,
                tick_which=params.cbar_tick_which,
                tick_dir=params.cbar_tick_dir,
                rasterized=kwargs.get('rasterized', None)
            )

    plot_ellipses(params.ellipses, ax)
    plot_vlines(params.vlines, ax, ref_unit)
    plot_hlines(params.hlines, ax, ref_unit)

    if params.plot_ellipse and im_list is not None:
        im = _cycle(im_list, params.reference_idx)
        data = im.get_array()
        if data is not None:
            if data.ndim == 2:
                X, Y = data.shape
            else:
                X, Y = data.shape[-2:]
            center = X//2, Y//2
            w = X//5
            h = Y//5
            plot_interactive_ellipse(
                center, w, h, ax, params.text_loc,
                params.text_color, params.highlight,
                rotation_step=kwargs.get('rotation_step', 5)
            )


def _has_color_mapping(mappable: ScalarMappable) -> bool:
    """Check that a ScalarMappable instance has valid data for a colormap"""
    return (
        mappable is not None
        and isinstance(mappable, ScalarMappable)
        and mappable.get_array() is not None
    )
