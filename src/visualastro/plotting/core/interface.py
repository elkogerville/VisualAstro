"""
Author: Elko Gerville-Reache
Date Created: 2026-05-26
Date Modified: 2026-07-09
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

from dataclasses import field, make_dataclass
from typing import Any

import astropy.units as u
from astropy.visualization.wcsaxes.core import WCSAxes
import matplotlib.axes as maxes

from visualastro.core.config import config
from visualastro.core.kwargs import _extract_kwargs, _kwarg
from visualastro.core.units import unit_2_string
from visualastro.plotting.core.axes import (
    set_axis_labels, set_axis_limits
)
from visualastro.plotting.core.colors import _has_color_mapping
from visualastro.plotting.core.utils import (
    add_colorbar,
    plot_ellipses,
    plot_hlines,
    plot_interactive_ellipse,
    plot_points,
    plot_vlines,
)
from visualastro.core.numerical_utils import _cycle


_PLOT_UTILS_KWARGS = [
    _kwarg('reference_idx', lambda: config.reference_idx),
    _kwarg('array_order', lambda: config.array_order),
    _kwarg('index_spec', lambda: config.index_specification),

    _kwarg('ellipses', lambda: None),
    _kwarg('plot_ellipse', lambda: False),
    _kwarg('highlight', lambda: config.highlight),
    _kwarg('text_loc', lambda: config.text_loc),
    _kwarg('text_color', lambda: config.text_color),

    _kwarg('points', lambda: None),

    _kwarg('vlines', lambda: None),
    _kwarg('hlines', lambda: None),

    _kwarg('legend_handles', lambda: config.legend.handles),
    _kwarg('legend_labels', lambda: config.legend.labels),
    _kwarg('legend_loc', lambda: config.legend.loc),
    _kwarg('legend_ncols', lambda: config.legend.ncols),
    _kwarg('legend_fontsize', lambda: config.legend.fontsize),
    _kwarg('legend_numpoints', lambda: config.legend.numpoints),
    _kwarg('legend_scatterpoints', lambda: config.legend.scatterpoints),
    _kwarg('legend_markerscale', lambda: config.legend.markerscale),
    _kwarg('legend_markerfirst', lambda: config.legend.markerfirst),
    _kwarg('legend_reverse', lambda: config.legend.reverse),
    _kwarg('legend_frameon', lambda: config.legend.frameon),
    _kwarg('legend_fancybox', lambda: config.legend.fancybox),
    _kwarg('legend_framealpha', lambda: config.legend.framealpha),
    _kwarg('legend_facecolor', lambda: config.legend.facecolor),
    _kwarg('legend_edgecolor', lambda: config.legend.edgecolor),
    _kwarg('legend_title', lambda: config.legend.title),
    _kwarg('legend_alignment', lambda: config.legend.alignment),
    _kwarg('legend_borderpad', lambda: config.legend.borderpad),
    _kwarg('legend_labelspacing', lambda: config.legend.labelspacing),
    _kwarg('legend_borderaxespad', lambda: config.legend.borderaxespad),
    _kwarg('legend_columnspacing', lambda: config.legend.columnspacing),
    _kwarg('legend_draggable', lambda: config.legend.draggable),

    _kwarg('xlabel', lambda: None),
    _kwarg('ylabel', lambda: None),
    _kwarg('unit_bracket_style', lambda: config.unit_bracket_style),
    _kwarg('show_physical_type', lambda: config.show_type_label),
    _kwarg('show_unit', lambda: config.show_unit_label),
    _kwarg('unit_fmt', lambda: config.unit_label_format),

    _kwarg('compute_limits', lambda: config.axes.compute_limits),
    _kwarg('scale', lambda: None),
    _kwarg('xlim', lambda: None),
    _kwarg('ylim', lambda: None),
    _kwarg('xpad', lambda: config.axes.xpad),
    _kwarg('ypad', lambda: config.axes.ypad),

    _kwarg('wcs_grid', lambda: config.wcs_grid),
    _kwarg('wcs_grid_color', lambda: config.wcs_grid_color),
    _kwarg('wcs_grid_linestyle', lambda: config.wcs_grid_linestyle),
    _kwarg('wcs_grid_linewidth', lambda: config.wcs_grid_linewidth),
    _kwarg('wcs_grid_alpha', lambda: config.wcs_grid_alpha),

    _kwarg('gridlines', lambda: config.gridlines),
    _kwarg('grid_which', lambda: config.grid_which),
    _kwarg('grid_color', lambda: config.grid_color),
    _kwarg('grid_linestyle', lambda: config.grid_linestyle),
    _kwarg('grid_linewidth', lambda: config.grid_linewidth),
    _kwarg('grid_alpha', lambda: config.grid_alpha),

    _kwarg('colorbar', lambda: config.colorbar.enable),
    _kwarg('cbar_width', lambda: config.colorbar.width),
    _kwarg('cbar_pad', lambda: config.colorbar.pad),
    _kwarg('cbar_label', lambda: config.colorbar.label),
    _kwarg('cbar_tick_which', lambda: config.colorbar.tick_which),
    _kwarg('cbar_tick_dir', lambda: config.colorbar.tick_dir),
]

PlotUtilParams = make_dataclass(
    'PlotUtilParams',
    [(kw[0], Any, field(default=None)) for kw in _PLOT_UTILS_KWARGS],
    slots=True
)


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
    # kw[1]() extracts the config from the lambda at runtime
    params = _extract_kwargs(
        kwargs,
        additional_kwargs=[
            _kwarg(
                kw[0],
                kw[1]() if callable(kw[1]) else kw[1]
            )
            for kw in _PLOT_UTILS_KWARGS
        ]
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
    compute_limits : bool, optional, default=config.axes.compute_limits
        If `False`, does not compute any limits based on data,
        and lets matplotlib decide axes limits.
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
    # PRE SETTING AXIS LIMITS
    # -----------------------
    if 'labels' in kwargs:
        if _cycle(kwargs['labels'], params.reference_idx) is not None:
            legend_kwargs = {
                'loc': params.legend_loc,
                'ncols': params.legend_ncols,
                'fontsize': params.legend_fontsize,
                'numpoints': params.legend_numpoints,
                'scatterpoints': params.legend_scatterpoints,
                'markerscale': params.legend_markerscale,
                'markerfirst': params.legend_markerfirst,
                'reverse': params.legend_reverse,
                'frameon': params.legend_frameon,
                'fancybox': params.legend_fancybox,
                'framealpha': params.legend_framealpha,
                'facecolor': params.legend_facecolor,
                'edgecolor': params.legend_edgecolor,
                'title': params.legend_title,
                'alignment': params.legend_alignment,
                'borderpad': params.legend_borderpad,
                'labelspacing': params.legend_labelspacing,
                'borderaxespad': params.legend_borderaxespad,
                'columnspacing': params.legend_columnspacing,
                'draggable': params.legend_draggable,
            }

            if params.legend_handles is not None:
                legend_kwargs['handles'] = params.legend_handles
            if params.legend_labels is not None:
                legend_kwargs['labels'] = params.legend_labels

            ax.legend(**legend_kwargs)

    plot_ellipses(params.ellipses, ax)
    plot_points(
        params.points,
        ax=ax,
        order=params.array_order,
        index_spec=params.index_spec
    )
    if params.compute_limits:
        set_axis_limits(
            xlist, ylist,
            ax=ax,
            scale=params.scale,
            xlim=params.xlim, ylim=params.ylim,
            xpad=params.xpad, ypad=params.ypad
        )

    # POST SETTING AXIS LIMITS
    # ------------------------
    if isinstance(ax, WCSAxes):
        xlabel = params.xlabel if params.xlabel is not None else config.right_ascension_label
        ylabel = params.ylabel if params.ylabel is not None else config.declination_label
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
