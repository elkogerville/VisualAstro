"""
Author: Elko Gerville-Reache
Date Created: 2026-05-26
Date Modified: 2026-10-05
Description:
    Interface for plotting functions. Handles kwargs and
    automatically adds functionality such as colorbar creation,
    plotting patches, etc...

    Any utility defined in `_apply_plot_utils` does not need to be
    manually added to a plotting function once the plotting interface
    is added to a visualastro plotting function.

To add a feature:
    - Add a attribute to `PlotUtilParams`
    - Register all relavent kwargs in `_extract_plot_util_kwargs`
    - Add feature to `_apply_plot_utils`

Examples
--------
To add the plotting interface to a visualastro plotting function:

    def plotting_function(X, Y, ax, **kwargs) -> None:
        plot_params = _extract_plot_util_kwargs(kwargs) -> removes any kwarg defined here from kwargs

        ** plotting logic **
        ie : ax.plot(X, Y, **kwargs)

        _apply_plot_utils(plot_params, ax=ax) -> applies the plotting utils to the plot
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import astropy.units as u
from astropy.visualization.wcsaxes.core import WCSAxes
import matplotlib.axes as maxes

from visualastro.core.config import _UNSET, config
from visualastro.core.kwargs import _pop_kwargs, _pop_prefixed, _pop_mapped
from visualastro.core.numerical_utils import _cycle
from visualastro.core.units import unit_2_string
from visualastro.plotting.core.axes import (
    set_axis_labels, set_axis_limits, set_title
)
from visualastro.plotting.core.colorbar import add_colorbar
from visualastro.plotting.core.colors import _has_color_mapping
from visualastro.plotting.core.legend import legend
from visualastro.plotting.core.utils import (
    axhline,
    axvline,
    plot_ellipses,
    plot_interactive_ellipse,
    plot_points,
)


@dataclass(slots=True)
class PlotUtilParams:
    reference_idx: int
    array_order: Literal['C', 'c', 'F', 'f', 'fortran']
    index_spec: tuple[int, int] | Literal['implicit', 'explicit']
    vlines: Sequence[float] | None
    hlines: Sequence[float] | None
    ticklabel_fontsize: float
    compute_limits: bool

    legend: dict
    title: dict
    limits: dict
    labels: dict
    gridlines: dict
    wcs_grid: dict
    colorbar: dict
    text: dict
    ellipses: dict
    points: dict


def _extract_plot_util_kwargs(kwargs: dict) -> PlotUtilParams:

    return PlotUtilParams(
        reference_idx=_pop_kwargs(kwargs, 'reference_idx', config.reference_idx),
        array_order=_pop_kwargs(kwargs, 'array_order', config.array_order),
        index_spec=_pop_kwargs(kwargs, 'index_spec', config.index_specification),
        vlines=_pop_kwargs(kwargs, 'vlines', None),
        hlines=_pop_kwargs(kwargs, 'hlines', None),
        ticklabel_fontsize=config.fontsizes.resolve('tick_labels'),
        compute_limits=_pop_kwargs(
            kwargs, 'compute_limits', config.axes.compute_limits
        ),

        legend=_pop_prefixed(kwargs, 'legend_', base='legend', default=True),
        title=_pop_prefixed(kwargs, 'title_', base='title', default=False),
        limits=_pop_mapped(kwargs,
            ('limits', 'xlim', 'ylim', 'xpad', 'ypad')
        ),
        labels=_pop_mapped(
            kwargs,
            (
                'xlabel', 'ylabel',
                'unit_bracket_style',
                'show_physical_type',
                'show_unit',
                'unit_fmt'
            ),
            prefix='label_',
            base='label_fontsize',
            default=config.fontsizes.resolve('axes_labels')
        ),
        wcs_grid=_pop_prefixed(
            kwargs, 'wcs_grid_', base='wcs_grid', default=config.wcs_grid
        ),
        gridlines=_pop_prefixed(
            kwargs, 'grid_', base='gridlines', default=config.gridlines
        ),
        colorbar=_pop_prefixed(
            kwargs, 'cbar_', base='colorbar', default=config.colorbar.enable
        ),
        text=_pop_mapped(kwargs, ('highlight',), prefix='text_'),
        ellipses=_pop_mapped(kwargs, ('ellipses', 'plot_ellipse')),
        points=_pop_mapped(kwargs, ('points',)),
    )


def _apply_plot_utils(
    params: PlotUtilParams,
    ax: maxes.Axes | WCSAxes,
    xlist: list | None = None,
    ylist: list | None = None,
    im_list: list | None = None,
    ref_unit: u.UnitBase | u.StructuredUnit | None = None,
    **kwargs
) -> None:
    # PRE SETTING AXIS LIMITS
    # -----------------------
    if 'labels' in kwargs and params.legend.pop('legend'):
        if _cycle(kwargs['labels'], params.reference_idx) is not None:
            legend(ax=ax, **params.legend)

    plot_ellipses(params.ellipses.pop('ellipses', None), ax)
    plot_points(
        params.points.pop('points', None),
        ax=ax,
        order=params.array_order,
        index_spec=params.index_spec
    )

    if params.compute_limits:
        set_axis_limits(
            xlist, ylist,
            ax=ax,
            **params.limits
        )

    # POST SETTING AXIS LIMITS
    # ------------------------
    ax.tick_params(
        axis='both',
        which='major',
        labelsize=params.ticklabel_fontsize,
    )

    if title := params.title.pop('title', None):
        set_title(
            title,
            ax=ax,
            **params.title
        )

    labels = params.labels

    label_fontsize = labels.pop('label_fontsize')
    if isinstance(ax, WCSAxes):
        xlabel = labels.get('xlabel')
        if xlabel is None:
            xlabel = config.right_ascension_label
        ylabel = labels.get('ylabel')
        if ylabel is None:
            ylabel = config.declination_label
        ax.coords['ra'].set_axislabel(xlabel, fontsize=label_fontsize)
        ax.coords['dec'].set_axislabel(ylabel, fontsize=label_fontsize)
        ax.coords['dec'].set_ticklabel(rotation=90)

    else:
        set_axis_labels(
            _cycle(xlist, params.reference_idx) if xlist is not None else None,
            _cycle(ylist, params.reference_idx) if ylist is not None else None,
            ax=ax,
            fontsize=label_fontsize,
            **labels,
        )

    if params.wcs_grid.pop('wcs_grid', None) and isinstance(ax, WCSAxes):
        ax.coords.grid(
            True,
            zorder=config.zorder.wcs_grid,
            **params.wcs_grid,
        )

    if params.gridlines.pop('gridlines', None):
        ax.grid(
            True,
            zorder=config.zorder.gridlines,
            **params.gridlines,
        )

    if params.colorbar.pop('colorbar', None):
        if im_list is not None and _has_color_mapping(_cycle(im_list, params.reference_idx)):
            cparams = params.colorbar
            cbar_unit = unit_2_string(ref_unit, fmt=config.unit_label_format)
            clabel = (
                cparams['cbar_label'] if isinstance(cparams['cbar_label'], str) else
                cbar_unit if cparams['cbar_label'] else None
            )
            add_colorbar(
                _cycle(im_list, params.reference_idx),
                ax=ax,
                label=clabel,
                rasterized=kwargs.pop('rasterized', _UNSET),
                **cparams
            )

    axvline(params.vlines, ax, ref_unit)
    axhline(params.hlines, ax, ref_unit)

    if params.ellipses.pop('plot_ellipse', None) and im_list is not None:
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
                center, w, h, ax, params.text.get('loc'),
                params.text.get('color'), params.text.get('highlight'),
                rotation_step=kwargs.get('rotation_step', 5)
            )
