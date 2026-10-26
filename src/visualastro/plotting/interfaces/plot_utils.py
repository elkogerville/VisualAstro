"""
Author: Elko Gerville-Reache
Date Created: 2026-05-26
Date Modified: 2026-10-25
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
from matplotlib.image import AxesImage
from numpy.typing import ArrayLike

from visualastro.core.config import _UNSET, config
from visualastro.core.kwargs import _pop_kwargs, _pop_prefixed, _pop_mapped
from visualastro.core.units import unit_2_string
from visualastro.plotting.core.axes import (
    set_axis_limits, set_title, _set_axis_labels
)
from visualastro.plotting.core.colorbar import add_colorbar
from visualastro.plotting.core.colors.plots import _has_color_mapping
from visualastro.plotting.core.legend import legend
from visualastro.plotting.core.lines import axhlines, axvlines, grid, wcs_grid
from visualastro.plotting.core.utils import (
    plot_ellipses,
    plot_interactive_ellipse,
    plot_points,
    _default_ellipse_geometry
)


@dataclass(slots=True)
class PlotUtilParams:
    """
    Plotting kwargs sent to `_apply_plot_utils`.

    Attributes
    ----------
    reference_idx : int
        Index of the artist/data/label used for the legend check, axis
        labels, colorbar, and interactive ellipse. Applied via `_cycle`.
    array_order : {'C', 'c', 'F', 'f', 'fortran'}
        Memory order forwarded to `plot_points`.
    index_spec : tuple[int, int] | {'implicit', 'explicit'}
        Index convention forwarded to `plot_points`.
    vlines : Sequence[float] | None
        Vertical line positions. Forwarded to `axvlines` with `ref_unit`.
    hlines : Sequence[float] | None
        Horizontal line positions. Forwarded to `axhline` with `ref_unit`.
    compute_limits : bool
        If False, `set_axis_limits` is skipped.
    legend : dict[str, Any]
        - `legend` : bool | None
            Draw the legend if truthy and `kwargs['labels']` is given.
        - remaining keys : forwarded to `legend`.

    title : dict[str, Any]
        - `title` : str | bool | None
            Title text. Falsy skips the title.
        - remaining keys : forwarded to `set_title`.

    limits : dict[str, Any]
        Forwarded to `set_axis_limits`. Used only if `compute_limits`.
    labels : dict[str, Any]
        - `label_fontsize` : float
            Axis label size.
        - `xlabel`, `ylabel` : str | None
            Override labels. For `WCSAxes`, None falls back to
            `config.right_ascension_label` / `config.declination_label`.
        - remaining keys : forwarded to `_set_axis_labels` (non-WCS only).

    gridlines : dict[str, Any]
        - `gridlines` : bool
            Enable `ax.grid`.
        - remaining keys : forwarded to `ax.grid`.

    wcs_grid : dict[str, Any]
        - `wcs_grid` : bool
            Enable `ax.coords.grid`. Applied for `WCSAxes` only.
        - remaining keys : forwarded to `ax.coords.grid`.

    colorbar : dict[str, Any]
        - `colorbar` : bool
            Add a colorbar.
        - `cbar_label` : str | bool | None
            `str`: used verbatim. `bool`: formatted `ref_unit`.
            Falsy: no label. Required when the colorbar is drawn.
        - remaining keys : forwarded to `add_colorbar`.

    text : dict[str, Any]
        Matplotlib `Text` attrbitures.
    ellipses : dict[str, Any]
        - `ellipses` : Any | None
            Ellipse specification forwarded to `plot_ellipses`.
        - `plot_ellipse` : bool
            Draw the interactive ellipse.

    points : dict[str, Any]
        - `points` : Any | None
            Point specification forwarded to `plot_points`.
    """
    reference_idx: int
    array_order: Literal['C', 'c', 'F', 'f', 'fortran']
    xy_indices: tuple[int, int] | None
    vlines: Sequence[float] | None
    hlines: Sequence[float] | None
    compute_limits: bool
    fontsize: float

    legend: dict
    title: dict
    limits: dict
    labels: dict
    tick: dict
    gridlines: dict
    wcs_grid: dict
    colorbar: dict
    text: dict
    ellipses: dict
    points: dict


def _extract_plot_util_kwargs(kwargs: dict) -> PlotUtilParams:
    """
    Extracts any keyword argument from a function related to
    visualastro plotting utilities. This way, kwargs can then
    be passed into a matplotlib function without any contamination
    from visualastro specific keyword arguments.
    """
    params = PlotUtilParams(
        reference_idx=_pop_kwargs(kwargs, 'reference_idx', config.reference_idx),
        array_order=_pop_kwargs(kwargs, 'array_order', config.array_order),
        xy_indices=_pop_kwargs(kwargs, 'xy_indices', config.xy_indices),
        vlines=_pop_kwargs(kwargs, 'vlines', None),
        hlines=_pop_kwargs(kwargs, 'hlines', None),
        compute_limits=_pop_kwargs(
            kwargs, 'compute_limits', config.axes.compute_limits
        ),
        fontsize=_pop_kwargs(kwargs, 'fontsize', config.fontsizes.size),

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
        ),
        tick=_pop_prefixed(kwargs, 'tick_'),
        wcs_grid=_pop_prefixed(
            kwargs, 'wcs_grid_', base='wcs_grid', default=config.wcs_grid.enable
        ),
        gridlines=_pop_prefixed(
            kwargs, 'grid_', base='gridlines', default=config.gridlines.enable
        ),
        colorbar=_pop_prefixed(
            kwargs, 'cbar_', base='colorbar', default=config.colorbar.enable
        ),
        text=_pop_mapped(kwargs, ('highlight',), prefix='text_'),
        ellipses=_pop_mapped(kwargs, ('ellipses', 'plot_ellipse')),
        points=_pop_mapped(kwargs, ('points',)),
    )

    _FONTSIZE_KEYS = [
        # (params attr, config key, field in params attr)
        ('labels', 'axes_labels', 'fontsize'),
        ('tick', 'tick_labels', 'fontsize'),
        ('title', 'title', 'fontsize'),
        ('legend', 'legend', 'fontsize'),
        ('legend', 'legend_title', 'title_fontsize'),
        ('text', 'text', 'fontsize'),
        ('colorbar', 'colorbar_label', 'fontsize'),
        ('colorbar', 'colorbar_tick_labels', 'tick_fontsize'),
    ]

    for attr, key, field in _FONTSIZE_KEYS:
        group = getattr(params, attr)
        if group.get(field) is None:
            group[field] = config.fontsizes.with_size(key, params.fontsize)

    return params


def _apply_plot_utils(
    params: PlotUtilParams,
    ax: maxes.Axes | WCSAxes,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    img: AxesImage | None = None,
    ref_unit: u.UnitBase | u.StructuredUnit | None = None,
    **kwargs
) -> None:
    """
    Apply plot utility functions to an `Axes`.

    Parameters
    ----------
    params : PlotUtilParams
        Container of grouped plotting options,
        returned by `_extract_plot_util_kwargs`.
    ax : matplotlib.axes.Axes | astropy.visualization.wcsaxes.WCSAxes
        Axes to decorate. `WCSAxes` triggers WCS-specific label, tick, and
        grid handling.
    x, y : ArrayLike | None, optional, default=None
        Plotted x and y data. Used for axis limits and for
        deriving the x and y-axis label (unit).
    img : AxesImage | None, optional, default=None
        Plotted artists. Required for the colorbar and
        interactive ellipses.
    ref_unit : astropy.units.UnitBase | astropy.units.StructuredUnit | None, optional, default=None
        Reference unit for the colorbar label and for `axvlines`/`axhlines`.
    label : sequence | None
        Plot label. The legend is drawn only if this key is not None,
        and `params.legend['legend']` is truthy.
    rasterized : bool, optional
        Forwarded to `add_colorbar`. Popped from `kwargs`.
    rotation_step : int | float, optional, default=5
        Rotation step of the interactive ellipse.

    Returns
    -------
    None
    """
    # PRE SETTING AXIS LIMITS
    # -----------------------
    if 'label' in kwargs and params.legend.pop('legend', None):
        if kwargs['label'] is not None:
            legend(ax=ax, **params.legend)

    plot_ellipses(params.ellipses.pop('ellipses', None), ax)
    plot_points(
        params.points.pop('points', None),
        ax=ax,
        order=params.array_order,
        xy_indices=params.xy_indices
    )

    if params.compute_limits:
        set_axis_limits(
            x, y,
            ax=ax,
            **params.limits
        )

    # POST SETTING AXIS LIMITS
    # ------------------------
    ax.tick_params(
        labelsize=params.tick.pop('fontsize'),
        **params.tick
    )

    if title := params.title.pop('title', None):
        set_title(
            title,
            ax=ax,
            **params.title
        )

    labels = params.labels

    label_fontsize = labels.pop('fontsize')
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
        _set_axis_labels(
            x, y,
            ax=ax,
            fontsize=label_fontsize,
            **labels,
        )

    if params.wcs_grid.pop('wcs_grid', None) and isinstance(ax, WCSAxes):
        wcs_grid(ax=ax, **params.wcs_grid)

    if params.gridlines.pop('gridlines', None):
        grid(ax=ax, **params.gridlines)

    if params.colorbar.pop('colorbar', None):
        if img is not None and _has_color_mapping(img):
            cparams = params.colorbar
            cbar_unit = unit_2_string(
                ref_unit, fmt=config.unit_label_format
            )
            clabel = (
                cb if isinstance(cb := cparams.get('cbar_label'), str) else
                cbar_unit if config.colorbar.label else None
            )
            add_colorbar(
                img,
                ax=ax,
                label=clabel,
                rasterized=kwargs.pop('rasterized', _UNSET),
                **cparams
            )

    axvlines(params.vlines, ax=ax)
    axhlines(params.hlines, ax=ax)

    if params.ellipses.pop('plot_ellipse', None) and img is not None:
        geometry = _default_ellipse_geometry(img)
        if geometry is not None:
            center, w, h = geometry
            plot_interactive_ellipse(
                center, w, h, ax,
                params.text.get('loc'),
                params.text.get('color'),
                params.text.get('highlight'),
                rotation_step=kwargs.get('rotation_step', 5),
            )
