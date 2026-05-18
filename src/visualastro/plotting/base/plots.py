"""
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2026-03-11
Description:
    General plotting functions.
Dependencies:
    - matplotlib
    - numpy
Module Structure:
    - Plotting Functions
        Functions for general plots.
"""

from collections import namedtuple
from types import SimpleNamespace
from typing import Literal, Sequence
import astropy.units as u
import matplotlib.axes as maxes
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import AutoMinorLocator, NullLocator
from matplotlib.typing import ColorType
import numpy as np
from numpy.typing import NDArray

from visualastro.core.config import (
    get_config_value,
    config,
    _resolve_default,
    _Unset,
    _UNSET
)
from visualastro.core.io import _kwarg, _param, _pop_kwargs, _resolve_kwargs
from visualastro.core.numerical_utils import (
    get_value,
    get_data,
    to_array,
    to_list,
    _cycle,
)
from visualastro.core.stats import normalize as _normalize
from visualastro.core.units import ensure_common_unit
from visualastro.plotting.core.colors import get_cmap, get_colors
from visualastro.plotting.core.plot_utils import (
    contour,
    _apply_plot_utils,
    _extract_plot_util_kwargs,
    _get_zorder,
    _normalize_plotting_input,
    _normalize_plotting_inputs,
)


def plot_density_histogram(X, Y, ax, ax_histx, ax_histy, bins=None,
                           xlog=None, ylog=None, xlog_hist=None,
                           ylog_hist=None, histtype=None,
                           normalize=None, colors=_UNSET, **kwargs):
    '''
    Plot a 2D scatter distribution with normalized density histograms.
    This function creates a scatter plot of `X` vs. `Y` along
    with normalizable histograms of `X` and `Y`.

    Parameters
    ----------
    X : array-like or list of arrays
        The x-axis data or list of data arrays.
    Y : array-like or list of arrays
        The y-axis data or list of data arrays.
    ax : matplotlib.axes.Axes
        Main axis for the 2D scatter plot.
    ax_histx : matplotlib.axes.Axes
        Axis for the top histogram (x-axis).
    ax_histy : matplotlib.axes.Axes
        Axis for the right histogram (y-axis).
    bins : int, sequence, str, or None, optional, default=None
        Histogram bin specification. Passed directly to
        `matplotlib.pyplot.hist`. If None, uses the default
        value from `config.bins`. If `bins` is a str, use
        one of the supported binning strategies 'auto', 'fd',
        'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
    xlog : bool or None, optional, default=None
        Whether to use a logarithmic x-axis scale for the scatter plot.
        If None, uses the default value from `config.axes.xlog`.
    ylog : bool or None, optional, default=None
        Whether to use a logarithmic y-axis scale for the scatter plot.
        If None, uses the default value from `config.axes.ylog`.
    xlog_hist : bool or None, optional, default=None
        Whether to use a logarithmic x-axis scale for the top histogram.
        If None, uses the default value from `config.axes.xlog_hist`.
    ylog_hist : bool or None, optional, default=None
        Whether to use a logarithmic y-axis scale for the right histogram.
        If None, uses the default value from `config.axes.ylog_hist`.
    histtype : {'bar', 'barstacked', 'step', 'stepfilled'} or None, optional, default=None
        Type of histogram to draw. If None, uses the default value from `config.histtype`.
    normalize : bool, optional, default=None
        If True, normalize histograms to a probability density.
        If None, uses the default value from `config.normalize_hist`.
    colors : list of colors, str, or None, optional, default=None
        Colors for each dataset. If None, uses the
        default color colorset from `config.default_colorset`.

    **kwargs : dict, optional
        Additional parameters.

        Supported keyword arguments include:

        - `rasterized` : bool, default=`config.rasterized`
            Whether to rasterize plot artists. Rasterization
            converts the artist to a bitmap when saving to
            vector formats (e.g., PDF, SVG), which can
            significantly reduce file size for complex plots.
        - `color`, `c` : list of colors, str, or None, optional, default=None
            aliases for `colors`.
        - `sizes`, `size`, `s` : float or list, optional, default=`config.scatter_size`
            Marker size(s) for scatter points.
        - `markers`, `marker`, `m` : str or list, optional, default=`config.marker`
            Marker style(s) for scatter points.
        - `alphas`, `alpha`, `a` : float or list, optional, default=`config.alpha`
            Transparency level(s).
        - `edgecolors`, `edgecolor`, `ec` : str or list, optional, default=`config.edgecolor`
            Edge colors for scatter points.
        - `linestyles`, `linestyle`, `ls` : str or list, optional, default=`config.linestyle`
            Line style(s) for histogram edges.
        - `linewidth`, `lw` : float or list, optional, default=`config.linewidth`
            Line width(s) for histogram edges.
        - `zorders`, `zorder` : int or list, optional, default=None
            Z-order(s) for drawing priority.
        - `cmap` : str, optional, default=`config.cmap`
            Colormap name for automatic color assignment.
        - `xlim`, `ylim` : tuple, optional, default=None
            Axis limits for the scatter plot.
        - `labels`, `label`, `l` : list or str, optional, default=None
            Labels for legend entries.
        - `loc` : str, optional, default=`config.legend_loc`
            Legend location.
        - `xlabel`, `ylabel` : str, optional, default=None
            Axis labels for the scatter plot.

    Returns
    -------
    handles : DensityHistogram
        A named tuple containing the created Matplotlib objects:

        - `scatters` : matplotlib.collections.PathCollection or list of PathCollection
            The scatter plot object(s) created on the main axis.
        - `histx` : tuple or list of tuple
            The result(s) from the top histograms, where each tuple is
            `(n, bins, patches)` as returned by `Axes.hist`.
        - `histy` : tuple or list of tuple
            The result(s) from the right-side histograms, with the same format
            as `histx`.

        If only a single dataset is provided, each field contains a single
        object or tuple instead of a list.
    '''
    # ---- KWARGS ----
    rasterized = kwargs.get('rasterized', config.rasterized)
    colors = _pop_kwargs(kwargs, 'color', 'c', default=colors)
    # scatter params
    sizes = _pop_kwargs(kwargs, 'size', 's', default=None)
    markers = _pop_kwargs(kwargs, 'marker', 'm', default=None)
    alphas = _pop_kwargs(kwargs, 'alpha', 'a', default=None)
    edgecolors = _pop_kwargs(kwargs, 'edgecolors', 'edgecolor', 'ec', default=None)
    # line params
    linestyles = _pop_kwargs(kwargs, 'linestyles', 'linestyle', 'ls', default=None)
    linewidths = _pop_kwargs(kwargs, 'linewidth', 'lw', default=None)
    zorders = _pop_kwargs(kwargs, 'zorders', 'zorder', default=None)
    cmap = kwargs.get('cmap', config.cmap)
    # figure params
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    # labels
    labels = _pop_kwargs(kwargs, 'labels', 'label', 'l', default=None)
    loc = kwargs.get('loc', config.legend_loc)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)

    # get default config values
    bins = get_config_value(bins, 'bins')
    xlog = get_config_value(xlog, 'xlog')
    ylog = get_config_value(ylog, 'ylog')
    xlog_hist = get_config_value(xlog_hist, 'xlog_hist')
    ylog_hist = get_config_value(ylog_hist, 'ylog_hist')
    histtype = get_config_value(histtype, 'histtype')
    normalize = get_config_value(normalize, 'normalize_hist')
    colors = _resolve_default(colors, config.colors)
    sizes = get_config_value(sizes, 'scatter_size')
    markers = get_config_value(markers, 'marker')
    alphas = get_config_value(alphas, 'alpha')
    edgecolors = get_config_value(edgecolors, 'edgecolor')
    linestyles = get_config_value(linestyles, 'linestyle')
    linewidths = get_config_value(linewidths, 'linewidth')

    X = to_list(X)
    Y = to_list(Y)
    ensure_common_unit(X)
    ensure_common_unit(Y)
    if np.ndim(X) == 1 and np.ndim(Y) >= 2:
        X = [X]
    if np.ndim(Y) == 1 and np.ndim(X) >= 2:
        Y = [Y]

    # configure scales and ticks
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    ax_histx.set_xlim(ax.get_xlim())
    ax_histy.set_ylim(ax.get_ylim())
    if xlog: ax.set_xscale('log')
    if ylog: ax.set_yscale('log')
    if xlog_hist: ax_histx.set_yscale('log')
    if ylog_hist: ax_histy.set_xscale('log')

    ax.minorticks_on()
    # tick parameters for main plot
    ax.tick_params(axis='both', length=2, direction='in', which='both',
                    pad=5, right=True, top=True)
    # tick parameters for top histogram (x-axis)
    ax_histx.tick_params(axis='x', direction='in', which='both',
                            labelbottom=False, bottom=True)
    ax_histx.tick_params(axis='y', direction = 'in', which='both',
                            left=True, right=True, labelleft=True, pad=5)
    ax_histx.yaxis.set_label_position('left')
    # tick parameters for right histogram (y-axis)
    ax_histy.tick_params(axis='y', direction='in', which='both',
                            labelleft=False, left=True)
    ax_histy.tick_params(axis='x', direction = 'in', which='both',
                            bottom=True, top=True, labelbottom=True, pad=5)
    ax_histy.xaxis.set_label_position('bottom')

    # set plot colors
    colors = get_colors(colors, cmap=cmap)

    sizes = sizes if isinstance(sizes, (list, np.ndarray, tuple)) else [sizes]
    markers = markers if isinstance(markers, (list, np.ndarray, tuple)) else [markers]
    alphas = alphas if isinstance(alphas, (list, np.ndarray, tuple)) else [alphas]
    edgecolors = edgecolors if isinstance(edgecolors, (list, np.ndarray, tuple)) else [edgecolors]

    linestyles = linestyles if isinstance(linestyles, (list, np.ndarray, tuple)) else [linestyles]
    linewidths = linewidths if isinstance(linewidths, (list, np.ndarray, tuple)) else [linewidths]
    zorders = zorders if isinstance(zorders, (list, np.ndarray, tuple)) else [zorders]
    labels = labels if isinstance(labels, (list, np.ndarray, tuple)) else [labels]

    scatters = []
    histx = []
    histy = []

    for i in range(len(Y)):
        x = X[i%len(X)]
        y = Y[i%len(Y)]
        color = colors[i%len(colors)]
        size = sizes[i%len(sizes)]
        marker = markers[i%len(markers)]
        alpha = alphas[i%len(alphas)]
        edgecolor = edgecolors[i%len(edgecolors)]
        linestyle = linestyles[i%len(linestyles)]
        linewidth = linewidths[i%len(linewidths)]
        zorder = zorders[i%len(zorders)] if zorders[i%len(zorders)] is not None else i
        label = labels[i] if (labels[i%len(labels)] is not None and i < len(labels)) else None

        sc = ax.scatter(x, y, c=color, s=size, marker=marker,
                        alpha=alpha, edgecolors=edgecolor,
                        label=label, rasterized=rasterized)
        # top histogram (x-axis)
        hx = ax_histx.hist(x, bins=bins, color=color, histtype=histtype,
                           ls=linestyle, lw=linewidth, alpha=alpha,
                           zorder=zorder, density=normalize,
                           rasterized=rasterized)
        # right histogram (y-axis)
        hy = ax_histy.hist(y, bins=bins, orientation='horizontal',
                           color=color, histtype=histtype, ls=linestyle,
                           lw=linewidth, alpha=alpha, zorder=zorder,
                           density=normalize, rasterized=rasterized)

        scatters.append(sc)
        histx.append(hx)
        histy.append(hy)

    if xlog_hist:
        ax_histx.set_ylabel('[Log]', labelpad=10)
    if ylog_hist:
        ax_histy.set_xlabel('[Log]', labelpad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels[0] is not None:
        ax.legend(loc=loc)

    # create return obect
    PlotHandles = namedtuple('DensityHistogram', ['scatters', 'histx', 'histy'])
    scatters = scatters[0] if len(scatters) == 1 else scatters
    histx = histx[0] if len(histx) == 1 else histx
    histy = histy[0] if len(histy) == 1 else histy

    return PlotHandles(scatters, histx, histy)


def hist(
    datas: u.Quantity | NDArray | list[u.Quantity | NDArray],
    ax: maxes.Axes,
    bins: int | Sequence[float] | str | _Unset = _UNSET,
    histtype: Literal['bar', 'barstacked', 'step', 'stepfilled'] | _Unset = _UNSET,
    normalize: bool | _Unset = _UNSET,
    align: Literal['left', 'mid', 'right'] = 'mid',
    color: ColorType | list[ColorType] | _Unset = _UNSET,
    xlog: bool | _Unset = _UNSET,
    ylog: bool | _Unset = _UNSET,
    zorder: float | list[float] | None = None,
    **kwargs
) -> list[SimpleNamespace]:
    """
    Plot one or more histograms on a given Axes object. 2D arrays are flattened.

    Parameters
    ----------
    datas : u.Quantity | NDArray | list[u.Quantity | NDArray]
        Input data to histogram. Can be a single 1D array or a
        list of 1D/2D arrays. 2D arrays are automatically flattened.
    ax : matplotlib.axes.Axes
        The Axes object on which to plot the histogram.
    bins : int | Sequence[float] | str | _Unset, optional, default=_UNSET
        From ax.hist documentation:

        If *bins* is an integer, it defines the number of equal-width bins in
        the range.

        If *bins* is a sequence, it defines the bin edges, including the left
        edge of the first bin and the right edge of the last bin; in this case,
        bins may be unequally spaced.  All but the last (righthand-most) bin is
        half-open.  In other words, if *bins* is:

        [1, 2, 3, 4]

        then the first bin is [1, 2) (including 1, but excluding 2) and the second
        [2, 3).  The last bin, however, is [3, 4], which *includes* 4.

        If *bins* is a string, it is one of the binning strategies supported by
        `numpy.histogram_bin_edges`: `'auto'`, `'fd'`, `'doane'`, `'scott'`,
        `'stone'`, `'rice'`, `'sturges'`, or `'sqrt'`.

    histtype : {'bar', 'barstacked', 'step', 'stepfilled'} | _Unset, optional, default=_UNSET
        Matplotlib histogram type. If `_UNSET`, uses `config.histtype`.
    normalize : bool or None, optional, default=None
        Alias for `density`. If `True`, normalize histograms to a probability density.
        If `_UNSET`, uses `config.normalize_hist`.
    align : {'left', 'mid', 'right'}, optional, default: 'mid'
        The horizontal alignment of the histogram bars.

            – 'left': bars are centered on the left bin edges.
            – 'mid': bars are centered between the bin edges.
            – 'right': bars are centered on the right bin edges.

    color : ColorType | list[ColorType] | int | _Unset, optional, default=_UNSET
        Color(s) for scatter markers. If `_UNSET`, uses `config.colors`.
    xlog : bool | _Unset, optional, default=_UNSET
        If `True`, uses logarithmic scale on x-axis.
        If `_UNSET`, uses `config.axes.xlog`.
    ylog : bool | _Unset, optional, default=_UNSET
        If `True`, use logarithmic scale on y-axis.
        If `_UNSET`, uses `config.axes.ylog`.
    colors : list of colors, str, or None, optional, default=None
        Colors to use for each dataset. If None,
        uses the default color colorset from `config.default_colorset`.
    label : str | list[str], optional, default=None
        Legend labels for scatter datasets.
    loc : str, optional, default=config.legend_loc
        Legend location.
    xlabel : str, optional, default=None
        Label for x-axis.
    ylabel : str, optional, default=None
        Label for y-axis.
    xlim : tuple[float, float], optional, default=None
        Limits for x-axis as (xmin, xmax).
    ylim : tuple[float, float], optional, default=None
        Limits for y-axis as (ymin, ymax).
    xpad : float, optional, default=config.axes.ypad
        Fractional padding added to the x-axis data range when computing axis limits.
    ypad : float, optional, default=config.axes.xpad
        Fractional padding added to the y-axis data range when computing axis limits.
    cmap : Colormap | str, optional, default=config.cmap
        Colormap used to generate colors if `color` is an int.
    bad_color : str, optional
        Fallback color for invalid values in colormap.
    rasterized : bool, optional, default=config.rasterized
        If `True`, rasterize artists when saving to vector formats.

    Returns
    -------
    hists : SimpleNamespace | list[SimpleNamespace]
        The result(s) returned by `Axes.hist`. Each SimpleNamespace has the form:

        - n : ndarray
          The values of the histogram (counts or densities).
        - bins : ndarray
          The edges of the bins (length = len(n) + 1).
        - patches : list[Patch]
          The artists created by the histogram (e.g., `Rectangle` or `Polygon`
          objects depending on `histtype`).

        If only one histogram is created, `hists` is a single tuple; otherwise,
        it is a list of tuples.
    """
    params = _resolve_kwargs(
        kwargs,
        [
            _param('bins', bins, config.bins),
            _param('histtype', histtype, config.histtype),
            _param('color', color, config.colors),
            _param('normalize', normalize, config.normalize_hist),
            _param('xlog', xlog, config.axes.xlog_hist),
            _param('ylog', ylog, config.axes.ylog_hist),
        ],
        [
            _kwarg('label', None),
            _kwarg('cmap', config.cmap),
            _kwarg('bad_color', None),
            _kwarg('rasterized', config.rasterized),
        ]
    )
    fig_params = _extract_plot_util_kwargs(kwargs)

    datas = to_list(datas)
    bins_list = to_list(params.bins)
    histtypes = to_list(params.histtype)
    labels = to_list(params.label)
    zorders = to_list(zorder)

    ref_unit = ensure_common_unit(datas)

    if xlog: ax.set_xscale('log')
    if ylog: ax.set_yscale('log')

    cmap = get_cmap(params.cmap, params.bad_color)
    colors = get_colors(params.color, cmap=cmap)
    data_list = []

    hists = []

    for i, data in enumerate(datas):
        bin = _cycle(bins_list, i)
        htype = _cycle(histtypes, i)
        color = _cycle(colors, i)
        z = _get_zorder(zorders, i, config.zorder.plot_data)
        label = labels[i] if (_cycle(labels, i) is not None and i < len(labels)) else None
        data = to_array(data)

        if data.ndim == 2:
            data = data.flatten()

        h = ax.hist(
            data,
            bins=bin,
            histtype=htype,
            density=params.normalize,
            align=align,
            color=color,
            label=label,
            rasterized=params.rasterized,
            zorder=z
        )
        if ref_unit is not None:
            data = data * ref_unit

        data_list.append(data)
        hists.append(
            SimpleNamespace(**{'n': h[0], 'bins': h[1], 'patches': h[2]})
        )

    _apply_plot_utils(fig_params, ax=ax, ref_unit=ref_unit, xlist=data_list, labels=labels)

    return hists


def plot(
    *data: float | u.Quantity | NDArray | list[float | u.Quantity | NDArray],
    ax: maxes.Axes,
    color: ColorType | list[ColorType] | _Unset = _UNSET,
    linestyle: Literal['-', '--', '-.', ':', ''] | list[Literal['-', '--', '-.', ':', '']] | _Unset = _UNSET,
    linewidth: float | list[float] | _Unset = _UNSET,
    alpha: float | list[float] | _Unset = _UNSET,
    normalize: bool | _Unset = _UNSET,
    xlog: bool | _Unset = _UNSET,
    ylog: bool | _Unset = _UNSET,
    zorder: float | list[float] | None = None,
    array_order: Literal['c', 'fortran'] | _Unset = _UNSET,
    **kwargs
) -> list[Line2D]:
    """
    Plot one or more lines on a given Axes object with flexible styling.

    Plot multiple datasets by passing in lists of data!

    Examples:

        – plot(x, y)
        – plot([1,2,3], [[4,5,6], [7,8,9]])
        – plot(radius, [vel1, vel2])
        – plot(flux)
        – plot(pos[:,0:2])

    Parameters
    ----------
    *data : float | u.Quantity | NDArray | list[float | u.Quantity | NDArray]
        Positional arguments specifying x and y data. Accepts either a single
        2D array or two separate arrays/list of arrays/values. 2D arrays can
        either be (N,2) (`order='c'`) or (2,N) (`order='fortran')`. If only
        one array is passed in, x values are automatically generated with
        np.arange(len(array)).
    ax : matplotlib.axes.Axes
        The Axes object to plot on.
    color : ColorType | list[ColorType] | int | _Unset, optional, default=_UNSET
        Color(s) for scatter markers. If `_UNSET`, uses `config.colors`.
    linestyle : str | list[str] | _Unset, optional, default=_UNSET
        Line style(s) to use for plotting. Can be a single string or a list of
        styles for multiple lines. Accepted values are:
        {'-', '--', '-.', ':', ''}. If `_UNSET`, uses `config.linestyle`.
    linewidth : float | list[float] | _Unset, optional, default=_UNSET
        Line width for the plotted lines. If `_UNSET`, uses the
        `config.linewidth`.
    alpha : float | list[float] | _Unset, optional, default=_UNSET
        Transparency value(s) in [0, 1]. If `_UNSET`, uses `config.alpha`.
    normalize : bool | _Unset, optional, default=_UNSET
        If `True`, normalize each dataset by its maximum value.
        If `_UNSET`, uses `config.normalize_data`.
    xlog : bool | _Unset, optional, default=_UNSET
        If `True`, uses logarithmic scale on x-axis.
        If `_UNSET`, uses `config.axes.xlog`.
    ylog : bool | _Unset, optional, default=_UNSET
        If `True`, use logarithmic scale on y-axis.
        If `_UNSET`, uses `config.axes.ylog`.
    zorder : float | list[float] | None, optional, default=None
        Order in which to plot lines in. Lines are drawn in order
        of greatest to lowest zorder. If None, starts at 0 and increments
        the zorder by 1 for each subsequent line drawn.
    array_order : {'C', 'c', 'F', 'fortran'} | _Unset, optional, default=_UNSET
        Array order of the input. `'C'` and `'c'` are for (N,2) shaped arrays
        while `'F'` and `'fortran'` are for (2,N) shaped arrays.
    label : str | list[str], optional, default=None
        Legend labels for scatter datasets.
    loc : str, optional, default=config.legend_loc
        Legend location.
    xlabel : str, optional, default=None
        Label for x-axis.
    ylabel : str, optional, default=None
        Label for y-axis.
    xlim : tuple[float, float], optional, default=None
        Limits for x-axis as (xmin, xmax).
    ylim : tuple[float, float], optional, default=None
        Limits for y-axis as (ymin, ymax).
    xpad : float, optional, default=config.axes.ypad
        Fractional padding added to the x-axis data range when computing axis limits.
    ypad : float, optional, default=config.axes.xpad
        Fractional padding added to the y-axis data range when computing axis limits.
    cmap : Colormap | str, optional, default=config.cmap
        Colormap used to generate colors if `color` is an int.
    bad_color : str, optional
        Fallback color for invalid values in colormap.
    rasterized : bool, optional, default=config.rasterized
        If `True`, rasterize artists when saving to vector formats.

    Returns
    -------
    lines : list[matplotlib.lines.Line2D]
        The line object(s) created by `Axes.plot`. Each element is a
        `matplotlib.lines.Line2D` instance representing one plotted line.
    """
    params = _resolve_kwargs(
        kwargs,
        [
            _param('color', color, config.colors),
            _param('linestyle', linestyle, config.linestyle),
            _param('linewidth', linewidth, config.linewidth),
            _param('alpha', alpha, config.alpha),
            _param('normalize', normalize, config.normalize_data),
            _param('xlog', xlog, config.axes.xlog),
            _param('ylog', ylog, config.axes.ylog),
            _param('array_order', array_order, config.array_order),
        ],
        [
            _kwarg('index_spec', 'implicit'),
            _kwarg('label', None),
            _kwarg('cmap', config.cmap),
            _kwarg('bad_color', None),
            _kwarg('rasterized', config.rasterized),
        ]
    )
    plot_params = _extract_plot_util_kwargs(kwargs)

    alphas = to_list(params.alpha)
    labels = to_list(params.label)
    linestyles = to_list(params.linestyle)
    linewidths = to_list(params.linewidth)
    zorders = to_list(zorder)

    xlist, ylist = _normalize_plotting_inputs(*data, order=params.array_order, index_spec=params.index_spec)

    ensure_common_unit(xlist, on_mismatch=config.unit_mismatch)
    ensure_common_unit(ylist, on_mismatch=config.unit_mismatch)

    if params.xlog: ax.set_xscale('log')
    if params.ylog: ax.set_yscale('log')

    cmap = get_cmap(params.cmap, params.bad_color)
    colors = get_colors(params.color, cmap=cmap)
    lines = []

    for i in range(len(ylist)):
        x = get_value(_cycle(xlist, i))
        y = get_value(_cycle(ylist, i))
        color = _cycle(colors, i)
        ls = _cycle(linestyles, i)
        lw = _cycle(linewidths, i)
        a = _cycle(alphas, i)
        z = _get_zorder(zorders, i, config.zorder.plot_data)
        label = labels[i] if (_cycle(labels, i) is not None and i < len(labels)) else None

        if params.normalize:
            y = _normalize(y)
            ylist[i] = y

        line = ax.plot(
            x, y,
            color=color,
            ls=ls,
            lw=lw,
            alpha=a,
            zorder=z,
            label=label,
            rasterized=params.rasterized,
            **kwargs
        )

        lines.append(line)

    _apply_plot_utils(plot_params, ax, xlist=xlist, ylist=ylist, labels=labels)

    return lines


def scatter(
    *data: float | u.Quantity | NDArray | list[float | u.Quantity | NDArray],
    ax: maxes.Axes,
    xerr: float | u.Quantity | NDArray | list[float | u.Quantity | NDArray] | None = None,
    yerr: float | u.Quantity | NDArray | list[float | u.Quantity | NDArray] | None = None,
    color: ColorType | list[ColorType] | int | _Unset = _UNSET,
    marker: MarkerStyle | list[MarkerStyle] | _Unset = _UNSET,
    size: float | list[float] | _Unset = _UNSET,
    alpha: float | list[float] | _Unset = _UNSET,
    edgecolor: Literal['face', 'none'] | ColorType | list[ColorType] | _Unset = _UNSET,
    facecolor: Literal['none'] | ColorType | list[ColorType] | _Unset = _UNSET,
    normalize: bool | _Unset = _UNSET,
    xlog: bool | _Unset = _UNSET,
    ylog: bool | _Unset = _UNSET,
    zorder: float | list[float] | None = None,
    array_order: Literal['C', 'c', 'F', 'fortran'] | _Unset = _UNSET,
    **kwargs
) ->  list[PatchCollection]:
    """
    Scatter plot data with optional error bars on a matplotlib Axes.

    Plot multiple datasets by passing in lists of data!

    Examples:

        – scatter(x, y)
        – scatter([1,2,3], [[4,5,6], [7,8,9]])
        – scatter(radius, [vel1, vel2])
        – scatter(flux)
        – scatter(pos[:,0:2])

    Parameters
    ----------
    *data : float | u.Quantity | NDArray | list[float | u.Quantity | NDArray]
        Positional arguments specifying x and y data. Accepts either a single
        2D array or two separate arrays/list of arrays/values. 2D arrays can
        either be (N,2) (`order='c'`) or (2,N) (`order='fortran')`. If only
        one array is passed in, x values are automatically generated with
        np.arange(len(array)).
    ax : matplotlib.axes.Axes
        Axes to plot on.
    xerr : array-like | list[array-like] | None, optional, default=None
        Errors on x-axis data. Must match shape of x data.
    yerr : array-like | list[array-like] | None, optional, default=None
        Errors on y-axis data. Must match shape of y data.
    color : ColorType | list[ColorType] | int | _Unset, optional, default=_UNSET
        Color(s) for scatter markers. If `_UNSET`, uses `config.colors`.
    marker : str | list[str] | _Unset, optional, default=_UNSET
        Marker style(s). If `_UNSET`, uses config.marker.
    size : float | list[float] | _Unset, optional, default=_UNSET
        Marker size(s). If `_UNSET`, uses `config.scatter_size`.
    alpha : float | list[float] | _Unset, optional, default=_UNSET
        Transparency value(s) in [0, 1]. If `_UNSET`, uses `config.alpha`.
    edgecolor : {'face', 'none'} | ColorType | list[ColorType] | _Unset, optional, default=_UNSET
        Edge color of markers.

            – 'face': Match face color
            – 'none': No edge
            – color or sequence: Explicit color(s)

        If not set, uses `config.edgecolor`.

    facecolor : {'none'} | ColorType | list[ColorType] | _Unset, optional, default=_UNSET
        Face color of markers.

            – 'none': Transparent
            – color or sequence: Explicit color(s)

        If not set, uses `config.facecolor`.

    normalize : bool | _Unset, optional, default=_UNSET
        If `True`, normalize each dataset by its maximum value.
        If `_UNSET`, uses `config.normalize_data`.
    xlog : bool | _Unset, optional, default=_UNSET
        If `True`, uses logarithmic scale on x-axis.
        If `_UNSET`, uses `config.axes.xlog`.
    ylog : bool | _Unset, optional, default=_UNSET
        If `True`, use logarithmic scale on y-axis.
        If `_UNSET`, uses `config.axes.ylog`.
    array_order : {'C', 'c', 'F', 'fortran'} | _Unset, optional, default=_UNSET
        Array order of the input. `'C'` and `'c'` are for (N,2) shaped arrays
        while `'F'` and `'fortran'` are for (2,N) shaped arrays.
    label : str | list[str], optional, default=None
        Legend labels for scatter datasets.
    loc : str, optional, default=config.legend_loc
        Legend location.
    xlabel : str, optional, default=None
        Label for x-axis.
    ylabel : str, optional, default=None
        Label for y-axis.
    xlim : tuple[float, float], optional, default=None
        Limits for x-axis as (xmin, xmax).
    ylim : tuple[float, float], optional, default=None
        Limits for y-axis as (ymin, ymax).
    xpad : float, optional, default=config.axes.ypad
        Fractional padding added to the x-axis data range when computing axis limits.
    ypad : float, optional, default=config.axes.xpad
        Fractional padding added to the y-axis data range when computing axis limits.
    cmap : Colormap | str, optional, default=config.cmap
        Colormap used to generate colors if `color` is an int.
    bad_color : str, optional
        Fallback color for invalid values in colormap.
    ecolor : ColorType, optional, default=config.errorbar.colors
        Error bar color.
    markeredgecolor : ColorTpe, optional, default=config.errorbar.markeredgecolor
        Plot marker edge color.
    elinewidth : float, optional, default=config.errorbar.linewidth
        Error bar line width in points.
    capsize : float, optional, default=config.errorbar.capsize
        Length of error bar caps in points.
    capthick : float, optional, default=config.errorbar.capthick
        Thickness of error bar caps in points.
    barsabove : bool, optional, default=config.errorbar.barsabove
        If `True`, draw error bars above plot symbols.
    rasterized : bool, optional, default=config.rasterized
        If `True`, rasterize artists when saving to vector formats.

    Returns
    -------
    list[matplotlib.collections.PathCollection]
        Scatter plot collection(s). Each element represents one scatter plot.
        Length matches the number of datasets plotted.

    See Also
    --------
    matplotlib.axes.Axes.scatter
    matplotlib.axes.Axes.errorbar
    """
    params = _resolve_kwargs(
        kwargs,
        [
            _param('color', color, config.colors),
            _param('marker', marker, config.marker),
            _param('size', size, config.scatter_size),
            _param('alpha', alpha, config.alpha),
            _param('edgecolor', edgecolor, config.edgecolor),
            _param('facecolor', facecolor, config.facecolor),
            _param('normalize', normalize, config.normalize_data),
            _param('xlog', xlog, config.axes.xlog),
            _param('ylog', ylog, config.axes.ylog),
            _param('array_order', array_order, config.array_order),
        ],
        [
            _kwarg('index_spec', 'implicit'),
            _kwarg('label', None),
            _kwarg('cmap', config.cmap),
            _kwarg('bad_color', None),
            _kwarg('ecolor', config.errorbar.colors),
            _kwarg('markeredgecolor', config.errorbar.markeredgecolor),
            _kwarg('elinewidth', config.errorbar.linewidth),
            _kwarg('capsize', config.errorbar.capsize),
            _kwarg('capthick', config.errorbar.capthick),
            _kwarg('barsabove', config.errorbar.barsabove),
            _kwarg('rasterized', config.rasterized),
        ]
    )
    plot_params = _extract_plot_util_kwargs(kwargs)

    alphas = to_list(params.alpha)
    edgecolors = to_list(params.edgecolor)
    facecolors = to_list(params.facecolor)
    labels = to_list(params.label)
    sizes = to_list(params.size)
    markers = to_list(params.marker)
    zorders = to_list(zorder)

    xlist, ylist = _normalize_plotting_inputs(*data, order=params.array_order, index_spec=params.index_spec)

    ensure_common_unit(xlist, on_mismatch=config.unit_mismatch)
    ensure_common_unit(ylist, on_mismatch=config.unit_mismatch)

    xerrs = _normalize_plotting_input(xerr) if xerr is not None else xerr
    yerrs = _normalize_plotting_input(yerr) if yerr is not None else yerr

    if params.xlog: ax.set_xscale('log')
    if params.ylog: ax.set_yscale('log')

    cmap = get_cmap(params.cmap, params.bad_color)
    colors = get_colors(params.color, cmap=cmap)
    scatters = []

    for i in range(len(ylist)):
        x = get_value(_cycle(xlist, i))
        y = get_value(_cycle(ylist, i))
        color = _cycle(colors, i)
        s = _cycle(sizes, i)
        m = _cycle(markers, i)
        a = _cycle(alphas, i)
        ec = _cycle(edgecolors, i)
        fc = _cycle(facecolors, i)
        z = _get_zorder(zorders, i, config.zorder.plot_data)
        label = labels[i] if (_cycle(labels, i) is not None and i < len(labels)) else None
        params.ecolor = color if params.ecolor is None else params.ecolor
        params.markeredgecolor = (
            color if params.markeredgecolor is None else params.markeredgecolor
        )

        if fc == 'none' and ec is None:
            ec = color

        if params.normalize:
            y = _normalize(y)
            ylist[i] = y

        scatter = ax.scatter(
            x, y,
            color=color,
            s=s,
            marker=m,
            alpha=a,
            edgecolors=ec,
            facecolors=fc,
            label=label,
            rasterized=params.rasterized,
            zorder=z,
            **kwargs
        )

        scatters.append(scatter)

        xerror = _cycle(xerrs, i) if xerrs is not None else None
        yerror = _cycle(yerrs, i) if yerrs is not None else None

        if xerror is not None or yerror is not None:
            ax.errorbar(
                x, y, yerror, xerror,
                fmt=config.errorbar.fmt,
                mfc=params.ecolor,
                ecolor=params.ecolor,
                elinewidth=params.elinewidth,
                mec=params.markeredgecolor,
                capsize=params.capsize,
                capthick=params.capthick,
                barsabove=params.barsabove,
                rasterized=params.rasterized,
                zorder=z,
            )

    _apply_plot_utils(plot_params, ax, xlist=xlist, ylist=ylist, labels=labels)

    return scatters


def scatter_fit(
    *data: float | u.Quantity | NDArray | list[float | u.Quantity | NDArray],
    ax: maxes.Axes,
    deg: int,
    xerr: float | u.Quantity | NDArray | list[float | u.Quantity | NDArray] | None = None,
    yerr: float | u.Quantity | NDArray | list[float | u.Quantity | NDArray] | None = None,
    color: ColorType | list[ColorType] | int | _Unset =_UNSET,
    marker: MarkerStyle | list[MarkerStyle] | _Unset = _UNSET,
    size: float | list[float] | _Unset = _UNSET,
    alpha: float | list[float] | _Unset = _UNSET,
    edgecolor: Literal['face', 'none'] | ColorType | list[ColorType] | _Unset = _UNSET,
    facecolor: Literal['none'] | ColorType | list[ColorType] | _Unset = _UNSET,
    linecolor: ColorType | list[ColorType] | int | _Unset = _UNSET,
    linestyle: Literal['-', '--', '-.', ':', ''] | list[Literal['-', '--', '-.', ':', '']] | _Unset = _UNSET,
    linewidth: float | list[float] | _Unset = _UNSET,
    linealpha: float | list[float] | _Unset = _UNSET,
    normalize: bool | _Unset = _UNSET,
    xlog: bool | _Unset = _UNSET,
    ylog: bool | _Unset = _UNSET,
    zorder: float | list[float] | None = None,
    array_order: Literal['C', 'c', 'F', 'fortran'] | _Unset = _UNSET,
    **kwargs
) -> SimpleNamespace:
    """
    Plot scatter data together with polynomial fits.

    Parameters
    ----------
    *data : float | u.Quantity | NDArray | list[float | u.Quantity | NDArray]
        Positional arguments specifying x and y data. Accepts either a single
        2D array or two separate arrays/list of arrays/values. 2D arrays can
        either be (N,2) (`order='c'`) or (2,N) (`order='fortran')`. If only
        one array is passed in, x values are automatically generated with
        np.arange(len(array)).
    ax : matplotlib.axes.Axes
        Axes to plot on.
    deg : int
        Degree of the polynomial fit.
    xerr : array-like | list[array-like] | None, optional, default=None
        Errors on x-axis data. Must match shape of x data.
    yerr : array-like | list[array-like] | None, optional, default=None
        Errors on y-axis data. Must match shape of y data.
    color : ColorType | list[ColorType] | int | _Unset, optional, default=_UNSET
        Color(s) for scatter markers. If `_UNSET`, uses `config.colors`.
    marker : str | list[str] | _Unset, optional, default=_UNSET
        Marker style(s). If `_UNSET`, uses config.marker.
    size : float | list[float] | _Unset, optional, default=_UNSET
        Marker size(s). If `_UNSET`, uses `config.scatter_size`.
    alpha : float | list[float] | _Unset, optional, default=_UNSET
        Transparency value(s) in [0, 1]. If `_UNSET`, uses `config.alpha`.
    edgecolor : {'face', 'none'} | ColorType | list[ColorType] | _Unset, optional, default=_UNSET
        Edge color of markers.

            – 'face': Match face color
            – 'none': No edge
            – color or sequence: Explicit color(s)

        If not set, uses `config.edgecolor`.

    facecolor : {'none'} | ColorType | list[ColorType] | _Unset, optional, default=_UNSET
        Face color of markers.

            – 'none': Transparent
            – color or sequence: Explicit color(s)

        If not set, uses `config.facecolor`.


    linecolor : ColorType | list[ColorType] | int | _Unset, optional, default=_UNSET
        Polynomial fit line colors. If unset, uses `color`.
    linestyle : str | list[str] | _Unset, optional, default=_UNSET
        Line style(s) to use for plotting. Can be a single string or a list of
        styles for multiple lines. Accepted values are:
        {'-', '--', '-.', ':', ''}. If `_UNSET`, uses `config.linestyle`.
    linewidth : float | list[float] | _Unset, optional, default=_UNSET
        Line width for the plotted lines. If `_UNSET`, uses the
        `config.linewidth`.
    linealpha : float | list[float], optional
        Alpha values for fitted lines.
    normalize : bool | _Unset, optional, default=_UNSET
        If `True`, normalize each dataset by its maximum value.
        If `_UNSET`, uses `config.normalize_data`.
    xlog : bool | _Unset, optional, default=_UNSET
        If `True`, uses logarithmic scale on x-axis.
        If `_UNSET`, uses `config.axes.xlog`.
    ylog : bool | _Unset, optional, default=_UNSET
        If `True`, use logarithmic scale on y-axis.
        If `_UNSET`, uses `config.axes.ylog`.
    zorder : float | list[float] | None, optional, default=None
        Order in which to plot lines in. Lines are drawn in order
        of greatest to lowest zorder. If None, starts at 0 and increments
        the zorder by 1 for each subsequent line drawn.
    array_order : {'C', 'c', 'F', 'fortran'} | _Unset, optional, default=_UNSET
        Array order of the input. `'C'` and `'c'` are for (N,2) shaped arrays
        while `'F'` and `'fortran'` are for (2,N) shaped arrays.
    label : str | list[str], optional, default=None
        Legend labels for scatter datasets.
    loc : str, optional, default=config.legend_loc
        Legend location.
    xlabel : str, optional, default=None
        Label for x-axis.
    ylabel : str, optional, default=None
        Label for y-axis.
    xlim : tuple[float, float], optional, default=None
        Limits for x-axis as (xmin, xmax).
    ylim : tuple[float, float], optional, default=None
        Limits for y-axis as (ymin, ymax).
    xpad : float, optional, default=config.axes.ypad
        Fractional padding added to the x-axis data range when computing axis limits.
    ypad : float, optional, default=config.axes.xpad
        Fractional padding added to the y-axis data range when computing axis limits.
    cmap : Colormap | str, optional, default=config.cmap
        Colormap used to generate colors if `color` is an int.
    bad_color : str, optional
        Fallback color for invalid values in colormap.
    ecolor : ColorType, optional, default=config.errorbar.colors
        Error bar color.
    markeredgecolor : ColorTpe, optional, default=config.errorbar.markeredgecolor
        Plot marker edge color.
    elinewidth : float, optional, default=config.errorbar.linewidth
        Error bar line width in points.
    capsize : float, optional, default=config.errorbar.capsize
        Length of error bar caps in points.
    capthick : float, optional, default=config.errorbar.capthick
        Thickness of error bar caps in points.
    barsabove : bool, optional, default=config.errorbar.barsabove
        If `True`, draw error bars above plot symbols.
    rasterized : bool, optional, default=config.rasterized
        If `True`, rasterize artists when saving to vector formats.

    Returns
    -------
    SimpleNamespace
        Namespace containing:

        scatter : list[matplotlib.collections.PathCollection]
            Scatter plot artists.

        line : list[list[matplotlib.lines.Line2D]]
            Polynomial fit line artists returned by
            :meth:`matplotlib.axes.Axes.plot`.
    """
    params = _resolve_kwargs(
        kwargs,
        [
            _param('linestyle', linestyle, config.linestyle),
            _param('linewidth', linewidth, config.linewidth),
            _param('linealpha', linealpha, 1),
        ]
    )
    linestyles = to_list(params.linestyle)
    linewidths = to_list(params.linewidth)
    linealphas = to_list(params.linealpha)
    zorders = to_list(zorder)
    paths = scatter(
        *data,
        ax=ax,
        xerr=xerr,
        yerr=yerr,
        color=color,
        marker=marker,
        size=size,
        alpha=alpha,
        edgecolor=edgecolor,
        facecolor=facecolor,
        normalize=normalize,
        xlog=xlog,
        ylog=ylog,
        array_order=array_order,
        **kwargs
    )

    datas = [get_data(path.get_offsets()) for path in paths]
    if linecolor is _UNSET:
        colors = [
            path.get_facecolors() if path.get_facecolors().size > 0 else
            path.get_edgecolors() for path in paths
        ]
    else:
        colors = get_colors(linecolor)
    lines = []

    for i, array in enumerate(datas):
        color = _cycle(colors, i)
        ls = _cycle(linestyles, i)
        lw = _cycle(linewidths, i)
        a = _cycle(linealphas, i)
        z = _get_zorder(zorders, i, config.zorder.plot_data)

        array = np.asarray(array)
        x = array[:,0]
        y = array[:,1]
        fn = np.polynomial.Polynomial.fit(x, y, deg=deg)

        l = ax.plot(x, fn(x), color=color, ls=ls, lw=lw, zorder=z, alpha=a)
        lines.append(l)

    return SimpleNamespace(**{'scatter': paths, 'line': lines})


def scatter3D(X, Y, Z, ax, elev=30, azim=45, roll=0,
              scale=None, axes_off=False, grid_lines=False,
              colors=_UNSET, size=None, marker=None, alpha=None,
              edgecolors=_UNSET, plot_contours=None, **kwargs):
    '''
    Scatter plot in 3D with support for multiple datasets.

    Parameters
    ----------
    X, Y, Z : array-like or list of array-like
        Coordinates of the data points. Each of `X`, `Y`, and `Z`
        may be a single array or a list of arrays for plotting
        multiple groups. All three must have the same number of arrays.
    ax : `matplotlib.axes._subplots.Axes3DSubplot`
        The 3D axes object on which to draw the scatter plot.
    elev : float, default=30
        Elevation angle in degrees (rotation around camera x-axis).
    azim : float, default=45
        Azimuth angle in degrees (rotation around the z-axis).
    roll : float, default=0
        Roll angle in degrees (rotation around the view direction).
    scale : float or None, default=None
        If given, sets symmetric limits for all axes as `[-scale, scale]`.
    axes_off : bool, default=False
        If True, hides all axes spines, ticks, and labels.
    grid_lines : bool, default=False
        If False, disables gridlines on the 3D plot.
    colors : list of colors, str, or None, optional, default=None
        Colors to use for each scatter group or dataset.
        If None, uses the default color colorset from
        `config.default_colorset`.
    size : float, list of float, or None, optional, default=None
        Size of scatter dots. If None, uses the default
        value in `config.scatter_size`.
    marker : str, list of str, or None, optional, default=None
        Marker style for scatter dots. If None, uses the
        default value in `config.marker`.
    alpha : float, list of float, or None, default=None
        The alpha blending value, between 0 (transparent) and 1 (opaque).
        If None, uses the default value from `config.alpha`.
    edgecolors : {'face', 'none', None}, color, list of color, or None, default=`_UNSET`
        The edge color of the marker. Possible values:
        - 'face': The edge color will always be the same as the face color.
        - 'none': No patch boundary will be drawn.
        - A color or sequence of colors.
        If `_UNSET`, uses the default value in `config.edgecolor`.
    plot_contours : {'x', 'y', 'z', 'all'}, sequence of {'x', 'y', 'z'}, or None, optional, default=None
        Specifies which contour projections to draw onto the side planes of the 3D axes.
        Each entry indicates the axis *normal* to the projection plane:
        - 'x' : Project onto the **YZ** plane at a fixed X offset.
        - 'y' : Project onto the **XZ** plane at a fixed Y offset.
        - 'z' : Project onto the **XY** plane at a fixed Z offset.
        - 'all' : Equivalent to `['x', 'y', 'z']`.
        If None, no contour projections are drawn.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `rasterized` : bool, default=`config.rasterized`
            Whether to rasterize plot artists. Rasterization
            converts the artist to a bitmap when saving to
            vector formats (e.g., PDF, SVG), which can
            significantly reduce file size for complex plots.
        - `color`, `c` : str, list of str or None, optional, default=`config.colors`
            Aliases for `colors`.
        - `sizes`, `s` : float or list of float, optional, default=`config.scatter_size`
            Aliases for `size`.
        - `markers`, `m` : str or list of str, optional, default=`config.marker`
            Aliases for `marker`.
        - `alphas`, `a` : float or list of float default=`config.alpha`
            Aliases for `alpha`.
        - `edgecolor`, `ec` : {'face', 'none', None}, color, list of color, or None, default=`config.edgecolor`
            Aliases for `edgecolors`.
        - `cmap` : str, optional, default=`config.cmap`
            Colormap to use if `colors` is not provided.
        - `xlim` : tuple of two floats or None
            Limits for the x-axis.
        - `ylim` : tuple of two floats or None
            Limits for the y-axis.
        - `zlim` : tuple of two floats or None
            Limits for the z-axis.
        - `plot_contour_offset` : float or sequence of float, optional, default=None
            Manual positional offsets for the contour projection planes.
            If a single float is given, the same offset is used for all projections.
            If a sequence is given (e.g., array-like), its length must match
            the number of entries in `plot_contours`, providing one offset per projection
            in the same order. If None, offsets are automatically chosen based
            on current axis limits (i.e., `ax.get_xlim()[0]`, `ax.get_ylim()[0]`,
            `ax.get_zlim()[0]`).
        - `xlabel` : str or None
            Label for the x-axis.
        - `ylabel` : str or None
            Label for the y-axis.
        - `zlabel` : str or None
            Label for the z-axis.
        - `minor_ticks` : bool, default=False
            If True, sets minor ticks for all axes.

    Returns
    -------
    scatter : `matplotlib.collections.Path3DCollection` or list of them
        The created scatter artist(s). Returns a single object
        if only one dataset is plotted.

    Raises
    ------
    ValueError
        If `X`, `Y`, and `Z` do not have the same number of arrays
        after unit consistency checks.

    Notes
    -----
    - The function cycles through `colors`, `sizes`, `markers`,
      `alphas`, and `edgecolors` if fewer values are given than
      datasets.
    - Pane backgrounds are set to white (`(1, 1, 1, 1)`).
    - Axis limits are applied in the order of `xlim`, `ylim`, `zlim`,
      and finally `scale` if provided.
    '''
    # ---- KWARGS ----
    rasterized = kwargs.get('rasterized', config.rasterized)
    # scatter params
    colors = _pop_kwargs(kwargs, 'color', 'c', default=colors)
    sizes = _pop_kwargs(kwargs, 'sizes', 's', default=size)
    markers = _pop_kwargs(kwargs, 'markers', 'm', default=marker)
    alphas = _pop_kwargs(kwargs, 'alphas', 'a', default=alpha)
    edgecolors = _pop_kwargs(kwargs, 'edgecolor', 'ec', default=edgecolors)
    cmap = kwargs.get('cmap', config.cmap)
    # figure params
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    zlim = kwargs.get('zlim', None)
    plot_contour_offset = kwargs.get('contour_offset', None)
    # labels
    xlabel = kwargs.get('xlabel', 'X')
    ylabel = kwargs.get('ylabel', 'Y')
    zlabel = kwargs.get('zlabel', 'Z')
    # ticks
    minor_ticks = kwargs.get('minor_ticks', False)

    # get default config values
    colors = _resolve_default(colors, config.colors)
    sizes = get_config_value(sizes, 'scatter_size')
    markers = get_config_value(markers, 'marker')
    alphas = get_config_value(alphas, 'alpha')
    edgecolors = config.edgecolor if edgecolors is _UNSET else edgecolors

    X = to_list(X)
    Y = to_list(Y)
    Z = to_list(Z)
    ensure_common_unit(X)
    ensure_common_unit(Y)
    ensure_common_unit(Z)

    if not (len(X) == len(Y) == len(Z)):
        raise ValueError(
            f'`x`, `y`, and `z` must have the same number of arrays '
            f'\n(got {len(X)}, {len(Y)}, and {len(Z)}).'
        )

    colors = get_colors(colors, cmap=cmap)
    sizes = sizes if isinstance(sizes, (list, np.ndarray, tuple)) else [sizes]
    markers = markers if isinstance(markers, (list, np.ndarray, tuple)) else [markers]
    alphas = alphas if isinstance(alphas, (list, np.ndarray, tuple)) else [alphas]
    edgecolors = edgecolors if isinstance(edgecolors, (list, np.ndarray, tuple)) else [edgecolors]

    ax.view_init(elev=elev, azim=azim, roll=roll)

    # set axes
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if zlim: ax.set_zlim(zlim)
    if scale is not None:
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_zlim(-scale, scale)

    scatters = []

    for i in range(len(X)):
        x = X[i%len(X)]
        y = Y[i%len(Y)]
        z = Z[i%len(Z)]

        color = colors[i%len(colors)]
        size = sizes[i%len(sizes)]
        marker = markers[i%len(markers)]
        alpha = alphas[i%len(alphas)]
        edgecolor = edgecolors[i%len(edgecolors)]

        sc = ax.scatter3D(x, y, z, c=color, s=size,
                          marker=marker, alpha=alpha,
                          edgecolors=edgecolor,
                          rasterized=rasterized)
        scatters.append(sc)

        # plot contours
        pairs = {
            'x': (Y, Z, 'x', lambda ax: ax.get_xlim()[0]),
            'y': (X, Z, 'y', lambda ax: ax.get_ylim()[0]),
            'z': (X, Y, 'z', lambda ax: ax.get_zlim()[0]),
        }
        if plot_contours is not None:
            if plot_contours.lower() == 'all': # type: ignore
                plot_contours = ['x', 'y', 'z']
            else:
                plot_contours = [c.lower() for c in plot_contours]

            if isinstance(plot_contour_offset, (list, tuple, np.ndarray)):
                if len(plot_contours) != len(plot_contour_offset):
                    raise ValueError(
                        'If `plot_contour_offset` is provided, it must have exactly '
                        'one offset per contour direction.'
                    )
            elif isinstance(plot_contour_offset, (float, int)):
                plot_contour_offset = [plot_contour_offset]*len(plot_contours)
            for i, zdir in enumerate(plot_contours):
                try:
                    data1, data2, axis_name, offset_fn = pairs[zdir.lower()]
                except KeyError:
                    raise ValueError(
                        f"Invalid contour projection '{zdir}'. "
                        "Use 'x', 'y', 'z', or 'all'."
                    )

                offset = offset_fn(ax) if plot_contour_offset is None else plot_contour_offset[i]
                contour(data1, data2, ax, zdir=axis_name, offset=offset)

    # set labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    # set border color
    border_color = (1.0, 1.0, 1.0, 1.0)
    ax.xaxis.set_pane_color(border_color)
    ax.yaxis.set_pane_color(border_color)
    ax.zaxis.set_pane_color(border_color)

    # hide gridlines
    if not grid_lines: ax.grid(False)
    if minor_ticks:
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='minor', direction='in', length=2, width=0.5)
    else:
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_minor_locator(NullLocator())
    if axes_off: ax.set_axis_off()

    scatters = scatters[0] if len(scatters) == 1 else scatters

    return scatters
