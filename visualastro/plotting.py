from astropy.visualization.wcsaxes.core import WCSAxes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from .data_cube import slice_cube
from .io import get_kwargs
from .numerical_utils import check_is_array, check_units_consistency, get_units
from .plot_utils import (
    add_colorbar, plot_circles, plot_ellipses,
    plot_interactive_ellipse, plot_points, return_array_values,
    return_imshow_norm, return_stylename, set_axis_limits,
    set_plot_colors, set_unit_labels, set_vmin_vmax,
)


# Plotting Functions
# ––––––––––––––––––
def imshow(datas, ax, idx=None, vmin=None, vmax=None, norm=None,
           percentile=[3,99.5], origin='lower', cmap='turbo',
           aspect=None, **kwargs):
    '''
    Display 2D image data with optional overlays and customization.
    Parameters
    ––––––––––
    datas : np.ndarray or list of np.ndarray
        Image array or list of image arrays to plot. Each array should
        be 2D (Ny, Nx) or 3D (Nz, Nx, Ny) if using 'idx' to slice a cube.
    ax : matplotlib.axes.Axes or WCSAxes
        Matplotlib axis on which to plot the image(s).
    idx : int or list of int, optional, default=None
        Index for slicing along the first axis if 'datas'
        contains a cube.
        - i -> returns cube[i]
        - [i] -> returns cube[i]
        - [i, j] -> returns the sum of cube[i:j+1] along axis 0
        If 'datas' is a list of cubes, you may also pass a list of
        indeces.
        ex: passing indeces for 2 cubes-> [[i,j], k].
    vmin, vmax : float, optional, default=None
        Lower and upper limits for colormap scaling. If not provided,
        values are determined from 'percentile'.
    norm : str, optional, default=None
        Normalization algorithm for colormap scaling.
        - 'asinh' -> AsinhStretch using 'ImageNormalize'
        - 'log' -> logarithmic scaling using 'LogNorm'
        - 'none' or None -> no normalization applied
    percentile : list of float, default=[3, 99.5]
        Default percentile range used to determine 'vmin' and 'vmax'.
    origin : str, {'upper', 'lower'}, default='lower'
        Pixel origin convention for imshow.
    cmap : str or list of str, default='turbo'
        Matplotlib colormap name or list of colormaps, cycled across images.
        ex: ['turbo', 'RdPu_r']
    aspect : str, {'auto', 'equal'} or float, optional, default=None
        Aspect ratio passed to imshow.

    **kwargs : dict, optional
        Additional plotting parameters.

        Supported keywords:

        - `invert_xaxis` : bool, optional, default=False
            Invert the x-axis if True.
        - `invert_yaxis` : bool, optional, default=False
            Invert the y-axis if True.
        - `text_loc` : list of float, optional, default=[0.03, 0.03]
            Relative axes coordinates for text placement when plotting interactive ellipses.
        - `text_color` : str, optional, default='k'
            Color of the ellipse annotation text.
        - `xlabel` : str, optional, default=None
            X-axis label.
        - `ylabel` : str, optional, default=None
            Y-axis label.
        - `colorbar` : bool, optional, default=True
            Add colorbar if True.
        - `clabel` : str or bool, optional, default=True
            Colorbar label. If True, use default label; if None or False, no label.
        - `cbar_width` : float, optional, default=0.03
            Width of the colorbar.
        - `cbar_pad` : float, optional, default=0.015
            Padding between plot and colorbar.
        - `circles` : list, optional, default=None
            List of Circle objects (e.g., `matplotlib.patches.Circle`) to overplot on the axes.
        - `ellipses` : list, optional, default=None
            List of Ellipse objects (e.g., `matplotlib.patches.Ellipse`) to overplot on the axes.
            Single Ellipse objects can also be passed directly.
        - `points` : array-like, shape (2,) or (N, 2), optional, default=None
            Coordinates of points to overplot. Can be a single point `[x, y]`
            or a list/array of points `[[x1, y1], [x2, y2], ...]`.
            Points are plotted as red stars by default.
        - `plot_ellipse` : bool, optional, default=False
            If True, plot an interactive ellipse overlay. Requires an interactive backend.
        - `center` : list of float, optional, default=[Nx//2, Ny//2]
            Center of the default interactive ellipse (x, y).
        - `w` : float, optional, default=X//5
            Width of the default interactive ellipse.
        - `h` : float, optional, default=Y//5
            Height of the default interactive ellipse.
    '''
    # –––– KWARGS ––––
    # figure params
    invert_xaxis = kwargs.get('invert_xaxis', False)
    invert_yaxis = kwargs.get('invert_yaxis', False)
    # labels
    text_loc = kwargs.get('text_loc', [0.03,0.03])
    text_color = kwargs.get('text_color', 'k')
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    colorbar = kwargs.get('colorbar', True)
    clabel = kwargs.get('clabel', True)
    cbar_width = kwargs.get('cbar_width', 0.03)
    cbar_pad = kwargs.get('cbar_pad', 0.015)
    # plot objects
    circles = kwargs.get('circles', None)
    points = kwargs.get('points', None)
    # plot ellipse
    ellipses = kwargs.get('ellipses', None)
    plot_ellipse = kwargs.get('plot_ellipse', False)
    # default ellipse parameters
    data = check_is_array(datas)
    X, Y = (data[0].shape if isinstance(datas, list) else data.shape)[-2:]
    center = kwargs.get('center', [X//2, Y//2])
    w = kwargs.get('w', X//5)
    h = kwargs.get('h', Y//5)

    # ensure inputs are iterable or conform to standard
    datas = check_units_consistency(datas)
    cmap = cmap if isinstance(cmap, (list, tuple)) else [cmap]
    if idx is not None:
        idx = idx if isinstance(idx, (list, tuple)) else [idx]

    # if wcsaxes are used, origin can only be 'lower'
    if isinstance(ax, WCSAxes) and origin == 'upper':
        origin = 'lower'
        invert_yaxis = True

    # loop over data list
    for i, data in enumerate(datas):
        # ensure data is an array
        data = check_is_array(data)
        # slice data with index if provided
        if idx is not None:
            data = slice_cube(data, idx[i%len(idx)])

        # set image stretch
        vmin, vmax = set_vmin_vmax(data, percentile, vmin, vmax)
        img_norm = return_imshow_norm(vmin, vmax, norm)

        # imshow image
        if img_norm is None:
            im = ax.imshow(data, origin=origin, vmin=vmin, vmax=vmax,
                           cmap=cmap[i%len(cmap)], aspect=aspect)
        else:
            im = ax.imshow(data, origin=origin, norm=img_norm,
                           cmap=cmap[i%len(cmap)], aspect=aspect)

    # overplot
    plot_circles(circles, ax)
    plot_points(points, ax)
    plot_ellipses(ellipses, ax)
    if plot_ellipse:
        plot_interactive_ellipse(center, w, h, ax, text_loc, text_color)

    # rotate tick labels
    if isinstance(ax, WCSAxes):
        ax.coords['ra'].set_axislabel('RA')
        ax.coords['ra'].set_axislabel_position('b')
        ax.coords['ra'].set_ticklabel(rotation=0)
        ax.coords['dec'].set_axislabel('DEC')
        ax.coords['dec'].set_axislabel_position('l')
        ax.coords['dec'].set_ticklabel(rotation=90)

    # set axes labels
    if isinstance(xlabel, str):
        ax.set_xlabel(xlabel)
    if isinstance(ylabel, str):
        ax.set_ylabel(ylabel)
    # add colorbar
    cbar_unit = set_unit_labels(get_units(datas[0]))
    if clabel is True:
        clabel = f'${cbar_unit}$' if cbar_unit is not None else None
    if colorbar:
        add_colorbar(im, ax, cbar_width, cbar_pad, clabel)
    # invert axes
    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()


def plot_density_histogram(X, Y, ax, ax_histx, ax_histy, bins='auto',
                           xlog=True, ylog=True, xlog_hist=None,
                           ylog_hist=None, histtype='step', colors=None,
                           **kwargs):
    # –––– KWARGS ––––
    colors = get_kwargs(kwargs, 'colors', 'color', 'c', default=colors)
    # scatter params
    sizes = get_kwargs(kwargs, 'size', 's', default=10)
    markers = get_kwargs(kwargs, 'marker', 'ms', default='o')
    alphas = get_kwargs(kwargs, 'alpha', 'a', default=1)
    edgecolors = get_kwargs(kwargs, 'edgecolors', 'edgecolor', 'ec', default=None)
    # line params
    linestyles = get_kwargs(kwargs, 'linestyles', 'linestyle', 'ls', default='-')
    linewidths = get_kwargs(kwargs, 'linewidth', 'lw', default=0.8)
    zorders = get_kwargs(kwargs, 'zorders', 'zorder', default=None)
    cmap = kwargs.get('cmap', 'turbo')
    # figure params
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    # labels
    labels = get_kwargs(kwargs, 'labels', 'label', 'l', default=None)
    loc = kwargs.get('loc', 'best')
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)

    X = check_units_consistency(X)
    Y = check_units_consistency(Y)
    if np.ndim(X) == 1 and np.ndim(Y) >= 2:
        X = [X]
    if np.ndim(Y) == 1 and np.ndim(X) >= 2:
        Y = [Y]

    if xlog_hist is None:
        xhistlog = xlog
    if ylog_hist is None:
        ylog_hist = ylog

    # configure scales and ticks
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if xlog: ax.set_xscale('log')
    if ylog: ax.set_yscale('log')
    if xlog_hist: ax_histx.set_xscale('log')
    if ylog_hist: ax_histy.set_yscale('log')

    ax.minorticks_on()
    # tick parameters for main plot
    ax.tick_params(axis='both', length=2, direction='in', which='both',
                    pad=5, right=True, top=True)
    # tick parameters for top histogram (x-axis)
    ax_histx.tick_params(axis='x', direction='in', which='both',
                            labelbottom=False, bottom=True)
    ax_histx.tick_params(axis='y', direction = 'in', which='both',
                            left=True, right=True, labelleft=True, pad=5)
    ax_histx.yaxis.set_label_position("left")
    # tick parameters for right histogram (y-axis)
    ax_histy.tick_params(axis='y', direction='in', which='both',
                            labelleft=False, left=True)
    ax_histy.tick_params(axis='x', direction = 'in', which='both',
                            bottom=True, top=True, labelbottom=True, pad=5)
    ax_histy.xaxis.set_label_position("bottom")
    # set plot colors
    colors, _ = set_plot_colors(colors, cmap=cmap)

    sizes = sizes if isinstance(sizes, (list, tuple)) else [sizes]
    markers = markers if isinstance(markers, (list, tuple)) else [markers]
    alphas = alphas if isinstance(alphas, (list, tuple)) else [alphas]
    edgecolors = edgecolors if isinstance(edgecolors, (list, tuple)) else [edgecolors]

    linestyles = linestyles if isinstance(linestyles, (list, tuple)) else [linestyles]
    linewidths = linewidths if isinstance(linewidths, (list, tuple)) else [linewidths]
    zorders = zorders if isinstance(zorders, (list, tuple)) else [zorders]
    labels = labels if isinstance(labels, (list, tuple)) else [labels]

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
        alpha = alphas[i%len(alphas)]
        zorder = zorders[i%len(zorders)] if zorders[i%len(zorders)] is not None else i
        label = labels[i] if (labels[i%len(labels)] is not None and i < len(labels)) else None

        ax.scatter(x, y, c=color, s=size, marker=marker,
                   alpha=alpha, edgecolors=edgecolor, label=label)
        # top histogram (x-axis)
        ax_histx.hist(x, bins=bins, color=color, histtype=histtype,
                      ls=linestyle, lw=linewidth, alpha=alpha,
                      zorder=zorder, density=True)
        # right histogram (y-axis)
        ax_histy.hist(y, bins=bins, orientation='horizontal',
                      color=color, histtype=histtype, ls=linestyle,
                      lw=linewidth, alpha=alpha, zorder=zorder,
                      density=True)

    if xlog_hist:
        ax_histx.set_ylabel('[Log]', labelpad=10)
    if ylog_hist:
        ax_histy.set_xlabel('[Log]', labelpad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels[0] is not None:
        ax.legend(loc=loc)


def plot_histogram(datas, ax, bins='auto', xlog=False,
                   ylog=False, histtype='step', colors=None, **kwargs):
    '''
    Plot one or more histograms on a given Axes object.
    Parameters
    ––––––––––
    datas : array-like or list of array-like
        Input data to histogram. Can be a single 1D array or a
        list of 1D/2D arrays. 2D arrays are automatically flattened.
    ax : matplotlib.axes.Axes
        The Axes object on which to plot the histogram.
    bins : int, sequence, or str, optional, default='auto'
        Number of bins or binning method. Passed to 'ax.hist'.
    xlog : bool, optional, default=False
        If True, set x-axis to logarithmic scale.
    ylog : bool, optional, Default=False
        If True, set y-axis to logarithmic scale.
    histtype : str, {'bar', 'barstacked', 'step', 'stepfilled'}, optional, default='step'
        Matplotlib histogram type.
    colors : list of colors or None, optional, default=None
        Colors to use for each dataset. If None, default
        color cycle is used.

    **kwargs : dict, optional
        Additional plotting parameters.

        Supported keywords:

        - `colors`, `color`, `c` : str, list of str or None, optional, default=None
            Colors to use for each line. If None, default color cycle is used.
        - `cmap` : str, optional, default='turbo'
            Colormap to use if `colors` is not provided.
        - `xlim` : tuple, optional
            X data range to display.
        - `ylim` : tuple, optional
            Y data range to display.
        - `labels`, `label`, `l` : str or list of str, default=None
            Legend labels.
        - `loc` : str, default='best'
            Location of legend.
        - `xlabel` : str or None, optional
            Label for the x-axis.
        - `ylabel` : str or None, optional
            Label for the y-axis.
    '''
    # –––– KWARGS ––––
    colors = get_kwargs(kwargs, 'colors', 'color', 'c', default=colors)
    cmap = kwargs.get('cmap', 'turbo')
    # figure params
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    # labels
    labels = get_kwargs(kwargs, 'labels', 'label', 'l', default=None)
    loc = kwargs.get('loc', 'best')
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)

    # ensure inputs are iterable or conform to standard
    datas = check_units_consistency(datas)
    labels = labels if isinstance(labels, (list, tuple)) else [labels]

    colors, _ = set_plot_colors(colors, cmap=cmap)
    data_list = []
    # loop over data list
    for i, data in enumerate(datas):
        color = colors[i%len(colors)]
        label = labels[i] if (labels[i%len(labels)] is not None and i < len(labels)) else None
        # ensure data is an array and is 1D
        data = check_is_array(data)
        if data.ndim == 2:
            data = data.flatten()
        data_list.append(data)
        ax.hist(data, bins=bins, color=color,
                histtype=histtype, label=label)

    # set axes parameters
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if xlog: ax.set_xscale('log')
    if ylog: ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels[0] is not None:
        ax.legend(loc=loc)


def plot_lines(X, Y, ax, normalize=False, xlog=False,
               ylog=False, colors=None, linestyle='-',
               linewidth=0.8, alpha=1, zorder=None, **kwargs):
    '''
    Plot one or more lines on a given Axes object with flexible styling.
    Parameters
    ––––––––––
    X : array-like or list of array-like
        x-axis data for the lines. Can be a single array or a list of arrays.
    Y : array-like or list of array-like
        y-axis data for the lines. Must match the length of X if lists are provided.
    ax : matplotlib.axes.Axes
        The Axes object to plot on.
    normalize : bool, optional, default=False
        If True, normalize each line to its maximum value.
    xlog : bool, optional, default=False
        If True, set the x-axis to logarithmic scale.
    ylog : bool, optional, default=False
        If True, set the y-axis to logarithmic scale.
    colors : str, list of str or None, optional, default=None
        Colors to use for each line. If None, default color cycle is used.
    linestyle : str or list of str, {'-', '--', '-.', ':', ''}, default='-'
        Line style of plotted lines.
    linewidth : float or list of float, optional, default=0.8
        Line width for the plotted lines.
    alpha : float or list of float default=None
        The alpha blending value, between 0 (transparent) and 1 (opaque).
    zorder : float or list of float, optional, default=None
        Order in which to plot lines in. Lines are drawn in order
        of greatest to lowest zorder. If None, starts at 0 and increments
        the zorder by 1 for each subsequent line drawn.

    **kwargs : dict, optional
        Additional plotting parameters.

        Supported keywords:

        - `colors`, `color`, `c` : str, list of str or None, optional, default=None
            Colors to use for each line. If None, default color cycle is used.
        - `linestyles`, `linestyle`, `ls` : str or list of str, {'-', '--', '-.', ':', ''}, default='-'
            Line style of plotted lines.
        - `linewidths`, `linewidth`, `lw` : float or list of float, optional, default=0.8
            Line width for the plotted lines.
        - `alphas`, `alpha`, `a` : float or list of float default=None
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        - `cmap` : str, optional, default='turbo'
            Colormap to use if `colors` is not provided.
        - `xlim` : tuple of two floats or None
            Limits for the x-axis.
        - `ylim` : tuple of two floats or None
            Limits for the y-axis.
        - `labels`, `label`, `l` : str or list of str, default=None
            Legend labels.
        - `loc` : str, default='best'
            Location of legend.
        - `xlabel` : str or None
            Label for the x-axis.
        - `ylabel` : str or None
            Label for the y-axis.
    '''
    # –––– KWARGS ––––
    colors = get_kwargs(kwargs, 'colors', 'color', 'c', default=colors)
    linestyles = get_kwargs(kwargs, 'linestyles', 'linestyle', 'ls', default=linestyle)
    linewidths = get_kwargs(kwargs, 'linewidth', 'lw', default=linewidth)
    alphas = get_kwargs(kwargs, 'alpha', 'a', default=1)
    cmap = kwargs.get('cmap', 'turbo')
    # figure params
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    # labels
    labels = get_kwargs(kwargs, 'labels', 'label', 'l', default=None)
    loc = kwargs.get('loc', 'best')
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)

    X = check_units_consistency(X)
    Y = check_units_consistency(Y)
    if np.ndim(X[0]) == 0:
        X = [X]
    if np.ndim(Y[0]) == 0:
        Y = [Y]

    colors, _ = set_plot_colors(colors, cmap=cmap)
    linestyles = linestyles if isinstance(linestyles, (list, tuple)) else [linestyles]
    linewidths = linewidths if isinstance(linewidths, (list, tuple)) else [linewidths]
    alphas = alphas if isinstance(alphas, (list, tuple)) else [alphas]
    zorders = zorder if isinstance(zorder, (list, tuple)) else [zorder]
    labels = labels if isinstance(labels, (list, tuple)) else [labels]

    y_list = []
    for i in range(len(Y)):
        x = X[i%len(X)]
        y = Y[i%len(Y)]
        color = colors[i%len(colors)]
        linestyle = linestyles[i%len(linestyles)]
        linewidth = linewidths[i%len(linewidths)]
        alpha = alphas[i%len(alphas)]
        zorder = zorders[i%len(zorders)] if zorders[i%len(zorders)] is not None else i
        label = labels[i] if (labels[i%len(labels)] is not None and i < len(labels)) else None

        if normalize:
            y = y / np.nanmax(y)
        y_list.append(y)

        ax.plot(x, y, c=color, ls=linestyle, lw=linewidth,
                alpha=alpha, zorder=zorder, label=label)

    # set axes parameters
    if xlog: ax.set_xscale('log')
    if ylog: ax.set_yscale('log')
    set_axis_limits(X, y_list, ax, xlim, ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels[0] is not None:
        ax.legend(loc=loc)


def scatter_plot(X, Y, ax, xerr=None, yerr=None, normalize=False,
                 xlog=False, ylog=False, colors=None, size=10,
                 marker='o', alpha=1, edgecolors='face', **kwargs):
    '''
    Plot one or more lines on a given Axes object with flexible styling.
    Parameters
    ––––––––––
    X : array-like or list of array-like
        x-axis data for the lines. Can be a single array or a list of arrays.
    Y : array-like or list of array-like
        y-axis data for the lines. Must match the length of X if lists are provided.
    ax : matplotlib.axes.Axes
        The Axes object to plot on.
    normalize : bool, optional, default=False
        If True, normalize each line to its maximum value.
    xlog : bool, optional, default=False
        If True, set the x-axis to logarithmic scale.
    ylog : bool, optional, default=False
        If True, set the y-axis to logarithmic scale.
    colors : list of str or None, optional, default=None
        Colors to use for each line. If None, default color cycle is used.
    size : float or list of float, optional, default=10
        Size of scatter dots.
    marker : str or list of str, optional, default='o'
        Marker style for scatter dots.
    alpha : float or list of float default=None
        The alpha blending value, between 0 (transparent) and 1 (opaque).
    edgecolors : {'face', 'none', None} or color or list of color, default='face'
        The edge color of the marker. Possible values:
        - 'face': The edge color will always be the same as the face color.
        - 'none': No patch boundary will be drawn.
        - A color or sequence of colors.

    **kwargs : dict, optional
        Additional plotting parameters.

        Supported keywords:

        - `colors`, `color`, `c` : str, list of str or None, optional, default=None
            Colors to use for each line. If None, default color cycle is used.
        - `sizes`, `size`, `s` : float or list of float, optional, default=10
            Size of scatter dots.
        - `markers`, `marker` : str or list of str, optional, default='o'
            Marker style for scatter dots.
        - `alphas`, `alpha`, `a` : float or list of float default=None
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        - `edgecolors`, `edgecolor`, `ec` : {'face', 'none', None} or color or list of color, default='face'
            The edge color of the marker.
        - `cmap` : str, optional, default='turbo'
            Colormap to use if `colors` is not provided.
        - `xlim` : tuple of two floats or None
            Limits for the x-axis.
        - `ylim` : tuple of two floats or None
            Limits for the y-axis.
        - `labels`, `label`, `l` : str or list of str, default=None
            Legend labels.
        - `loc` : str, default='best'
            Location of legend.
        - `xlabel` : str or None
            Label for the x-axis.
        - `ylabel` : str or None
            Label for the y-axis.
    '''
    # –––– KWARGS ––––
    # scatter params
    colors = get_kwargs(kwargs, 'colors', 'color', 'c', default=colors)
    sizes = get_kwargs(kwargs, 'size', 's', default=size)
    markers = get_kwargs(kwargs, 'marker', 'm', default=marker)
    alphas = get_kwargs(kwargs, 'alpha', 'a', default=alpha)
    edgecolors = get_kwargs(kwargs, 'edgecolors', 'edgecolor', 'ec', default=edgecolors)
    cmap = kwargs.get('cmap', 'turbo')
    # figure params
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    # labels
    labels = get_kwargs(kwargs, 'labels', 'label', 'l', default=None)
    loc = kwargs.get('loc', 'best')
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    # errorbars
    ecolors = get_kwargs(kwargs, 'ecolors', 'ecolor', default=None)
    elinewidth = kwargs.get('elinewidth', 1)
    capsize = kwargs.get('capsize', 1)
    capthick = kwargs.get('capthick', 1)
    barsabove = kwargs.get('barsabove', False)

    X = check_units_consistency(X)
    Y = check_units_consistency(Y)
    if np.ndim(X) == 1 and np.ndim(Y) >= 2:
        X = [X]
    if np.ndim(Y) == 1 and np.ndim(X) >= 2:
        Y = [Y]

    if xerr is not None:
        xerr = xerr if isinstance(xerr, (list, tuple)) else [xerr]
    if yerr is not None:
        yerr = yerr if isinstance(yerr, (list, tuple)) else [yerr]

    xerror, yerror = None, None
    colors, _ = set_plot_colors(colors, cmap=cmap)
    sizes = sizes if isinstance(sizes, (list, tuple)) else [sizes]
    markers = markers if isinstance(markers, (list, tuple)) else [markers]
    alphas = alphas if isinstance(alphas, (list, tuple)) else [alphas]
    edgecolors = edgecolors if isinstance(edgecolors, (list, tuple)) else [edgecolors]
    labels = labels if isinstance(labels, (list, tuple)) else [labels]
    ecolors = ecolors if isinstance(ecolors, (list, tuple)) else [ecolors]

    for i in range(len(Y)):
        x = X[i%len(X)]
        y = Y[i%len(Y)]
        color = colors[i%len(colors)]
        size = sizes[i%len(sizes)]
        marker = markers[i%len(markers)]
        alpha = alphas[i%len(alphas)]
        edgecolor = edgecolors[i%len(edgecolors)]
        ecolor = ecolors[i%len(ecolors)]
        label = labels[i] if (labels[i%len(labels)] is not None and i < len(labels)) else None

        if normalize:
            y = y / np.nanmax(y)

        ax.scatter(x, y, c=color, s=size, marker=marker,
                   alpha=alpha, edgecolors=edgecolor, label=label)

        if xerr is not None:
            xerror = xerr[i%len(xerr)]
        if yerr is not None:
            yerror = yerr[i%len(yerr)]

        ax.errorbar(x, y, yerror, xerror, fmt='None', ecolor=ecolor, elinewidth=elinewidth,
                    capsize=capsize, capthick=capthick, barsabove=barsabove)

    # set axes parameters
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if xlog: ax.set_xscale('log')
    if ylog: ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels[0] is not None:
        ax.legend(loc=loc)


# Figure Functions
# ––––––––––––––––
def panel_axes(nrows_ncols=[2,2], figsize=(6,6), sharex=False, sharey=False,
               hspace=None, wspace=None, fancy_axes=False, Nticks=4):
    Nx, Ny = nrows_ncols
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(Nx, Ny, hspace=hspace, wspace=wspace)
    axs = gs.subplots(sharex=sharex, sharey=sharey)
    axs = np.atleast_1d(axs).ravel()

    if fancy_axes:
        labeltop = [[True if i == 0 else False for j in range(Ny)] for i in range(Nx)]
        labelright = [[True if i == Ny-1 else False for i in range(Ny)] for j in range(Nx)]

    for i in range(Nx):
        for j in range(Ny):
            ax = axs[j + Ny*i]

            if fancy_axes:
                ax.minorticks_on()
                ax.tick_params(axis='both', length=2, direction='in',
                               which='both', labeltop=labeltop[i][j],
                               labelright=labelright[i][j],
                               right=True, top=True)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(Nticks))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(Nticks))
            ax.set_box_aspect(1)

    return axs
