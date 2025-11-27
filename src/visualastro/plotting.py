'''
Author: Elko Gerville-Reache
Date Created: 2025-09-22
Date Modified: 2025-10-20
Description:
    Plotting functions.
Dependencies:
    - astropy
    - matplotlib
    - numpy
Module Structure:
    - Plotting Functions
        Functions for general plots.
'''

from collections import namedtuple
from astropy.visualization.wcsaxes.core import WCSAxes
from matplotlib.ticker import AutoMinorLocator, NullLocator
import numpy as np
from .data_cube import slice_cube
from .io import get_kwargs
from .numerical_utils import check_is_array, check_units_consistency, get_units
from .plot_utils import (
    add_colorbar, add_contours, format_unit_labels,
    plot_circles, plot_ellipses,
    plot_interactive_ellipse, plot_points,
    return_imshow_norm, set_axis_limits,
    set_plot_colors, set_vmin_vmax
)
from .va_config import get_config_value, va_config, _default_flag


# Plotting Functions
# ––––––––––––––––––
def imshow(datas, ax, idx=None, vmin=_default_flag,
           vmax=_default_flag, norm=_default_flag,
           percentile=_default_flag, origin=None,
           cmap=None, aspect=_default_flag,
           mask_non_pos=None, wcs_grid=None,
           **kwargs):
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
    vmin : float or None, optional, default=`_default_flag`
        Lower limit for colormap scaling; overides `percentile[0]`.
        If None, values are determined from `percentile[0]`.
        If `_default_flag`, uses the default value in `va_config.vmin`.
    vmax : float or None, optional, default=`_default_flag`
        Upper limit for colormap scaling; overides `percentile[1]`.
        If None, values are determined from `percentile[1]`.
        If `_default_flag`, uses the default value in `va_config.vmax`.
    norm : str or None, optional, default=`_default_flag`
        Normalization algorithm for colormap scaling.
        - 'asinh' -> asinh stretch using 'ImageNormalize'
        - 'asinhnorm' -> asinh stretch using 'AsinhNorm'
        - 'log' -> logarithmic scaling using 'LogNorm'
        - 'powernorm' -> power-law normalization using 'PowerNorm'
        - 'linear', 'none', or None -> no normalization applied
        If `_default_flag`, uses the default value in `va_config.norm`.
    percentile : list or tuple of two floats, or None, default=`_default_flag`
        Default percentile range used to determine 'vmin' and 'vmax'.
        If `_default_flag`, uses default value from `va_config.percentile`.
        If None, use no percentile stretch.
    origin : {'upper', 'lower'} or None, default=None
        Pixel origin convention for imshow. If None,
        uses the default value from `va_config.origin`.
    cmap : str, list of str or None, default=None
        Matplotlib colormap name or list of colormaps, cycled across images.
        If None, uses the default value from `va_config.cmap`.
        ex: ['turbo', 'RdPu_r']
    aspect : {'auto', 'equal'}, float, or None, optional, default=`_default_flag`
        Aspect ratio passed to imshow, shortcut for `Axes.set_aspect`. 'auto'
        results in fixed axes with the aspect adjusted to fit the axes. 'equal`
        sets an aspect ratio of 1. None defaults to 'equal', however, if the
        image uses a transform that does not contain the axes data transform,
        then None means to not modify the axes aspect at all. If `_default_flag`,
        uses the default value from `va_config.aspect`.
    mask_non_pos : bool or None, optional, default=None
        If True, mask out non-positive data values. Useful for displaying
        log scaling of images with non-positive values. If None, uses the
        default value set by `va_config.mask_non_positive`.
    wcs_grid : bool or None, optional, default=None
        If True, display WCS grid ontop of plot. Requires
        using WCSAxes for `ax`. If None, uses the default
        value set by `va_config.wcs_grid`.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `rasterized` : bool, default=`va_config.rasterized`
            Whether to rasterize plot artists. Rasterization
            converts the artist to a bitmap when saving to
            vector formats (e.g., PDF, SVG), which can
            significantly reduce file size for complex plots.
        - `invert_xaxis` : bool, optional, default=False
            Invert the x-axis if True.
        - `invert_yaxis` : bool, optional, default=False
            Invert the y-axis if True.
        - `text_loc` : list of float, optional, default=`va_config.text_loc`
            Relative axes coordinates for text placement when
            plotting interactive ellipses.
        - `text_color` : str, optional, default=`va_config.text_color`
            Color of the ellipse annotation text.
        - `xlabel` : str, optional, default=None
            X-axis label.
        - `ylabel` : str, optional, default=None
            Y-axis label.
        - `colorbar` : bool, optional, default=`va_config.cbar`
            Add colorbar if True.
        - `clabel` : str or bool, optional, default=`va_config.clabel`
            Colorbar label. If True, use default label; if None or False, no label.
        - `cbar_width` : float, optional, default=`va_config.cbar_width`
            Width of the colorbar.
        - `cbar_pad` : float, optional, default=`va_config.cbar_pad`
            Padding between plot and colorbar.
        - `mask_out_val` : float, optional, default=`va_config.mask_out_value`
            Value to use when masking out non-positive values.
            Ex: np.nan, 1e-6, np.inf
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
    Returns
    –––––––
    images : matplotlib.image.AxesImage or list of matplotlib.image.AxesImage
            Image object if a single array is provided, otherwise a list of image
            objects created by `imshow`.
    '''
    # –––– KWARGS ––––
    # figure params
    rasterized = kwargs.get('rasterized', va_config.rasterized)
    invert_xaxis = kwargs.get('invert_xaxis', False)
    invert_yaxis = kwargs.get('invert_yaxis', False)
    # labels
    text_loc = kwargs.get('text_loc', va_config.text_loc)
    text_color = kwargs.get('text_color', va_config.text_color)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    colorbar = kwargs.get('colorbar', va_config.cbar)
    clabel = kwargs.get('clabel', va_config.clabel)
    cbar_width = kwargs.get('cbar_width', va_config.cbar_width)
    cbar_pad = kwargs.get('cbar_pad', va_config.cbar_pad)
    # mask out value
    mask_out_val = kwargs.get('mask_out_val', va_config.mask_out_value)
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

    # get default va_config values
    vmin = va_config.vmin if vmin is _default_flag else vmin
    vmax = va_config.vmax if vmax is _default_flag else vmax
    norm = va_config.norm if norm is _default_flag else norm
    percentile = va_config.percentile if percentile is _default_flag else percentile
    origin = get_config_value(origin, 'origin')
    cmap = get_config_value(cmap, 'cmap')
    aspect = va_config.aspect if aspect is _default_flag else aspect
    mask_non_pos = get_config_value(mask_non_pos, 'mask_non_positive')
    wcs_grid = get_config_value(wcs_grid, 'wcs_grid')

    # ensure inputs are iterable or conform to standard
    datas = check_units_consistency(datas)
    cmap = cmap if isinstance(cmap, (list, np.ndarray, tuple)) else [cmap]
    if idx is not None:
        idx = idx if isinstance(idx, (list, np.ndarray, tuple)) else [idx]

    # if wcsaxes are used, origin can only be 'lower'
    if isinstance(ax, WCSAxes) and origin == 'upper':
        origin = 'lower'
        invert_yaxis = True

    images = []

    # loop over data list
    for i, data in enumerate(datas):
        # ensure data is an array
        data = check_is_array(data)
        # slice data with index if provided
        if idx is not None:
            data = slice_cube(data, idx[i%len(idx)])

        if mask_non_pos:
            data = np.where(data > 0.0, data, mask_out_val)

        # set image stretch
        vmin, vmax = set_vmin_vmax(data, percentile, vmin, vmax)
        img_norm = return_imshow_norm(vmin, vmax, norm, **kwargs)

        # imshow image
        if img_norm is None:
            im = ax.imshow(data, origin=origin, vmin=vmin,
                           vmax=vmax, cmap=cmap[i%len(cmap)],
                           aspect=aspect, rasterized=rasterized)
        else:
            im = ax.imshow(
                data,
                origin=origin,
                norm=img_norm,
                cmap=cmap[i%len(cmap)],
                aspect=aspect,
                rasterized=rasterized
            )

        images.append(im)

    # overplot
    plot_circles(circles, ax)
    plot_points(points, ax)
    plot_ellipses(ellipses, ax)
    if plot_ellipse:
        plot_interactive_ellipse(center, w, h, ax, text_loc, text_color)

    # invert axes
    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()

    # rotate tick labels and set optional grid
    if isinstance(ax, WCSAxes):
        for coord in ax.coords:
            coord_type = coord.coord_type
            coord_index = coord.coord_index

            # set label based on coordinate type
            if coord_type == 'longitude':
                coord.set_axislabel(va_config.right_ascension)
            elif coord_type == 'latitude':
                coord.set_axislabel(va_config.declination)

            # determine which pixel axis this world coordinate
            # primarily affects by checking the PC/CD matrix
            wcs = ax.wcs
            if hasattr(wcs, 'pixel_to_world_values'):
                # check which pixel axis (0=x, 1=y) this world axis varies most with
                # look at the absolute values in the transformation matrix
                if hasattr(wcs.wcs, 'pc'):
                    pc_matrix = wcs.wcs.pc
                elif hasattr(wcs.wcs, 'cd'):
                    pc_matrix = wcs.wcs.cd
                else:
                    pc_matrix = None

                if pc_matrix is not None:
                    # for this world axis, see which pixel axis it affects most
                    world_axis_row = pc_matrix[coord_index, :]
                    dominant_pixel_axis = abs(world_axis_row).argmax()

                    # dominant_pixel_axis: 0 = x-axis (horizontal), 1 = y-axis (vertical)
                    if dominant_pixel_axis == 0:
                        coord.set_axislabel_position('b')
                        coord.set_ticklabel(rotation=0)
                    elif dominant_pixel_axis == 1:
                        coord.set_axislabel_position('l')
                        coord.set_ticklabel(rotation=90)

        if wcs_grid:
            ax.coords.grid(True, color='white', ls='dotted')

    # set axes labels
    if isinstance(xlabel, str):
        ax.set_xlabel(xlabel)
    if isinstance(ylabel, str):
        ax.set_ylabel(ylabel)
    # add colorbar
    cbar_unit = format_unit_labels(get_units(datas[0]))
    if clabel is True:
        clabel = cbar_unit if cbar_unit is not None else None
    if colorbar:
        add_colorbar(im, ax, cbar_width, cbar_pad, clabel)

    images = images[0] if len(images) == 1 else images

    return images


def plot_density_histogram(X, Y, ax, ax_histx, ax_histy, bins=None,
                           xlog=None, ylog=None, xlog_hist=None,
                           ylog_hist=None, histtype=None,
                           normalize=None, colors=None, **kwargs):
    '''
    Plot a 2D scatter distribution with normalized density histograms.
    This function creates a scatter plot of `X` vs. `Y` along
    with normalizable histograms of `X` and `Y`.
    Parameters
    ––––––––––
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
        value from `va_config.bins`. If `bins` is a str, use
        one of the supported binning strategies 'auto', 'fd',
        'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
    xlog : bool or None, optional, default=None
        Whether to use a logarithmic x-axis scale for the scatter plot.
        If None, uses the default value from `va_config.xlog`.
    ylog : bool or None, optional, default=None
        Whether to use a logarithmic y-axis scale for the scatter plot.
        If None, uses the default value from `va_config.ylog`.
    xlog_hist : bool or None, optional, default=None
        Whether to use a logarithmic x-axis scale for the top histogram.
        If None, uses the default value from `va_config.xlog_hist`.
    ylog_hist : bool or None, optional, default=None
        Whether to use a logarithmic y-axis scale for the right histogram.
        If None, uses the default value from `va_config.ylog_hist`.
    histtype : {'bar', 'barstacked', 'step', 'stepfilled'} or None, optional, default=None
        Type of histogram to draw. If None, uses the default value from `va_config.histtype`.
    normalize : bool, optional, default=None
        If True, normalize histograms to a probability density.
        If None, uses the default value from `va_config.normalize_hist`.
    colors : list of colors, str, or None, optional, default=None
        Colors for each dataset. If None, uses the
        default color palette from `va_config.default_palette`.

    **kwargs : dict, optional
        Additional parameters.

        Supported keyword arguments include:

        - `rasterized` : bool, default=`va_config.rasterized`
            Whether to rasterize plot artists. Rasterization
            converts the artist to a bitmap when saving to
            vector formats (e.g., PDF, SVG), which can
            significantly reduce file size for complex plots.
        - `color`, `c` : list of colors, str, or None, optional, default=None
            aliases for `colors`.
        - `sizes`, `size`, `s` : float or list, optional, default=`va_config.scatter_size`
            Marker size(s) for scatter points.
        - `markers`, `marker`, `m` : str or list, optional, default=`va_config.marker`
            Marker style(s) for scatter points.
        - `alphas`, `alpha`, `a` : float or list, optional, default=`va_config.alpha`
            Transparency level(s).
        - `edgecolors`, `edgecolor`, `ec` : str or list, optional, default=`va_config.edgecolor`
            Edge colors for scatter points.
        - `linestyles`, `linestyle`, `ls` : str or list, optional, default=`va_config.linestyle`
            Line style(s) for histogram edges.
        - `linewidth`, `lw` : float or list, optional, default=`va_config.linewidth`
            Line width(s) for histogram edges.
        - `zorders`, `zorder` : int or list, optional, default=None
            Z-order(s) for drawing priority.
        - `cmap` : str, optional, default=`va_config.cmap`
            Colormap name for automatic color assignment.
        - `xlim`, `ylim` : tuple, optional, default=None
            Axis limits for the scatter plot.
        - `labels`, `label`, `l` : list or str, optional, default=None
            Labels for legend entries.
        - `loc` : str, optional, default=`va_config.loc`
            Legend location.
        - `xlabel`, `ylabel` : str, optional, default=None
            Axis labels for the scatter plot.
    Returns
    –––––––
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
    # –––– KWARGS ––––
    rasterized = kwargs.get('rasterized', va_config.rasterized)
    colors = get_kwargs(kwargs, 'color', 'c', default=colors)
    # scatter params
    sizes = get_kwargs(kwargs, 'size', 's', default=None)
    markers = get_kwargs(kwargs, 'marker', 'm', default=None)
    alphas = get_kwargs(kwargs, 'alpha', 'a', default=None)
    edgecolors = get_kwargs(kwargs, 'edgecolors', 'edgecolor', 'ec', default=None)
    # line params
    linestyles = get_kwargs(kwargs, 'linestyles', 'linestyle', 'ls', default=None)
    linewidths = get_kwargs(kwargs, 'linewidth', 'lw', default=None)
    zorders = get_kwargs(kwargs, 'zorders', 'zorder', default=None)
    cmap = kwargs.get('cmap', va_config.cmap)
    # figure params
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    # labels
    labels = get_kwargs(kwargs, 'labels', 'label', 'l', default=None)
    loc = kwargs.get('loc', va_config.loc)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)

    # get default va_config values
    bins = get_config_value(bins, 'bins')
    xlog = get_config_value(xlog, 'xlog')
    ylog = get_config_value(ylog, 'ylog')
    xlog_hist = get_config_value(xlog_hist, 'xlog_hist')
    ylog_hist = get_config_value(ylog_hist, 'ylog_hist')
    histtype = get_config_value(histtype, 'histtype')
    normalize = get_config_value(normalize, 'normalize_hist')
    colors = get_config_value(colors, 'colors')
    sizes = get_config_value(sizes, 'scatter_size')
    markers = get_config_value(markers, 'marker')
    alphas = get_config_value(alphas, 'alpha')
    edgecolors = get_config_value(edgecolors, 'edgecolor')
    linestyles = get_config_value(linestyles, 'linestyle')
    linewidths = get_config_value(linewidths, 'linewidth')

    X = check_units_consistency(X)
    Y = check_units_consistency(Y)
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
    colors, _ = set_plot_colors(colors, cmap=cmap)

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


def plot_histogram(datas, ax,
                   bins=None,
                   xlog=None,
                   ylog=None,
                   histtype=None,
                   normalize=None,
                   colors=None,
                   **kwargs):
    '''
    Plot one or more histograms on a given Axes object.
    Parameters
    ––––––––––
    datas : array-like or list of array-like
        Input data to histogram. Can be a single 1D array or a
        list of 1D/2D arrays. 2D arrays are automatically flattened.
    ax : matplotlib.axes.Axes
        The Axes object on which to plot the histogram.
    bins : int, sequence, str, or None, optional, default=None
        Histogram bin specification. Passed directly to
        `matplotlib.pyplot.hist`. If None, uses the default
        value from `va_config.bins`. If `bins` is a str, use
        one of the supported binning strategies 'auto', 'fd',
        'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
    xlog : bool or None, optional, default=None
        If True, set x-axis to logarithmic scale.
        If None, uses the default value from `va_config.xlog`.
    ylog : bool or None, optional, default=None
        If True, set y-axis to logarithmic scale.
        If None, uses the default value from `va_config.ylog`.
    histtype : {'bar', 'barstacked', 'step', 'stepfilled'} or None, optional, default=None
        Matplotlib histogram type. If None, uses the default value from `va_config.histtype`.
    normalize : bool or None, optional, default=None
        If True, normalize histograms to a probability density.
        If None, uses the default value from `va_config.normalize_hist`.
    colors : list of colors, str, or None, optional, default=None
        Colors to use for each dataset. If None,
        uses the default color palette from `va_config.default_palette`.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `rasterized` : bool, default=`va_config.rasterized`
            Whether to rasterize plot artists. Rasterization
            converts the artist to a bitmap when saving to
            vector formats (e.g., PDF, SVG), which can
            significantly reduce file size for complex plots.
        - `color`, `c` : str, list of str or None, optional, default=`va_config.colors`.
            Aliases for `colors`.
        - `cmap` : str, optional, default=`va_config.cmap`
            Colormap to use if `colors` is not provided.
        - `xlim` : tuple, optional
            X data range to display.
        - `ylim` : tuple, optional
            Y data range to display.
        - `labels`, `label`, `l` : str or list of str, default=None
            Legend labels.
        - `loc` : str, default=`va_config.loc`
            Location of legend.
        - `xlabel` : str or None, optional
            Label for the x-axis.
        - `ylabel` : str or None, optional
            Label for the y-axis.

    Returns
    –––––––
    hists : tuple or list of tuple
        The result(s) returned by `Axes.hist`. Each tuple has the form
        `(n, bins, patches)`, where:

        - `n` : ndarray
          The values of the histogram (counts or densities).
        - `bins` : ndarray
          The edges of the bins (length = len(n) + 1).
        - `patches` : list or list-like of Patch
          The artists created by the histogram (e.g., `Rectangle` or `Polygon`
          objects depending on `histtype`).

        If only one histogram is created, `hists` is a single tuple; otherwise,
        it is a list of tuples.
    '''
    # –––– KWARGS ––––
    rasterized = kwargs.get('rasterized', va_config.rasterized)
    colors = get_kwargs(kwargs, 'color', 'c', default=colors)
    cmap = kwargs.get('cmap', va_config.cmap)
    # figure params
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    # labels
    labels = get_kwargs(kwargs, 'labels', 'label', 'l', default=None)
    loc = kwargs.get('loc', va_config.loc)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)

    # get default va_config values
    bins = get_config_value(bins, 'bins')
    xlog = get_config_value(xlog, 'xlog')
    ylog = get_config_value(ylog, 'ylog')
    histtype = get_config_value(histtype, 'histtype')
    normalize = get_config_value(normalize, 'normalize_hist')
    colors = get_config_value(colors, 'colors')

    # ensure inputs are iterable or conform to standard
    datas = check_units_consistency(datas)
    labels = labels if isinstance(labels, (list, np.ndarray, tuple)) else [labels]

    colors, _ = set_plot_colors(colors, cmap=cmap)
    data_list = []

    # set axes
    if xlog: ax.set_xscale('log')
    if ylog: ax.set_yscale('log')
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    hists = []

    # loop over data list
    for i, data in enumerate(datas):
        color = colors[i%len(colors)]
        label = labels[i] if (labels[i%len(labels)] is not None and i < len(labels)) else None
        # ensure data is an array and is 1D
        data = check_is_array(data)
        if data.ndim == 2:
            data = data.flatten()
        data_list.append(data)
        h = ax.hist(
            data,
            bins=bins,
            color=color,
            histtype=histtype,
            density=normalize,
            label=label,
            rasterized=rasterized
        )

        hists.append(h)

    # set axes labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels[0] is not None:
        ax.legend(loc=loc)

    hists = hists[0] if len(hists) == 1 else hists

    return hists


def plot_lines(X, Y, ax, normalize=None,
               xlog=None, ylog=None,
               colors=None, linestyle=None,
               linewidth=None, alpha=None,
               zorder=None, **kwargs):
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
    normalize : bool or None, optional, default=None
        If True, normalize each line by its maximum value.
        If None, uses the default value from `va_config.normalize_data`.
    xlog : bool or None, optional, default=None
        If True, set the x-axis to logarithmic scale.
        If None, uses the default value from `va_config.xlog`.
    ylog : bool or None, optional, default=None
        If True, set the y-axis to logarithmic scale.
        If None, uses the default value from `va_config.ylog`.
    colors : list of colors, str, or None, optional, default=None
        Colors to use for each line. If None, uses the
        default color palette from `va_config.default_palette`.
    linestyle : str, list of str, or None, optional, default=None
        Line style(s) to use for plotting. Can be a single string or a list of
        styles for multiple lines. Accepted values are:
        {'-', '--', '-.', ':', ''}. If None, uses the default
        value set in `va_config.linestyle`.
    linewidth : float, list of float, or None, optional, default=None
        Line width for the plotted lines. If None, uses the
        default value set in `va_config.linewidth`.
    alpha : float, list of float or None, optional, default=None
        The alpha blending value, between 0 (transparent) and 1 (opaque).
        If None, uses the default value set in `va_config.alpha`.
    zorder : float or list of float, optional, default=None
        Order in which to plot lines in. Lines are drawn in order
        of greatest to lowest zorder. If None, starts at 0 and increments
        the zorder by 1 for each subsequent line drawn.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `rasterized` : bool, default=`va_config.rasterized`
            Whether to rasterize plot artists. Rasterization
            converts the artist to a bitmap when saving to
            vector formats (e.g., PDF, SVG), which can
            significantly reduce file size for complex plots.
        - `color`, `c` : str, list of str or None, optional, default=`va_config.colors`
            Aliases for `colors`.
        - `linestyles`, `ls` : str or list of str, default=`va_config.linestyle`
            Aliases for `linestyle`.
        - `linewidths`, `lw` : float or list of float, optional, default=`va_config.linewidth`
            Aliases for `linewidth`.
        - `alphas`, `a` : float or list of float, default=`va_config.alpha`
            Aliases for `alpha`.
        - `cmap` : str, optional, default=`va_config.cmap`
            Colormap to use if `colors` is not provided.
        - `xlim` : tuple of two floats or None
            Limits for the x-axis.
        - `ylim` : tuple of two floats or None
            Limits for the y-axis.
        - `labels`, `label`, `l` : str or list of str, default=None
            Legend labels.
        - `loc` : str, default=`va_config.loc`
            Location of legend.
        - `xlabel` : str or None
            Label for the x-axis.
        - `ylabel` : str or None
            Label for the y-axis.
        - `xpad`/`ypad` : float
            padding along x and y axis used when computing
            axis limits. Defined as:
                xmax/min ±= xpad * (xmax - xmin)
                ymax/min ±= ypad * (ymax - ymin)

    Returns
    –––––––
    lines : Line2D or list of Line2D
        The line object(s) created by `Axes.plot`. Each element is a
        `matplotlib.lines.Line2D` instance representing one plotted line.
        If only one line is created, `lines` is a single `Line2D` object;
        otherwise, it is a list of `Line2D` objects.
    '''
    # –––– KWARGS ––––
    rasterized = kwargs.get('rasterized', va_config.rasterized)
    colors = get_kwargs(kwargs, 'color', 'c', default=colors)
    linestyles = get_kwargs(kwargs, 'linestyles', 'ls', default=linestyle)
    linewidths = get_kwargs(kwargs, 'linewidths', 'lw', default=linewidth)
    alphas = get_kwargs(kwargs, 'alphas', 'a', default=alpha)
    cmap = kwargs.get('cmap', va_config.cmap)
    # figure params
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    # labels
    labels = get_kwargs(kwargs, 'labels', 'label', 'l', default=None)
    loc = kwargs.get('loc', va_config.loc)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    # axes
    xpad = kwargs.get('xpad', 0.0)
    ypad = kwargs.get('ypad', 0.0)

    # get default va_config values
    normalize = get_config_value(normalize, 'normalize_data')
    xlog = get_config_value(xlog, 'xlog')
    ylog = get_config_value(ylog, 'ylog')
    colors = get_config_value(colors, 'colors')
    linestyles = get_config_value(linestyles, 'linestyle')
    linewidths = get_config_value(linewidths, 'linewidth')
    alphas = get_config_value(alphas, 'alpha')

    X = check_units_consistency(X)
    Y = check_units_consistency(Y)
    if np.ndim(X[0]) == 0:
        X = [X]
    if np.ndim(Y[0]) == 0:
        Y = [Y]

    colors, _ = set_plot_colors(colors, cmap=cmap)
    linestyles = linestyles if isinstance(linestyles, (list, np.ndarray, tuple)) else [linestyles]
    linewidths = linewidths if isinstance(linewidths, (list, np.ndarray, tuple)) else [linewidths]
    alphas = alphas if isinstance(alphas, (list, np.ndarray, tuple)) else [alphas]
    zorders = zorder if isinstance(zorder, (list, np.ndarray, tuple)) else [zorder]
    labels = labels if isinstance(labels, (list, np.ndarray, tuple)) else [labels]

    if xlog: ax.set_xscale('log')
    if ylog: ax.set_yscale('log')

    lines = []
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

        l = ax.plot(x, y, c=color, ls=linestyle,
                    lw=linewidth, alpha=alpha,
                    zorder=zorder, label=label,
                    rasterized=rasterized)

        lines.append(l)

    # set axes parameters
    set_axis_limits(X, y_list, ax, xlim, ylim, xpad=xpad, ypad=ypad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels[0] is not None:
        ax.legend(loc=loc)

    lines = lines[0] if len(lines) == 1 else lines

    return lines


def plot_scatter(X, Y, ax, xerr=None, yerr=None, normalize=None,
                 xlog=None, ylog=None, colors=None, size=None,
                 marker=None, alpha=None, edgecolors=_default_flag,
                 facecolors=_default_flag, **kwargs):
    '''
    Plot a scatter plot (optionally with error bars) on a given Axes object.
    Parameters
    ––––––––––
    X : array-like or list of array-like
        x-axis data for the lines. Can be a single array or a list of arrays.
    Y : array-like or list of array-like
        y-axis data for the lines. Must match the length of X if lists are provided.
    ax : matplotlib.axes.Axes
        The Axes object to plot on.
    xerr : array-like or list of array-like, optional, default=None
        x-axis errors on `X`. Should be same shape as `X`.
    yerr : array-like or list of array-like, optional, default=None
        x-axis errors on `Y`. Should be same shape as `Y`.
    normalize : bool or None, optional, default=None
        If True, normalize each line by its maximum value.
        If None, uses the default value from `va_config.normalize_data`.
    xlog : bool or None, optional, default=None
        If True, set the x-axis to logarithmic scale. If
        None, uses the default value in `va_config.xlog`.
    ylog : bool or None, optional, default=None
        If True, set the y-axis to logarithmic scale. If
        None, uses the default value in `va_config.ylog`.
    colors : list of colors, str, or None, optional, default=None
        Colors to use for each scatter group or dataset.
        If None, uses the default color palette from
        `va_config.default_palette`.
    size : float, list of float, or None, optional, default=None
        Size of scatter dots. If None, uses the default
        value in `va_config.scatter_size`.
    marker : str, list of str, or None, optional, default=None
        Marker style for scatter dots. If None, uses the
        default value in `va_config.marker`.
    alpha : float, list of float, or None, default=None
        The alpha blending value, between 0 (transparent) and 1 (opaque).
        If None, uses the default value from `va_config.alpha`.
    edgecolors : {'face', 'none'}, color, list of color, or None, default=`_default_flag`
        The edge color of the marker. Possible values:
        - 'face': The edge color will always be the same as the face color.
        - 'none': No patch boundary will be drawn.
        - A color or sequence of colors.
        - None: no edgecolor is set (edgecolor is set to marker color).
        If `_default_flag`, uses the default value in `va_config.edgecolor`.
    facecolors : {'none'}, color, list of colors, or None, default=`_default_flag`
        The face color of the marker. Possible values:
        - 'none': Sets the face color to transparent
        - A color or sequence of colors
        - None: No facecolor is set (facecolor is set to marker color).
        If `_default_flag`, uses the default value in `va_config.facecolor`.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `rasterized` : bool, default=`va_config.rasterized`
            Whether to rasterize plot artists. Rasterization
            converts the artist to a bitmap when saving to
            vector formats (e.g., PDF, SVG), which can
            significantly reduce file size for complex plots.
        - `color`, `c` : str, list of str or None, optional, default=`va_config.colors`
            Aliases for `colors`.
        - `sizes`, `s` : float or list of float, optional, default=`va_config.scatter_size`
            Aliases for `size`.
        - `markers`, `m` : str or list of str, optional, default=`va_config.marker`
            Aliases for `marker`.
        - `alphas`, `a` : float or list of float default=`va_config.alpha`
            Aliases for `alpha`.
        - `edgecolor`, `ec` : {'face', 'none', None}, color, list of color, or None, default=`va_config.edgecolor`
            Aliases for `edgecolors`.
        - `facecolor`, `fc` : {'none'}, color, list of colors, or None, default=`_default_flag`
            Aliases for `facecolors`.
        - `cmap` : str, optional, default=`va_config.cmap`
            Colormap to use if `colors` is not provided.
        - `xlim` : tuple of two floats or None
            Limits for the x-axis.
        - `ylim` : tuple of two floats or None
            Limits for the y-axis.
        - `labels`, `label`, `l` : str or list of str, default=None
            Legend labels.
        - `loc` : str, default=`va_config.loc`
            Location of legend.
        - `xlabel` : str or None
            Label for the x-axis.
        - `ylabel` : str or None
            Label for the y-axis.
        - `ecolors`, `ecolor` : color or list of color, optional, default=`va_config.ecolors`
            Color(s) of the error bars.
        - `elinewidth` : float, default=`va_config.elinewidth`
            Line width of the error bars.
        - `capsize` : float, default=`va_config.capsize`
            Length of the error bar caps in points.
        - `capthick` : float, default=`va_config.capthick`
            Thickness of the error bar caps in points.
        - `barsabove` : bool, default=`va_config.barsabove`
            If True, draw error bars above the plot symbols; otherwise, below.

    Returns
    –––––––
    lines : PathCollection or list of PathCollection
        The scatter plot object(s) created by `Axes.scatter`. Each element
        is a `matplotlib.collections.PathCollection` instance representing
        one scatter plot. If only one scatter is created, `lines` is a single
        `PathCollection`; otherwise, it is a list of `PathCollection` objects.
    '''
    # –––– KWARGS ––––
    rasterized = kwargs.get('rasterized', va_config.rasterized)
    # scatter params
    colors = get_kwargs(kwargs, 'color', 'c', default=colors)
    sizes = get_kwargs(kwargs, 'sizes', 's', default=size)
    markers = get_kwargs(kwargs, 'markers', 'm', default=marker)
    alphas = get_kwargs(kwargs, 'alphas', 'a', default=alpha)
    edgecolors = get_kwargs(kwargs, 'edgecolor', 'ec', default=edgecolors)
    facecolors = get_kwargs(kwargs, 'facecolor', 'fc', default=facecolors)
    cmap = kwargs.get('cmap', va_config.cmap)
    # figure params
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    # labels
    labels = get_kwargs(kwargs, 'labels', 'label', 'l', default=None)
    loc = kwargs.get('loc', va_config.loc)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    # errorbars
    ecolors = get_kwargs(kwargs, 'ecolors', 'ecolor', default=None)
    elinewidth = kwargs.get('elinewidth', va_config.elinewidth)
    capsize = kwargs.get('capsize', va_config.capsize)
    capthick = kwargs.get('capthick', va_config.capthick)
    barsabove = kwargs.get('barsabove', va_config.barsabove)

    # get default va_config values
    xlog = get_config_value(xlog, 'xlog')
    ylog = get_config_value(ylog, 'ylog')
    colors = get_config_value(colors, 'colors')
    sizes = get_config_value(sizes, 'scatter_size')
    markers = get_config_value(markers, 'marker')
    alphas = get_config_value(alphas, 'alpha')
    edgecolors = va_config.edgecolor if edgecolors is _default_flag else edgecolors
    facecolors = va_config.facecolor if facecolors is _default_flag else facecolors
    ecolors = get_config_value(ecolors, 'ecolors')

    X = check_units_consistency(X)
    Y = check_units_consistency(Y)
    if np.ndim(X) == 1 and np.ndim(Y) >= 2:
        X = [X]
    if np.ndim(Y) == 1 and np.ndim(X) >= 2:
        Y = [Y]

    if xerr is not None:
        xerr = xerr if isinstance(xerr, (list, np.ndarray, tuple)) else [xerr]
    if yerr is not None:
        yerr = yerr if isinstance(yerr, (list, np.ndarray, tuple)) else [yerr]

    xerror, yerror = None, None
    colors, _ = set_plot_colors(colors, cmap=cmap)
    sizes = sizes if isinstance(sizes, (list, np.ndarray, tuple)) else [sizes]
    markers = markers if isinstance(markers, (list, np.ndarray, tuple)) else [markers]
    alphas = alphas if isinstance(alphas, (list, np.ndarray, tuple)) else [alphas]
    edgecolors = edgecolors if isinstance(edgecolors, (list, np.ndarray, tuple)) else [edgecolors]
    facecolors = facecolors if isinstance(facecolors, (list, np.ndarray, tuple)) else [facecolors]
    labels = labels if isinstance(labels, (list, np.ndarray, tuple)) else [labels]
    ecolors = ecolors if isinstance(ecolors, (list, np.ndarray, tuple)) else [ecolors]

    # set axes
    if xlog: ax.set_xscale('log')
    if ylog: ax.set_yscale('log')
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    scatters = []

    for i in range(len(Y)):
        x = X[i%len(X)]
        y = Y[i%len(Y)]
        color = colors[i%len(colors)]
        size = sizes[i%len(sizes)]
        marker = markers[i%len(markers)]
        alpha = alphas[i%len(alphas)]
        edgecolor = edgecolors[i%len(edgecolors)]
        facecolor = facecolors[i%len(facecolors)]
        ecolor = ecolors[i%len(ecolors)]
        label = labels[i] if (labels[i%len(labels)] is not None and i < len(labels)) else None

        if facecolor == 'none' and edgecolor is None:
            edgecolor = color

        if normalize:
            y = y / np.nanmax(y)
        s = ax.scatter(x, y, color=color, s=size, marker=marker,
                       alpha=alpha, edgecolors=edgecolor,
                       facecolors=facecolor, label=label,
                       rasterized=rasterized)

        scatters.append(s)

        if xerr is not None:
            xerror = xerr[i%len(xerr)]
        if yerr is not None:
            yerror = yerr[i%len(yerr)]

        ax.errorbar(x, y, yerror, xerror, fmt=va_config.eb_fmt, ecolor=ecolor, elinewidth=elinewidth,
                    capsize=capsize, capthick=capthick, barsabove=barsabove)

    # set axes labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels[0] is not None:
        ax.legend(loc=loc)

    scatters = scatters[0] if len(scatters) == 1 else scatters

    return scatters


def scatter3D(X, Y, Z, ax, elev=30, azim=45, roll=0,
              scale=None, axes_off=False, grid_lines=False,
              colors=None, size=None, marker=None, alpha=None,
              edgecolors=_default_flag, plot_contours=None, **kwargs):
    '''
    Scatter plot in 3D with support for multiple datasets.
    Parameters
    ––––––––––
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
        If None, uses the default color palette from
        `va_config.default_palette`.
    size : float, list of float, or None, optional, default=None
        Size of scatter dots. If None, uses the default
        value in `va_config.scatter_size`.
    marker : str, list of str, or None, optional, default=None
        Marker style for scatter dots. If None, uses the
        default value in `va_config.marker`.
    alpha : float, list of float, or None, default=None
        The alpha blending value, between 0 (transparent) and 1 (opaque).
        If None, uses the default value from `va_config.alpha`.
    edgecolors : {'face', 'none', None}, color, list of color, or None, default=`_default_flag`
        The edge color of the marker. Possible values:
        - 'face': The edge color will always be the same as the face color.
        - 'none': No patch boundary will be drawn.
        - A color or sequence of colors.
        If `_default_flag`, uses the default value in `va_config.edgecolor`.
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

        - `rasterized` : bool, default=`va_config.rasterized`
            Whether to rasterize plot artists. Rasterization
            converts the artist to a bitmap when saving to
            vector formats (e.g., PDF, SVG), which can
            significantly reduce file size for complex plots.
        - `color`, `c` : str, list of str or None, optional, default=`va_config.colors`
            Aliases for `colors`.
        - `sizes`, `s` : float or list of float, optional, default=`va_config.scatter_size`
            Aliases for `size`.
        - `markers`, `m` : str or list of str, optional, default=`va_config.marker`
            Aliases for `marker`.
        - `alphas`, `a` : float or list of float default=`va_config.alpha`
            Aliases for `alpha`.
        - `edgecolor`, `ec` : {'face', 'none', None}, color, list of color, or None, default=`va_config.edgecolor`
            Aliases for `edgecolors`.
        - `cmap` : str, optional, default=`va_config.cmap`
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
    –––––––
    scatter : `matplotlib.collections.Path3DCollection` or list of them
        The created scatter artist(s). Returns a single object
        if only one dataset is plotted.

    Raises
    ––––––
    ValueError
        If `X`, `Y`, and `Z` do not have the same number of arrays
        after unit consistency checks.

    Notes
    –––––
    - The function cycles through `colors`, `sizes`, `markers`,
      `alphas`, and `edgecolors` if fewer values are given than
      datasets.
    - Pane backgrounds are set to white (`(1, 1, 1, 1)`).
    - Axis limits are applied in the order of `xlim`, `ylim`, `zlim`,
      and finally `scale` if provided.
    '''
    # –––– KWARGS ––––
    rasterized = kwargs.get('rasterized', va_config.rasterized)
    # scatter params
    colors = get_kwargs(kwargs, 'color', 'c', default=colors)
    sizes = get_kwargs(kwargs, 'sizes', 's', default=size)
    markers = get_kwargs(kwargs, 'markers', 'm', default=marker)
    alphas = get_kwargs(kwargs, 'alphas', 'a', default=alpha)
    edgecolors = get_kwargs(kwargs, 'edgecolor', 'ec', default=edgecolors)
    cmap = kwargs.get('cmap', va_config.cmap)
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

    # get default va_config values
    colors = get_config_value(colors, 'colors')
    sizes = get_config_value(sizes, 'scatter_size')
    markers = get_config_value(markers, 'marker')
    alphas = get_config_value(alphas, 'alpha')
    edgecolors = va_config.edgecolor if edgecolors is _default_flag else edgecolors

    X = check_units_consistency(X)
    Y = check_units_consistency(Y)
    Z = check_units_consistency(Z)

    if not (len(X) == len(Y) == len(Z)):
        raise ValueError(
            f'`x`, `y`, and `z` must have the same number of arrays '
            f'\n(got {len(X)}, {len(Y)}, and {len(Z)}).'
        )

    colors, _ = set_plot_colors(colors, cmap=cmap)
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
                add_contours(data1, data2, ax, zdir=axis_name, offset=offset)

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
