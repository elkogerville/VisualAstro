'''
Author: Elko Gerville-Reache
Date Created: 2025-05-24
Date Modified: 2025-10-20
Description:
    Plotting utility functions.
Dependencies:
    - astropy
    - matplotlib
    - numpy
    - regions
Module Structure:
    - Plot Style and Color Functions
        Utility functions to set plotting style.
    - Imshow Stretch Functions
        Utility functions related to plot stretches.
    - Axes Labels, Format, and Styling
        Axes related utility functions.
    - Plot Matplotlib Patches and Shapes
        Plotting matplotlib shapes utility functions.
    - Notebook Utils
        Notebook utility functions.
'''

import os
import warnings
from functools import partial
import astropy.units as u
from astropy.units import physical
from astropy.visualization import AsinhStretch, ImageNormalize
from matplotlib import colors as mcolors
from matplotlib.colors import AsinhNorm, LogNorm, PowerNorm
from matplotlib.patches import Circle, Ellipse
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import matplotlib.ticker as ticker
import numpy as np
from regions import PixCoord, EllipsePixelRegion
from .data_cube_utils import slice_cube
from .numerical_utils import (
    check_is_array, compute_density_kde, get_data, get_physical_type, get_units, return_array_values
)
from .va_config import get_config_value, va_config, _default_flag


# Plot Style and Color Functions
# ––––––––––––––––––––––––––––––
def return_stylename(style):
    '''
    Returns the path to a visualastro mpl stylesheet for
    consistent plotting parameters.
    Avaliable styles:
        - 'astro'
        - 'default'
        - 'latex'
        - 'minimal'

    Matplotlib styles are also allowed (ex: 'classic').

    To add custom user defined mpl sheets, add files in:
    VisualAstro/visualastro/stylelib/
    Ensure the stylesheet follows the naming convention:
        mystylesheet.mplstyle

    If a style is unable to load due to missing fonts
    or other errors, `va_config.style_fallback` is used.

    Parameters
    ––––––––––
    style : str
        Name of the mpl stylesheet without the extension.
        ex: 'astro'
    Returns
    –––––––
    style_path : str
        Path to matplotlib stylesheet.
    '''
    # if style is a default matplotlib stylesheet
    if style in mplstyle.available:
        return style
    # if style is a visualastro stylesheet
    dir_path = os.path.dirname(os.path.realpath(__file__))
    style_path = os.path.join(dir_path, 'stylelib', f'{style}.mplstyle')
    # ensure that style works on computer, otherwise return default style
    try:
        with plt.style.context(style_path):
            # pass if can load style successfully on computer
            pass
        return style_path
    except Exception as e:
        warnings.warn(
            f"[visualastro] Could not apply style '{style}' ({e}). "
            "Falling back to 'default' style."
        )
        fallback = os.path.join(dir_path, 'stylelib', va_config.style_fallback)
        return fallback


def lighten_color(color, mix=0.5):
    '''
    Lightens the given matplotlib color by mixing it with white.
    Parameters
    ––––––––––
    color : matplotlib color, str
        Matplotlib named color, hex color, html color or rgb tuple.
    mix : float or int
        Ratio of color to white in mix.
        mix=0 returns the original color,
        mix=1 returns pure white.
    '''
    # convert to rgb
    rgb = np.array(mcolors.to_rgb(color))
    white = np.array([1, 1, 1])
    # mix color with white
    mixed = (1 - mix) * rgb + mix * white

    return mcolors.to_hex(mixed)


def sample_cmap(N, cmap=None, return_hex=False):
    '''
    Sample N distinct colors from a given matplotlib colormap
    returned as RGBA tuples in an array of shape (N,4).
    Parameters
    ––––––––––
    N : int
        Number of colors to sample.
    cmap : str, Colormap, or None, optional, default=None
        Name of the matplotlib colormap. If None,
        uses the default value in `va_config.cmap`.
    return_hex : bool, optional, default=False
        If True, return colors as hex strings.
    Returns
    –––––––
    list of tuple
        A list of RGBA colors sampled evenly from the colormap.
    '''
    # get default va_config values
    cmap = get_config_value(cmap, 'cmap')

    colors = plt.get_cmap(cmap)(np.linspace(0, 1, N))
    if return_hex:
        colors = np.array([mcolors.to_hex(c) for c in colors])

    return colors


def set_plot_colors(user_colors=None, cmap=None):
    '''
    Returns plot and model colors based on predefined palettes or user input.
    Parameters
    ––––––––––
    user_colors : str, list, or None, optional, default=None
        - None: returns the default palette (`va_config.default_palette`).
        - str:
            * If the string matches a palette name, returns that palette.
            * If the string ends with '_r', returns the reversed version of the palette.
            * If the string is a single color (hex or matplotlib color name), returns
              that color and a lighter version for the model.
        - list:
            * A list of colors (hex or matplotlib color names). Returns the list
              for plotting and lighter versions for models.
        - int:
            * An integer specifying how many colors to sample from a matplolib cmap
              using sample_cmap(). By default uses 'turbo'.
    cmap : str, list of str, or None, default=None
        Matplotlib colormap name. If None, uses
        the default value in `va_config.cmap`.
    Returns
    –––––––
    plot_colors : list of str
        Colors for plotting the data.
    model_colors : list of str
        Colors for plotting the model (contrasting or lighter versions).
    '''
    # default visualastro color palettes
    palettes = {
        'visualastro': {
            'plot':  ['#483D8B', '#DC267F', '#648FFF', '#FFB000', '#26DCBA'],
            'model': ['#D62728', '#1F77B4', '#E45756', '#17BECF', '#9467BD']
        },
        'va': {
            'plot':  ['#483D8B', '#DC267F', '#648FFF', '#FFB000', '#26DCBA'],
            'model': ['#D62728', '#1F77B4', '#E45756', '#17BECF', '#9467BD']
        },
        'ibm_contrast': {
            'plot':  ['#648FFF', '#DC267F', '#785EF0', '#26DCBA', '#FFB000', '#FE6100'],
            'model': ['#D62728', '#2CA02C', '#9467BD', '#17BECF', '#1F77B4', '#8C564B']
        },
        'astro': {
            'plot':  ['#9FB7FF', '#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000', '#CFE23C', '#26DCBA'],
            'model': ['#D62728', '#1F77B4', '#9467BD', '#2CA02C', '#E45756', '#17BECF', '#8C564B', '#FFD700']
        },
        'MSG': {
            'plot':  ['#483D8B', '#DC267F', '#DBB0FF', '#26DCBA', '#7D7FF3'],
            'model': ['#D62728', '#1F77B4', '#2CA02C', '#9467BD', '#17BECF']
        },
        'ibm': {
            'plot':  ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000'],
            'model': ['#D62728', '#2CA02C', '#9467BD', '#17BECF', '#E45756']
        },
        'smplot': {
            'plot': ['k', '#FF0000', '#0000FF', '#00FF00', '#00FFFF', '#FF00FF', '#FFFF00'],
            'model': ['#808080', '#FF6B6B', '#6B6BFF', '#6BFF6B', '#6BFFFF', '#FF6BFF', '#FFFF6B']
        }
    }
    # get default va_config values
    cmap = get_config_value(cmap, 'cmap')
    default_palette = va_config.default_palette

    # default case
    if user_colors is None:
        palette = palettes[default_palette]
        return palette['plot'], palette['model']
    # if user passes a color string
    if isinstance(user_colors, str):
        # if palette in visualastro palettes
        # return a reversed palette if palette
        # ends with '_r'
        if user_colors.rstrip('_r') in palettes:
            base_name = user_colors.rstrip('_r')
            palette = palettes[base_name]
            plot_colors = palette['plot']
            model_colors = palette['model']
            # if '_r', reverse palette
            if user_colors.endswith('_r'):
                plot_colors = plot_colors[::-1]
                model_colors = model_colors[::-1]
            return plot_colors, model_colors
        else:
            return [user_colors], [lighten_color(user_colors)]
    # if user passes a list or array of colors
    if isinstance(user_colors, (list, np.ndarray)):
        return user_colors, [lighten_color(c) for c in user_colors]
    # if user passes an integer N, sample a cmap for N colors
    if isinstance(user_colors, int):
        colors = sample_cmap(user_colors, cmap=cmap)
        return colors, [lighten_color(c) for c in colors]
    raise ValueError(
        'user_colors must be None, a str palette name, a str color, a list of colors, or an integer'
    )


# Imshow Stretch Functions
# ––––––––––––––––––––––––
def return_imshow_norm(vmin, vmax, norm, **kwargs):
    '''
    Return a matplotlib or astropy normalization object for image display.
    Parameters
    ––––––––––
    vmin : float or None
        Minimum value for normalization.
    vmax : float or None
        Maximum value for normalization.
    norm : str or None
        Normalization algorithm for colormap scaling.
        - 'asinh' -> asinh stretch using 'ImageNormalize'
        - 'asinhnorm' -> asinh stretch using 'AsinhNorm'
        - 'linear' -> no normalization applied
        - 'log' -> logarithmic scaling using 'LogNorm'
        - 'powernorm' -> power-law normalization using 'PowerNorm'
        - 'none' -> no normalization applied

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `linear_width` : float, optional, default=`va_config.linear_width`
            The effective width of the linear region, beyond
            which the transformation becomes asymptotically logarithmic.
            Only used in 'asinhnorm'.
        - `gamma` : float, optional, default=`va_config.gamma`
            Power law exponent.
    Returns
    –––––––
    norm_obj : None or matplotlib.colors.Normalize or astropy.visualization.ImageNormalize
        Normalization object to pass to `imshow`. None if `norm` is 'none'.
    '''
    linear_width = kwargs.get('linear_width', va_config.linear_width)
    gamma = kwargs.get('gamma', va_config.gamma)

    # use linear stretch if plotting boolean array
    if vmin==0 and vmax==1:
        return None

    # ensure norm is a string
    norm = 'none' if norm is None else norm
    # ensure case insensitivity
    norm = norm.lower()
    # dict containing possible stretch algorithms
    norm_map = {
        'asinh': ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch()), # type: ignore
        'asinhnorm': AsinhNorm(vmin=vmin, vmax=vmax, linear_width=linear_width),
        'log': LogNorm(vmin=vmin, vmax=vmax),
        'powernorm': PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax),
        'linear': None,
        'none': None
    }
    if norm not in norm_map:
        raise ValueError(
            f'ERROR: unsupported norm: {norm}. '
            f'\nsupported norms are {list(norm_map.keys())}'
        )

    return norm_map[norm]


def set_vmin_vmax(data, percentile=_default_flag, vmin=None, vmax=None):
    '''
    Compute vmin and vmax for image display. By default uses the
    data nanpercentile using `percentile`, but optionally vmin and/or
    vmax can be set by the user. Setting percentile to None results in
    no stretch. Passing in a boolean array uses vmin=0, vmax=1. This
    function is used internally by plotting functions.
    Parameters
    ––––––––––
    data : array-like
        Input data array (e.g., 2D image) for which to compute vmin and vmax.
    percentile : list or tuple of two floats, or None, default=`_default_flag`
        Percentile range '[pmin, pmax]' to compute vmin and vmax.
        If None, sets vmin and vmax to None. If `_default_flag`, uses
        default value from `va_config.percentile`.
    vmin : float or None, default=None
        If provided, overrides the computed vmin.
    vmax : float or None, default=None
        If provided, overrides the computed vmax.
    Returns
    –––––––
    vmin : float or None
        Minimum value for image scaling.
    vmax : float or None
        Maximum value for image scaling.
    '''
    percentile = va_config.percentile if percentile is _default_flag else percentile
    # check if data is an array
    data = check_is_array(data)
    # check if data is boolean
    if data.dtype == bool:
        return 0, 1

    # by default use percentile range
    if percentile is not None:
        if vmin is None:
            vmin = np.nanpercentile(data, percentile[0])
        if vmax is None:
            vmax = np.nanpercentile(data, percentile[1])
    # if vmin or vmax is provided overide and use those instead
    elif vmin is None and vmax is None:
        vmin = None
        vmax = None

    return vmin, vmax


def compute_cube_percentile(cube, slice_idx, vmin, vmax):
    '''
    Compute percentile-based intensity limits from a data cube slice.
    This function is intended to be used to compute an image scaling.
    Parameters
    ––––––––––
    cube : ndarray, SpectralCube, or DataCube
        Input data cube of shape (N_frames, N, M).
    slice_idx : int or list of int, optional
        Index or indices specifying the slice along the first axis:
        - i -> returns 'cube[i]'
        - [i] -> returns 'cube[i]'
        - [i, j] -> returns 'cube[i:j+1].sum(axis=0)'
    vmin : float
        Lower percentile (0–100) for intensity scaling.
    vmax : float
        Upper percentile (0–100) for intensity scaling.
    Returns
    –––––––
    vmin : float
        Computed lower intensity value corresponding to the
        specified 'vmin' percentile.
    vmax : float
        Computed upper intensity value corresponding to the
        specified 'vmax' percentile.
    '''
    # ensure cube is stripped of metadata
    cube = get_data(cube)
    # slice cube
    data = slice_cube(cube, slice_idx)
    data = return_array_values(data)
    # compute vmin and vmax
    vmin = np.nanpercentile(data, vmin) #type: ignore
    vmax = np.nanpercentile(data, vmax) #type: ignore

    return vmin, vmax


# Axes Labels, Format, and Styling
# ––––––––––––––––––––––––––––––––
def make_plot_grid(nrows=None, ncols=None, figsize=None,
                   sharex=None, sharey=None, hspace=_default_flag,
                   wspace=_default_flag, width_ratios=None, height_ratios=None,
                   fancy_axes=False, Nticks=_default_flag, aspect=None):
    '''
    Create a grid of Matplotlib axes panels with consistent sizing
    and optional fancy tick styling.
    Parameters
    ––––––––––
    nrows : int or None, default=None
        Number of subplot rows. If None, uses
        the default value set in `va_config.nrows`.
    ncols : int or None, default=None
        Number of subplot columns. If None, uses
        the default value set in `va_config.ncols`.
    figsize : tuple of float or None, default=None
        Figure size in inches as (width, height). If None,
        uses the default value set in `va_config.grid_figsize`.
    sharex : bool or None, default=None
        If True, share the x-axis among all subplots. If None,
        uses the default value set in `va_config.sharex`.
    sharey : bool or None, default=None
        If True, share the y-axis among all subplots. If None,
        uses the default value set in `va_config.sharey`.
    hspace : float or None, default=`_default_flag`
        Height padding between subplots. If None,
        Matplotlib’s default spacing is used. If
        `_default_flag`, uses the default value set in
        `va_config.hspace`.
    wspace : float or None, default=`_default_flag`
        Width padding between subplots. If None,
        Matplotlib’s default spacing is used. If
        `_default_flag`, uses the default value set in
        `va_config.wspace`.
    width_ratios : array-like of length `ncols`, optional, default=None
        Width padding between subplots. If None, Matplotlib’s default spacing is used.
        Defines the relative widths of the columns. Each column gets a relative width
        of width_ratios[i] / sum(width_ratios). If not given, all columns will have the same width.
    height_ratios : array-like of length `nrows`, optional
        Defines the relative heights of the rows. Each row gets a relative height of
        height_ratios[i] / sum(height_ratios). If not given, all rows will have the same height.
    fancy_axes : bool, default=False
        If True, enables "fancy" axes styling:
        - minor ticks on
        - inward ticks on all sides
        - axes labels on outer grid axes
        - h/wspace = 0.0
    Nticks : int or None, default=`_default_flag`
        Maximum number of major ticks per axis. If None,
        uses the default matplotlib settings. If `_default_flag`,
        uses the default value set in `va_config.Nticks`.
    aspect : float or None, default=None
        Changes the physical dimensions of the Axes,
        such that the ratio of the Axes height to the
        Axes width in physical units is equal to aspect.
        None will disable a fixed box aspect so that height
        and width of the Axes are chosen independently.
    Returns
    –––––––
    fig : `~matplotlib.figure.Figure`
        The created Matplotlib Figure instance.
    axs : ndarray of `~matplotlib.axes.Axes`
        Flattened array of Axes objects, ordered row-wise.
    '''
    # get default va_config values
    nrows = get_config_value(nrows, 'nrows')
    ncols = get_config_value(ncols, 'ncols')
    figsize = get_config_value(figsize, 'grid_figsize')
    sharex = get_config_value(sharex, 'sharex')
    sharey = get_config_value(sharey, 'sharey')
    hspace = va_config.hspace if hspace is _default_flag else hspace
    wspace = va_config.wspace if wspace is _default_flag else wspace
    Nticks = va_config.Nticks if Nticks is _default_flag else Nticks

    Nx = nrows
    Ny = ncols

    if fancy_axes:
        labeltop = [[True if i == 0 else False for j in range(Ny)] for i in range(Nx)]
        labelright = [[True if i == Ny-1 else False for i in range(Ny)] for j in range(Nx)]
        hspace = 0.0 if hspace is None else hspace
        wspace = 0.0 if wspace is None else wspace

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(Nx, Ny, hspace=hspace, wspace=wspace,
                          width_ratios=width_ratios,
                          height_ratios=height_ratios)
    axs = gs.subplots(sharex=sharex, sharey=sharey)
    axs = np.atleast_1d(axs).ravel()

    for i in range(Nx):
        for j in range(Ny):
            ax = axs[j + Ny*i]

            if fancy_axes:
                ax.minorticks_on()
                ax.tick_params(axis='both', length=2, direction='in',
                               which='both', labeltop=labeltop[i][j],
                               labelright=labelright[i][j],
                               right=True, top=True)
            if Nticks is not None:
                ax.xaxis.set_major_locator(ticker.MaxNLocator(Nticks))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(Nticks))
            ax.set_box_aspect(aspect)

    return fig, axs


def add_subplot(
    shape=111,
    fig=None,
    figsize=None,
    projection=None,
    return_fig=False,
    **kwargs):
    '''
    Add a subplot to a figure, optionally creating a new figure.
    Parameters
    ––––––––––
    shape : int or tuple, default: 111
        The subplot specification. Can be given as a three-digit integer
        (e.g., 211 means 2 rows, 1 column, subplot index 1) or a tuple
        `(nrows, ncols, index)`.
    fig : matplotlib.figure.Figure or None, optional, default=None
        Existing figure to add the subplot to. If None,
        a new figure is created.
    figsize : tuple of float, optional, default=None
        Figure size in inches. If None, uses the
        default value set by `va_config.figsize`.
    projection : str or None, optional, default=None
        Projection type for the subplot. Examples include WCSAxes or
        {None, '3d', 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar',
        'rectilinear', str}. If None, defaults to 'rectilinear'.
    return_fig : bool, optional, default=False
        If True, return both `(fig, ax)`. Otherwise return only `ax`.

    **kwargs
        Additional keyword arguments passed directly to
        `matplotlib.figure.Figure.add_subplot`. This allows supplying any
        subplot or axes-related parameters supported by Matplotlib (e.g.,
        `aspect`, `facecolor`, etc.).

    Returns
    –––––––
    ax : matplotlib.axes.Axes
        The created or retrieved subplot axes.
    fig : matplotlib.figure.Figure, optional
        The figure object containing the subplot.
        Returned only if `return_fig=True`.

    Examples
    ––––––––
    Create a new figure and subplot:
    >>> fig, ax = add_subplot(return_fig=True)

    Add a subplot to an existing figure:
    >>> fig = plt.figure()
    >>> ax = add_subplot(fig=fig, shape=121)

    Create a 3D subplot:
    >>> fig, ax = add_subplot(projection='3d', return_fig=True)
    '''
    # get default va_config values
    figsize = get_config_value(figsize, 'figsize')
    # create figure if not passed in
    if fig is None:
        fig = plt.figure(figsize=figsize)
    # add desired subplot with projection
    ax = fig.add_subplot(shape, projection=projection, **kwargs)

    return (fig, ax) if return_fig else ax


def add_colorbar(im, ax, cbar_width=None,
                 cbar_pad=None, clabel=None):
    '''
    Add a colorbar next to an Axes.
    Parameters
    ––––––––––
    im : matplotlib.cm.ScalarMappable
        The image, contour set, or mappable object returned by
        a plotting function (e.g., 'imshow', 'scatter', etc...).
    ax : matplotlib.axes.Axes
        The axes to which the colorbar will be attached.
    cbar_width : float or None, optional, default=None
        Width of the colorbar in figure coordinates.
        If None, uses the default value set in `va_config.cbar_width`.
    cbar_pad : float or None, optional, default=None
        Padding between the main axes and the colorbar
        in figure coordinates. If None, uses the default
        value set in `va_config.cbar_pad`.
    clabel : str, optional
        Label for the colorbar. If None, no label is set.
    '''
    # get default va_config values
    cbar_width = get_config_value(cbar_width, 'cbar_width')
    cbar_pad = get_config_value(cbar_pad, 'cbar_pad')

    # extract figure from axes
    fig = ax.figure
    # add colorbar axes
    cax = fig.add_axes([ax.get_position().x1+cbar_pad, ax.get_position().y0,
                        cbar_width, ax.get_position().height])
    # add colorbar
    cbar = fig.colorbar(im, cax=cax, pad=0.04)
    # formatting and label
    cbar.ax.tick_params(which=va_config.cbar_tick_which, direction=va_config.cbar_tick_dir)
    if clabel is not None:
        cbar.set_label(fr'{clabel}')


def add_contours(x, y, ax, levels=20, contour_method='contour',
                 bw_method='scott', resolution=200, padding=0.2,
                 cslabel=False, zdir=None, offset=None, cmap=None,
                 **kwargs):
    '''
    Add 2D or 3D Gaussian KDE density contours to an axis.
    This function computes a 2D Gaussian kernel density estimate (KDE)
    from input data (`x`, `y`) using `compute_density_kde` and plots
    contour lines or filled contours using either `ax.contour` or
    `ax.contourf`. If `zdir` and `offset` are provided, the contours
    are projected onto a plane in 3D space.
    Parameters
    ––––––––––
    x : array-like
        1D array of x-values for the dataset.
    y : array-like
        1D array of y-values for the dataset.
    ax : matplotlib.axes.Axes or mpl_toolkits.mplot3d.axes3d.Axes3D
        Axis on which to draw the contours.
    levels : int or array-like, default=20
        Number or list of contour levels to draw.
    contour_method : {'contour', 'contourf'}, default='contour'
        Method used to draw contours. 'contour' draws lines, while
        'contourf' draws filled contours.
    bw_method : str, scalar or callable, optional, default='scott'
        The method used to calculate the bandwidth factor for the Gaussian KDE.
        Can be one of:
        - 'scott' or 'silverman': use standard rules of thumb.
        - a scalar constant: directly used as the bandwidth factor.
        - a callable: should take a `scipy.stats.gaussian_kde` instance as its
            sole argument and return a scalar bandwidth factor.
    resolution : int, default=200
        Number of grid points used per axis for density estimation.
    padding : float, default=0.2
        Fractional padding applied to the data range when generating
        the KDE grid.
    cslabel : bool, default=False
        If True, label contour levels with their corresponding values.
        Only works in 2D plots.
    zdir : {'x', 'y', 'z'} or None, default=None
        Direction normal to the plane where contours are drawn.
        If None, contours are plotted in 2D.
    offset : float or None, default=None
        Offset along the `zdir` direction for projecting contours in 3D space.
    cmap : str, optional, default=`va_config.cmap`
        Colormap used for plotting contours.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `fontsize` : float, default=`va_config.fontsize`
            Fontsize of contour labels.

    Returns
    –––––––
    cs : matplotlib.contour.QuadContourSet or mpl_toolkits.mplot3d.art3d.QuadContourSet3D
        The contour set object created by Matplotlib.
    '''
    # –––– KWARGS ––––
    fontsize = kwargs.get('fontsize', va_config.fontsize)
    # get default va_config values
    cmap = get_config_value(cmap, 'cmap')
    # get contour plotting method
    contour_method = {
        'contour': ax.contour,
        'contourf': ax.contourf
    }.get(contour_method.lower(), ax.contour)

    # compute kde density
    X, Y, Z = compute_density_kde(x, y, bw_method=bw_method, resolution=resolution, padding=padding)

    # plot contours as either 3D projections or a simple 2D plot
    valid_zdirs = {'x', 'y', 'z'}
    zdir = zdir.lower() if isinstance(zdir, str) else None
    if zdir in valid_zdirs and offset is not None:
        if zdir == 'z':
            cs = contour_method(X, Y, Z, levels=levels, cmap=cmap, zdir=zdir, offset=offset)
        elif zdir == 'y':
            cs = contour_method(X, Z, Y, levels=levels, cmap=cmap, zdir=zdir, offset=offset)
        else:
            cs = contour_method(Z, Y, X, levels=levels, cmap=cmap, zdir=zdir, offset=offset)
    else:
        cs = contour_method(X, Y, Z, levels=levels, cmap=cmap)

    # add labels
    if cslabel:
        ax.clabel(cs, fontsize=fontsize)

    return cs


def format_unit_labels(unit, fmt=None):
    '''
    Convert an astropy unit string into a LaTeX-formatted label
    for plotting. Returns None if no unit is found.
    Parameters
    ––––––––––
    unit : str
        The unit string to convert.
    fmt : {'latex', 'latex_inline', 'inline'} or None, optional, default=None
        The format of the unit label. 'latex_inline' and 'inline' uses
        negative exponents while 'latex' uses fractions. If None, uses
        the default value set by `va_config.unit_label_format`.
    Returns
    –––––––
    str or None
        A LaTeX-formatted unit label if the input is recognized.
        Returns None if the unit is invalid.
    '''
    fmt = get_config_value(fmt, 'unit_label_format')

    if fmt.lower() == 'inline':
        fmt = 'latex_inline'

    try:
        return u.Unit(unit).to_string(fmt)
    except Exception:
        return None


def set_axis_limits(xdata, ydata, ax, xlim=None, ylim=None, **kwargs):
    '''
    Set axis limits based on concatenated data or user-provided limits.
    Parameters
    ––––––––––
    xdata : list/tuple of arrays or array
        X-axis data from multiple datasets.
    ydata : list/tuple of arrays or array
        Y-axis data from multiple datasets.
    ax : matplotlib axis
        The matplotlib axes object on which to set the axis limits.
    xlim : tuple/list, optional
        User-defined x-axis limits.
    ylim : tuple/list, optional
        User-defined y-axis limits.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `xpad`/`ypad` : float, optional, default=`va_config.xpad`/`va_config.ypad`
            Padding along x and y axis used when computing axis limits.
            Defined as:
                xmax/min ±= xpad * (xmax - xmin)
                ymax/min ±= ypad * (ymax - ymin)
    '''
    # –––– KWARGS ––––
    xpad = kwargs.get('xpad', va_config.xpad)
    ypad = kwargs.get('ypad', va_config.ypad)

    if xdata is not None:
        # concatenate list of data into single array
        if isinstance(xdata, (list, tuple)):
            xdata = np.concatenate(xdata)
        else:
            xdata = np.asarray(xdata)
        # min and max values across data sets
        xmin = return_array_values(np.nanmin(xdata))
        xmax = return_array_values(np.nanmax(xdata))
        # pad xlim
        if xpad > 0:
            dx = xmax - xmin
            xmin -= xpad * dx
            xmax += xpad * dx
        # use computed limits unless user overides
        xlim = xlim if xlim is not None else [xmin, xmax]

    if ydata is not None:
        # concatenate list of data into single array
        if isinstance(ydata, (list, tuple)):
            ydata = np.concatenate(ydata)
        else:
            ydata = np.asarray(ydata)
        # min and max values across data sets
        ymin = return_array_values(np.nanmin(ydata))
        ymax = return_array_values(np.nanmax(ydata))
        # pad ylim
        if ypad > 0:
            dy = ymax - ymin
            ymin -= ypad * dy
            ymax += ypad * dy
        # use computed limits unless user overides
        ylim = ylim if ylim is not None else [ymin, ymax]

    # set x and y limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def set_axis_labels(
    X, Y, ax, xlabel=None, ylabel=None, use_brackets=None,
    use_type_label=None, use_unit_label=None, fmt=None
):
    '''
    Automatically generate and set axis labels from Quantity objects with units.
    Creates formatted axis labels by combining the physical type (e.g., 'Wavelength',
    'Flux Density') and units (e.g., 'μm', 'MJy/sr') of the input data. Labels can be
    customized to show only the type, only units, or both. Units are encapsulated with
    either [] or ().
    Parameters
    ––––––––––
    X : '~astropy.units.Quantity' or object with 'unit' attribute
        The data for the x-axis, typically a spectral axis (frequency, wavelength, or velocity).
    Y : '~astropy.units.Quantity' or object with 'unit' or 'spectral_unit' attribute
        The data for the y-axis, typically flux or intensity.
    ax : 'matplotlib.axes.Axes'
        The matplotlib axes object on which to set the labels.
    xlabel : str or None, optional, default=None
        Custom label for the x-axis. If None, the label is inferred from 'X'.
    ylabel : str or None, optional, default=None
        Custom label for the y-axis. If None, the label is inferred from 'Y'.
    use_brackets : bool or None, optional, default=None
        If True, wrap units in square brackets '[ ]'. If False, use parentheses '( )'.
        If None, uses the default value set in `va_config.use_brackets`.
    use_type_label: bool or None, optional, default=None
        If True, include the physical type of the X and Y for the axis label if
        available. If None, uses the default value set by `va_config.use_type_label`.
    use_unit_label: bool or None, optional, default=None
        If True, include the unit of the X and Y for the axis label if
        available. If None, uses the default value set by `va_config.use_unit_label`.
    fmt : {'latex', 'latex_inline', 'inline'} or None, optional, default=None
        The format of the unit label. 'latex_inline' and 'inline' uses
        negative exponents while 'latex' uses fractions. If None, uses
        the default value set by `va_config.unit_label_format`.

    Examples
    ––––––––
    >>> import astropy.units as u
    >>> wavelength = np.linspace(1, 10, 100) * u.um
    >>> flux = np.random.random(100) * u.MJy / u.sr
    >>> fig, ax = plt.subplots()
    >>> ax.plot(wavelength, flux)
    >>> set_axis_labels(wavelength, flux, ax)
    # Sets xlabel to 'Wavelength [μm]' and ylabel to 'Surface Brightness [MJy/sr]'

    >>> # Custom label with only units
    >>> set_axis_labels(wavelength, flux, ax, use_type_label=False)
    # Sets xlabel to '[μm]' and ylabel to '[MJy/sr]'

    >>> # Override with custom label
    >>> set_axis_labels(wavelength, flux, ax, ylabel='Custom Flux')
    # Uses 'Custom Flux [MJy/sr]' for y-axis

    Notes
    –––––
    - Units are formatted using 'format_unit_labels', which provides LaTeX-friendly labels.
      The labels are formatted as either 'latex_inline' or 'latex', which displays fractions
      with either negative exponents or with fractions.
    '''
    # get default va_config values
    use_brackets = get_config_value(use_brackets, 'use_brackets')
    use_type_label = get_config_value(use_type_label, 'use_type_label')
    use_unit_label = get_config_value(use_unit_label, 'use_unit_label')
    fmt = get_config_value(fmt, 'unit_label_format')

    # unit bracket type [] or ()
    brackets = [r'[',r']'] if use_brackets else [r'(',r')']

    TYPE_MAP = {
        u.adu.physical_type: 'ADU',
        u.count.physical_type: 'Counts',
        u.electron.physical_type: 'Counts',
        u.mag.physical_type: 'Mag',
        physical.energy: 'Energy',
        physical.frequency: 'Frequency',
        physical.length: 'Wavelength',
        physical.power_density: 'Flux',
        physical.spectral_flux_density: 'Flux Density',
        physical.speed: 'Velocity',
        physical.surface_brightness: 'Surface Brightness'
    }

    def _create_label(
        obj, type_map, label, brackets, use_type_label, use_unit_label, fmt
    ):
        '''Creates the axis label based on the object being plotted'''

        # get physical type and unit of object
        physical_type = get_physical_type(obj)
        unit = get_units(obj)

        # get custom label if exists
        if isinstance(label, str):
            type_label = label
        elif use_type_label and physical_type in type_map:
            type_label = type_map[physical_type]
        # use physical type if exists
        elif use_type_label and physical_type is not None:
            type_label = str(physical_type).replace('_', ' ').title()
        else:
            type_label = ''

        # set unit label
        unit_label = format_unit_labels(unit, fmt=fmt)

        # add brackets to unit if exists
        if use_unit_label and unit_label is not None:
            unit_label = fr'{brackets[0]}{unit_label}{brackets[1]}'
        else:
            unit_label = ''

        axis_label = fr'{type_label} {unit_label}'.strip()

        return axis_label

    xlabel = _create_label(
        X, TYPE_MAP, xlabel, brackets,
        use_type_label, use_unit_label, fmt
    )

    ylabel = _create_label(
        Y, TYPE_MAP, ylabel, brackets,
        use_type_label, use_unit_label, fmt
    )

    # set plot labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# Plot Matplotlib Patches and Shapes
# ––––––––––––––––––––––––––––––––––
def plot_circles(
    circles,
    ax,
    colors=None,
    linewidth=None,
    fill=None,
    cmap=None
):
    '''
    Plot one or more circles on a Matplotlib axis with customizable style.
    Parameters
    ––––––––––
    circles : array-like or None
        Circle coordinates and radii. Can be a single circle `[x, y, r]`
        or a list/array of circles `[[x1, y1, r1], [x2, y2, r2], ...]`.
        If None, no circles are plotted.
    ax : matplotlib.axes.Axes
        The Matplotlib axis on which to plot the circles.
    colors : list of colors, str, or None, optional, default=None
        List of colors to cycle through for each circle. None defaults
        to ['r', 'mediumvioletred', 'magenta']. A single color can also
        be passed. If there are more circles than colors, colors are
        sampled from a colormap using sample_cmap(cmap=`cmap`).
    linewidth : float or None, optional, default=None
        Width of the circle edge lines. If None,
        uses the default value set in `va_config.linewidth`.
    fill : bool or None, optional, default=None
        Whether the circles are filled. If None,
        uses the default value set in `va_config.circle_fill`.
    cmap : str or None, optional, default=None
        matplolib cmap used to sample default circle colors.
        If None, uses the default value set in `va_config.cmap`.
    '''
    # get default va_config values
    linewidth = get_config_value(linewidth, 'linewidth')
    fill = get_config_value(fill, 'circle_fill')
    cmap = get_config_value(cmap, 'cmap')

    if circles is not None:
        # ensure circles is list [x,y,r] or list of list [[x,y,r],[x,y,r]...]
        circles = np.atleast_2d(circles)
        if circles.shape[1] != 3:
            raise ValueError(
                'Circles must be either [x, y, r] or [[x1, y1, r1], [x2, y2, r2], ...]'
            )
        # number of circles to plot
        N = circles.shape[0]
        # set circle colors
        if colors is None:
            colors = ['r', 'mediumvioletred', 'magenta'] if N<=3 else sample_cmap(N, cmap=cmap)
        if isinstance(colors, str):
            colors = [colors]

        # plot each cirlce
        for i, circle in enumerate(circles):
            x, y, r = circle
            color = colors[i%len(colors)]
            circle_patch = Circle((x, y), radius=r, fill=fill, linewidth=linewidth, color=color)
            ax.add_patch(circle_patch)


def copy_ellipse(ellipse):
    '''
    Returns a copy of an Ellipse object.
    Parameters
    ––––––––––
    ellipse : matplotlib.patches.Ellipse
        The Ellipse object to copy.
    Returns
    ––––––––––
    matplotlib.patches.Ellipse
        A new Ellipse object with the same properties as the input.
    '''
    return Ellipse(
        xy=ellipse.center,
        width=ellipse.width,
        height=ellipse.height,
        angle=ellipse.angle,
        edgecolor=ellipse.get_edgecolor(),
        facecolor=ellipse.get_facecolor(),
        lw=ellipse.get_linewidth(),
        ls=ellipse.get_linestyle(),
        alpha=ellipse.get_alpha()
    )


def plot_ellipses(ellipses, ax):
    '''
    Plots an ellipse or list of ellipses to an axes.
    Parameters
    ––––––––––
    ellipses : matplotlib.patches.Ellipse or list
        The Ellipse or list of Ellipses to plot.
    ax : matplotlib.axes.Axes
        Matplotlib axis on which to plot the ellipses(s).
    '''
    if ellipses is not None:
        # ensure ellipses is iterable
        ellipses = ellipses if isinstance(ellipses, list) else [ellipses]
        # plot each ellipse
        for ellipse in ellipses:
            ax.add_patch(copy_ellipse(ellipse))


def plot_interactive_ellipse(center, w, h, ax, text_loc=None,
                             text_color=None, highlight=None):
    '''
    Create an interactive ellipse selector on an Axes
    along with an interactive text window displaying
    the current ellipse center, width, and height.
    Parameters
    ––––––––––
    center : tuple of float
        (x, y) coordinates of the ellipse center in data units.
    w : float
        Width of the ellipse.
    h : float
        Height of the ellipse.
    ax : matplotlib.axes.Axes
        The Axes on which to draw the ellipse selector.
    text_loc : list of float or None, optional, default=None
        Position of the text label in Axes coordinates, given as [x, y].
        If None, uses the default value set in `va_config.text_loc`.
    text_color : str or None, optional, default=None
        Color of the annotation text. If None, uses
        the default value set in `va_config.text_color`.
    highlight : bool or None, optional, default=None
        If True, adds a bbox to highlight the text. If None,
        uses the default value set in `va_config.highlight`.
    Notes
    –––––
    Ensure an interactive backend is active. This can be
    activated with use_interactive().
    '''
    # get default va_config values
    text_loc = get_config_value(text_loc, 'ellipse_label_loc')
    text_color = get_config_value(text_color, 'text_color')
    highlight = get_config_value(highlight, 'highlight')

    # define text for ellipse data display
    facecolor = 'k' if text_color == 'w' else 'w'
    bbox = dict(facecolor=facecolor, alpha=0.6, edgecolor="none") if highlight else None
    text = ax.text(text_loc[0], text_loc[1], '',
                   transform=ax.transAxes,
                   size='small', color=text_color,
                   bbox=bbox)
    # define ellipse
    ellipse_region = EllipsePixelRegion(center=PixCoord(x=center[0], y=center[1]),
                                        width=w, height=h)
    # define interactive ellipse
    selector = ellipse_region.as_mpl_selector(ax, callback=partial(_update_ellipse_region, text=text))
    # bind ellipse to axes
    ax._ellipse_selector = selector


def _update_ellipse_region(region, text):
    '''
    Update ellipse information text when the
    interactive region is modified.
    Parameters
    ––––––––––
    region : regions.EllipsePixelRegion
        The ellipse region being updated.
    text : matplotlib.text.Text
        The text object used to display ellipse parameters.
    '''
    # extract properties from ellipse object
    x_center = region.center.x
    y_center = region.center.y
    width = region.width
    height = region.height
    major = max(width, height)
    minor = min(width, height)
    # display properties
    text.set_text(
        f'Center: [{x_center:.1f}, {y_center:.1f}]\n'
        f'Major: {major:.1f}\n'
        f'Minor: {minor:.1f}\n'
    )


def return_ellipse_region(center, w, h, angle=0, fill=False):
    '''
    Create a matplotlib.patches.Ellipse object.
    Parameters
    ––––––––––
    center : tuple of float
        (x, y) coordinates of the ellipse center.
    w : float
        Width of the ellipse (along x-axis before rotation).
    h : float
        Height of the ellipse (along y-axis before rotation).
    angle : float, default=0
        Rotation angle of the ellipse in degrees (counterclockwise).
    fill : bool, default=False
        Whether the ellipse should be filled (True) or only outlined (False).
    Returns
    –––––––
    matplotlib.patches.Ellipse
        An Ellipse patch that can be added to a matplotlib Axes.
    '''
    ellipse = Ellipse(xy=(center[0], center[1]), width=w, height=h, angle=angle, fill=fill)

    return ellipse


def plot_points(points, ax, color='r', size=20, marker='*'):
    '''
    Plot points on a given Matplotlib axis with customizable style.
    Parameters
    ––––––––––
    points : array-like or None
        Coordinates of points to plot. Can be a single point `[x, y]`
        or a list/array of points `[[x1, y1], [x2, y2], ...]`.
        If None, no points are plotted.
    ax : matplotlib.axes.Axes
        The Matplotlib axis on which to plot the points.
    color : str or list or int, optional, default='r'
        Color of the points. If an integer, will draw colors
        from sample_cmap().
    size : float, optional, default=20
        Marker size.
    marker : str, optional, default='*'
        Matplotlib marker style.
    '''
    if points is not None:
        points = np.asarray(points)
        # ensure points is list [x,y] or list of list [[x,y],[x,y]...]
        if points.ndim == 1 and points.shape[0] == 2:
            points = points[np.newaxis, :]
        elif points.ndim != 2 or points.shape[1] != 2:
            error = 'Points must be either [x, y] or [[x1, y1], [x2, y2], ...]'
            raise ValueError(error)
        if isinstance(color, int):
            color = sample_cmap(color)
        color = color if isinstance(color, list) else [color]
        # loop through each set of points in points and plot
        for i, point in enumerate(points):
            ax.scatter(point[0], point[1], s=size, marker=marker, c=color[i%len(color)])


# Notebook Utils
# ––––––––––––––
def use_inline():
    '''
    Start an inline IPython backend session.
    Allows for inline plots in IPython sessions
    like Jupyter Notebook.
    '''
    try:
        from IPython.core.getipython import get_ipython
    except ImportError:
        raise ImportError(
            'IPython is not installed. Install it to use this feature'
        )

    ipython = get_ipython()
    if ipython is None:
        print('Not inside an IPython environment')
        return None

    try:
        ipython.run_line_magic('matplotlib', 'inline')
    except Exception as e:
        print(f'Unable to set inline backend: {e}')


def use_interactive():
    '''
    Start an interactive IPython backend session.
    Allows for interactive plots in IPython sessions
    like Jupyter Notebook.
    Ensure ipympl is installed:
    >>> $ conda install -c conda-forge ipympl
    '''
    try:
        from IPython.core.getipython import get_ipython
    except ImportError:
        raise ImportError(
            'IPython is not installed. Install it to use this feature'
        )

    ipython = get_ipython()
    if ipython is None:
        print('Not inside an IPython environment')
        return None

    try:
        ipython.run_line_magic('matplotlib', 'ipympl')
    except Exception as e:
        print(
            f'ipympl backend unavailable: {e}. Please install with:\n'
            f'$ conda install -c conda-forge ipympl'
        )


def plt_close():
    '''
    Closes all interactive plots in session.
    '''
    plt.close('all')
