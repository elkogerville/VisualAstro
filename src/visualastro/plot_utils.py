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

from collections.abc import Sequence
import os
from typing import Any
import warnings
from functools import partial
import astropy.units as u
from astropy.units import Quantity, UnitBase
from astropy.visualization import AsinhStretch, ImageNormalize
from matplotlib import colors as mcolors
from matplotlib.colors import AsinhNorm, LogNorm, PowerNorm
from matplotlib.patches import Circle, Ellipse
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import matplotlib.ticker as ticker
import numpy as np
from numpy.typing import NDArray
from regions import PixCoord, EllipsePixelRegion
from specutils import SpectralAxis
from .config import get_config_value, config, _default_flag
from .data_cube_utils import stack_cube
from .numerical_utils import (
    compute_density_kde, flatten, get_data, get_value, shift_by_radial_vel, to_array, to_list
)
from .spectra_utils import get_spectral_axis, spectral_idx_2_world
from .units import (
    convert_quantity, to_latex_unit, get_physical_type,
    get_unit, _infer_physical_type_label, to_unit
)
from .utils import _type_name


# Plot Style and Color Functions
# ------------------------------
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
    or other errors, `config.style_fallback` is used.

    Parameters
    ----------
    style : str
        Name of the mpl stylesheet without the extension.
        ex: 'astro'

    Returns
    -------
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
        fallback = os.path.join(dir_path, 'stylelib', config.style_fallback)
        return fallback


def lighten_color(color, mix=0.5):
    '''
    Lightens the given matplotlib color by mixing it with white.

    Parameters
    ----------
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
    ----------
    N : int
        Number of colors to sample.
    cmap : str, Colormap, or None, optional, default=None
        Name of the matplotlib colormap. If None,
        uses the default value in `config.cmap`.
    return_hex : bool, optional, default=False
        If True, return colors as hex strings.

    Returns
    -------
    array
        An array of RGBA colors sampled evenly from the colormap,
        or an array of hex colors if `return_hex=True`.
    '''
    # get default config values
    cmap = get_config_value(cmap, 'cmap')

    colors = plt.get_cmap(cmap)(np.linspace(0, 1, N))
    if return_hex:
        colors = np.array([mcolors.to_hex(c) for c in colors])

    return colors


def set_plot_colors(user_colors=None, cmap=None):
    '''
    Returns plot and model colors based on predefined palettes or user input.

    Parameters
    ----------
    user_colors : str, list, or None, optional, default=None
        - None: returns the default palette (`config.default_palette`).
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
        the default value in `config.cmap`.

    Returns
    -------
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
    # get default config values
    cmap = get_config_value(cmap, 'cmap')
    default_palette = config.default_palette

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
# ------------------------
def return_imshow_norm(vmin, vmax, norm, **kwargs):
    '''
    Return a matplotlib or astropy normalization object for image display.

    Parameters
    ----------
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

        - `linear_width` : float, optional, default=`config.linear_width`
            The effective width of the linear region, beyond
            which the transformation becomes asymptotically logarithmic.
            Only used in 'asinhnorm'.
        - `gamma` : float, optional, default=`config.gamma`
            Power law exponent.

    Returns
    -------
    norm_obj : None or matplotlib.colors.Normalize or astropy.visualization.ImageNormalize
        Normalization object to pass to `imshow`. None if `norm` is 'none'.
    '''
    linear_width = kwargs.get('linear_width', config.linear_width)
    gamma = kwargs.get('gamma', config.gamma)

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
    ----------
    data : array-like
        Input data array (e.g., 2D image) for which to compute vmin and vmax.
    percentile : list or tuple of two floats, or None, default=`_default_flag`
        Percentile range '[pmin, pmax]' to compute vmin and vmax.
        If None, sets vmin and vmax to None. If `_default_flag`, uses
        default value from `config.percentile`.
    vmin : float or None, default=None
        If provided, overrides the computed vmin.
    vmax : float or None, default=None
        If provided, overrides the computed vmax.

    Returns
    -------
    vmin : float or None
        Minimum value for image scaling.
    vmax : float or None
        Maximum value for image scaling.
    '''
    percentile = config.percentile if percentile is _default_flag else percentile
    # check if data is an array
    data = to_array(data)
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


def nanpercentile_limits(
    data,
    vmin,
    vmax,
    *,
    slice_idx=None,
    stack_method=None,
    axis=0,
):
    """
    Compute NaN-safe percentile-based color limits.

    If input is 3D, it is reduced to 2D before computing limits.

    Parameters
    ----------
    data : array-like, SpectralCube, or DataCube
        Input image or cube.
    vmin, vmax : float
        Lower and upper percentiles (0–100).
    slice_idx : int, list of int, or None, optional, default=None
        Optional slicing for cube-like inputs. If None,
        is ignored.
    stack_method : {'mean', 'median', 'sum', 'max', 'min', 'std'} or None, default=None
        Reduction method if stacking is required. If None,
        uses the default value set by ``config.stack_cube_method``.
    axis : int, default=0
        Axis to reduce if data.ndim > 2 and no slice_idx is given.

    Returns
    -------
    vmin_val, vmax_val : float
    """
    stack_method = get_config_value(stack_method, 'stack_cube_method')

    data = get_data(data)

    if slice_idx is not None:
        data = stack_cube(
            data, idx=slice_idx, method=stack_method, axis=axis
        )

    arr = to_array(get_value(data))

    if arr.ndim > 2:
        arr = getattr(np, f'nan{stack_method}')(arr, axis=axis)

    return (
        np.nanpercentile(arr, vmin),
        np.nanpercentile(arr, vmax),
    )


# Axes Labels, Format, and Styling
# --------------------------------
def make_plot_grid(nrows=None, ncols=None, figsize=None,
                   sharex=None, sharey=None, hspace=_default_flag,
                   wspace=_default_flag, width_ratios=None, height_ratios=None,
                   fancy_axes=False, Nticks=_default_flag, aspect=None):
    '''
    Create a grid of Matplotlib axes panels with consistent sizing
    and optional fancy tick styling.

    Parameters
    ----------
    nrows : int or None, default=None
        Number of subplot rows. If None, uses
        the default value set in `config.nrows`.
    ncols : int or None, default=None
        Number of subplot columns. If None, uses
        the default value set in `config.ncols`.
    figsize : tuple of float or None, default=None
        Figure size in inches as (width, height). If None,
        uses the default value set in `config.grid_figsize`.
    sharex : bool or None, default=None
        If True, share the x-axis among all subplots. If None,
        uses the default value set in `config.sharex`.
    sharey : bool or None, default=None
        If True, share the y-axis among all subplots. If None,
        uses the default value set in `config.sharey`.
    hspace : float or None, default=`_default_flag`
        Height padding between subplots. If None,
        Matplotlib’s default spacing is used. If
        `_default_flag`, uses the default value set in
        `config.hspace`.
    wspace : float or None, default=`_default_flag`
        Width padding between subplots. If None,
        Matplotlib’s default spacing is used. If
        `_default_flag`, uses the default value set in
        `config.wspace`.
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
        uses the default value set in `config.Nticks`.
    aspect : float or None, default=None
        Changes the physical dimensions of the Axes,
        such that the ratio of the Axes height to the
        Axes width in physical units is equal to aspect.
        None will disable a fixed box aspect so that height
        and width of the Axes are chosen independently.

    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        The created Matplotlib Figure instance.
    axs : ndarray of `~matplotlib.axes.Axes`
        Flattened array of Axes objects, ordered row-wise.
    '''
    # get default config values
    nrows = get_config_value(nrows, 'nrows')
    ncols = get_config_value(ncols, 'ncols')
    figsize = get_config_value(figsize, 'grid_figsize')
    sharex = get_config_value(sharex, 'sharex')
    sharey = get_config_value(sharey, 'sharey')
    hspace = config.hspace if hspace is _default_flag else hspace
    wspace = config.wspace if wspace is _default_flag else wspace
    Nticks = config.Nticks if Nticks is _default_flag else Nticks

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
    ----------
    shape : int or tuple, default: 111
        The subplot specification. Can be given as a three-digit integer
        (e.g., 211 means 2 rows, 1 column, subplot index 1) or a tuple
        `(nrows, ncols, index)`.
    fig : matplotlib.figure.Figure or None, optional, default=None
        Existing figure to add the subplot to. If None,
        a new figure is created.
    figsize : tuple of float, optional, default=None
        Figure size in inches. If None, uses the
        default value set by `config.figsize`.
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
    -------
    ax : matplotlib.axes.Axes
        The created or retrieved subplot axes.
    fig : matplotlib.figure.Figure, optional
        The figure object containing the subplot.
        Returned only if `return_fig=True`.

    Examples
    --------
    Create a new figure and subplot:
    >>> fig, ax = add_subplot(return_fig=True)

    Add a subplot to an existing figure:
    >>> fig = plt.figure()
    >>> ax = add_subplot(fig=fig, shape=121)

    Create a 3D subplot:
    >>> fig, ax = add_subplot(projection='3d', return_fig=True)
    '''
    # get default config values
    figsize = get_config_value(figsize, 'figsize')
    # create figure if not passed in
    if fig is None:
        fig = plt.figure(figsize=figsize)
    # add desired subplot with projection
    ax = fig.add_subplot(shape, projection=projection, **kwargs)

    return (fig, ax) if return_fig else ax


def add_colorbar(im, ax, cbar_width=None,
                 cbar_pad=None, clabel=None,
                 rasterized=None):
    '''
    Add a colorbar next to an Axes.

    Parameters
    ----------
    im : matplotlib.cm.ScalarMappable
        The image, contour set, or mappable object returned by
        a plotting function (e.g., 'imshow', 'scatter', etc...).
    ax : matplotlib.axes.Axes
        The axes to which the colorbar will be attached.
    cbar_width : float or None, optional, default=None
        Width of the colorbar in figure coordinates.
        If None, uses the default value set in `config.cbar_width`.
    cbar_pad : float or None, optional, default=None
        Padding between the main axes and the colorbar
        in figure coordinates. If None, uses the default
        value set in `config.cbar_pad`.
    clabel : str, optional
        Label for the colorbar. If None, no label is set.
    rasterized : bool or None, default=None
        Whether to rasterize colorbar. Rasterization
        converts the artist to a bitmap when saving to
        vector formats (e.g., PDF, SVG), which can
        significantly reduce file size for complex plots.
        If None, uses default value set by `config.rasterized`
    '''
    # get default config values
    cbar_width = get_config_value(cbar_width, 'cbar_width')
    cbar_pad = get_config_value(cbar_pad, 'cbar_pad')
    rasterized = get_config_value(rasterized, 'rasterized')

    # extract figure from axes
    fig = ax.figure
    # add colorbar axes
    cax = fig.add_axes([ax.get_position().x1+cbar_pad, ax.get_position().y0,
                        cbar_width, ax.get_position().height])
    # add colorbar
    cbar = fig.colorbar(im, cax=cax, pad=0.04)
    # formatting and label
    cbar.ax.tick_params(which=config.cbar_tick_which, direction=config.cbar_tick_dir)
    if clabel is not None:
        cbar.set_label(fr'{clabel}')

    if rasterized:
        cbar.solids.set_rasterized(True)


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
    ----------
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
    cmap : str, optional, default=`config.cmap`
        Colormap used for plotting contours.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `fontsize` : float, default=`config.fontsize`
            Fontsize of contour labels.

    Returns
    -------
    cs : matplotlib.contour.QuadContourSet or mpl_toolkits.mplot3d.art3d.QuadContourSet3D
        The contour set object created by Matplotlib.
    '''
    # ---- KWARGS ----
    fontsize = kwargs.get('fontsize', config.fontsize)
    # get default config values
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


def set_axis_limits(
    xdata=None,
    ydata=None,
    *,
    ax=None,
    xlim=None,
    ylim=None,
    **kwargs,
):
    """
    Set axis limits on a Matplotlib Axes based on data range
    and optional user-specified limits.

    Parameters
    ----------
    xdata : array-like, list of array-like, or None, optional
        X-axis data. Can be a single array, a list of arrays, or None.
        If None, the X-axis will be autoscaled unless ``xlim`` is provided.
    ydata : array-like, list of array-like, or None, optional
        Y-axis data. Can be a single array, a list of arrays, or None.
        If None, the Y-axis will be autoscaled unless ``ylim`` is provided.
    ax : matplotlib.axes.Axes
        The Axes object on which to set the limits. Required.
    xlim : tuple of float, optional
        User-defined X-axis limits. If provided, only data within this range
        is considered when computing Y-axis limits automatically.
    ylim : tuple of float, optional
        User-defined Y-axis limits. If provided, only data within this range
        is considered when computing X-axis limits automatically.
    xpad : float or None, optional, default=None
        Fractional padding to apply to X-axis limits. If None,
        uses the default value from ``config.xpad``.
    ypad : float or None, optional, default=None
        Fractional padding to apply to Y-axis limits. If None,
        uses the default value from ``config.ypad``.

    Returns
    -------
    xlim_out : tuple of float
        The X-axis limits that were applied to the Axes.
    ylim_out : tuple of float
        The Y-axis limits that were applied to the Axes.

    Notes
    -----
    - If both ``xdata`` and ``ydata`` are provided as lists, they must have the same length
        or be broadcastable (single array broadcast across multiple arrays of the other axis).
    - Data outside user-provided ``xlim`` or ``ylim`` is ignored when computing automatic limits.
    - Both scalar and multi-dimensional arrays are flattened before processing.
    - If all data is ``None`` or empty, the corresponding axis will not be modified.
    """
    xpad_frac = kwargs.get('xpad', config.xpad)
    ypad_frac = kwargs.get('ypad', config.ypad)

    if ax is None:
        raise ValueError('ax must be an axes instance')

    if xdata is not None and ydata is not None:
        xdata = to_list(xdata)
        ydata = to_list(ydata)

        if len(xdata) == 1:
            xdata = xdata * len(ydata)
        if len(ydata) == 1:
            ydata = ydata * len(xdata)
        if len(xdata) != len(ydata):
            raise ValueError('Cannot broadcast xdata and ydata')

        xs, ys = [], []
        for x, y in zip(xdata, ydata):
            if x is None or y is None:
                continue

            x = np.asarray(x).ravel()
            y = np.asarray(y).ravel()
            if x.shape != y.shape:
                raise ValueError(
                    'Each x/y pair must match in shape! '
                    f'Got: x.shape: {x.shape}, y.shape: {y.shape}'
                )

            mask = np.ones_like(x, bool)
            if xlim is not None:
                mask &= (x >= get_value(xlim[0])) & (x <= get_value(xlim[1]))
            if ylim is not None:
                mask &= (y >= get_value(ylim[0])) & (y <= get_value(ylim[1]))

            if np.any(mask):
                xs.append(x[mask])
                ys.append(y[mask])

        xvals = np.concatenate(xs) if xs else None
        yvals = np.concatenate(ys) if ys else None

    else:
        xvals = None
        if xdata is not None:
            xvals = flatten(xdata)
            if xvals is not None and xlim is not None:
                mask = (xvals >= get_value(xlim[0])) & (xvals <= get_value(xlim[1]))
                xvals = xvals[mask] if np.any(mask) else xvals

        yvals = None
        if ydata is not None:
            yvals = flatten(ydata)
            if yvals is not None and ylim is not None:
                mask = (yvals >= get_value(ylim[0])) & (yvals <= get_value(ylim[1]))
                yvals = yvals[mask] if np.any(mask) else yvals

    if xlim is None and xvals is not None and len(xvals) > 0:
        xmin, xmax = np.nanmin(xvals), np.nanmax(xvals)
        xpad = xpad_frac * (xmax - xmin)
        xlim = (xmin - xpad, xmax + xpad)

    if ylim is None and yvals is not None and len(yvals) > 0:
        ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
        ypad = ypad_frac * (ymax - ymin)
        ylim = (ymin - ypad, ymax + ypad)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    return xlim, ylim


def set_axis_labels(
    X, Y, ax, xlabel=None, ylabel=None, use_brackets=None,
    show_physical_label=None, show_unit=None, fmt=None
):
    """
    Automatically generate and set axis labels from objects with physical
    units.

    This function creates formatted axis labels by combining an inferred
    physical type (e.g., 'Wavelength', 'Flux Density') with a formatted unit
    string (e.g., 'μm', 'MJy/sr'). Each component can be enabled or disabled
    independently, and custom labels may be provided.

    Parameters
    ----------
    X : '~astropy.units.Quantity' or object with a unit
        Data for the x-axis, typically a spectral axis (frequency,
        wavelength, or velocity).
    Y : '~astropy.units.Quantity' or object with a unit
        Data for the y-axis, typically flux, intensity, or surface brightness.
    ax : 'matplotlib.axes.Axes'
        Matplotlib axes object on which to set the labels.
    xlabel : str or None, optional
        Custom label for the x-axis. If None, the label is inferred from `X`.
    ylabel : str or None, optional
        Custom label for the y-axis. If None, the label is inferred from `Y`.
    use_brackets : bool or None, optional
        If True, wrap units in square brackets '[ ]'. If False, use
        parentheses '( )'. If None, uses the default value from
        `config.use_brackets`.
    show_physical_label : bool or None, optional
        If True, include the inferred physical type in the axis label.
        If None, uses the default value from `config.use_type_label`.
    show_unit : bool or None, optional
        If True, include the unit in the axis label. If None, uses the
        default value from `config.use_unit_label`.
    fmt : {'latex', 'latex_inline', 'inline'} or None, optional
        Format for unit rendering. Passed to `to_latex_unit`. If None,
        uses the default value from `config.unit_label_format`.

    Examples
    --------
    >>> import astropy.units as u
    >>> wavelength = np.linspace(1, 10, 100) * u.um
    >>> flux = np.random.random(100) * u.MJy / u.sr
    >>> fig, ax = plt.subplots()
    >>> ax.plot(wavelength, flux)
    >>> set_axis_labels(wavelength, flux, ax)
    # xlabel: 'Wavelength [μm]'
    # ylabel: 'Surface Brightness [MJy/sr]'

    >>> set_axis_labels(wavelength, flux, ax, show_physical_label=False)
    # xlabel: '[μm]'
    # ylabel: '[MJy/sr]'

    >>> set_axis_labels(wavelength, flux, ax, ylabel='Custom Flux')
    # ylabel: 'Custom Flux [MJy/sr]'

    Notes
    -----
    - Units are formatted using `to_latex_unit`, which provides LaTeX-friendly
      representations.
    - If both `show_physical_label` and `show_unit` are False, the resulting
      axis label is an empty string.
    """
    # get default config values
    use_brackets = get_config_value(use_brackets, 'use_brackets')
    show_physical_label = get_config_value(show_physical_label, 'use_type_label')
    show_unit = get_config_value(show_unit, 'use_unit_label')
    fmt = get_config_value(fmt, 'unit_label_format')

    # unit bracket type [] or ()
    brackets = [r'[', r']'] if use_brackets else [r'(', r')']

    xlabel = _format_axis_label(
        X, xlabel, brackets, show_physical_label, show_unit, fmt
    )
    ylabel = _format_axis_label(
        Y, ylabel, brackets, show_physical_label, show_unit, fmt
    )

    # set plot labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _format_axis_label(
    obj: Any,
    label: str | None,
    brackets: Sequence[str],
    show_physical_label: bool,
    show_unit: bool,
    fmt: str
) -> str:
    """
    Create a scientific axis label with physical type and unit information.

    This function generates axis labels in the format `<physical label> [<unit>]`,
    where the physical label describes what the axis represents (e.g., 'Wavelength',
    'Flux') and the unit is formatted in LaTeX notation. Both components can be
    customized or disabled independently.

    Parameters
    ----------
    obj : any
        An object from which physical type and unit information can be extracted.
        This may be an Astropy `Quantity`, a Spectrum-like object, or any object
        compatible with `get_unit` and `get_physical_type`.
    label : str or None
        If a string is provided, use it as the physical label directly, overriding
        any auto-detected physical type. If None, the physical label is inferred
        from the object's physical type (when `show_physical_label=True`).
    brackets : tuple of str
        A 2-element tuple specifying the opening and closing characters to wrap
        around the unit string. Common choices include `('[', ']')`, `('(', ')')`,
        or `('', '')` for no brackets.
    show_physical_label : bool
        If True, include the physical type label in the output. If False, omit
        the physical label entirely (useful for creating unit-only labels).
    show_unit : bool
        If True, include the unit in the output. If False, omit the unit
        (useful for creating label-only outputs).
    fmt : str or None
        Format string for unit rendering, passed to `to_latex_unit`. Common
        values include 'latex', 'latex_inline', or 'inline'. See `to_latex_unit`
        documentation for details.

    Returns
    -------
    str
        A formatted axis label string. The format depends on the parameters:
        - Label only: 'Wavelength'
        - Unit only: '[$\\mu$m]'
        - Both enabled: 'Wavelength ($\\mathrm{\\mu m}$)'
        - Neither: '' (empty string)

    Notes
    -----
    - If the object has no unit or the unit cannot be formatted, the unit
      portion is omitted regardless of `show_unit`.
    - The output is stripped of leading/trailing whitespace.
    """
    physical_type = get_physical_type(obj)

    # format physical label
    if isinstance(label, str):
        physical_label = label

    elif show_physical_label:
        inferred = _infer_physical_type_label(obj)
        if inferred is not None:
            physical_label = inferred
        elif physical_type is not None:
            physical_label = str(physical_type).replace('_', ' ').title()
        else:
            physical_label = ''

    else:
        physical_label = ''

    # format unit label
    unit = get_unit(obj)
    unit_str = to_latex_unit(unit, fmt=fmt)

    # create axis label
    if show_unit and unit_str is not None:
        unit_label = fr'{brackets[0]}{unit_str}{brackets[1]}'
    else:
        unit_label = ''

    return fr'{physical_label} {unit_label}'.strip()


def spectral_axis_label(
    spectral_axis: SpectralAxis | Quantity,
    idx,
    ax,
    *,
    ref_unit,
    radial_vel=None,
    emission_line=None,
    as_title=False,
    **kwargs
):
    """
    Add a label indicating the spectral coordinate of a slice.

    This function computes a representative spectral coordinate value for a
    given index or index range along a spectral axis and renders a LaTeX-formatted
    label on a matplotlib Axes. The spectral axis is first converted to the
    specified reference unit and optionally shifted by a radial velocity.

    The label can be displayed either as an axes title or as text positioned
    within the axes.

    Parameters
    ----------
    spectral_axis : SpectralAxis or Quantity
        Spectral axis array representing wavelength, frequency, or velocity.
        Must have valid physical units convertible via ``astropy.units.spectral()``
        equivalencies.
    idx : int, list of int, or None
        Index or index range specifying the slice:
        - ``i`` → label corresponding to spectral_axis[i]
        - ``[i]`` → label corresponding to spectral_axis[i]
        - ``[i, j]`` → label corresponding to midpoint of spectral_axis[i:j]
        - ``None`` → label corresponding to midpoint of entire spectral axis
    ax : matplotlib.axes.Axes
        Target matplotlib Axes on which the label will be rendered.
    ref_unit : UnitBase or str
        Reference unit to which the spectral axis will be converted prior to
        computing the label (e.g., ``u.nm``, ``u.AA``, ``u.Hz``, ``u.km/u.s``).
    radial_vel : Quantity or None, optional, default=None
        Radial velocity used to Doppler-shift the spectral axis before computing
        the representative value. Must be velocity-compatible if provided.
    emission_line : str or None, optional, default=None
        Optional emission line identifier to include in the label
        (e.g., ``"H alpha"``, ``"[O III]"``). If provided, this replaces the
        default spectral symbol prefix.
    as_title : bool, optional, default=False
        If True, render the label as the axes title. Otherwise, render as text
        inside the axes.
    text_loc : tuple of float, optional
        Axes-relative coordinates (x, y) for text placement. Default is
        ``config.text_loc``.
    text_color : str, optional
        Text color. Default is ``config.text_color``.
    highlight : bool, optional
        If True, draw a white background box behind the label text.
        Default is ``config.highlight``.

    Raises
    ------
    ValueError
        If spectral_axis is None or does not have valid units.
    """
    text_loc = kwargs.get('text_loc', config.text_loc)
    text_color = kwargs.get('text_color', config.text_color)
    highlight = kwargs.get('highlight', config.highlight)

    spectral_axis = get_spectral_axis(spectral_axis)
    if spectral_axis is None:
        raise ValueError(
            'spectral_axis cannot be None! '
            f'got: {_type_name(spectral_axis)}'
        )

    # compute spectral axis value of slice for label
    spectral_axis = convert_quantity(spectral_axis, ref_unit, equivalencies=u.spectral())
    spectral_unit = spectral_axis.unit
    if spectral_unit is None:
        raise ValueError(
            'spectral_axis must have a unit!'
        )

    spectral_axis = shift_by_radial_vel(spectral_axis, radial_vel)
    spectral_value = spectral_idx_2_world(spectral_axis, idx, keep_unit=False)

    slice_label = _format_spectral_label(
        spectral_value, spectral_unit, emission_line=emission_line
    )

    if as_title:
        ax.set_title(slice_label, color=text_color, loc='center')
    else:
        bbox = dict(facecolor='white', edgecolor='w') if highlight else None
        ax.text(
            text_loc[0], text_loc[1], slice_label,
            transform=ax.transAxes, color=text_color, bbox=bbox
        )


def _format_spectral_label(
    spectral_value: float,
    spectral_unit,
    *,
    emission_line: str | None = None
) -> str:
    """
    Format a LaTeX label representing a spectral coordinate value.

    This function generates a LaTeX-formatted string suitable for use as a
    matplotlib text label or title. The label represents a spectral axis value
    (e.g., wavelength, frequency, or velocity), optionally prefixed with an
    emission line identifier.

    Used internally by ``spectral_axis_label``.

    Parameters
    ----------
    spectral_value : float
        Spectral axis value expressed in the specified ``spectral_unit``.

    spectral_unit : Unit
        Unit associated with ``value``. Must have a valid physical type such
        as ``length``, ``frequency``, or ``speed``.

    emission_line : str or None, optional, default=None
        Optional emission line identifier (e.g., ``"H alpha"``, ``"[O III]"``).
        If provided, this replaces the default spectral symbol prefix.

    Returns
    -------
    label : str
        LaTeX-formatted spectral label string enclosed in math mode delimiters.
    """

    unit_label = to_latex_unit(spectral_unit)

    spectral_type = {
        'length': r'\lambda = ',
        'frequency': r'f = ',
        'speed': r'v = '
    }.get(str(spectral_unit.physical_type))

    if emission_line is None:
        return fr"${spectral_type}{spectral_value:0.2f}\,{unit_label.strip('$')}$"

    # replace spaces with latex format
    emission_label = emission_line.replace(' ', r'\ ')
    return (
        fr"$\mathrm{{{emission_label}}}\,{spectral_value:0.2f}\,{unit_label.strip('$')}$"
        )


def _figure_utils(ax, **kwargs):

    ellipses = kwargs.get('ellipses', None)
    vlines = kwargs.get('vlines', None)
    hlines = kwargs.get('hlines', None)

    plot_ellipses(ellipses, ax)

    plot_vlines(vlines, ax)
    plot_hlines(hlines, ax)



# Plot Matplotlib Patches and Shapes
# ----------------------------------
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
    ----------
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
        uses the default value set in `config.linewidth`.
    fill : bool or None, optional, default=None
        Whether the circles are filled. If None,
        uses the default value set in `config.circle_fill`.
    cmap : str or None, optional, default=None
        matplolib cmap used to sample default circle colors.
        If None, uses the default value set in `config.cmap`.
    '''
    # get default config values
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


def plot_ellipses(ellipses, ax):
    """
    Plots an ellipse or list of ellipses to an axes.

    Parameters
    ----------
    ellipses : matplotlib.patches.Ellipse or list
        The Ellipse or list of Ellipses to plot.
    ax : matplotlib.axes.Axes
        Matplotlib axis on which to plot the ellipses(s).
    """
    if ellipses is not None:
        ellipses = to_list(ellipses)

        for ellipse in ellipses:
            if not isinstance(ellipse, Ellipse):
                raise ValueError(
                    'ellipses must contain matplotlib.patches.Ellipse instances! '
                    f'got: {_type_name(ellipse)}'
                )
            ax.add_patch(_copy_ellipse(ellipse))


def _copy_ellipse(ellipse):
    """
    Returns a copy of an Ellipse object.

    This function is used to avoid running into
    a matplotlib error when plotting the same
    artist onto multiple figures.

    Parameters
    ----------
    ellipse : matplotlib.patches.Ellipse
        The Ellipse object to copy.

    Returns
    -------
    matplotlib.patches.Ellipse
        A new Ellipse object with the same properties as the input.
    """
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


def plot_interactive_ellipse(center, w, h, ax, text_loc=None,
                             text_color=None, highlight=None,
                             angle=0.0, rotation_step=5.0):
    """
    Create an interactive ellipse selector on an Axes
    along with an interactive text window displaying
    the current ellipse center, width, and height.

    Parameters
    ----------
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
        If None, uses the default value set in `config.text_loc`.
    text_color : str or None, optional, default=None
        Color of the annotation text. If None, uses
        the default value set in `config.text_color`.
    highlight : bool or None, optional, default=None
        If True, adds a bbox to highlight the text. If None,
        uses the default value set in `config.highlight`.

    Notes
    -----
    Ensure an interactive backend is active. This can be
    activated with use_interactive().
    """
    text_loc = get_config_value(text_loc, 'ellipse_label_loc')
    text_color = get_config_value(text_color, 'text_color')
    highlight = get_config_value(highlight, 'highlight')

    facecolor = 'k' if text_color == 'w' else 'w'
    bbox = dict(facecolor=facecolor, alpha=0.6, edgecolor="none") if highlight else None

    text = ax.text(
        text_loc[0], text_loc[1], '',
        transform=ax.transAxes,
        size='small',
        color=text_color,
        bbox=bbox
    )

    # create and store region explicitly
    region = EllipsePixelRegion(
        center=PixCoord(x=center[0], y=center[1]),
        width=w,
        height=h,
        angle=angle * u.deg
    )

    selector = region.as_mpl_selector(
        ax,
        callback=partial(_update_ellipse_region, text=text)
    )

    ax._ellipse_selector = selector
    ax._ellipse_region = region

    artist = selector._selection_artist
    artist.angle = angle

    # initialize display
    _update_ellipse_region(region, text=text)

    def _rotate(event):

        key = event.key

        if key not in ('e', 'f', 't', 'o'):
            return

        artist = selector._selection_artist
        region = ax._ellipse_region

        if key == 'e':
            artist.angle -= rotation_step

        elif key == 'f':
            artist.angle += rotation_step

        elif key == 't':
            artist.angle = 0.0

        elif key == 'o':
            selector.set_active(not selector.active)
            return

        # synchronize region ← artist
        region.angle = -artist.angle * u.deg

        artist.figure.canvas.draw_idle()

        _update_ellipse_region(region, text=text)

    ax.figure.canvas.mpl_connect("key_press_event", _rotate)

    return selector


def _update_ellipse_region(region, text):
    """
    Update ellipse information text when the
    interactive region is modified.

    Parameters
    ----------
    region : regions.EllipsePixelRegion
        The ellipse region being updated.
    text : matplotlib.text.Text
        The text object used to display ellipse parameters.
    """
    x_center = region.center.x
    y_center = region.center.y
    width = region.width
    height = region.height
    angle = region.angle
    major = max(width, height)
    minor = min(width, height)

    # display properties
    text.set_text(
        f'Center: [{x_center:.1f}, {y_center:.1f}]\n'
        f'Major: {major:.1f}\n'
        f'Minor: {minor:.1f}\n'
        f'Angle: {angle:.1f}\n'
    )


def ellipse_patch(
    center, w, h, angle=0, fill=False,
    edgecolor='k', facecolor=None, **kwargs
):
    """
    Create a matplotlib.patches.Ellipse object.

    Parameters
    ----------
    center : tuple of float
        (x, y) coordinates of the ellipse center.
    w : float
        Width of the ellipse (along x-axis before rotation).
    h : float
        Height of the ellipse (along y-axis before rotation).
    angle : float, optional, default=0
        Rotation angle of the ellipse in degrees (counterclockwise).
    fill : bool, optional, default=False
        Whether the ellipse should be filled (True) or only outlined (False).
    edgecolor : str, optional, default='k'
        Color of patch edges.
    facecolor : str, optional, default=None
        Color of patch face. If not ``None``,
        sets ``fill=True``.
    linestyle or ls : {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}, optional
        Linestyle of ellipse patch.
    linewidth or lw : float or None, optional
        Linewidth of ellipse patch.
    Any other kwargs accepted by matplotlib.patches.Ellipse

    Returns
    -------
    ellipse : matplotlib.patches.Ellipse
        An Ellipse patch that can be added to a matplotlib Axes.
    """
    if facecolor is not None:
        fill = True

    return Ellipse(
        xy=(center[0], center[1]), width=w, height=h,
        angle=angle, fill=fill, edgecolor=edgecolor,
        facecolor=facecolor, **kwargs
    )


def plot_points(points, ax, color='r', size=20, marker='*'):
    '''
    Plot points on a given Matplotlib axis with customizable style.

    Parameters
    ----------
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


def plot_vlines(vlines, ax, unit=None):
    """
    Plot one or more vertical reference lines on a Matplotlib axis.

    Parameters
    ----------
    vlines : float, astropy.units.Quantity, iterable of float or Quantity, or None
        X-axis coordinate(s) at which to draw vertical line(s). If a Quantity,
        each value is converted to ``unit`` before plotting. If an iterable is
        provided, a vertical line is drawn for each element. If None, no lines
        are drawn.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object on which to draw the vertical line(s).
    unit : astropy.units.UnitBase or str, optional, default=None
        Unit to which Quantity values in ``vlines`` are converted before plotting.
        If None, Quantity inputs must already be in the axis unit system or must
        not require conversion.

    Notes
    -----
    Vertical lines are drawn using ``ax.axvline`` with a dotted linestyle,
    linewidth of 1.0, black color, alpha of 0.7, and z-order of 0.
    """
    if vlines is not None:
        vlines = to_list(vlines)
        unit = to_unit(unit)

        for vline in vlines:
            if isinstance(vline, Quantity):
                vline = vline.to(unit).value
            ax.axvline(
                vline,
                ls=':',
                lw=1.0,
                color='k',
                alpha=0.7,
                zorder=0,
            )


def plot_hlines(hlines, ax, unit=None):
    """
    Plot one or more horizontal reference lines on a Matplotlib axis.

    Parameters
    ----------
    vlines : float, astropy.units.Quantity, iterable of float or Quantity, or None
        Y-axis coordinate(s) at which to draw horizontal line(s). If a Quantity,
        each value is converted to ``unit`` before plotting. If an iterable is
        provided, a horizontal line is drawn for each element. If None, no lines
        are drawn.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object on which to draw the horizontal line(s).
    unit : astropy.units.UnitBase or str, optional, default=None
        Unit to which Quantity values in ``hlines`` are converted before plotting.
        If None, Quantity inputs must already be in the axis unit system or must
        not require conversion.

    Notes
    -----
    Horizontal lines are drawn using ``ax.axhline`` with a dotted linestyle,
    linewidth of 1.0, black color, alpha of 0.7, and z-order of 0.
    """
    if hlines is not None:
        hlines = to_list(hlines)
        unit = to_unit(unit)

        for hline in hlines:
            if isinstance(hline, Quantity):
                hline = hline.to(unit).value
            ax.axhline(
                hline,
                ls=':',
                lw=1.0,
                color='k',
                alpha=0.7,
                zorder=0,
            )


# Notebook Utils
# --------------
def inline():
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
        close()
        ipython.run_line_magic('matplotlib', 'inline')
    except Exception as e:
        print(f'Unable to set inline backend: {e}')


def interactive():
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


def close():
    '''
    Closes all interactive plots in session.
    '''
    plt.close('all')
