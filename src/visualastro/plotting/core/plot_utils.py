"""
Author: Elko Gerville-Reache
Date Created: 2025-05-24
Date Modified: 2026-03-14
Description:
    Plotting utility functions.
"""

from contextlib import contextmanager
from importlib.resources import files
from typing import Literal
import warnings
from functools import partial

import astropy.units as u
from astropy.units import Quantity
from astropy.visualization import ImageNormalize
from astropy.visualization.wcsaxes.core import WCSAxes
import matplotlib.axes as maxes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import AsinhNorm, LogNorm, PowerNorm, TwoSlopeNorm
from matplotlib.patches import Circle, Ellipse
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import matplotlib.ticker as ticker
import numpy as np
from numpy.typing import ArrayLike, NDArray
from regions import PixCoord, EllipsePixelRegion
from spectral_cube import SpectralCube

from visualastro.analysis.image_utils import stack_cube
from visualastro.core.config import (
    get_config_value,
    config,
    _Unset,
    _UNSET,
    _resolve_default
)
from visualastro.core.io import _extract_kwargs, _kwarg
from visualastro.core.numerical import kde2d
from visualastro.core.numerical_utils import (
    get_data,
    get_value,
    to_array,
    to_list,
    _cycle,
    _is_iterable,
    _is_scalar,
    _is_1d,
    _is_2d
)
from visualastro.core.units import to_unit
from visualastro.datamodels.datacube import DataCube
from visualastro.datamodels.fitsfile import FitsFile
from visualastro.plotting.core.colors import sample_cmap


@contextmanager
def style(name: str | _Unset = _UNSET, rc: dict | None = None, **rc_kwargs):
    """
    Context manager to temporarily apply a Matplotlib or VisualAstro style,
    with optional rcParams overrides.

    Parameters
    ----------
    name : str | _Unset, optional, default=_UNSET
        Matplotlib or VisualAstro style name. If `_UNSET`,
        uses `config.style`. Ex: 'astro' or 'latex'.
    rc : dict, optional
        Dictionary of rcParams overrides.
        Ex: {'font.size': 14}
    **rc_kwargs :
        Additional rcParams overrides supplied as keyword arguments.
        Use underscores in place of dots: font_size → font.size

    Examples
    --------
    >>> with style('latex', font_size=23, axes_labelsize=40):
    ...     plt.plot(x, y)

    >>> with style('paper', rc={'font.size': 14, 'lines.linewidth': 2}):
    ...     fig, ax = plt.subplots()

    >>> with style('astro', rc={'font.size': 12}, xtick_labelsize=10):
    ...     # rc dict and kwargs are merged (kwargs take precedence)
    ...     plt.plot(x, y)
    """
    name = _resolve_default(name, config.style)
    style_name = _get_stylepath(name)

    # update rcParams, with priority to kwargs
    rc_combined = {}
    if rc is not None:
        rc_combined.update(rc)
    if rc_kwargs:
        # replace '_' with '.' for rcParams
        rc_combined.update({
            k.replace('_', '.'): v for k, v in rc_kwargs.items()
        })

    context = [style_name, rc_combined] if rc_combined else style_name

    with plt.style.context(context):
        yield


def _get_stylepath(style: str) -> str:
    """
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
    """
    # if style is a default matplotlib stylesheet
    if style in mplstyle.available:
        return style

    # if style is a visualastro stylesheet
    stylelib = files('visualastro').joinpath('stylelib')
    base_style = style.split('_')[0] if '_' in style else style
    style_path = stylelib.joinpath(f'{base_style}.mplstyle')

    # ensure that style works on computer, otherwise return default style
    try:
        with plt.style.context(str(style_path)):
            # pass if can load style successfully on computer
            pass
        return str(style_path)
    except Exception as e:
        warnings.warn(
            f"[visualastro] Could not apply style '{style}' ({e}). "
            f"Falling back to '{config.style_fallback}' style."
        )
        style = config.style_fallback
        base_style = style.split('_')[0] if '_' in style else style
        return str(stylelib.joinpath(f'{base_style}.mplstyle'))


def apply_style_modifiers(ax, style: str):
    """
    Apply programmatic style modifiers based on underscore-separated suffixes.
    This updates an axes instance in place with stylistic modifiers.

    Modifiers are appended to the base style name with underscores and can be
    chained together in any order (e.g., 'astro_minimal_grid' or 'latex_bare').
    This function is mostly for internal use by plotting functions in the
    visualastro.plotting.ax module.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or astropy.visualization.wcsaxes.WCSAxes
        Axes object to apply style modifiers to.
    style : str
        Full style string including base style and optional modifiers.
        Format: 'basestyle_modifier1_modifier2_...'
        Example: 'astro_minimal_grid'

    Notes
    -----
    Supported modifiers:
        - minimal : Remove minor ticks and show ticks only on bottom-left axes.
                    For WCSAxes, uses coords positioning. For regular axes,
                    disables top and right ticks.
        - nominor : Remove minor tick marks only, keeping major ticks unchanged.
        - bare : Remove all frame elements including ticks, tick labels, and
                spines/frame. Creates a minimal plot with data only.
        - grid : Add a background grid. Uses config settings for color, alpha,
                and linestyle. Grid style differs between WCSAxes and regular axes.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> apply_style_modifiers(ax, 'astro_minimal')
    >>> apply_style_modifiers(ax, 'latex_grid_nominor')
    >>> apply_style_modifiers(ax, 'default_bare')
    """
    if '_' not in style:
        return

    parts = style.split('_')
    modifiers = parts[1:]

    for modifier in modifiers:

        modifier = modifier.lower()

        if modifier == 'minimal':
            # remove minor ticks
            ax.tick_params(which='minor', length=0)
            if isinstance(ax, WCSAxes):
                ax.coords['ra'].set_ticks_position('bl')
                ax.coords['dec'].set_ticks_position('bl')
            else:
                ax.tick_params(top=False, right=False)

        elif modifier == 'nominor':
            ax.tick_params(which='minor', length=0)

        elif modifier == 'bare':
            # remove the frame, ticks, and ticklabels
            if isinstance(ax, WCSAxes):
                ax.coords['ra'].set_ticklabel_visible(False)
                ax.coords['dec'].set_ticklabel_visible(False)
                ax.coords['ra'].set_ticks_visible(False)
                ax.coords['dec'].set_ticks_visible(False)
                ax.coords.frame.set_linewidth(0)
            else:
                ax.tick_params(which='both', length=0)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

        elif modifier == 'grid':
            if isinstance(ax, WCSAxes):
                 ax.coords.grid(
                     True,
                     color=config.wcs_grid_color,
                     alpha=config.grid_alpha,
                     ls=config.wcs_grid_linestyle
                 )
            else:
                ax.grid(
                    True,
                    color=config.grid_color,
                    alpha=config.grid_alpha,
                    ls=config.grid_linestyle
                )


# Axes Labels, Format, and Styling
# --------------------------------
def gridspec(
    nrows: int | _Unset = _UNSET,
    ncols: int | _Unset = _UNSET,
    figsize: tuple[float, float] | _Unset = _UNSET,
    sharex: bool | _Unset = _UNSET,
    sharey: bool | _Unset = _UNSET,
    hspace: float | None | _Unset = _UNSET,
    wspace: float | None | _Unset = _UNSET,
    width_ratios: ArrayLike | None = None,
    height_ratios: ArrayLike | None = None,
    fancy_axes: bool = False,
    Nticks: int | None | _Unset = _UNSET,
    aspect: float | None = None
):
    """
    Create a grid of Matplotlib axes panels with consistent sizing
    and optional fancy tick styling.

    Parameters
    ----------
    nrows : int | _Unset, optional, default=_UNSET
        Number of subplot rows. If `_UNSET`, uses `config.nrows`.
    ncols : int | _Unset, optional, default=_UNSET
        Number of subplot columns. If `_UNSET`, uses `config.ncols`.
    figsize : tuple[float, float] | _Unset, optional, default=_UNSET
        Figure size in inches as (width, height). If `_UNSET`,
        uses `config.grid_figsize`.
    sharex : bool | _Unset, optional, default=_UNSET
        If `True`, share the x-axis among all subplots. If `_UNSET`,
        uses `config.axes.sharex`.
    sharey : bool | _Unset, optional, default=_UNSET
        If `True`, share the y-axis among all subplots. If `_UNSET`,
        uses `config.axes.sharey`.
    hspace : float | None | _Unset, optional, default=`_UNSET`
        Height padding between subplots. If `None`,
        Matplotlib’s default spacing is used. If `_UNSET`,
        uses `config.axes.hspace`.
    wspace : float | None | _Unset, optional, default=`_UNSET`
        Width padding between subplots. If `None`,
        Matplotlib’s default spacing is used. If `_UNSET`,
        uses `config.axes.wspace`.
    width_ratios : ArrayLike | None, optional, default=None
        ArrayLike of length `ncols` defining the width padding between
        subplots. If `None`, Matplotlib’s default spacing is used.
        Defines the relative widths of the columns. Each column gets a
        relative width of `width_ratios[i] / sum(width_ratios)`. If not
        given, all columns will have the same width.
    height_ratios : ArrayLike | None, optional, default=None
        ArrayLike of length `nrows` defining the relative heights of the rows.
        Each row gets a relative height of `height_ratios[i] / sum(height_ratios)`.
        If not given, all rows will have the same height.
    fancy_axes : bool, optional, default=False
        If True, enables 'fancy' axes styling:

        * minor ticks on
        * inward ticks on all sides
        * axes labels on outer grid axes
        * h/wspace = 0.0

    Nticks : int | None | _Unset, optional, default=`_UNSET`
        Maximum number of major ticks per axis. If `None`,
        uses the default matplotlib settings. If `_UNSET`,
        uses `config.axes.Nticks`.
    aspect : float | None, optional, default=None
        Changes the physical dimensions of the Axes,
        such that the ratio of the Axes height to the
        Axes width in physical units is equal to aspect.
        None will disable a fixed box aspect so that height
        and width of the Axes are chosen independently.

    Returns
    -------
    fig : ~matplotlib.figure.Figure
        The created Matplotlib Figure instance.
    axs : ndarray[~matplotlib.axes.Axes]
        Flattened array of Axes objects, ordered row-wise.
    """
    nrows = _resolve_default(nrows, config.nrows)
    ncols = _resolve_default(ncols, config.ncols)
    figsize = _resolve_default(figsize, config.grid_figsize)
    sharex = _resolve_default(sharex, config.axes.sharex)
    sharey = _resolve_default(sharey, config.axes.sharey)
    hspace = _resolve_default(hspace, config.axes.hspace)
    wspace = _resolve_default(wspace, config.axes.wspace)
    Nticks = _resolve_default(Nticks, config.axes.Nticks)

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
        Figure size in inches. If None, uses `config.figsize`.
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


def add_colorbar(
    im: ScalarMappable,
    ax: maxes.Axes,
    cbar_width: float | _Unset = _UNSET,
    cbar_pad: float | _Unset = _UNSET,
    label: str | None = None,
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
        cbar.set_label(fr'{label}')

    if rasterized:
        cbar.solids.set_rasterized(True)


def legend(*args, ax, **kwargs) -> None:
    """
    Create a legend on the specified axes with configuration defaults.

    Parameters
    ----------
    *args : tuple
        Positional arguments for legend specification:

        * If 1 arg: labels only
        * If 2 args: handles, labels

        Maximum of 2 positional arguments allowed.
    ax : matplotlib.axes.Axes
        The axes object on which to place the legend.
    handles : Sequence, optional
        Artists (lines, patches) to display in legend.
    labels : Sequence, optional
        Text labels corresponding to artists.
    loc : str, optional, default=config.legend.loc
        Legend location.
    ncols : int, optional, default=config.legend.ncols
        Number of columns.
    fontsize : int | str, optional, default=config.legend.fontsize
        Font size for legend text.
    fancybox : bool, optional, default=config.legend.fancybox
        Enable rounded box frame.
    framealpha : float, optional, default=config.legend.framealpha
        Frame alpha transparency [0, 1].
    facecolor : str, optional, default=config.legend.facecolor
        Frame background color.
    edgecolor : str, optional, default=config.legend.edgecolor
        Frame edge color.
    title : str, optional, default=config.legend.title
        Legend title.
    alignment : {'center', 'left', 'right'}, optional, default=config.legend.alignment
        Legend alignment.
    columnspacing : float, optional, default=config.legend.columnspacing
        Spacing between columns in units of fontsize.
    draggable : bool, optional, default=config.legend.draggable
        Enable legend dragging.

    Raises
    ------
    ValueError
        If more than 2 positional arguments provided.

    Returns
    -------
    None
    """

    legend_kwargs = _extract_kwargs(
        kwargs,
        additional_kwargs=[
            _kwarg('loc', config.legend.loc),
            _kwarg('ncols', config.legend.ncols),
            _kwarg('fontsize', config.legend.fontsize),
            _kwarg('fancybox', config.legend.fancybox),
            _kwarg('framealpha', config.legend.framealpha),
            _kwarg('facecolor', config.legend.facecolor),
            _kwarg('edgecolor', config.legend.edgecolor),
            _kwarg('title', config.legend.title),
            _kwarg('alignment', config.legend.alignment),
            _kwarg('columnspacing', config.legend.columnspacing),
            _kwarg('draggable', config.legend.draggable),
        ]
    )

    handles = None
    labels = None

    if len(args) == 1:
        labels = args[0]
    elif len(args) == 2:
        handles, labels = args
    elif len(args) > 2:
        raise ValueError('legend() takes at most 2 positional arguments')

    handles = kwargs.pop('handles', handles)
    labels = kwargs.pop('labels', labels)

    if handles is not None:
        legend_kwargs['handles'] = handles
    if labels is not None:
        legend_kwargs['labels'] = labels

    ax.legend(**legend_kwargs)


def contour(x, y, ax, levels=20, contour_method='contour',
            bw_method: Literal['scott', 'silverman']='scott',
            gridsize=200, padding=0.2,
            cslabel=False, zdir=None, offset=None, cmap=None,
            zorder=None, xlim=None, ylim=None, **kwargs):
    """
    Add 2D or 3D Gaussian KDE density contours to an axis.
    This function computes a 2D Gaussian kernel density estimate (KDE)
    from input data (`x`, `y`) using `kde2d` and plots
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
        `'contourf'` draws filled contours.
    bw_method : str, scalar or callable, optional, default='scott'
        The method used to calculate the bandwidth factor for the Gaussian KDE.
        Can be one of:
        - `'scott'` or `'silverman'`: use standard rules of thumb.
        - a scalar constant: directly used as the bandwidth factor.
        - a callable: should take a `scipy.stats.gaussian_kde` instance as its
            sole argument and return a scalar bandwidth factor.
    gridsize : int, default=200
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
    cmap : str, optional, default=config.cmap
        Colormap used for plotting contours.

    **kwargs : dict, optional
        Additional parameters.

        Supported keywords:

        - `fontsize` : float, default=`config.fontsize`
            Fontsize of contour labels.

    Returns
    -------
    cs : `matplotlib.contour.QuadContourSet` or `mpl_toolkits.mplot3d.art3d.QuadContourSet3D`
        The contour set object created by Matplotlib.
    """
    # ---- KWARGS ----
    fontsize = kwargs.get('fontsize', config.fontsize)
    cmap = get_config_value(cmap, 'cmap')

    c_method = contour_method.lower()
    contour_methods = {
        'contour': ax.contour,
        'contourf': ax.contourf
    }
    contour_func = contour_methods.get(c_method, ax.contour)
    c_method_name = c_method if c_method in contour_methods else 'contour'

    contour_method = {
        'contour': ax.contour,
        'contourf': ax.contourf
    }.get(contour_method.lower(), ax.contour)

    # compute kde density
    X, Y, Z = kde2d(
        x, y,
        bw_method=bw_method,
        gridsize=gridsize,
        padding=padding,
        xlim=xlim, ylim=ylim
    )

    if zorder is None:
        zorder = config.zorder.contour if c_method_name == 'contour' else config.zorder.contourf

    # plot contours as either 3D projections or a simple 2D plot
    valid_zdirs = {'x', 'y', 'z'}
    zdir = zdir.lower() if isinstance(zdir, str) else None
    if zdir in valid_zdirs and offset is not None:
        if zdir == 'z':
            cs = contour_func(
                X, Y, Z, levels=levels, cmap=cmap, zdir=zdir, offset=offset, zorder=zorder
            )
        elif zdir == 'y':
            cs = contour_func(
                X, Z, Y, levels=levels, cmap=cmap, zdir=zdir, offset=offset, zorder=zorder
            )
        else:
            cs = contour_func(
                Z, Y, X, levels=levels, cmap=cmap, zdir=zdir, offset=offset, zorder=zorder
            )
    else:
        cs = contour_func(X, Y, Z, levels=levels, cmap=cmap, zorder=zorder)

    if cslabel:
        ax.clabel(cs, fontsize=fontsize)

    return cs


def contourf(
    x,
    y,
    ax,
    levels=20,
    bw_method: Literal['scott', 'silverman'] ='scott',
    gridsize=200,
    padding=0.2,
    cslabel=False,
    zdir=None,
    offset=None,
    cmap=None,
    zorder=None,
    **kwargs,
):
    """
    Filled contour wrapper around `contour`.

    Equivalent to calling `contour(..., contour_method='contourf')`.

    See Also
    --------
    contour : Full parameter documentation.
    """
    return contour(
        x,
        y,
        ax,
        levels=levels,
        contour_method='contourf',
        bw_method=bw_method,
        gridsize=gridsize,
        padding=padding,
        cslabel=cslabel,
        zdir=zdir,
        offset=offset,
        cmap=cmap,
        zorder=zorder,
        **kwargs,
    )


def _extract_xy(
    *data: float | u.Quantity | NDArray | list[float | u.Quantity | NDArray],
    order: Literal['c', 'fortran'] | _Unset = _UNSET,
    index_spec: Literal['implicit', 'explicit'] | tuple[int, int] = 'implicit'
) -> tuple[
    float | u.Quantity | NDArray | list[float | u.Quantity | NDArray] | None,
    float | u.Quantity | NDArray | list[float | u.Quantity | NDArray],
]:
    """
    Extract X and Y coordinates from flexible numeric inputs.

    Interprets input based on dimensionality and structure. For 2D arrays,
    extracts columns according to `order` and `index_spec`. Returns X as None
    if only Y values are detected.

    This function is used by `_normalize_plotting_input(s)`.

    Parameters
    ----------
    *data : tuple
        Input data. Supported forms:

        * Single argument:
            * 1D array-like or Quantity: Y values, X = None
            * 2D array or Quantity: extract X, Y according to `order` and `index_spec`
            * list/tuple of scalars: Y values, X = None
            * scalar or scalar Quantity: single Y value, X = None
            * Two arguments: (X, Y) pairs passed through unchanged

    order : {'c', 'fortran'} | _Unset, optional, default=_UNSET
        Memory layout for 2D input interpretation. Defines what a
        column is for `index_spec`.

        * 'c': row-major, shape (N, 2)
        * 'fortran': column-major, shape (2, N)

        If `_UNSET`, uses `config.array_order`.
    index_spec : {'implicit', 'explicit'} | tuple[int, int], optional
        Column extraction mode for 2D inputs.

        * 'implicit': return (None, [col_0, col_1, ...])
        * 'explicit': return (col_0, col_1)
        * tuple (i, j): return (col_i, col_j)

    Returns
    -------
    X : ndarray | Quantity | scalar | list | None
        X coordinates. None indicates implicit indexing (caller should generate).
        Type preserves input semantics.
    Y : ndarray | Quantity | scalar | list
        Y coordinates. Preserves input type.

    Raises
    ------
    ValueError
        If input has unsupported dimensionality, structure, or count.

    Notes
    -----
    For 2D inputs with 'implicit' mode, X is returned as None and Y as a list
    of column arrays, delegating index generation to the caller.
    """
    array_order = _resolve_default(order, config.array_order)

    if len(data) == 1:
        obj = data[0]
        if isinstance(obj, (np.ndarray, u.Quantity)):
            if obj.ndim == 1:
                return None, obj
            if obj.ndim == 2:
                if array_order.lower() == 'c':
                    axis = 0
                    get_col = lambda i: obj[:, i]
                else:
                    axis = 1
                    get_col = lambda i: obj[i, :]

                if isinstance(index_spec, (list, tuple)):
                    ix, iy = index_spec
                    return get_col(ix), get_col(iy)
                elif index_spec == 'explicit':
                    return get_col(0), get_col(1)
                elif index_spec == 'implicit':
                    y = [
                        get_col(i) for i in range(
                            obj.shape[1] if axis == 0 else obj.shape[0]
                        )
                    ]
                    return None, y

        if _is_scalar(obj):
            return None, obj

        if isinstance(obj, (list, tuple)):
            if all(_is_scalar(x) for x in obj):
                return None, obj

        raise ValueError(f'Unsupported input type {type(obj).__name__}')

    if len(data) == 2:
        return data[0], data[1]

    raise ValueError(f'Expected 1 or 2 arguments, got {len(data)}')


def _normalize_plotting_inputs(
    *data: float | u.Quantity | NDArray | list[float | u.Quantity | NDArray],
    order: Literal['c', 'fortran'] | _Unset = _UNSET,
    index_spec: Literal['implicit', 'explicit'] | tuple[int, int] = 'implicit',
    mode: Literal['plot', 'scatter'] = 'scatter'
):
    """
    Extract and normalize X, Y inputs for plotting.

    Handles variable input formats and ensures X and Y have compatible
    dimensionality for broadcasting. Generates implicit X indices when not
    provided. Wraps scalars and 1D data in lists to match 2D counterparts.

    See `visualastro.plotting.core.plot_utils._extract_xy` for documentation
    on how `*data` is interpreted.

    Parameters
    ----------
    *data : float | Quantity | ndarray | list thereof
        Plotting data. Accepts 1 or 2 arguments:

        - Single argument: interpreted as Y (X set to None)
        - Two arguments: interpreted as (X, Y)

    order : {'c', 'fortran'} | _Unset, optional, default=_UNSET
        Memory layout for 2D input interpretation. Defines what a
        column is for `index_spec`.

        * 'c': row-major, shape (N, 2)
        * 'fortran': column-major, shape (2, N)

        If `_UNSET`, uses `config.array_order`.
    index_spec : {'implicit', 'explicit'} | tuple[int, int], optional
        Specifies which columns to extract from 2D input.
        - `'implicit'`: extract all columns as separate Y arrays
        - `'explicit'`: extract columns 0 and 1 as X, Y
        - `tuple (i, j)`: extract columns i and j

    Returns
    -------
    X : ArrayLike | list[ArrayLike]
        X coordinates. Generated as np.arange(n) if not provided.
    Y : ArrayLike | list[ArrayLike]
        Y coordinates.

    Raises
    ------
    ValueError
        If 2D input has inconsistent row lengths.
    """
    X, Y = _extract_xy(*data, order=order, index_spec=index_spec)

    if X is None:
        if _is_2d(Y):
            lengths = [len(y) for y in Y]
            if not all(length == lengths[0] for length in lengths):
                raise ValueError(
                    f'2D input has inconsistent row lengths: {lengths}. '
                    f'All rows must have length {lengths[0]}.'
                )
            length = lengths[0]
        else:
            length = len(Y) if hasattr(Y, '__len__') else 1
        X = np.arange(length)

    if _is_2d(Y) and not _is_2d(X):
        X = [X]
    if _is_2d(X) and not _is_2d(Y):
        Y = [Y]

    if _is_scalar(X) or isinstance(X, np.ndarray):
        X = [X]
    if _is_scalar(Y) or isinstance(Y, np.ndarray):
        Y = [Y]

    if mode == 'plot':
        if _is_1d(X):
            X = [X]
        if _is_1d(Y):
            Y = [Y]

    return X, Y


def _normalize_plotting_input(data):
    """
    Normalize single input to consistent dimensionality structure.

    Ensures error arrays and similar inputs match the dimensionality of
    corresponding data arrays. Scalars are wrapped in lists, 2D structures
    are validated for consistent row lengths.

    Parameters
    ----------
    data : scalar | Sequence | None
        Input data to normalize. Can be a scalar, 1D array-like, 2D array-like,
        or `None`.

    Returns
    -------
    normalized : scalar | list | np.ndarray
        Normalized input. Scalars wrapped in lists, 2D data validated and
        returned unchanged, 1D iterables returned as-is, None returned as-is.

    Raises
    ------
    ValueError
        If 2D input has inconsistent row lengths.
    """
    if data is None:
        return None

    if _is_2d(data):
        lengths = [len(d) for d in data]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError(
                f'2D input has inconsistent row lengths: {lengths}. '
                f'All rows must have length {lengths[0]}.'
            )
        return data

    if _is_scalar(data) or isinstance(data, np.ndarray):
        return [data]

    if _is_iterable(data):
        return data

    return [data]


def _get_zorder(zorders: list[float] | None, i: int, fallback: float):
    """
    Get zorder value from a list of zorders with a fallback zorder.
    Increments the fallback value by i.
    """
    if zorders is None:
        return fallback + 1
    return _cycle(zorders, i) if _cycle(zorders, i) is not None else fallback+i


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
        uses `config.linewidth`.
    fill : bool or None, optional, default=None
        Whether the circles are filled. If None,
        uses `config.circle_fill`.
    cmap : str or None, optional, default=None
        matplolib cmap used to sample default circle colors.
        If None, uses `config.cmap`.
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
                    f'got: {type(ellipse).__name__}'
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
        If None, uses `config.text_loc`.
    text_color : str or None, optional, default=None
        Color of the annotation text. If None, uses `config.text_color`.
    highlight : bool or None, optional, default=None
        If True, adds a bbox to highlight the text. If None,
        uses `config.highlight`.

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
        """
        Rotate the ellipse region based on key press events.

        Parameters
        ----------
        event :
        """
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

        # synchronize region <- artist
        region.angle = -artist.angle * u.deg

        artist.figure.canvas.draw_idle()

        _update_ellipse_region(region, text=text)

    ax.figure.canvas.mpl_connect('key_press_event', _rotate)

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
        Color of patch face. If not `None`,
        sets `fill=True`.
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


def plot_vlines(vlines, ax, unit=None, equivalencies=None) -> None:
    """
    Plot one or more vertical reference lines on a Matplotlib axis.

    Parameters
    ----------
    vlines : float | Quantity | Sequence[float | Quantity] | None
        X-axis coordinate(s) at which to draw vertical line(s). If a Quantity,
        each value is converted to `unit` before plotting. If an iterable is
        provided, a vertical line is drawn for each element. If None, no lines
        are drawn.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object on which to draw the vertical line(s).
    unit : astropy.units.UnitBase or str, optional, default=None
        Unit to which Quantity values in `vlines` are converted before plotting.
        If None, Quantity inputs must already be in the axis unit system or must
        not require conversion.
    equivalencies : astropy.units.equivalencies or None, optional, default=None
        Equivalencies for converting units. If None, is ignored.

    """
    if vlines is not None:
        vlines = to_list(vlines)
        unit = to_unit(unit)

        for vline in vlines:
            if isinstance(vline, Quantity) and unit is not None:
                vline = vline.to(unit, equivalencies=equivalencies).value
            else:
                vline = get_value(vline)
            ax.axvline(
                vline,
                ls=config.axline.linestyle,
                lw=config.axline.linewidth,
                color=config.axline.color,
                alpha=config.axline.alpha,
                zorder=config.zorder.vlines
            )


def plot_hlines(hlines, ax, unit=None, equivalencies=None):
    """
    Plot one or more horizontal reference lines on a Matplotlib axis.

    Parameters
    ----------
    hlines : float | Quantity | Sequence[float | Quantity] | None
        Y-axis coordinate(s) at which to draw horizontal line(s). If a Quantity,
        each value is converted to `unit` before plotting. If an iterable is
        provided, a horizontal line is drawn for each element. If None, no lines
        are drawn.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object on which to draw the horizontal line(s).
    unit : astropy.units.UnitBase or str, optional, default=None
        Unit to which Quantity values in `hlines` are converted before plotting.
        If None, Quantity inputs must already be in the axis unit system or must
        not require conversion.
    equivalencies : astropy.units.equivalencies or None, optional, default=None
        Equivalencies for converting units. If None, is ignored.

    Notes
    -----
    Horizontal lines are drawn using `ax.axhline` with a dotted linestyle,
    linewidth of 1.0, black color, alpha of 0.7, and z-order of 0.
    """
    if hlines is not None:
        hlines = to_list(hlines)
        unit = to_unit(unit)

        for hline in hlines:
            if isinstance(hline, Quantity) and unit is not None:
                hline = hline.to(unit, equivalencies=equivalencies).value
            else:
                hline = get_value(hline)
            ax.axhline(
                hline,
                ls=config.axline.linestyle,
                lw=config.axline.linewidth,
                color=config.axline.color,
                alpha=config.axline.alpha,
                zorder=config.zorder.hlines
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
