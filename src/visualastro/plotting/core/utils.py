"""
Author: Elko Gerville-Reache
Date Created: 2025-05-24
Date Modified: 2026-06-13
Description:
    Plotting utility functions.
"""

from collections.abc import Sequence
from contextlib import contextmanager
from importlib.resources import files
from typing import Callable, Literal
import warnings
from functools import partial

import astropy.units as u
from astropy.units import Quantity
from astropy.visualization.wcsaxes.core import WCSAxes
import matplotlib.axes as maxes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap
from matplotlib.contour import QuadContourSet
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Circle, Ellipse
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.typing import ColorType
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.typing import NDArray
from regions import PixCoord, EllipsePixelRegion

from visualastro.core.config import (
    get_config_value,
    config,
    _Unset,
    _UNSET,
    _resolve_default
)
from visualastro.core.kwargs import _extract_kwargs, _kwarg, _param, _resolve_kwargs
from visualastro.core.numerical import kde2d
from visualastro.core.numerical_utils import (
    get_value,
    to_list,
    _cycle,
    _extract_xy,
    _is_1d,
    _is_2d,
    _is_iterable,
    _is_ndarray_or_quantity_array,
    _is_scalar,
)
from visualastro.core.units import to_unit
from visualastro.plotting.core.colors import get_cmap, get_colors, sample_cmap


@contextmanager
def style(name: str | _Unset = _UNSET, *additional_styles, rc: dict | None = None, **rc_kwargs):
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
    styles = [style_name, rc_combined]
    modifiers = []
    for style in additional_styles:
        if isinstance(style, str):
            modifiers.append(style)
        if len(modifiers) > 0:
            styles += modifiers
    context = styles if rc_combined else style_name

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
            f"Falling back to '{config.style_fallback}' style.",
            stacklevel=2
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


def contour(
    x,
    y,
    ax: maxes.Axes | Axes3D,
    levels: int | _Unset = _UNSET,
    contour_method: Literal['contour', 'contourf'] | _Unset = _UNSET,
    bw_method: Literal['scott', 'silverman'] | float | Callable | _Unset = _UNSET,
    gridsize: int | _Unset = _UNSET,
    padding: float | _Unset = _UNSET,
    cslabel: bool | _Unset = _UNSET,
    zdir=None,
    offset=None,
    cmap: Colormap | str | _Unset = _UNSET,
    zorder=None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    **kwargs
) -> QuadContourSet:
    """
    Add 2D Gaussian KDE density contours to an axis.
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
    ax : matplotlib.axes.Axes | mpl_toolkits.mplot3d.axes3d.Axes3D
        Axis on which to draw the contours.
    levels : int | array-like | _Unset, optional, default=_UNSET
        Number or list of contour levels to draw. If `_UNSET`,
        uses `config.contour.levels`.
    contour_method : {'contour', 'contourf'} | _Unset, optional, default=_UNSET
        Method used to draw contours. `'contour'` draws lines, while
        `'contourf'` draws filled contours. If `_UNSET`, uses
        `config.contour.method`.
    bw_method : str | float | Callable | _Unset, optional, default=_UNSET
        The method used to calculate the bandwidth factor for the Gaussian KDE.
        Can be one of:

        * `'scott'` or `'silverman'`: use standard rules of thumb.
        * a scalar constant: directly used as the bandwidth factor.
        * a callable: should take a `scipy.stats.gaussian_kde` instance as its
            sole argument and return a scalar bandwidth factor.

    gridsize : int | _Unset, optional, default=_UNSET
        Number of grid points used per axis for density estimation.
        If `_UNSET`, uses `config.contour.gridsize`.
    padding : float | _Unset, optional, default=_UNSET
        Fractional padding applied to the data range when generating
        the KDE grid. If `_UNSET`, uses `config.contour.padding`.
    cslabel : bool | _Unset, optional, default=_UNSET
        If `True`, label contour levels with their corresponding values.
        Only works in 2D plots. If `_UNSET`, uses `config.contour.clabel`.
    zdir : {'x', 'y', 'z'} | None, default=None
        Direction normal to the plane where contours are drawn.
        If None, contours are plotted in 2D.
    offset : float or None, default=None
        Offset along the `zdir` direction for projecting contours in 3D space.
    cmap : Colormap | str | _Unset, optional, default=_UNSET
        Colormap used for plotting contours. If `_UNSET`,
        uses `config.cmap`.
    fontsize : float, optional, default=config.fontsize
        Fontsize of contour labels.

    Returns
    -------
    cs : matplotlib.contour.QuadContourSet | mpl_toolkits.mplot3d.art3d.QuadContourSet3D
        The contour set object created by Matplotlib.
    """
    params = _resolve_kwargs(
        kwargs,
        params=[
            _param('levels', levels, config.contour.levels),
            _param('contour_method', contour_method, config.contour.method),
            _param('bw_method', bw_method, config.contour.bw_method),
            _param('gridsize', gridsize, config.contour.gridsize),
            _param('padding', padding, config.contour.padding),
            _param('cslabel', cslabel, config.contour.clabel),
            _param('cmap', cmap, config.cmap),
        ],
        additional_kwargs=[
            _kwarg('fontsize', config.fontsize),
            _kwarg('bad_color', None),
        ]

    )
    cmap = get_cmap(params.cmap, params.bad_color)

    c_method = params.contour_method.lower()
    contour_methods = {
        'contour': ax.contour,
        'contourf': ax.contourf
    }
    contour_func = contour_methods.get(c_method, ax.contour)
    c_method_name = c_method if c_method in contour_methods else 'contour'

    # compute kde density
    X, Y, Z = kde2d(
        x, y,
        bw_method=params.bw_method,
        gridsize=params.gridsize,
        padding=params.padding,
        xlim=xlim, ylim=ylim
    )

    if zorder is None:
        zorder = config.zorder.contour if c_method_name == 'contour' else config.zorder.contourf

    # plot contours as either 3D projections or a simple 2D plot
    valid_zdirs = {'x', 'y', 'z'}
    zdir = zdir.lower() if isinstance(zdir, str) else None
    if zdir in valid_zdirs and offset is not None:
        input_data = {
            'z': (X, Y, Z),
            'y': (X, Z, Y),
            'x': (Z, Y, X),
        }.get(zdir, (X, Y, Z))

        cs = contour_func(
            *input_data,
            levels=params.levels,
            cmap=cmap,
            zdir=zdir,
            offset=offset,
            zorder=zorder,
            **kwargs
        )

    else:
        cs = contour_func(
            X, Y, Z,
            levels=params.levels,
            cmap=cmap,
            zorder=zorder,
            **kwargs
        )

    if params.cslabel:
        ax.clabel(cs, fontsize=params.fontsize)

    return cs


def contourf(
    x,
    y,
    ax: maxes.Axes | Axes3D,
    levels: int | _Unset = _UNSET,
    bw_method: Literal['scott', 'silverman'] | float | Callable | _Unset = _UNSET,
    gridsize: int | _Unset = _UNSET,
    padding: float | _Unset = _UNSET,
    cslabel: bool | _Unset = _UNSET,
    zdir=None,
    offset=None,
    cmap: Colormap | str | _Unset = _UNSET,
    zorder=None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    **kwargs
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
        xlim=xlim,
        ylim=ylim,
        **kwargs,
    )


def _normalize_plotting_inputs(
    *data: float | u.Quantity | NDArray | Sequence[float | u.Quantity | NDArray],
    order: Literal['c', 'fortran'] | _Unset = _UNSET,
    index_spec: Literal['implicit', 'explicit'] | tuple[int, int] | _Unset = _UNSET,
    mode: Literal['plot', 'scatter'] = 'scatter'
):
    """
    Extract and normalize X, Y inputs for plotting.

    Handles variable input formats and ensures X and Y have compatible
    dimensionality for broadcasting. Generates implicit X indices when not
    provided. Wraps scalars and 1D data in lists to match 2D counterparts.

    See `visualastro.plotting.core.utils._extract_xy` for documentation
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
        Y = to_list(Y)
        xlist, ylist = [], []
        for y in Y:
            xi, yi = _normalize_xy_plotting_inputs(None, y, mode=mode)
            xlist.append(xi)
            ylist.append(yi)

        if all(isinstance(y, (list, tuple)) for y in ylist):
            if all(_is_ndarray_or_quantity_array(item) for y in ylist for item in y):
                ylist = [item for sublist in ylist for item in sublist]

        return xlist, ylist

    return _normalize_xy_plotting_inputs(X, Y, mode=mode)


def _normalize_xy_plotting_inputs(X, Y, mode):
    """Helper method for _normalize_plotting_inputs."""
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

    Parameters
    ----------
    zorders : list[float] | None
        List of zorders to cycle through. If `None`, returns
        `fallback + 1`.
    fallback : float
        Fallback value if `zorders` or `zorders[i]` is `None`.
        Always incremented by 1.
    """
    if zorders is None:
        return fallback + 1
    return _cycle(zorders, i) if _cycle(zorders, i) is not None else fallback+i


# Plot Matplotlib Patches and Shapes
# ----------------------------------
def plot_circles(
    circles,
    ax: maxes.Axes,
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


def plot_ellipses(ellipses: Ellipse | list[Ellipse], ax: maxes.Axes) -> None:
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


def _copy_ellipse(ellipse: Ellipse) -> Ellipse:
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


def plot_points(
    *points: float | u.Quantity | NDArray | Sequence[float | u.Quantity | NDArray],
    ax: maxes.Axes,
    color: ColorType | list[ColorType] | int = 'astro',
    size: float | list[float] = 20,
    marker: MarkerStyle | list[MarkerStyle] = '*',
    order: Literal['c', 'fortran'] | _Unset = _UNSET,
    index_spec: Literal['implicit', 'explicit'] | tuple[int, int] = 'implicit',
) -> None:
    """
    Plot points on a given Matplotlib axis.

    Parameters
    ----------
    points : float | u.Quantity | NDArray | Sequence[float | u.Quantity | NDArray]
        Input data. Supported forms:

        * Single argument:
            * 1D array-like or Quantity: Y values, X = None
            * 2D array or Quantity: extract X, Y according to `order` and `index_spec`
            * list/tuple of scalars: Y values, X = None
            * scalar or scalar Quantity: single Y value, X = None

        * Two arguments:
            * (X, Y) pairs passed through unchanged

    ax : matplotlib.axes.Axes
        The Matplotlib axis on which to plot the points.
    color : ColorType | list[ColorType] | int, optional, default='astro'
        Color of the points. If an integer, will draw colors using `config.cmap`.
    size : float | list[float], optional, default=20
        Marker size(s).
    marker : MarkerStyle | list[MarkerStyle], optional, default='*'
        Matplotlib marker style(s).
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
    """
    if points[0] is not None:
        xlist, ylist = _normalize_plotting_inputs(
            *points, order=order, index_spec=index_spec
        )

        colors = get_colors(color)
        sizes = to_list(size)
        markers = to_list(marker)
        for i in range(len(ylist)):
            x = get_value(_cycle(xlist, i))
            y = get_value(_cycle(ylist, i))

            ax.scatter(
                x, y,
                marker=_cycle(markers, i),
                s=_cycle(sizes, i),
                color=_cycle(colors, i)
            )


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
