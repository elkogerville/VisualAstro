"""
Author: Elko Gerville-Reache
Date Created: 2026-07-30
Date Modified: 2026-07-30
Description:
    Contour plotting functions.
"""

from typing import Callable, Literal

import matplotlib.axes as maxes
from matplotlib.colors import Colormap
from matplotlib.contour import QuadContourSet
from mpl_toolkits.mplot3d import Axes3D

from visualastro.core.config import (
    config, _Unset, _UNSET,
)
from visualastro.core.kwargs import (
    _kwarg, _param,
    _resolve_kwargs
)
from visualastro.core.numerical import kde2d
from visualastro.plotting.core.colormaps import get_cmap


def contour_kde(
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
    x : ArrayLike
        1D array of x-values for the dataset.
    y : ArrayLike
        1D array of y-values for the dataset.
    ax : matplotlib.axes.Axes | mpl_toolkits.mplot3d.axes3d.Axes3D
        Axis on which to draw the contours.
    levels : int | ArrayLike | _Unset, optional, default=_UNSET
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
    offset : float | None, default=None
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
            _kwarg('fontsize', config.fontsizes.resolve('text')),
            _kwarg('bad_color', None),
        ]

    )
    cmap = get_cmap(params.cmap, bad_color=params.bad_color)

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


def contourf_kde(
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

    Equivalent to calling `contour_kde(..., contour_method='contourf')`.

    See Also
    --------
    contour : Full parameter documentation.
    """
    return contour_kde(
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
