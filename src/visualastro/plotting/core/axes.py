"""
Author: Elko Gerville-Reache
Date Created: 2026-06-02
Date Modified: 2026-06-22
Description:
    Functions related to matplotlib axes.
"""

from typing import Any, Literal

import astropy.units as u
import matplotlib.axes as maxes
from matplotlib.figure import Figure, SubFigure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.typing import ColorType
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.typing import ArrayLike

from visualastro.core.config import (
    config,
    _Unset,
    _UNSET,
    _resolve_default
)
from visualastro.core.numerical_utils import (
    as_list,
    flatten,
    get_value,
    _is_iterable,
    to_list,
)
from visualastro.core.units import (
    get_physical_type,
    get_unit_label,
    _infer_physical_type_label,
)
from visualastro.plotting.core.colors import as_color, get_colors


def subplot(
    *args : int | tuple[int, int, int],
    figsize : tuple[float, float] | _Unset = _UNSET,
    projection: str | None = None,
    **kwargs
) -> tuple[Figure, maxes.Axes]:
    """
    Create a 3D plot instance.

    Parameters
    ----------
    *args :  int | tuple[int, int, *index*], optional, default=(1, 1, 1)
        The position of the subplot described by one of:

        * three integers `(*nrows*, *ncols*, *index*)`. The subplot will
        take the *index* position on a grid with *nrows* rows and *ncols* columns.
        *index* starts at 1 in the upper left corner and increases to the right.
        *index* can also be a two-tuple specifying the (*first*, *last*) indices
        (1-based, and including *last*) of the subplot, ie. `(3, 1, (1, 2))`
        makes a subplot that spans the upper 2/3 of the figure.
        * A 3-digit integer. The digits are interpreted as if given separately as
        three single-digit integers, i.e. `235` is the same as `(2, 3, 5)`.
        Note that this can only be used if there are no more than 9 subplots.

    figsize : tuple[float, float]
        Figure size. If `_UNSET`, uses `config.figsize`.
    projection : str | None, optional, default=None
        The projection type of the subplot. Can be one of the accepted values:
        `'aitoff'`, `'hammer'`, `'lambert'`, `'mollweide'`, `'polar'`, `'rectilinear'`.
        If `None`, uses `'rectilinear'` projection.
    **kwargs :
        Additional keyword arguments passed to `plt.figure`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure instance.
    ax : matplotlib.axes.Axes
        Axes instance.
    """
    figsize = _resolve_default(figsize, config.figsize)

    fig = plt.figure(figsize=figsize, **kwargs)
    ax = fig.add_subplot(*args, projection=projection)

    return fig, ax


def add_subplot(
    *args : int | tuple[int, int, int],
    ax: maxes.Axes | None = None,
    fig: Figure | SubFigure | None = None,
    projection: str | None = None,
    **kwargs
) -> maxes.Axes | tuple[Figure, maxes.Axes]:
    """
    Add a subplot to a figure, optionally creating a new figure.

    Parameters
    ----------
    *args :  int | tuple[int, int, *index*], optional, default=(1, 1, 1)
        The position of the subplot described by one of:

        * three integers `(*nrows*, *ncols*, *index*)`. The subplot will
        take the *index* position on a grid with *nrows* rows and *ncols* columns.
        *index* starts at 1 in the upper left corner and increases to the right.
        *index* can also be a two-tuple specifying the (*first*, *last*) indices
        (1-based, and including *last*) of the subplot, ie. `(3, 1, (1, 2))`
        makes a subplot that spans the upper 2/3 of the figure.
        * A 3-digit integer. The digits are interpreted as if given separately as
        three single-digit integers, i.e. `235` is the same as `(2, 3, 5)`.
        Note that this can only be used if there are no more than 9 subplots.

    ax : Axes | None, optional, default=None
        Axes instance. Either `ax` or `fig` should be provided.
    fig : Figure | SubFigure | None, optional, default=None
        Figure instance.  Either `ax` or `fig` should be provided.
    projection : str or None, optional, default=None
        Projection type for the subplot. Examples include WCSAxes or
        {None, '3d', 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar',
        'rectilinear', str}. If None, defaults to 'rectilinear'.
    **kwargs :
        Additional keyword arguments passed to `fig.add_subplot`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Created subplot axes.

    Examples
    --------
    Create a new figure and subplot:
    >>> fig, ax = add_subplot(return_fig=True)

    Add a subplot to an existing figure:
    >>> fig = plt.figure()
    >>> ax = add_subplot(121, fig=fig)

    Create a 3D subplot:
    >>> fig, ax = add_subplot(projection='3d', return_fig=True)
    """
    fig = ax.figure if ax is not None else fig
    if fig is None:
        raise ValueError(
            'Provide at least fig or ax!'
        )

    ax = fig.add_subplot(*args, projection=projection, **kwargs)

    return ax


def ax3d(
    *args : int | tuple[int, int, int],
    figsize: tuple[float, float] | _Unset = _UNSET,
    **kwargs
) -> tuple[Figure, Axes3D]:
    """
    Create a 3D plot instance.

    Parameters
    ----------
    *args :  int | tuple[int, int, *index*], optional, default=(1, 1, 1)
        The position of the subplot described by one of:

        * three integers `(*nrows*, *ncols*, *index*)`. The subplot will
        take the *index* position on a grid with *nrows* rows and *ncols* columns.
        *index* starts at 1 in the upper left corner and increases to the right.
        *index* can also be a two-tuple specifying the (*first*, *last*) indices
        (1-based, and including *last*) of the subplot, ie. `(3, 1, (1, 2))`
        makes a subplot that spans the upper 2/3 of the figure.
        * A 3-digit integer. The digits are interpreted as if given separately as
        three single-digit integers, i.e. `235` is the same as `(2, 3, 5)`.
        Note that this can only be used if there are no more than 9 subplots.

    figsize : tuple[float, float]
        Figure size. If `_UNSET`, uses `config.figsize3D`.
    **kwargs :
        Additional keyword arguments passed to `plt.figure`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure instance.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        3D axes instance.
    """
    figsize = _resolve_default(figsize, config.figsize3D)

    fig = plt.figure(figsize=figsize, **kwargs)
    ax = fig.add_subplot(*args, projection='3d')

    return fig, ax


def add_ax3d(
    *args : int | tuple[int, int, int],
    ax: maxes.Axes | Axes3D | None = None,
    fig: Figure | SubFigure | None = None,
    **kwargs
) -> Axes3D:
    """
    Add a 3D subplot to a figure.

    Parameters
    ----------
    *args :  int | tuple[int, int, *index*], optional, default=(1, 1, 1)
        The position of the subplot described by one of:

        * three integers `(*nrows*, *ncols*, *index*)`. The subplot will
        take the *index* position on a grid with *nrows* rows and *ncols* columns.
        *index* starts at 1 in the upper left corner and increases to the right.
        *index* can also be a two-tuple specifying the (*first*, *last*) indices
        (1-based, and including *last*) of the subplot, ie. `(3, 1, (1, 2))`
        makes a subplot that spans the upper 2/3 of the figure.
        * A 3-digit integer. The digits are interpreted as if given separately as
        three single-digit integers, i.e. `235` is the same as `(2, 3, 5)`.
        Note that this can only be used if there are no more than 9 subplots.

    ax : Axes | Axes3D | None, optional, default=None
        Axes instance. Either `ax` or `fig` should be provided.
    fig : Figure | SubFigure | None, optional, default=None
        Figure instance.  Either `ax` or `fig` should be provided.
    **kwargs :
        Additional keyword arguments passed to `fig.add_subplot`.

    Returns
    -------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        Created 3D axes instance.
    """
    fig = ax.figure if ax is not None else fig
    if fig is None:
        raise ValueError(
            'Provide at least fig or ax!'
        )
    ax = fig.add_subplot(*args, projection='3d', **kwargs)

    return ax


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
        uses `config.figsize_gridspec`.
    sharex : bool | _Unset, optional, default=_UNSET
        If `True`, share the x-axis among all subplots. If `_UNSET`,
        uses `config.axes.sharex`.
    sharey : bool | _Unset, optional, default=_UNSET
        If `True`, share the y-axis among all subplots. If `_UNSET`,
        uses `config.axes.sharey`.
    hspace : float | None | _Unset, optional, default=`_UNSET`
        Height padding between subplots. If `None`,
        Matplotlib's default spacing is used. If `_UNSET`,
        uses `config.axes.hspace`.
    wspace : float | None | _Unset, optional, default=`_UNSET`
        Width padding between subplots. If `None`,
        Matplotlib's default spacing is used. If `_UNSET`,
        uses `config.axes.wspace`.
    width_ratios : ArrayLike | None, optional, default=None
        ArrayLike of length `ncols` defining the width padding between
        subplots. If `None`, Matplotlib's default spacing is used.
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
    figsize = _resolve_default(figsize, config.figsize_gridspec)
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
    axs = np.atleast_1d(np.asarray(axs)).ravel()

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


def set_axis_limits(
    xdata=None,
    ydata=None,
    *,
    ax=None,
    xlim=None,
    ylim=None,
    **kwargs,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """
    Set axis limits on a Matplotlib Axes based on data range
    and optional user-specified limits.

    Parameters
    ----------
    xdata : array-like, list of array-like, or None, optional
        X-axis data. Can be a single array, a list of arrays, or None.
        If None, the X-axis will be autoscaled unless `xlim` is provided.
    ydata : array-like, list of array-like, or None, optional
        Y-axis data. Can be a single array, a list of arrays, or None.
        If None, the Y-axis will be autoscaled unless `ylim` is provided.
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
        uses `config.axes.xpad`.
    ypad : float or None, optional, default=None
        Fractional padding to apply to Y-axis limits. If None,
        uses `config.axes.ypad`.

    Returns
    -------
    xlim_out : tuple of float
        The X-axis limits that were applied to the Axes.
    ylim_out : tuple of float
        The Y-axis limits that were applied to the Axes.

    Notes
    -----
    - If both `xdata` and `ydata` are provided as lists, they must have the same length
        or be broadcastable (single array broadcast across multiple arrays of the other axis).
    - Data outside user-provided `xlim` or `ylim` is ignored when computing automatic limits.
    - Both scalar and multi-dimensional arrays are flattened before processing.
    - If all data is `None` or empty, the corresponding axis will not be modified.
    """
    xpad_frac = kwargs.get('xpad', config.axes.xpad)
    ypad_frac = kwargs.get('ypad', config.axes.ypad)

    if ax is None:
        raise ValueError('ax must be an axes instance')

    if xdata is not None and ydata is not None:
        if not _is_iterable(xdata):
            xdata = as_list(xdata)
        if not _is_iterable(ydata):
            ydata = as_list(ydata)

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
        if xmin == xmax:
            xpad = (
                abs(xmin) * xpad_frac if xmin != 0 else
                (xpad_frac if xpad_frac > 0 else 0.1)
            )
        else:
            xpad = xpad_frac * (xmax - xmin)
        xlim = (xmin - xpad, xmax + xpad)

    if ylim is None and yvals is not None and len(yvals) > 0:
        ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
        if ymin == ymax:
            ypad = (
                abs(ymin) * ypad_frac if ymin != 0 else
                (ypad_frac if ypad_frac > 0 else 0.1)
            )
        else:
            ypad = ypad_frac * (ymax - ymin)
        ylim = (ymin - ypad, ymax + ypad)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    return xlim, ylim


def set_axis_labels(
    X: u.Quantity | ArrayLike | None,
    Y: u.Quantity | ArrayLike | None,
    ax: maxes.Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    unit_bracket_style: Literal['round', 'square'] | _Unset = _UNSET,
    show_physical_type: bool | _Unset = _UNSET,
    show_unit: bool | _Unset = _UNSET,
    fmt: Literal['latex', 'latex_inline', 'fits', 'unicode', 'console', 'vounit', 'cds', 'ogip'] | _Unset = _UNSET
) -> None:
    """
    Automatically generate and set axis labels from objects with physical
    units.

    This function creates formatted axis labels by combining an inferred
    physical type (e.g., 'Wavelength', 'Flux Density') with a formatted unit
    string (e.g., 'μm', 'MJy/sr'). Each component can be enabled or disabled
    independently, and custom labels may be provided.

    Parameters
    ----------
    X : u.Quantity | ArrayLike | None
        Object with a unit exposable with `get_data`. If `None`,
        does not set any labels unless `xlabel` is set.
    Y : '~astropy.units.Quantity' or object with a unit
        Object with a unit exposable with `get_data`. If `None`,
        does not set any labels unless `ylabel` is set.
    ax : matplotlib.axes.Axes
        Matplotlib axes object on which to set the labels.
    xlabel : str | None, optional, default=None
        Custom label for the x-axis. If None, the label is inferred from `X`.
    ylabel : str | None, optional, default=None
        Custom label for the y-axis. If None, the label is inferred from `Y`.
    unit_bracket_style : Literal['round', 'square'] | _Unset, optional, default=_UNSET
        If `'round`' displays the unit of `X` and `Y` as (unit). If `'square`' as [unit].
    show_physical_type : bool | _Unset, optional, default=_UNSET
        If `True`, include the inferred physical type in the axis label.
        If `_UNSET`, uses `config.show_type_label`.
    show_unit : bool | _Unset, optional, default=_UNSET
        If `True`, include the unit in the axis label. If `_UNSET`, uses
        `config.show_unit_label`.
    fmt : str | _Unset, optional, default=_UNSET
        Format for unit rendering. Passed to `to_latex_unit`.

        Accepted options are `'latex'`, `'latex_inline'`, `'fits'`,
        `'unicode'`, `'console'`, `'vounit'`, `'cds'`, `'ogip'`

        If `_UNSET`, uses `config.unit_label_format`.

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

    >>> set_axis_labels(wavelength, flux, ax, show_physical_type=False)
    # xlabel: '[μm]'
    # ylabel: '[MJy/sr]'

    >>> set_axis_labels(wavelength, flux, ax, ylabel='Custom Flux')
    # ylabel: 'Custom Flux [MJy/sr]'

    Notes
    -----
    - Units are formatted using `to_latex_unit`, which provides LaTeX-friendly
      representations.
    - If both `show_physical_type` and `show_unit` are False, the resulting
      axis label is an empty string.
    """
    unit_bracket_style = _resolve_default(unit_bracket_style, config.unit_bracket_style)
    show_physical_type = _resolve_default(show_physical_type, config.show_type_label)
    show_unit = _resolve_default(show_unit, config.show_unit_label)
    fmt = _resolve_default(fmt, config.unit_label_format)

    xlabel = _format_axis_label(
        X, xlabel, unit_bracket_style, show_physical_type, show_unit, fmt
    )
    ylabel = _format_axis_label(
        Y, ylabel, unit_bracket_style, show_physical_type, show_unit, fmt
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _format_axis_label(
    obj: Any,
    label: str | None,
    bracket_style: Literal['round', 'square'],
    show_physical_type: bool,
    show_unit: bool,
    fmt: Literal['latex', 'latex_inline', 'fits', 'unicode', 'console', 'vounit', 'cds', 'ogip']
) -> str:
    r"""
    Create a scientific axis label with physical type and unit information.

    This function generates axis labels in the format `<physical label> [<unit>]`
    or `<physical label> (<unit>)`, where the physical label describes what the axis
    represents (e.g., 'Wavelength', 'Flux') and the unit is formatted in LaTeX notation.
    Both components can be customized or disabled independently.

    Parameters
    ----------
    obj : any
        An object from which physical type and unit information can be extracted.
        This may be an Astropy `Quantity`, a Spectrum-like object, or any object
        compatible with `get_unit` and `get_physical_type`.
    label : str | None
        If a string is provided, use it as the physical label directly, overriding
        any auto-detected physical type. If None, the physical label is inferred
        from the object's physical type (when `show_physical_type=True`).
    bracket_style : Literal['round', 'square']
        If `'round`' displays the unit of `obj` as (unit). If `'square`' as [unit].
    show_physical_type : bool
        If `True`, include the physical type label in the output. If `False`, omit
        the physical label entirely (useful for creating unit-only labels).
    show_unit : bool
        If `True`, include the unit in the output. If `False`, omit the unit
        (useful for creating label-only outputs).
    fmt : str, default=_UNSET
        Format for unit rendering. Passed to `to_latex_unit`.

        Accepted options are `'latex'`, `'latex_inline'`, `'fits'`,
        `'unicode'`, `'console'`, `'vounit'`, `'cds'`, `'ogip'`

        If `_UNSET`, uses `config.unit_label_format`.

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

    elif show_physical_type:
        inferred = _infer_physical_type_label(obj)
        if inferred is not None:
            physical_label = inferred
        elif physical_type is not None:
            physical_label = str(physical_type).replace('_', ' ').title()
        else:
            physical_label = ''

    else:
        physical_label = ''

    if show_unit:
        unit_label = get_unit_label(obj, bracket_style=bracket_style, fmt=fmt)
    else:
        unit_label = ''

    return fr'{physical_label} {unit_label}'.strip()


def ax3d_axis_style(
    ax: Axes3D,
    style: str | None = 'cube'
) -> None:
    """
    Set the spine style of a 3D matplotlib axes by drawing explicit edge lines.

    Overrides matplotlib axes rendering while keeping the ticks and tick labels.
    For this reason these styles are experimental and may break in interactive mode.

    The styles are defined with respect to the default matplotlib viewing angles of
    `elev=30`, `azim=-60`, and `roll=0`.

    Parameters
    ----------
    ax : Axes3D
        Target 3D axes object.
    style : str | None, optional, default='cube'
        Spine layout to apply.

        * `'triad'`        : 3 edges from the front-right-bottom corner (matplotlib default-like).
        * `'floor'`        : 4 bottom edges of the bounding box.
        * `'ceiling'`      : 4 top edges of the bounding box.
        * `'cube'`         : all 12 edges of the bounding box.
        * `'panel_back'`   : floor + back vertical edges + back-top edges (alias `'panel'`).
        * `'panel_front'`  : floor + front vertical edges + front-top edges (alias `'panel_r'`).
        * `'x_panel'`      : floor + back x-aligned top edge + right vertical edges.
        * `'x_panel_front'`: floor + front x-aligned top edge + left vertical edges (alias `'x_panel_r'`).
        * `'y_panel'`      : floor + back y-aligned top edge + left vertical edges.
        * `'y_panel_front'`: floor + front y-aligned top edge + right vertical edges (alias `'y_panel_r'`).

    If `None`, uses matplotlib rendering engine.
    """
    if style is None: return

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    zmin, zmax = ax.get_zlim()

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    color = ax.spines['left'].get_edgecolor()
    lw = ax.spines['left'].get_linewidth()

    x_top_front = ([xmin, xmax], [ymin, ymin], [zmax, zmax])
    x_top_back = ([xmin, xmax], [ymax, ymax], [zmax, zmax])
    x_bottom_front = ([xmin, xmax], [ymin, ymin], [zmin, zmin])
    x_bottom_back = ([xmin, xmax], [ymax, ymax], [zmin, zmin])

    y_top_front = ([xmax, xmax], [ymin, ymax], [zmax, zmax])
    y_top_back = ([xmin, xmin], [ymin, ymax], [zmax, zmax])
    y_bottom_front = ([xmax, xmax], [ymin, ymax], [zmin, zmin])
    y_bottom_back = ([xmin, xmin], [ymin, ymax], [zmin, zmin])

    z_center_front = ([xmax, xmax], [ymin, ymin], [zmin, zmax])
    z_center_back = ([xmin, xmin], [ymax, ymax], [zmin, zmax])
    z_left = ([xmin, xmin], [ymin, ymin], [zmin, zmax])
    z_right = ([xmax, xmax], [ymax, ymax], [zmin, zmax])

    triad = [x_bottom_front, y_bottom_front, z_right]
    floor = [x_bottom_front, x_bottom_back, y_bottom_front, y_bottom_back]
    ceiling = [x_top_front, x_top_back, y_top_front, y_top_back]
    sides = [z_center_front, z_center_back, z_left, z_right]
    panel_back = [*floor, x_top_back, y_top_back, z_left, z_right, z_center_back]
    panel_front = [*floor, x_top_front, y_top_front, z_left, z_right, z_center_front]
    x_panel_back = [*floor, x_top_back, z_center_back, z_right]
    x_panel_front = [*floor, x_top_front, z_center_front, z_left]
    y_panel_back = [*floor, y_top_back, z_center_back, z_left]
    y_panel_front = [*floor, y_top_front, z_center_front, z_right]
    cube = [*floor, *ceiling, *sides]

    axis_modes = {
        'triad': triad,
        'floor': floor,
        'ceiling': ceiling,
        'panel': panel_back,
        'panel_r': panel_front,
        'panel_back': panel_back,
        'panel_front': panel_front,
        'x_panel': x_panel_back,
        'x_panel_r': x_panel_front,
        'x_panel_front': x_panel_front,
        'y_panel': y_panel_back,
        'y_panel_r': y_panel_front,
        'y_panel_front': y_panel_front,
        'cube': cube,
    }
    edges = axis_modes.get(style, None)
    if edges is None: return

    ax.xaxis.line.set_linewidth(0)
    ax.yaxis.line.set_linewidth(0)
    if not style == 'floor' and not style == 'ceiling':
        ax.zaxis.line.set_linewidth(0)

    for xs, ys, zs in edges:
        ax.plot(xs, ys, zs, lw=lw, color=color, zorder=config.zorder.axes)


def ax3d_pane_color(
    pane_color: ColorType | tuple[ColorType, ColorType, ColorType],
    ax: Axes3D
) -> None:
    """
    Set 3D plot pane color.

    Parameters
    ----------
    pane_color: ColorType | tuple[ColorType, ColorType, ColorType]
        Either color for all panels or a tuple of X, Y, Z panel colors.
    ax : Axes3D
        3D axes instance on which to set panel colors.

    Examples
    --------
    # set all pane colors to red
    >>> fig = plt.figure(figsize=figsize, **kwargs)
    >>> ax = fig.add_subplot(*args, projection='3d')
    >>> ax3d_border_color('r', ax)

    # set each x,y,z color separately
    >>> fig = plt.figure(figsize=figsize, **kwargs)
    >>> ax = fig.add_subplot(*args, projection='3d')
    >>> ax3d_border_color(('r', 'g', 'b'), ax)
    """
    colors = to_list(pane_color)
    if len(colors) >= 3:
        colors = [as_color(get_colors(c), fmt='rgba') for c in colors]
    else:
        colors = [colors[0]]*3

    ax.xaxis.set_pane_color(colors[0])
    ax.yaxis.set_pane_color(colors[1])
    ax.zaxis.set_pane_color(colors[2])
