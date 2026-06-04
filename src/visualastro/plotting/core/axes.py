"""
Author: Elko Gerville-Reache
Date Created: 2026-06-02
Date Modified: 2026-06-02
Description:
    Functions related to matplotlib axes.
"""

from typing import Any, Literal

import astropy.units as u
import matplotlib.axes as maxes
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
)
from visualastro.core.units import (
    get_physical_type,
    get_unit_label,
    _infer_physical_type_label,
)


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



def ax3d_axis_style(ax: Axes3D, mode: Literal['triad', 'semi', 'cube']='cube') -> None:
    """
    Configure the bounding box style of a 3D matplotlib axes.

    Parameters
    ----------
    ax : Axes3D
        Target 3D axes object.
    mode : str, optional, default='triad'
        Box style to apply:
        - 'triad'   : 3 edges from origin corner only (default matplotlib-like)
        - 'cube'    : all 12 edges of the bounding box
        - 'pillar'  : triad + 3 opposing vertical/depth edges (semi-open box)
    """

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    zmin, zmax = ax.get_zlim()

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    color = ax.spines['left'].get_edgecolor()
    lw = ax.spines['left'].get_linewidth()

    ax.xaxis.line.set_linewidth(0)
    ax.yaxis.line.set_linewidth(0)
    ax.zaxis.line.set_linewidth(0)

    # original
    triad = [
        ([xmin, xmax], [ymin, ymin], [zmin, zmin]),  # x-axis
        ([xmax, xmax], [ymin, ymax], [zmin, zmin]),  # y-axis
        ([xmax, xmax], [ymax, ymax], [zmin, zmax]),  # z-axis
    ]
    semi = [
        # x-axis
        ([xmin, xmax], [ymax, ymax], [zmin, zmin]),
        ([xmin, xmax], [ymax, ymax], [zmax, zmax]),
        # y-axis
        ([xmin, xmin], [ymin, ymax], [zmin, zmin]),
        ([xmin, xmin], [ymin, ymax], [zmax, zmax]),
        # z-axis
        ([xmin, xmin], [ymax, ymax], [zmin, zmax]),
        ([xmin, xmin], [ymin, ymin], [zmin, zmax]),
    ]
    cube = [
        ([xmin, xmax], [ymin, ymin], [zmax, zmax]), # x-axis
        ([xmax, xmax], [ymin, ymax], [zmax, zmax]), # y-axis
        ([xmax, xmax], [ymin, ymin], [zmin, zmax]), # z-axis
    ]

    if mode == 'triad':
        edges = triad
    elif mode == 'semi':
        edges = triad + semi
    elif mode == 'cube':
        edges = triad + semi + cube
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose from: 'triad', 'cube', 'pillar'.")

    for xs, ys, zs in edges:
        ax.plot(xs, ys, zs, lw=lw, color=color, zorder=config.zorder.axes)
