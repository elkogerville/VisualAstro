"""
Author: Elko Gerville-Reache
Date Created: 2026-06-02
Date Modified: 2026-10-28
Description:
    Module for setting axes labels.
"""

from typing import Any, Literal

import astropy.units as u
from astropy.visualization.wcsaxes.core import WCSAxes
import matplotlib.axes as maxes

from matplotlib.text import Text
from numpy.typing import ArrayLike

from visualastro.core.config import (
    config,
    _Unset,
    _UNSET,
    _resolve_default
)
from visualastro.core.units import (
    get_physical_type,
    get_unit_label,
    _infer_physical_type_label,
)
from visualastro.plotting.core.axes.axes import get_ax


def set_title(
    title: str,
    loc: Literal['center', 'left', 'right'] = 'center',
    pad: float | None = None,
    fontsize: float | _Unset = _UNSET,
    ax: maxes.Axes | WCSAxes | None = None,
    **kwargs
) -> Text:
    """
    Set the title for an `Axes` instance.

    Parameters
    ----------
    title : str
        Figure title.
    loc : {'center', 'left', 'right'} | optional, default='center'
        Title location.
    pad : float | None, optional, default=None
        Title offset from the top of the `Axes`, in points.
    fontsize : float | _Unset, optional, default=_UNSET
        Title fontsize in points. If `_UNSET`, uses
        `config.fontsizes.title`, expressed as a scaling
        factor relative to `config.fontsizes.size`.
    ax : Axes | WCSAxes | None, optional, default=None
        Sets title to either `plt.gca` if `ax=None`, or
        the instance passed in.
    **kwargs : dict, optional
        Additional keyword arguments passed on to `ax.set_title`.

    Returns
    -------
    Text :
        `Text` object returned by `ax.set_title`.
    """
    ax = get_ax(ax)
    fontsize = _resolve_default(fontsize, config.fontsizes.resolve('title'))
    return ax.set_title(title, loc=loc, pad=pad, fontsize=fontsize, **kwargs)


def _set_axis_labels(
    X: u.Quantity | ArrayLike | None,
    Y: u.Quantity | ArrayLike | None,
    ax: maxes.Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    fontsize: float | _Unset = _UNSET,
    unit_bracket_style: Literal['round', 'square'] | _Unset = _UNSET,
    show_physical_type: bool | _Unset = _UNSET,
    show_unit: bool | _Unset = _UNSET,
    unit_fmt: str | _Unset = _UNSET,
    **kwargs
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
    Y : ~astropy.units.Quantity | object with a unit
        Object with a unit exposable with `get_data`. If `None`,
        does not set any labels unless `ylabel` is set.
    ax : matplotlib.axes.Axes
        Matplotlib axes object on which to set the labels.
    xlabel, ylabel : str | None, optional, default=None
        Custom label for the x-axis and y-axis. If `None`,
        the label is inferred from `X` / `Y`.
    fontsize : float | _Unset, optional, default=_UNSET
        Fontsize of both axes labels in points. If `_UNSET`, uses
        `config.fontsizes.axes_labels`, expressed as a scaling
        factor relative to `config.fontsizes.size`.
    unit_bracket_style : {'round', 'square'} | _Unset, optional, default=_UNSET
        If `'round`' displays the unit of `X` and `Y` as (unit). If `'square`' as [unit].
    show_physical_type : bool | _Unset, optional, default=_UNSET
        If `True`, include the inferred physical type in the axis label.
        If `_UNSET`, uses `config.show_type_label`.
    show_unit : bool | _Unset, optional, default=_UNSET
        If `True`, include the unit in the axis label. If `_UNSET`, uses
        `config.show_unit_label`.
    unit_fmt : str | _Unset, optional, default=_UNSET
        Format for unit rendering. Passed to `to_latex_unit`.

        Accepted options are `'latex'`, `'latex_inline'`, `'fits'`,
        `'unicode'`, `'console'`, `'vounit'`, `'cds'`, `'ogip'`

        If `_UNSET`, uses `config.unit_label_format`.
    **kwargs : dict, optional
        Additional parameters passed on to `ax.set_xlabel`
        and `ax.set_ylabel`.

    Examples
    --------
    >>> import astropy.units as u
    >>> wavelength = np.linspace(1, 10, 100) * u.um
    >>> flux = np.random.random(100) * u.MJy / u.sr
    >>> fig, ax = plt.subplots()
    >>> ax.plot(wavelength, flux)
    >>> _set_axis_labels(wavelength, flux, ax)
    # xlabel: 'Wavelength [μm]'
    # ylabel: 'Surface Brightness [MJy/sr]'

    >>> _set_axis_labels(wavelength, flux, ax, show_physical_type=False)
    # xlabel: '[μm]'
    # ylabel: '[MJy/sr]'

    >>> _set_axis_labels(wavelength, flux, ax, ylabel='Custom Flux')
    # ylabel: 'Custom Flux [MJy/sr]'

    Notes
    -----

    - Units are formatted using `to_latex_unit`, which provides LaTeX-friendly
      representations.
    - If both `show_physical_type` and `show_unit` are False, the resulting
      axis label is an empty string.
    """
    fontsize = _resolve_default(
        fontsize, config.fontsizes.resolve('axes_labels')
    )
    unit_bracket_style = _resolve_default(unit_bracket_style, config.unit_bracket_style)
    show_physical_type = _resolve_default(show_physical_type, config.show_type_label)
    show_unit = _resolve_default(show_unit, config.show_unit_label)
    fmt = _resolve_default(unit_fmt, config.unit_label_format)

    xlabel = _format_axis_label(
        X, xlabel, unit_bracket_style, show_physical_type, show_unit, fmt
    )
    ylabel = _format_axis_label(
        Y, ylabel, unit_bracket_style, show_physical_type, show_unit, fmt
    )

    ax.set_xlabel(xlabel, fontsize=fontsize, **kwargs)
    ax.set_ylabel(ylabel, fontsize=fontsize, **kwargs)


def _format_axis_label(
    obj: Any,
    label: str | None,
    bracket_style: Literal['round', 'square'],
    show_physical_type: bool,
    show_unit: bool,
    fmt: str | _Unset = _UNSET
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
    bracket_style : {'round', 'square'}
        If `'round`' displays the unit of `obj` as (unit). If `'square`' as [unit].
    show_physical_type : bool
        If `True`, include the physical type label in the output. If `False`, omit
        the physical label entirely (useful for creating unit-only labels).
    show_unit : bool
        If `True`, include the unit in the output. If `False`, omit the unit
        (useful for creating label-only outputs).
    fmt : str | _Unset, optional, default=_UNSET
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
