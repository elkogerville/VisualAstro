"""
Author: Elko Gerville-Reache
Date Created: 2026-04-08
Date Modified: 2026-11-14
Description:
    Spectra plot utility functions.
"""

from __future__ import annotations

import astropy.units as u
import matplotlib.axes as maxes
from matplotlib.typing import ColorType

from visualastro.analysis.spectra import (
    shift_to_rest_frame,
    spectral_idx_2_world,
)
from visualastro.analysis.spectra.utils import _get_spectral_axis
from visualastro.core.configuration import _UNSET, _resolve_default, _Unset, config
from visualastro.core.units import (
    convert_quantity,
    unit_2_string,
)
from visualastro.datamodels.datacube import DataCube
from visualastro.optional_dependencies._spectralcube import SpectralCube
from visualastro.optional_dependencies._specutils import SpectralAxis


def spectral_axis_label(
    spectral_axis: SpectralAxis | u.Quantity | SpectralCube | DataCube,
    idx: int | tuple[int, int] | None,
    ax: maxes.Axes,
    *,
    ref_unit: u.UnitBase | u.StructuredUnit | str | None = None,
    radial_vel: u.Quantity | float | None = None,
    emission_line: str | None = None,
    as_title: bool = False,
    text_loc: tuple[float, float] | _Unset = _UNSET,
    text_color: ColorType | _Unset = _UNSET,
    highlight: bool | _Unset = _UNSET,
    unit_fmt: str | _Unset = _UNSET,
    fontsize: float | _Unset = _UNSET
) -> None:
    """
    Add a label indicating the spectral coordinate of a slice.

    This function computes a representative spectral coordinate value for a
    given index or index range along a spectral axis and renders a LaTeX-formatted
    label on a matplotlib Axes. The spectral axis is first converted to the
    specified reference unit and optionally shifted by a radial velocity.

    The label can be displayed either as an axes title or as text positioned
    within the axes.

    If no spectral axis is found, the function does not add a label.

    Parameters
    ----------
    spectral_axis : SpectralAxis or Quantity
        Spectral axis array representing wavelength, frequency, or velocity.
        Must have valid physical units convertible via ``astropy.units.spectral()``
        equivalencies.
    idx : int | tuple[int, int] | None
        Index or index range specifying the slice:

        - ``i`` → label corresponding to spectral_axis[i]
        - ``[i]`` → label corresponding to spectral_axis[i]
        - ``[i, j]`` → label corresponding to midpoint of spectral_axis[i:j]
        - ``None`` → label corresponding to midpoint of entire spectral axis

    ax : matplotlib.axes.Axes
        Target matplotlib Axes on which the label will be rendered.
    ref_unit : u.UnitBase | u.StructuredUnit | str | None, optional, default=None
        Reference unit to which the spectral axis will be converted prior to
        computing the label (e.g., ``u.nm``, ``u.AA``, ``u.Hz``, ``u.km/u.s``).
        If None, uses the unit from ``spectral_axis``.
    radial_vel : Quantity | float | None, optional, default=None
        Radial velocity used to Doppler-shift the spectral axis before computing
        the representative value. Must be velocity-compatible if provided.
    emission_line : str | None, optional, default=None
        Optional emission line identifier to include in the label
        (e.g., ``"H alpha"``, ``"[O III]"``). If provided, this replaces the
        default spectral symbol prefix.
    as_title : bool, optional, default=False
        If ``True``, render the label as the axes title. Otherwise, render as text
        inside the axes.
    text_loc : tuple[float, float], optional
        Axes-relative coordinates (x, y) for text placement. Default is
        ``config.text_loc``.
    text_color : str, optional
        Text color. Default is ``config.text_color``.
    highlight : bool, optional
        If ``True``, draw a white background box behind the label text.
        Default is ``config.highlight``.
    unit_fmt : str | _Unset, optional, default=_UNSET
        Format for unit rendering. Passed to ``to_latex_unit``.

        Accepted options are ``'latex'``, ``'latex_inline'``, ``'fits'``,
        ``'unicode'``, ``'console'``, ``'vounit'``, ``'cds'``, ``'ogip'``

        If ``_UNSET``, uses ``config.unit_label_format``.
    fontsize : float | _Unset, optional, default=_UNSET
        Fontsize of spectral label. If ``_UNSET``, uses
        ``config.fontsizes.text``, expressed as a scaling
        factor relative to ``config.fontsizes.size``.

    Raises
    ------
    ValueError
        If ``spectral_axis`` is ``None`` or does not have valid units.
    """
    text_loc = _resolve_default(text_loc, config.text_loc)
    text_color = _resolve_default(text_color, config.text_color)
    highlight = _resolve_default(highlight, config.highlight)
    fontsize = _resolve_default(fontsize, config.fontsizes.resolve('text'))

    spectral_axis = _get_spectral_axis(spectral_axis)
    if spectral_axis is None:
        return

    # compute spectral axis value of slice for label
    ref_unit = spectral_axis.unit if ref_unit is None else ref_unit
    spectral_axis = convert_quantity(spectral_axis, ref_unit, equivalencies=u.spectral())
    spectral_unit = spectral_axis.unit
    if spectral_unit is None:
        raise ValueError(
            'spectral_axis must have a unit!'
        )

    spectral_axis = shift_to_rest_frame(spectral_axis, radial_vel)
    spectral_value = spectral_idx_2_world(spectral_axis, idx, keep_unit=False)

    slice_label = _format_spectral_label(
        spectral_value, spectral_unit, emission_line=emission_line, fmt=unit_fmt
    )

    if as_title:
        ax.set_title(slice_label, color=text_color, loc='center')
    else:
        bbox = dict(facecolor='white', edgecolor='w') if highlight else None
        ax.text(
            text_loc[0], text_loc[1], slice_label,
            transform=ax.transAxes,
            color=text_color,
            bbox=bbox,
            fontsize=fontsize,
        )


def _format_spectral_label(
    spectral_value: float,
    spectral_unit: u.UnitBase | u.StructuredUnit,
    *,
    emission_line: str | None = None,
    fmt: str | _Unset = _UNSET
) -> str:
    """
    Format a LaTeX label representing a spectral coordinate value.

    This function generates a LaTeX-formatted string suitable for use as a
    matplotlib text label or title. The label represents a spectral axis value
    (e.g., wavelength, frequency, or velocity), optionally prefixed with an
    emission line identifier.

    Used internally by `spectral_axis_label`.

    Parameters
    ----------
    spectral_value : float
        Spectral axis value expressed in the specified `spectral_unit`.
    spectral_unit : Unit | StructuredUnit
        Unit associated with `value`. Must have a valid physical type such
        as `length`, `frequency`, or `speed`.
    emission_line : str | None, optional, default=None
        Optional emission line identifier (e.g., `'H alpha'`, `'[O III]'`).
        If provided, this replaces the default spectral symbol prefix.
    fmt : str | _Unset, optional, default=_UNSET
        Format for unit rendering. Passed to `to_latex_unit`.

        Accepted options are `'latex'`, `'latex_inline'`, `'fits'`,
        `'unicode'`, `'console'`, `'vounit'`, `'cds'`, `'ogip'`

        If `_UNSET`, uses `config.unit_label_format`.

    Returns
    -------
    label : str
        LaTeX-formatted spectral label string enclosed in math mode delimiters.
    """
    fmt = _resolve_default(fmt, config.unit_label_format)

    unit_label = unit_2_string(spectral_unit, fmt=fmt)

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
