"""
Author: Elko Gerville-Reache
Date Created: 2026-04-08
Date Modified: 2026-04-08
Description:
    Spectra plot utility functions.
Dependencies:
    - astropy
    - matplotlib
    - specutils
"""

from typing import Literal, cast
import astropy.units as u
import matplotlib.axes as maxes
from matplotlib.figure import Figure
import matplotlib.transforms as mtransforms
from specutils import SpectralAxis
from visualastro.analysis.spectra_utils import (
    GaussianFitResult,
    get_spectral_axis,
    shift_by_radial_vel,
    spectral_idx_2_world
)
from visualastro.core.config import (
    config,
    _Unset,
    _UNSET
)
from visualastro.core.numerical_utils import get_value, to_list
from visualastro.core.units import (
    convert_quantity,
    ensure_common_unit,
    get_unit,
    to_latex_unit,
)
from visualastro.core.validation import _type_name


def spectral_line_marker(
    *x: float | u.Quantity,
    y: float | u.Quantity,
    h: float | u.Quantity,
    ax: maxes.Axes,
    label: str | None = None,
    direction: Literal['up', 'down'] | _Unset = _UNSET,
    label_offset_points: tuple[float, float] | _Unset = _UNSET,
    label_position: Literal['center', 'left', 'right'] | _Unset = _UNSET,
    label_anchor: Literal['center', 'left', 'right', 'auto'] | _Unset = _UNSET,
    label_reference: Literal['marker', 'hline', 'auto'] | _Unset = _UNSET,
    rotation: float | _Unset = _UNSET,
    hline_extend: float | u.Quantity | None = None,
    **kwargs
) -> None:
    """
    Plot multi-prong spectral line markers with an optional grouped label.

    Parameters
    ----------
    *x : float or astropy.units.Quantity
        X-coordinates of the spectral lines.
    y : float or astropy.units.Quantity
        Y-coordinate of the base of the markers.
    h : float or astropy.units.Quantity
        Height of the vertical markers.
    ax : matplotlib.axes.Axes
        Matplotlib Axes object on which to draw the markers.
    label : str, optional
        Label for the group of lines.
    direction : Literal['up', 'down'] | _Unset, optional, default=_UNSET
        Direction of the prongs. If ``_UNSET``, uses the default value
        from ``config.spectral_line_marker.marker_direction``.
    label_offset_points : tuple[float, float], optional, default=_UNSET
        (dx, dy) offset of the label in points (display coordinates).
        If ``_UNSET``, uses the default value from
        ``config.spectral_line_marker.label_offset_points``.
    label_position : {'center', 'left', 'right'}, optional, default=_UNSET
        Position of label relative to markers. If ``_UNSET``, uses the
        default value from ``config.spectral_line_marker.label_position``.
    label_anchor : {'center', 'left', 'right', 'auto'}, optional, default=_UNSET
        Text alignment (maps to matplotlib `ha`). Independent of position.
        If ``'auto'``, is set to ``label_position``. If ``_UNSET``, uses
        the default value from ``config.spectral_line_marker.label_anchor``.
    label_reference : {'marker', 'hline', 'auto'}, optional, default=_UNSET
        Reference point for label x-position:
        - ``'marker'``: Position relative to marker x-values (ignores hline extension)
        - ``'hline'``: Position relative to hline endpoints (includes extension)
        - ``'auto'``: Uses 'hline' if hline_extend is set, otherwise 'marker'
        If ``_UNSET``, uses the default value from
        ``config.spectral_line_marker.label_reference``.

    rotation : float, optional, default=_UNSET
        Rotation angle of the label in degrees. If ``_UNSET``, uses the
        default value from ``config.spectral_line_marker.label_rotation``.
    hline_extend : float or astropy.units.Quantity or None, optional
        Horizontal line extension distance. If None (default), draws connector
        between all x values when len(x) > 1. If provided, draws horizontal
        line extending in direction specified by ``label_position``:
        - ``'left'``: extends leftward from leftmost x
        - ``'right'``: extends rightward from rightmost x
        - ``'center'``: extends symmetrically from center
    **kwargs
        Additional keyword arguments passed to vlines and hlines.

    Returns
    -------
    None : Modifies ``ax`` in place.
    """
    marker_direction = (
        config.spectral_line_marker.marker_direction
        if direction is _UNSET else direction
    )
    label_offset_points = (
        config.spectral_line_marker.label_offset_points
        if label_offset_points is _UNSET else label_offset_points
    )
    label_position = (
        config.spectral_line_marker.label_position
        if label_position is _UNSET else label_position
    )
    label_anchor = (
        config.spectral_line_marker.label_anchor
        if label_anchor is _UNSET else label_anchor
    )
    label_reference = (
            config.spectral_line_marker.label_reference
            if label_reference is _UNSET else label_reference
        )
    rotation = (
        config.spectral_line_marker.label_rotation
        if rotation is _UNSET else rotation
    )
    if label_anchor == 'auto': label_anchor = label_position
    if label_reference == 'auto':
        label_reference = 'hline' if hline_extend is not None else 'marker'

    x_list = list(x)
    ensure_common_unit(x_list + [hline_extend])
    ensure_common_unit([y, h])

    x_vals = sorted(get_value(xval) for xval in x_list)
    y_val = get_value(y)
    h_val = get_value(h)
    h_val = -h_val if marker_direction == 'up' else h_val
    extend_val = get_value(hline_extend)

    for x_val in x_vals:
        ax.vlines(x_val, y_val, y_val + h_val, **kwargs)

    if extend_val is not None:
        if label_position == 'left':
            x0, x1 = x_vals[0] - extend_val, x_vals[0]
        elif label_position == 'right':
            x0, x1 = x_vals[-1], x_vals[-1] + extend_val
        else:
            x_mid = 0.5 * (x_vals[0] + x_vals[-1])
            x0, x1 = x_mid - extend_val / 2, x_mid + extend_val / 2

        ax.hlines(y_val + h_val, x0, x1, **kwargs)

    elif len(x_vals) > 1:
        ax.hlines(y_val + h_val, x_vals[0], x_vals[-1], **kwargs)

    if label is not None:
        if label_reference == 'marker':
            if label_position == 'left':
                x_text = x_vals[0]
            elif label_position == 'right':
                x_text = x_vals[-1]
            else:
                x_text = 0.5 * (x_vals[0] + x_vals[-1])
        else:
            if label_position == 'left':
                x_text = x_vals[0] - (extend_val if extend_val is not None else 0)
            elif label_position == 'right':
                x_text = x_vals[-1] + (extend_val if extend_val is not None else 0)
            else:
                x_text = 0.5 * (x_vals[0] + x_vals[-1])

        dx, dy = label_offset_points

        y_text = float(y_val) + float(h_val)
        if marker_direction == 'down':
            va = 'bottom'
            dy = abs(dy)
        else:
            va = 'top'
            dy = -abs(dy)

        text_transform = mtransforms.offset_copy(
            ax.transData,
            fig=cast(Figure, ax.figure),
            x=dx,
            y=dy,
            units='points'
        )

        ax.text(
            float(x_text),
            y_text,
            label,
            transform=text_transform,
            ha=label_anchor,
            va=va,
            rotation=rotation
        )

    return None


def mark_spectral_lines(
    fit_results: GaussianFitResult | list[GaussianFitResult],
    h: u.Quantity | float,
    ax: maxes.Axes,
    labels: list[str] | Literal['auto'] | None = None,
    y_offset: u.Quantity | float | None = None,
    y_reference: Literal['peak'] | float = 'peak',
    label_formatter=None,
    style_cycle=None,
    **kwargs
) -> None:
    """
    Mark spectral lines on a plot from Gaussian fit results.

    The function automatically labels each fitted spectral line
    using the fit parameters ``peak_height`` and ``mu``.

    Parameters
    ----------
    fit_results : GaussianFitResult | list[GaussianFitResult]
        Fitted line results containing mu and peak_height.
    h : u.Quantity | float
        Height of marker prongs.
    ax : matplotlib.axes.Axes
        Axes to draw on.
    labels : list[str] | None, optional, default=None
        Line labels. If ``None``, no labels shown. Can also be
        ``'auto'`` to format from mu values.
    y_offset : u.Quantity | float | None, optional, default=None
        Offset to add to y-position.
    y_reference : {'peak'} | float, optional, default='peak'
        Base y-position for markers:
        - 'peak': Use peak_height from fit
        - float | Quantity: Start from ``y=float``
    label_formatter : callable or None, optional
        Function taking (result, index) and returning label string.
        Overrides labels parameter if provided.
    style_cycle : list of dict or None, optional
        List of style dicts to cycle through for different lines.
        Each dict can contain color, linestyle, linewidth, etc.
    **kwargs
        Additional arguments passed to spectral_line_marker.

    Examples
    --------
    # Simple usage
    mark_spectral_lines(fits, h=0.5, ax=ax, labels=['Hα', 'Hβ'])

    # Auto-format labels from wavelength
    mark_spectral_lines(
        fits, h=0.5, ax=ax,
        label_formatter=lambda r, i: f"{r.mu.value:.2f} μm"
    )

    # Cycle through colors
    mark_spectral_lines(
        fits, h=0.5, ax=ax, labels=['Hα', 'Hβ'],
        style_cycle=[{'color': 'red'}, {'color': 'blue'}]
    )
    """
    if not isinstance(fit_results, list):
        fit_results = [fit_results]

    if labels == 'auto':
        labels = [f'{r.mu.value:.3f}' for r in fit_results]

    for i, result in enumerate(fit_results):
        if y_reference == 'peak':
            y_pos = result.peak_height

        elif isinstance(y_reference, (float, int, u.Quantity)):
            y_pos = get_value(y_reference)
            unit = get_unit(result.peak_height)
            if unit is not None:
                y_pos *= unit

        else:
            raise ValueError(f'Invalid y_reference: {y_reference}')

        if y_offset is not None:
            y_pos = get_value(y_pos) + get_value(y_offset)

        if label_formatter is not None:
            label = label_formatter(result, i)
        elif labels is not None:
            label = labels[i]
        else:
            label = None

        marker_kwargs = kwargs.copy()
        if style_cycle is not None:
            style = style_cycle[i % len(style_cycle)]
            marker_kwargs.update(style)

        spectral_line_marker(
            result.mu,
            y=y_pos,
            h=h,
            ax=ax,
            label=label,
            **marker_kwargs
        )

    return None


def spectral_axis_label(
    spectral_axis: SpectralAxis | u.Quantity,
    idx: int | tuple[int, int] | None,
    ax: maxes.Axes,
    *,
    ref_unit: u.UnitBase | u.StructuredUnit | str | None = None,
    radial_vel: u.Quantity | float | None = None,
    emission_line: str | None = None,
    as_title: bool = False,
    **kwargs
) -> None:
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
        If True, render the label as the axes title. Otherwise, render as text
        inside the axes.
    text_loc : tuple[float, float], optional
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
    text_loc: tuple[float, float] = kwargs.get('text_loc', config.text_loc)
    text_color: str = kwargs.get('text_color', config.text_color)
    highlight: bool = kwargs.get('highlight', config.highlight)

    spectral_axis = get_spectral_axis(spectral_axis)
    if spectral_axis is None:
        raise ValueError(
            'spectral_axis cannot be None! '
            f'got: {_type_name(spectral_axis)}'
        )

    # compute spectral axis value of slice for label
    ref_unit = spectral_axis.unit if ref_unit is None else ref_unit
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

    return None


def _format_spectral_label(
    spectral_value: float,
    spectral_unit: u.UnitBase | u.StructuredUnit,
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
