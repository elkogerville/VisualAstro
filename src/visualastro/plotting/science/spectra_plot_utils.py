"""
Author: Elko Gerville-Reache
Date Created: 2026-04-08
Date Modified: 2026-10-25
Description:
    Spectra plot utility functions.
"""

from __future__ import annotations
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal

import astropy.units as u
import matplotlib.axes as maxes
from matplotlib.colors import is_color_like
from matplotlib.typing import ColorType, LineStyleType
import numpy as np

from visualastro.analysis.spectra_utils import (
    GaussianFitResult,
    shift_by_radial_vel,
    spectral_idx_2_world,
    _get_spectral_axis,
)
from visualastro.core.config import (
    config,
    _Unset,
    _UNSET,
    _resolve_default
)
from visualastro.core.data import get_value, _ndim
from visualastro.core.kwargs import _param, _resolve_kwargs
from visualastro.core.sequences import _broadcast, _common_value, _cycle
from visualastro.core.units import (
    convert_quantity,
    ensure_common_unit,
    get_unit,
    unit_2_string,
)
from visualastro.datamodels.datacube import DataCube
from visualastro.optional_dependencies._spectralcube import SpectralCube
from visualastro.optional_dependencies._specutils import SpectralAxis
from visualastro.plotting.core.axes import get_ax
from visualastro.plotting.core.colors.colors import get_colors
from visualastro.plotting.core.lines import axvlines


def spectral_line_marker(
    *x: float | u.Quantity,
    y: float | u.Quantity | None = None,
    h: float | u.Quantity | None = None,
    ax: maxes.Axes | None = None,
    style: Literal['marker', 'vline'] = 'marker',
    label: str | Sequence[str | None] | None = None,
    direction: Literal['up', 'down'] | _Unset = _UNSET,
    label_offset_points: tuple[float, float] | _Unset = _UNSET,
    label_position: Literal['center', 'left', 'right'] | _Unset = _UNSET,
    label_anchor: Literal['center', 'left', 'right', 'auto'] | _Unset = _UNSET,
    label_reference: Literal['marker', 'hline', 'auto'] | _Unset = _UNSET,
    rotation: float | _Unset = _UNSET,
    draw_hline: bool = True,
    hline_extend: float | u.Quantity | None = None,
    fontsize: float | _Unset = _UNSET,
    color: ColorType | _Unset = _UNSET,
    linewidth: float | _Unset = _UNSET,
    linestyle: LineStyleType | _Unset = _UNSET,
    va: Literal['top', 'bottom', 'center'] = 'top',
    **kwargs
) -> None:
    """
    Plot spectral line prong markers or vertical lines with optional labels.

    Parameters
    ----------
    *x : float | astropy.units.Quantity
        X-coordinates of the spectral lines. At least one is required.
    y : float | astropy.units.Quantity | None, optional, default=None
        - style='marker': y-coordinate of the base of the markers (required).
        - style='vline': y-coordinate (data) of the label anchor.
          If None, labels anchor at the top of the axes.

    h : float | astropy.units.Quantity | None, optional, default=None
        Height of the markers. Required if `style='marker'`; ignored
        if `style='vline'`.
    ax : matplotlib.axes.Axes | None, optional, default=None
        Axes on which to draw. If `None`, uses `plt.gca()`.
    style : {'marker', 'vline'}, optional, default='marker'
        - `'marker'`: multi-prong marker delineating each line.
        - `'vline'`: `ax.axvline` at each x-value, spanning the full
          axes height.

    label : str | Sequence[str | None] | None, optional, default=None
        Label text.

        - `style='marker'`: `str` only; one label for the whole group.
        - `style='vline'`: `str` is broadcast to every line; a
          sequence must satisfy `len(label) == len(x)` and is paired
          with `x` in the order given. Individual entries may be `None`
          to skip a line.

    direction : {'up', 'down'} | _Unset, optional, default=_UNSET
        Direction in which the prongs point away from the connecting
        hline. `'up'`: prongs extend upward from `y` to `y - h`;
        `'down'`: from `y` to `y + h`. Marker mode only. If `_UNSET`, uses
        `config.spectral_line_marker.marker_direction`.
    label_offset_points : tuple[float, float] | _Unset, optional, default=_UNSET
        (dx, dy) label offset in points. The sign of `dy` is overridden
        so that the label always sits outside the marker (marker mode)
        or below the top axes edge (vline mode). If `_UNSET`, uses
        `config.spectral_line_marker.label_offset_points`.
    label_position : {'center', 'left', 'right'} | _Unset, optional, default=_UNSET
        Label position relative to the marker group (marker mode). If
        `_UNSET`, uses `config.spectral_line_marker.label_position`.
    label_anchor : {'center', 'left', 'right', 'auto'} | _Unset, optional, default=_UNSET
        Text horizontal alignment (matplotlib `ha`). If `'auto'`, equals
        `label_position` in marker mode; in vline mode, `'left'` and
        `'right'` are swapped so that text sits beside the line. If
        `_UNSET`, uses `config.spectral_line_marker.label_anchor`.
    label_reference : {'marker', 'hline', 'auto'} | _Unset, optional, default=_UNSET
        Reference for label x-position (marker mode only).

        - `'marker'`: marker x-extent (ignores hline extension).
        - `'hline'`: hline x-extent (includes extension).
        - `'auto'`: `'hline'` if `hline_extend` is not `None`, else
          `'marker'`.

        If `_UNSET`, uses `config.spectral_line_marker.label_reference`.
    rotation : float | _Unset, optional, default=_UNSET
        Label rotation in degrees. If `_UNSET`, uses
        `config.spectral_line_marker.label_rotation`.
    draw_hline : bool, optional, default=True
        Marker mode only. If `False`, only the vertical prongs are drawn
        and the connecting horizontal bar is omitted. `hline_extend` is
        then ignored, including for label placement
        (`label_reference='hline'` and `'auto'` resolve to the prong
        extent).
    hline_extend : float | astropy.units.Quantity | None, optional, default=None
        Marker mode only. If `None`, the connector spans all x-values
        (drawn only if `len(x) > 1`). Otherwise, the connector spans the
        prongs and is extended by this distance in the direction set by
        `label_position`:

        - `'left'`: extends leftward from the leftmost x.
        - `'right'`: extends rightward from the rightmost x.
        - `'center'`: extends symmetrically about the center; the
            connector never shrinks below the prong span.

    fontsize : float | _Unset, optional, default=_UNSET
        Label font size. If `_UNSET`, uses `config.fontsizes.resolve('text')`.
    color : ColorType | _Unset, optional, default=_UNSET
        Color of markers/lines and label text. If `_UNSET`, uses
        `config.axline.color`.
    linewidth : float, optional, default=0.7
        Linewidth of vlines/markers. If `_UNSET` uses
        `config.spectral_line_marker.linewidth`.
    linestyle : LineStyleType, optional, default='dashed'
        Linestyle of the vlines/markers. If `_UNSET` uses
        `config.spectral_line_marker.marker_linestyle` if
        `style='marker'` else `config.spectral_line_marker.vline_linestyle`
    va : {'top', 'bottom', 'center'}, optional, default='top'
        Vertical alignment of the label in `style='vline'`.
    **kwargs
        Additional keyword arguments passed to `ax.vlines`, `ax.hlines`
        (marker mode) or `ax.axvline` (vline mode).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        - `style` is not `'marker'` or `'vline'`.
        - No `x` values given.
        - `y` or `h` missing in marker mode.
        - `label` is a sequence in marker mode.
        - `len(label) != len(x)` in vline mode.

    Notes
    -----
    - Modifies `ax` in place.
    - `y`, `h`, `direction`, `draw_hline`, `hline_extend`, `label_reference` and
      `label_position` have no effect in vline mode.
    - Vline labels are placed at the top of the axes (axes-fraction
      y = 1.0) via the x-axis blended transform.
    - Marker mode assumes a non-inverted y-axis.
    """
    marker_style = str(style).lower()
    if marker_style not in {'vline', 'marker'}:
        raise ValueError(
            f"style must be either 'vline' or 'marker'! got: {marker_style}"
        )
    if not x:
        raise ValueError('At least one x value is required.')
    if marker_style == 'marker':
        if y is None or h is None:
            raise ValueError("`y` and `h` are required for style='marker'.")
        if label is not None and not isinstance(label, str):
            raise ValueError(
                "`label` must be a str for style='marker'! "
                f'got: {type(label).__name__}'
            )
    if style == 'marker':
        ls_default = config.spectral_line_marker.marker_linestyle
    else:
        ls_default = config.spectral_line_marker.vline_linestyle

    params = _resolve_kwargs(
        kwargs,
        [
            _param('marker_direction', direction, config.spectral_line_marker.marker_direction),
            _param('label_offset_points', label_offset_points, config.spectral_line_marker.label_offset_points),
            _param('label_position', label_position, config.spectral_line_marker.label_position),
            _param('label_anchor', label_anchor, config.spectral_line_marker.label_anchor),
            _param('label_reference', label_reference, config.spectral_line_marker.label_reference),
            _param('rotation', rotation, config.spectral_line_marker.label_rotation),
            _param('fontsize', fontsize, config.fontsizes.resolve('text')),
            _param('color', color, config.axline.color),
            _param('linewidth', linewidth, config.spectral_line_marker.linewidth),
            _param('linestyle', linestyle, ls_default),
        ],
    )
    ax = get_ax(ax)

    if params.label_anchor == 'auto':
        if marker_style == 'vline':
            params.label_anchor = {'left': 'right', 'right': 'left'}.get(
                params.label_position, params.label_position
            )
        else:
            params.label_anchor = params.label_position

    if params.label_reference == 'auto':
        params.label_reference = (
            'hline' if hline_extend is not None and draw_hline else 'marker'
        )

    x_list = list(x)
    ensure_common_unit([v for v in (*x_list, hline_extend) if v is not None])
    if marker_style == 'marker':
        ensure_common_unit([y, h])

    dx, dy = params.label_offset_points

    # #####
    # vline
    # #####
    if marker_style == 'vline':
        x_raw = [get_value(v) for v in x_list]
        n = len(x_raw)

        if label is None:
            labels = [None] * n
        elif isinstance(label, str):
            labels = [label] * n
        else:
            labels = list(label)
            if len(labels) != n:
                raise ValueError(
                    f'len(label)={len(labels)} != number of lines={n}'
                )

        if y is None:
            anchor_y, xycoords = 1.0, ax.get_xaxis_transform()
        else:
            anchor_y, xycoords = float(get_value(y)), 'data'

        for x_val, lab in zip(x_raw, labels):
            axvlines(
                x_val,
                ax=ax,
                color=params.color,
                lw=params.linewidth,
                ls=params.linestyle,
                **kwargs
            )
            if lab is None:
                continue
            ax.annotate(
                lab,
                xy=(float(x_val), anchor_y),
                xycoords=xycoords,
                xytext=(dx, -abs(dy)),
                textcoords='offset points',
                ha=params.label_anchor,
                va=va,
                rotation=params.rotation,
                color=params.color,
                fontsize=params.fontsize,
            )
        return None

    # ######
    # marker
    # ######
    assert h is not None and y is not None
    x_vals = sorted(get_value(v) for v in x_list)
    y_val = get_value(y)
    h_val = get_value(h)
    h_val = -h_val if params.marker_direction == 'up' else h_val
    extend_val = get_value(hline_extend) if draw_hline else None

    x_lo, x_hi = x_vals[0], x_vals[-1]

    ax.vlines(
        x_vals, y_val, y_val + h_val,
        color=params.color,
        lw=params.linewidth,
        ls=params.linestyle,
        **kwargs
    )

    h_lo, h_hi = x_lo, x_hi
    if extend_val is not None:
        if params.label_position == 'left':
            h_lo = x_lo - extend_val
        elif params.label_position == 'right':
            h_hi = x_hi + extend_val
        else:
            x_mid = 0.5 * (x_lo + x_hi)
            h_lo = min(x_lo, x_mid - extend_val / 2)
            h_hi = max(x_hi, x_mid + extend_val / 2)

    if draw_hline and (extend_val is not None or len(x_vals) > 1):
        ax.hlines(
            y_val + h_val, h_lo, h_hi,
            color=params.color,
            lw=params.linewidth,
            ls=params.linestyle,
            **kwargs
        )

    if label is not None:
        ref_lo, ref_hi = (
            (x_lo, x_hi) if params.label_reference == 'marker' else (h_lo, h_hi)
        )
        if params.label_position == 'left':
            x_text = ref_lo
        elif params.label_position == 'right':
            x_text = ref_hi
        else:
            x_text = 0.5 * (ref_lo + ref_hi)

        if params.marker_direction == 'down':
            va, dy = 'bottom', abs(dy)
        else:
            va, dy = 'top', -abs(dy)

        ax.annotate(
            label,
            xy=(float(x_text), float(y_val) + float(h_val)),
            xycoords='data',
            xytext=(dx, dy),
            textcoords='offset points',
            ha=params.label_anchor,
            va=va,
            rotation=params.rotation,
            color=params.color,
            fontsize=params.fontsize,
        )

    return None


def spectral_lines(
    wavelengths: Sequence[float | u.Quantity] | u.Quantity | float,
    labels: Sequence[str | None] | None = None,
    *,
    style: Literal['marker', 'vline'] = 'vline',
    ax: maxes.Axes | None = None,
    y: float | u.Quantity | Sequence[float | u.Quantity] | None = None,
    h: float | u.Quantity | Sequence[float | u.Quantity] | None = None,
    color: (
        ColorType | Sequence[ColorType] | Mapping[str, ColorType] | _Unset | None
    ) = _UNSET,
    unlabeled_color: ColorType | _Unset | None = _UNSET,
    draw_hline: bool = False,
    **kwargs,
) -> None:
    """
    Plot multiple spectral lines, grouped and colored by label.

    Parameters
    ----------
    wavelengths : Sequence[float | astropy.units.Quantity] | astropy.units.Quantity | float
        Line positions. A scalar is treated as a single line.
    labels : Sequence[str | None] | None, optional, default=None
        One label per wavelength; `None` entries are unlabeled.
    style : {'marker', 'vline'}, optional, default='vline'
        - `'marker'`: if `draw_hline=True`, lines sharing a label are
            drawn as one multi-prong marker with a single label.
            Unlabeled lines are drawn as individual markers. If
            `draw_hline=False`, every line is drawn as an individual
            marker with its own label.
        - `'vline'`: one vertical line (and label) per wavelength.

    ax : matplotlib.axes.Axes | None, optional, default=None
        Axes on which to draw. If `None`, uses `plt.gca()`.
    y : float | astropy.units.Quantity | Sequence[float | astropy.units.Quantity] | None, optional, default=None
        Scalar (broadcast) or per-line y-coordinate. Required in
        marker mode. If `draw_hline=True`, must be identical for all
        lines sharing a label.
        - style='marker': y-coordinate of the base of the markers (required).
        - style='vline': y-coordinate (data) of the label anchor.
          If None, labels anchor at the top of the axes.

    h : float | astropy.units.Quantity | Sequence[float | astropy.units.Quantity] | None, optional, default=None
        Scalar (broadcast) or per-line marker height. Required in
        marker mode, ignored in vline mode. If `draw_hline=True`, must
        be identical for all lines sharing a label.
    color : ColorType | Sequence[ColorType] | Mapping[str, ColorType] | _Unset | None, optional, default=_UNSET
        - single color: applied to all lines.
        - mapping `{label: color}`: per-label color; missing labels
            fall back to the `spectral_line_marker` default.
        - sequence of colors: assigned to unique labels in order of
            first appearance, cycled if too short. Must be a `list`;
            a tuple of 3 or 4 numbers is read as one RGB(A) color.

        `None` is treated as `_UNSET`. With a palette or mapping,
        unlabeled lines use the `spectral_line_marker` default unless
        `unlabeled_color` is given.
    unlabeled_color : ColorType | _Unset | None, optional, default=_UNSET
        Color of lines without a label. Overrides `color` for those
        lines, including when `color` is a single color. If `_UNSET`
        or `None`, unlabeled lines follow `color` (single color) or the
        `spectral_line_marker` default (palette or mapping).
    draw_hline : bool, optional, default=True
            Marker mode only; ignored in vline mode. If `False`, the
            connecting horizontal bar is omitted, grouping by label is
            disabled, and each line gets its own prong and label
            (equivalent to vline-mode labeling).
    **kwargs
        Forwarded to `spectral_line_marker`.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        - `labels` is a `str`.
    ValueError
        - `style` is not `'marker'` or `'vline'`.
        - `wavelengths` is empty.
        - `len(labels) != len(wavelengths)`.
        - `y` or `h` differ among lines sharing a label (marker mode).
        - `color` is an empty sequence.
        - `unlabeled_color` is not a valid color.
        - Units are inconsistent among wavelengths, or among `y`/`h`.
        - `y` or `h` differ among lines sharing a label
          (marker mode, `draw_hline=True`).

    Notes
    -----
    - Modifies `ax` in place.
    - Units of `wavelengths` and of `y`/`h` are checked separately,
      since they may differ (e.g. wavelength vs. flux).
    """
    style = str(style).lower()
    if style not in {'marker', 'vline'}:
        raise ValueError(f"`style` must be 'marker' or 'vline'; got {style!r}.")

    if isinstance(labels, str):
        raise TypeError('`labels` must be a sequence of str, not a str.')

    if _ndim(wavelengths) == 0:
        wavelengths = np.atleast_1d(wavelengths)
    wavelengths = list(wavelengths)
    n = len(wavelengths)
    if n == 0:
        raise ValueError('At least one wavelength is required.')

    labels = [None] * n if labels is None else list(labels)
    if len(labels) != n:
        raise ValueError(f'len(wavelengths)={n} != len(labels)={len(labels)}')

    ys = _broadcast(y, n, 'y')
    hs = _broadcast(h, n, 'h') if style == 'marker' else [None] * n

    ensure_common_unit(wavelengths)
    y_h = [v for v in (*ys, *hs) if v is not None]
    if y_h: ensure_common_unit(y_h)

    ax = get_ax(ax)
    color = _UNSET if color is None else color
    color_by_label, fallback = _label_colors(color, labels)

    if unlabeled_color is not None and unlabeled_color is not _UNSET:
        if not is_color_like(unlabeled_color):
            raise ValueError(
                f'`unlabeled_color` is not a valid color; got {unlabeled_color!r}.'
            )
        fallback = unlabeled_color

    if style == 'vline':
        for w, lab, yi in zip(wavelengths, labels, ys):
            spectral_line_marker(
                w,
                label=lab,
                y=yi,
                color=color_by_label.get(lab, fallback),
                style='vline',
                ax=ax,
                **kwargs,
            )
        return None

    records = list(zip(wavelengths, labels, ys, hs))

    if not draw_hline:
        for w, lab, yi, hi in records:
            spectral_line_marker(
                w,
                label=lab,
                y=yi,
                h=hi,
                color=color_by_label.get(lab, fallback),
                style='marker',
                draw_hline=False,
                ax=ax,
                **kwargs,
            )
        return None

    labeled = defaultdict(list)
    unlabeled_groups = []
    for rec in records:
        lab = rec[1]
        if lab is None:
            unlabeled_groups.append((None, [rec]))
        else:
            labeled[lab].append(rec)

    for lab, members in [*labeled.items(), *unlabeled_groups]:
        ws, _, ys_g, hs_g = zip(*members)
        spectral_line_marker(
            *ws,
            label=lab,
            y=_common_value(ys_g, 'y', lab),
            h=_common_value(hs_g, 'h', lab),
            color=color_by_label.get(lab, fallback),
            style='marker',
            draw_hline=True,
            ax=ax,
            **kwargs,
        )

    return None


def _label_colors(color, labels):
    """
    Map each unique (non-None) label to a color.

    Parameters
    ----------
    color : ColorType | Sequence[ColorType] | Mapping[str, ColorType] | _Unset | None
        Single color, `{label: color}` mapping, or list of colors
        assigned to unique labels in order of first appearance
        (cycled if too short, extras ignored).
    labels : Sequence[str | None]
        Per-line labels.

    Returns
    -------
    mapping : dict[str, ColorType | _Unset]
        Color per unique label.
    fallback : ColorType | _Unset | None
        Color for lines without a label or missing from `mapping`.

    Raises
    ------
    ValueError
        If `color` is an empty sequence.
    """
    unique = list(dict.fromkeys(lab for lab in labels if lab is not None))

    if color is _UNSET or color is None or is_color_like(color):
        return {lab: color for lab in unique}, color
    if isinstance(color, Mapping):
        return {lab: color.get(lab, _UNSET) for lab in unique}, _UNSET

    palette = get_colors(color)
    if not palette:
        raise ValueError('`color` sequence is empty.')
    return (
        {lab: palette[i % len(palette)] for i, lab in enumerate(unique)},
        _UNSET,
    )


def mark_spectral_lines(
    fit_results: GaussianFitResult | list[GaussianFitResult],
    h: u.Quantity | float,
    ax: maxes.Axes,
    style: Literal['vline', 'marker'] = 'marker',
    labels: list[str] | Literal['auto'] | None = 'auto',
    y_offset: u.Quantity | float | None = None,
    y_reference: Literal['peak'] | u.Quantity | float | Sequence[u.Quantity | float] = 'peak',
    style_cycle=None,
    **kwargs
) -> None:
    """
    Mark spectral lines on a plot from Gaussian fit results.

    The function automatically labels each fitted spectral line
    using the fit parameters `peak_height` and `mu`.

    Internally calls `spectral_line_marker`.

    Parameters
    ----------
    fit_results : GaussianFitResult | list[GaussianFitResult]
        Fitted line results containing mu and peak_height.
    h : u.Quantity | float
        Height of marker prongs.
    ax : matplotlib.axes.Axes
        Axes to draw on.
    style : {'marker', 'vline'}, optional, default='marker'
        If `'marker'`, plots a multi-prong marker delineating
        each spectral peak location. If `'vline'`, plots a
        ax.axvline at the location of the peak of each line.
    labels : list[str] | None, optional, default=None
        Line labels. If `None`, no labels shown. Can also be
        `'auto'` to format from mu values.
    y_offset : u.Quantity | float | None, optional, default=None
        Offset to add to y-position.
    y_reference : {'peak'} | float, optional, default='peak'
        Base y-position for markers:
        - `'peak'`: Use peak_height from fit
        - `u.Quantity` | `float`: Start from `y=float`
        - `Sequence[u.Quantity | float]`: Cycle through y-positions for each line
    style_cycle : list of dict or None, optional
        List of style dicts to cycle through for different lines.
        Each dict can contain color, linestyle, linewidth, etc.
    **kwargs
        Additional arguments passed to spectral_line_marker.

    Examples
    --------
    # simple usage
    mark_spectral_lines(fits, h=0.5, ax=ax, labels=['Hα', 'Hβ'])

    # cycle through colors
    mark_spectral_lines(
        fits, h=0.5, ax=ax, labels=['Hα', 'Hβ'],
        style_cycle=[{'color': 'red'}, {'color': 'blue'}]
    )
    """
    if str(style).lower() == 'vline':
        label_offset_points = kwargs.get('label_offset_points', [0, 0])
    else:
        label_offset_points = kwargs.get(
            'label_offset_points',
            config.spectral_line_marker.label_offset_points
        )

    if not isinstance(fit_results, list):
        fit_results = [fit_results]

    if labels == 'auto':
        labels = [f'{r.mu.value:.3f}' for r in fit_results]

    for i, result in enumerate(fit_results):
        if y_reference == 'peak':
            y_pos = result.peak_height

        elif isinstance(y_reference, Sequence):
            y_ref = _cycle(y_reference, i)
            y_pos = get_value(y_ref)
            unit = get_unit(result.peak_height)
            if unit is not None:
                y_pos *= unit

        elif isinstance(y_reference, (float, int, u.Quantity)):
            y_pos = get_value(y_reference)
            unit = get_unit(result.peak_height)
            if unit is not None:
                y_pos *= unit

        else:
            raise ValueError(f'Invalid y_reference: {y_reference}')

        if y_offset is not None:
            y_pos = get_value(y_pos) + get_value(y_offset)

        label = labels[i] if labels is not None else None

        marker_kwargs = kwargs.copy()
        if style_cycle is not None:
            style_i = _cycle(style_cycle, i)
            marker_kwargs.update(style_i)

        spectral_line_marker(
            result.mu,
            y=y_pos,
            h=h,
            ax=ax,
            style=style,
            label=label,
            label_offset_points=label_offset_points
            **marker_kwargs
        )

    return None


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
        Must have valid physical units convertible via `astropy.units.spectral()`
        equivalencies.
    idx : int | tuple[int, int] | None
        Index or index range specifying the slice:

        - `i` → label corresponding to spectral_axis[i]
        - `[i]` → label corresponding to spectral_axis[i]
        - `[i, j]` → label corresponding to midpoint of spectral_axis[i:j]
        - `None` → label corresponding to midpoint of entire spectral axis

    ax : matplotlib.axes.Axes
        Target matplotlib Axes on which the label will be rendered.
    ref_unit : u.UnitBase | u.StructuredUnit | str | None, optional, default=None
        Reference unit to which the spectral axis will be converted prior to
        computing the label (e.g., `u.nm`, `u.AA`, `u.Hz`, `u.km/u.s`).
        If None, uses the unit from `spectral_axis`.
    radial_vel : Quantity | float | None, optional, default=None
        Radial velocity used to Doppler-shift the spectral axis before computing
        the representative value. Must be velocity-compatible if provided.
    emission_line : str | None, optional, default=None
        Optional emission line identifier to include in the label
        (e.g., `"H alpha"`, `"[O III]"`). If provided, this replaces the
        default spectral symbol prefix.
    as_title : bool, optional, default=False
        If True, render the label as the axes title. Otherwise, render as text
        inside the axes.
    text_loc : tuple[float, float], optional
        Axes-relative coordinates (x, y) for text placement. Default is
        `config.text_loc`.
    text_color : str, optional
        Text color. Default is `config.text_color`.
    highlight : bool, optional
        If True, draw a white background box behind the label text.
        Default is `config.highlight`.
    unit_fmt : str | _Unset, optional, default=_UNSET
        Format for unit rendering. Passed to `to_latex_unit`.

        Accepted options are `'latex'`, `'latex_inline'`, `'fits'`,
        `'unicode'`, `'console'`, `'vounit'`, `'cds'`, `'ogip'`

        If `_UNSET`, uses `config.unit_label_format`.
    fontsize : float | _Unset, optional, default=_UNSET
        Fontsize of spectral label. If `_UNSET`, uses
        `config.fontsizes.text`, expressed as a scaling
        factor relative to `config.fontsizes.size`.

    Raises
    ------
    ValueError
        If spectral_axis is None or does not have valid units.
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

    spectral_axis = shift_by_radial_vel(spectral_axis, radial_vel)
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
