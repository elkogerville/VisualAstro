"""
Author: Elko Gerville-Reache
Date Created: 2026-04-08
Date Modified: 2026-11-14
Description:
    Spectra plot utility functions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import astropy.units as u
import matplotlib.axes as maxes
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
from matplotlib.typing import ColorType, LineStyleType

from visualastro.analysis.spectra import (
    GaussianFitResult,
    shift_to_rest_frame,
    spectral_idx_2_world,
)
from visualastro.analysis.spectra.utils import _get_spectral_axis
from visualastro.core.configuration import _UNSET, _resolve_default, _Unset, config
from visualastro.core.data import _ndim, get_value
from visualastro.core.kwargs import _param, _resolve_kwargs
from visualastro.core.sequences import _broadcast_param, _cycle, match_length
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
from visualastro.plotting.core.colors.utils import _is_colorset
from visualastro.plotting.core.lines import _axlines, axvlines


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
        - ``style='marker'``: y-coordinate of the base of the markers (required).
        - ``style='vline'``: y-coordinate (data) of the label anchor.
          If ``None``, labels anchor at the top of the axes.

    h : float | astropy.units.Quantity | None, optional, default=None
        Height of the markers. Required if ``style='marker'``; ignored
        if ``style='vline'``.
    ax : matplotlib.axes.Axes | None, optional, default=None
        Axes on which to draw. If ``None``, uses ``plt.gca()``.
    style : {'marker', 'vline'}, optional, default='marker'
        - ``'marker'``: multi-prong marker delineating each line.
        - ``'vline'``: ``ax.axvline`` at each x-value, spanning the full
          axes height.

    label : str | Sequence[str | None] | None, optional, default=None
        Label text.

        - ``style='marker'``: ``str`` only; one label for the whole group.
        - ``style='vline'``: ``str`` is broadcast to every line; a
          sequence must satisfy ``len(label) == len(x)`` and is paired
          with ``x`` in the order given. Individual entries may be ``None``
          to skip a line.

    direction : {'up', 'down'} | _Unset, optional, default=_UNSET
        Direction in which the prongs point away from the connecting
        hline. ``'up'``: prongs extend upward from ``y`` to ``y - h``;
        ``'down'``: from ``y`` to ``y + h``. Marker mode only. If ``_UNSET``, uses
        ``config.spectral_line_marker.marker_direction``.
    label_offset_points : tuple[float, float] | _Unset, optional, default=_UNSET
        (dx, dy) label offset in points. The sign of ``dy`` is overridden
        so that the label always sits outside the marker (marker mode)
        or below the top axes edge (vline mode). If ``_UNSET``, uses
        ``config.spectral_line_marker.label_offset_points``.
    label_position : {'center', 'left', 'right'} | _Unset, optional, default=_UNSET
        Label position relative to the marker group (marker mode). If
        ``_UNSET``, uses ``config.spectral_line_marker.label_position``.
    label_anchor : {'center', 'left', 'right', 'auto'} | _Unset, optional, default=_UNSET
        Text horizontal alignment (matplotlib ``ha``). If ``'auto'``, equals
        ``label_position`` in marker mode; in vline mode, ``'left'`` and
        ``'right'`` are swapped so that text sits beside the line. If
        ``_UNSET``, uses ``config.spectral_line_marker.label_anchor``.
    label_reference : {'marker', 'hline', 'auto'} | _Unset, optional, default=_UNSET
        Reference for label x-position (marker mode only).

        - ``'marker'``: marker x-extent (ignores hline extension).
        - ``'hline'``: hline x-extent (includes extension).
        - ``'auto'``: ``'hline'`` if ``hline_extend`` is not ``None``, else
          ``'marker'``.

        If ``_UNSET``, uses ``config.spectral_line_marker.label_reference``.
    rotation : float | _Unset, optional, default=_UNSET
        Label rotation in degrees. If ``_UNSET``, uses
        ``config.spectral_line_marker.label_rotation``.
    draw_hline : bool, optional, default=True
        Marker mode only. If ``False``, only the vertical prongs are drawn
        and the connecting horizontal bar is omitted. ``hline_extend`` is
        then ignored, including for label placement
        (``label_reference='hline'`` and ``'auto'`` resolve to the prong
        extent).
    hline_extend : float | astropy.units.Quantity | None, optional, default=None
        Marker mode only. If ``None``, the connector spans all x-values
        (drawn only if ``len(x) > 1``). Otherwise, the connector spans the
        prongs and is extended by this distance in the direction set by
        ``label_position``:

        - ``'left'``: extends leftward from the leftmost x.
        - ``'right'``: extends rightward from the rightmost x.
        - ``'center'``: extends symmetrically about the center; the
            connector never shrinks below the prong span.

    fontsize : float | _Unset, optional, default=_UNSET
        Label font size. If ``_UNSET``, uses ``config.fontsizes.resolve('text')``.
    color : ColorType | _Unset, optional, default=_UNSET
        Color of markers/lines and label text. If ``_UNSET``, uses
        ``config.axline.color``.
    linewidth : float, optional, default=0.7
        Linewidth of vlines/markers. If ``_UNSET`` uses
        ``config.spectral_line_marker.linewidth``.
    linestyle : LineStyleType, optional, default='dashed'
        Linestyle of the vlines/markers. If ``_UNSET`` uses
        ``config.spectral_line_marker.marker_linestyle`` if
        ``style='marker'`` else ``config.spectral_line_marker.vline_linestyle``
    va : {'top', 'bottom', 'center'}, optional, default='top'
        Vertical alignment of the label in ``style='vline'``.
    **kwargs
        Additional keyword arguments passed to ``ax.vlines``, ``ax.hlines``
        (marker mode) or ``ax.axvline`` (vline mode).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        - ``style`` is not ``'marker'`` or ``'vline'``.
        - No ``x`` values given.
        - ``y`` or ``h`` missing in marker mode.
        - ``label`` is a sequence in marker mode.
        - ``len(label) != len(x)`` in vline mode.

    Notes
    -----
    - Modifies ``ax`` in place.
    - ``y``, ``h``, ``direction``, ``draw_hline``, ``hline_extend``, ``label_reference`` and
      ``label_position`` have no effect in vline mode.
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


def spectral_vlines(
    x: float | u.Quantity | Sequence[float | u.Quantity],
    labels: str | Sequence[str | None] | None = None,
    *,
    y: float | u.Quantity | Sequence[float | u.Quantity | None] | None = None,
    ax: maxes.Axes | None = None,
    color: ColorType | str | list[ColorType] | _Unset = _UNSET,
    unlabeled_color: ColorType | _Unset = _UNSET,
    linewidth: float | list[float] | _Unset = _UNSET,
    linestyle: LineStyleType | list[LineStyleType] | _Unset = _UNSET,
    alpha: float | list[float] | _Unset = _UNSET,
    unit: u.UnitBase | None = None,
    equivalencies: u.Equivalency | None = None,
    fontsize: float | _Unset = _UNSET,
    rotation: float | _Unset = _UNSET,
    rotation_mode: Literal['default', 'anchor'] | _Unset = _UNSET,
    ha: Literal['left', 'center', 'right'] | _Unset = _UNSET,
    va: Literal['top', 'bottom', 'center', 'baseline'] | _Unset = _UNSET,
    label_offset_points: tuple[float, float] | _Unset = _UNSET,
    label_kw: Mapping[str, Any] | None = None,
    **kwargs,
) -> tuple[list[Line2D], list[Annotation]]:
    """
    Plot full-height vertical lines at spectral line positions with optional labels.

    Parameters
    ----------
    x : float | astropy.units.Quantity | Sequence[float | astropy.units.Quantity]
        Line positions. A scalar is treated as a single line.
    labels : str | Sequence[str | None] | None, optional, default=None
        - ``str``: broadcast to every line.
        - sequence: one entry per line (length 1 or ``len(x)``).
        - ``None``: no labels.

        ``None``, ``''``, whitespace-only strings and ``'none'``
        (case-insensitive) mark a line as unlabeled. Unlabeled lines get no
        annotation.
    y : float | Quantity | Sequence[float | Quantity | None] | None, optional, default=None
        Data y-coordinate of the label anchor. ``Quantity`` values are used
        as their bare values, without conversion. Scalar is broadcast; a
        sequence has length 1 or ``len(x)``. ``None`` (or a ``None`` entry)
        anchors the label at the top of the axes (axes fraction 1.0).
    ax : matplotlib.axes.Axes | None, optional, default=None
        Axes on which to draw. If ``None``, uses ``plt.gca()``.
    color : ColorType | str | list[ColorType] | _Unset, optional, default=_UNSET
        Color of lines and label text. Lines with identical labels share a color.

        - single color (``str`` | RGB(A) ``tuple``): all lines.
        - colorset alias (``str``): palette assigned to unique labels in order
          of first appearance, cycled or cropped.
        - ``list`` | ``numpy.ndarray``:

            - ``len(color) == len(x)``: one color per line, used verbatim.
              Identical labels are not grouped.
            - otherwise: palette assigned to unique labels in order of first
              appearance, cycled if shorter than the number of unique labels
              and cropped if longer. Identical labels share a color.

        If ``_UNSET``, uses ``config.axline.color``.
    unlabeled_color : ColorType | _Unset, optional, default=_UNSET
        Color of unlabeled lines when ``color`` is a colorset or a palette
        ``list``. Ignored if ``color`` is a single color or a per-line ``list``.
        If ``_UNSET``, uses ``config.spectral_line_marker.unlabeled_color``.
    linewidth : float | list[float] | _Unset, optional, default=_UNSET
        Scalar or per-line (``list``) width. If ``_UNSET``, uses
        ``config.spectral_line_marker.linewidth``.
    linestyle : LineStyleType | list[LineStyleType] | _Unset, optional, default=_UNSET
        Scalar or per-line (``list``) style. If ``_UNSET``, uses
        ``config.spectral_line_marker.vline_linestyle``.
    alpha : float | list[float] | _Unset, optional, default=_UNSET
        Scalar or per-line (``list``) alpha. If ``_UNSET``, uses
        ``config.axline.alpha``.
    unit : astropy.units.UnitBase | None, optional, default=None
        Unit to convert ``x`` to. If ``None``, ``Quantity`` values are used
        as their bare values, without conversion.
    equivalencies : astropy.units.Equivalency | None, optional, default=None
        Equivalencies used in the unit conversion of ``x``.
    fontsize : float | _Unset, optional, default=_UNSET
        Label font size. If ``_UNSET``, uses ``config.fontsizes.resolve('text')``.
    rotation : float | _Unset, optional, default=_UNSET
        Label rotation in degrees (positive turns CCW). If ``_UNSET``, uses
        ``config.spectral_line_marker.label_rotation``.
    rotation_mode : {'default', 'anchor'} | _Unset, optional, default=_UNSET
        Rotation order for the label (matplotlib ``rotation_mode``).

        - ``'default'``: rotate the text, then align its bbox with ``ha`` and
            ``va``.
        - ``'anchor'``: align with ``ha`` and ``va``, then rotate about the
            pinned point.

        If ``_UNSET``, uses ``config.spectral_line_marker.vline_label_rotation_mode``.
    ha : {'left', 'center', 'right'} | _Unset, optional, default=_UNSET
        Horizontal alignment of the label. Selects which point of the text
        bbox is pinned to the anchor plus offset. If ``_UNSET``, uses
        ``config.spectral_line_marker.label_ha``.
    va : {'top', 'bottom', 'center', 'baseline'} | _Unset, optional, default=_UNSET
        Vertical alignment of the label. Selects which point of the text
        bbox is pinned to the anchor plus offset (``'top'``: text hangs
        below the anchor). If ``_UNSET``, uses
        ``config.spectral_line_marker.label_va``.
    label_offset_points : tuple[float, float] | _Unset, optional, default=_UNSET
        Signed ``(dx, dy)`` shift of the label in points, applied after
        anchoring. Positive ``dx`` moves right, positive ``dy`` moves up,
        independent of ``ha``, ``va`` and ``rotation``. If ``_UNSET``,
        ``dx`` is ``config.spectral_line_marker.vline_label_offset_points[0]``
        and ``dy`` is chosen from ``va`` (``'top'``: -4, ``'bottom'`` and
        ``'baseline'``: 4, ``'center'``: 0) so the label clears the anchor.
    label_kw : Mapping[str, Any] | None, optional, default=None
        Extra keyword arguments for ``ax.annotate``. Override label defaults.
    **kwargs
        Forwarded to :func:`_axlines` (and from there to ``ax.axvline``).

    Returns
    -------
    lines : list[matplotlib.lines.Line2D]
        One per ``x`` value, in input order.
    annotations : list[matplotlib.text.Annotation]
        One per non-``None`` label, in input order.

    Raises
    ------
    ValueError
        - ``x`` is empty.
        - ``labels``, ``y``, or a ``list`` style parameter has length
          other than 1 or ``len(x)``.
        - ``color`` contains an invalid color.
        - Units are inconsistent among ``x`` (when ``unit`` is ``None``),
          or among ``y``.

    Notes
    -----
    - ``tuple`` inputs for ``labels`` and ``y`` are treated as sequences.
      For ``color``, a ``tuple`` is always a single RGB(A) color.
    - If no line is labeled, a colorset or palette is cycled per line.
    - The offset is not tied to ``va``. With ``va='bottom'``, use a positive
      ``dy`` to place the label above the anchor.
    - With ``rotation_mode='default'``, ``ha`` and ``va`` act on the rotated
      text bbox. With ``'anchor'``, they act on the unrotated text, and the
      rotation pivots about the pinned point.
    """
    xs = [x] if _ndim(x) == 0 else list(x)
    n = len(xs)
    if n == 0:
        raise ValueError('At least one x value is required.')

    labs = [
        _normalize_label(lab)
        for lab in _broadcast_param(_tuple_to_list(labels), n, 'labels')
    ]
    ys = _broadcast_param(_tuple_to_list(y), n, 'y')

    params = _resolve_kwargs(
        kwargs,
        [
            _param('color', color, config.axline.color),
            _param(
                'unlabeled_color', unlabeled_color,
                config.spectral_line_marker.unlabeled_color
            ),
            _param('ha', ha, config.spectral_line_marker.label_ha),
            _param('va', va, config.spectral_line_marker.label_va),
            _param('rotation', rotation, config.spectral_line_marker.label_rotation),
            _param(
                'rotation_mode', rotation_mode,
                config.spectral_line_marker.label_rotation_mode
            ),
            _param('fontsize', fontsize, config.fontsizes.resolve('text')),
            _param('linewidth', linewidth, config.spectral_line_marker.linewidth),
            _param('linestyle', linestyle, config.spectral_line_marker.vline_linestyle),
        ],
    )
    if label_offset_points is _UNSET:
        dx, _ = config.spectral_line_marker.vline_label_offset_points
        dy = _VA_DEFAULT_DY[params.va]
    else:
        dx, dy = label_offset_points

    ax = get_ax(ax)

    lines = _axlines(
        xs,
        orientation='v',
        ax=ax,
        color=_label_colors(params.color, labs, params.unlabeled_color),
        linewidth=params.linewidth,
        linestyle=params.linestyle,
        alpha=alpha,
        unit=unit,
        equivalencies=equivalencies,
        **kwargs,
    )
    assert lines is not None

    annotations: list[Annotation] = []
    for ln, yi, lab in zip(lines, ys, labs, strict=True):
        if lab is None:
            continue
        xv = float(ln.get_xdata()[0])
        if yi is None:
            xy, xycoords = (xv, 1.0), ax.get_xaxis_transform()
        else:
            xy, xycoords = (xv, float(get_value(yi))), 'data'

        text_kw = dict(
            xytext=(dx, dy),
            textcoords='offset points',
            ha=params.ha,
            va=params.va,
            rotation=params.rotation,
            rotation_mode=params.rotation_mode,
            color=ln.get_color(),
            fontsize=params.fontsize,
        )
        text_kw.update(label_kw or {})
        annotations.append(ax.annotate(lab, xy=xy, xycoords=xycoords, **text_kw))

    return lines, annotations


_VA_DEFAULT_DY = {'top': -4.0, 'bottom': 4.0, 'center': 0.0, 'baseline': 4.0}


def _tuple_to_list(v: Any) -> Any:
    """Convert a ``tuple`` to a ``list`` so ``_broadcast_param`` treats it as a sequence."""
    return list(v) if isinstance(v, tuple) else v


def _label_colors(
    color: Any, labs: Sequence[str | None], fallback: ColorType
) -> Any:
    """
    Expand a colorset or color sequence into one color per line, keyed by label.

    Parameters
    ----------
    color : ColorType | str | list[ColorType]
        - colorset alias (``str``): palette over unique labels.
        - ``list`` | ``numpy.ndarray`` with ``len(color) == len(labs)``:
          returned unchanged (one color per line).
        - other ``list`` | ``numpy.ndarray``: palette over unique labels.
        - anything else (single color): returned unchanged.
    labs : Sequence[str | None]
        Per-line labels. Lines with equal labels get equal colors.
    fallback : ColorType
        Color of unlabeled (``None``) lines.

    Returns
    -------
    ColorType | str | list[ColorType]
        ``color`` unchanged, or a ``list`` of ``len(labs)`` colors.

    Raises
    ------
    ValueError
        If ``color`` is an empty ``list`` or ``numpy.ndarray``.

    Notes
    -----
    - Palettes are cycled if shorter than the number of unique labels and
      cropped if longer.
    - A sequence of length ``len(labs)`` is always read as per-line, even
      when labels repeat. Use a colorset alias to force per-label grouping.
    """
    if isinstance(color, str) and _is_colorset(color):
        palette = get_colors(color)
    elif isinstance(color, (list, np.ndarray)) and np.ndim(color) > 0:
        if len(color) == 0:
            raise ValueError('`color` sequence is empty.')
        if len(color) == len(labs):
            return list(color)
        palette = list(color)
    else:
        return color

    unique = list(dict.fromkeys(lab for lab in labs if lab is not None))
    if not unique:
        return match_length(palette, labs)
    mapping = dict(zip(unique, match_length(palette, unique), strict=True))
    return [mapping.get(lab, fallback) for lab in labs]


_NO_LABEL = frozenset({'', 'none'})


def _normalize_label(lab: Any) -> str | None:
    """
    Map empty-like labels to ``None``.

    Parameters
    ----------
    lab : Any
        Label entry. ``None``, ``''``, whitespace-only strings and
        ``'none'`` (case-insensitive) are treated as no label.

    Returns
    -------
    str | None
        The label as given, or ``None`` if it is empty-like.
    """
    if lab is None:
        return None
    if isinstance(lab, str) and lab.strip().lower() in _NO_LABEL:
        return None
    return lab


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
        If ``'marker'``, plots a multi-prong marker delineating
        each spectral peak location. If ``'vline'``, plots a
        ax.axvline at the location of the peak of each line.
    labels : list[str] | None, optional, default=None
        Line labels. If ``None``, no labels shown. Can also be
        ``'auto'`` to format from mu values.
    y_offset : u.Quantity | float | None, optional, default=None
        Offset to add to y-position.
    y_reference : {'peak'} | float, optional, default='peak'
        Base y-position for markers:
        - ``'peak'``: Use peak_height from fit
        - ``u.Quantity`` | ``float``: Start from ``y=float``
        - ``Sequence[u.Quantity | float]``: Cycle through y-positions for each line
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
