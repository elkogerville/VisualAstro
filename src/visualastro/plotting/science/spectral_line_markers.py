"""
Author: Elko Gerville-Reache
Date Created: 2026-04-08
Date Modified: 2026-11-14
Description:
    Spectra plot utility functions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import astropy.units as u
import matplotlib.axes as maxes
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
from matplotlib.typing import ColorType, LineStyleType

from visualastro.analysis.spectra import (
    GaussianFitResult,
)
from visualastro.core.configuration import _UNSET, _Unset, config
from visualastro.core.data import _ndim, get_value
from visualastro.core.kwargs import _param, _resolve_kwargs
from visualastro.core.sequences import _broadcast_param, _cycle, match_length
from visualastro.core.units import (
    ensure_common_unit,
    get_unit,
)
from visualastro.plotting.core.axes import get_ax
from visualastro.plotting.core.colors.colors import get_colors
from visualastro.plotting.core.colors.utils import _is_colorset
from visualastro.plotting.core.lines import _axlines

__all__ = ['spectral_elbow_markers', 'spectral_markers', 'spectral_vlines']


ConnectorExtendType: TypeAlias = (
    float |
    u.Quantity |
    tuple[float | u.Quantity, float | u.Quantity] |
    list[float | u.Quantity | tuple[float | u.Quantity, float | u.Quantity] | None]
)


def spectral_markers(
    x: float | u.Quantity | Sequence[float | u.Quantity],
    labels: str | Sequence[str | None] | None = None,
    *,
    y: float | u.Quantity | Sequence[float | u.Quantity],
    h: float | u.Quantity | Sequence[float | u.Quantity],
    ax: maxes.Axes | None = None,
    group: bool = True,
    connector: bool = True,
    connector_extend: ConnectorExtendType | None = None,
    stem: float | u.Quantity | None = None,
    label_loc: Literal['left', 'center', 'right'] | _Unset = _UNSET,
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
) -> tuple[list[LineCollection], list[Annotation]]:
    """
    Plot prong markers at spectral line positions with optional labels.

    Each marker is a set of vertical prongs from ``y`` to ``y + h``, joined
    by a horizontal connector at ``y + h``. Lines sharing a label are drawn
    as one multi-prong marker. An optional vertical stem links the connector
    to the label.

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
        annotation and are never grouped.
    y : float | astropy.units.Quantity | Sequence[float | astropy.units.Quantity]
        Data y-coordinate of the prong tips. Scalar is broadcast; a
        sequence has length 1 or ``len(x)``. ``Quantity`` values are used
        as their bare values, without conversion.
    h : float | astropy.units.Quantity | Sequence[float | astropy.units.Quantity]
        Signed marker height in data units. The connector sits at ``y + h``:
        ``h > 0`` places it above the tips, ``h < 0`` below. Must be nonzero.
        Same broadcasting and unit handling as ``y``.
    ax : matplotlib.axes.Axes | None, optional, default=None
        Axes on which to draw. If ``None``, uses ``plt.gca()``.
    group : bool, optional, default=True
        If ``True``, lines sharing a label form one marker with a single
        label. If ``False``, every line is its own marker and is labeled
        individually.
    connector : bool, optional, default=True
        If ``False``, only the prongs are drawn and ``connector_extend`` and
        ``stem`` are ignored. The connector is drawn only if its length is positive.
    connector_extend : ConnectorExtendType | None, optional, default=None
        Extension of the connector beyond the outermost prongs, in x data units. A
        scalar extends both sides; a tuple is ``(left, right)``. ``Quantity`` values
        are converted to ``unit`` if given, without equivalencies.
    stem : float | astropy.units.Quantity | None, optional, default=None
        Length of a vertical segment from the connector anchor to the label,
        in y data units. The stem is drawn at the label anchor x (see
        ``label_loc``), points away from the tips (direction ``sign(h)``)
        and moves the label anchor to ``y + h + sign(h) * stem``. Drawn only
        for labeled markers and only where a connector is drawn (``connector=True``
        and positive connector length). ``None``and ``0`` draw no stem. Same
        unit handling as ``y`` and ``h``.
    label_loc : {'left', 'center', 'right'} | _Unset, optional, default=_UNSET
        Position of the label anchor along the connector extent (including
        ``connector_extend``). If ``_UNSET``, uses ``config.spectral_line_marker.label_ha``.
    color : ColorType | str | list[ColorType] | _Unset, optional, default=_UNSET
        Color of markers and label text. Lines with identical labels share a color.

        - single color (``str`` | RGB(A) ``tuple``): all markers.
        - colorset alias (``str``): palette assigned to unique labels in
          order of first appearance, cycled or cropped.
        - ``list`` | ``numpy.ndarray``:

          - ``len(color) == len(x)``: one color per line, used verbatim.
            Identical labels are not grouped.
          - otherwise: palette assigned to unique labels, cycled if shorter
            than the number of unique labels and cropped if longer.
            Identical labels share a color.

        A marker uses the color of its first line. If ``_UNSET``, uses
        ``config.axline.color``.
    unlabeled_color : ColorType | _Unset, optional, default=_UNSET
        Color of unlabeled lines when ``color`` is a colorset or a palette
        ``list``. Ignored if ``color`` is a single color or a per-line
        ``list``. If ``_UNSET``, uses ``config.spectral_line_marker.unlabeled_color``.
    linewidth : float | list[float] | _Unset, optional, default=_UNSET
        Scalar or per-line (``list``) width. A marker uses the value of its
        first line. If ``_UNSET``, uses ``config.spectral_line_marker.linewidth``.
    linestyle : LineStyleType | list[LineStyleType] | _Unset, optional, default=_UNSET
        Scalar or per-line (``list``) style. A marker uses the value of its
        first line. If ``_UNSET``, uses ``config.spectral_line_marker.marker_linestyle``.
    alpha : float | list[float] | _Unset, optional, default=_UNSET
        Scalar or per-line (``list``) alpha of the markers (not applied to
        label text). A marker uses the value of its first line. If
        ``_UNSET``, uses ``config.axline.alpha``.
    unit : astropy.units.UnitBase | None, optional, default=None
        Unit to convert ``x`` and ``connector_extend`` to. If ``None``,
        ``Quantity`` values are used as their bare values, without conversion.
    equivalencies : astropy.units.Equivalency | None, optional, default=None
        Equivalencies used in the unit conversion of ``x``.
    fontsize : float | _Unset, optional, default=_UNSET
        Label font size. If ``_UNSET``, uses ``config.fontsizes.resolve('text')``.
    rotation : float | _Unset, optional, default=_UNSET
        Label rotation in degrees (positive turns CCW). If ``_UNSET``, uses
        ``config.spectral_line_marker.label_rotation``.
    rotation_mode : {'default', 'anchor'} | _Unset, optional, default=_UNSET
        Rotation order for the label (matplotlib ``rotation_mode``).

        - ``'default'``: rotate the text, then align its bbox with ``ha`` and ``va``.
        - ``'anchor'``: align with ``ha`` and ``va``, then rotate about the pinned point.

        If ``_UNSET``, uses ``config.spectral_line_marker.label_rotation_mode``.
    ha : {'left', 'center', 'right'} | _Unset, optional, default=_UNSET
        Horizontal alignment of the label. Selects which point of the text
        bbox is pinned to the anchor plus offset. If ``_UNSET``, equals
        ``label_loc``.
    va : {'top', 'bottom', 'center', 'baseline'} | _Unset, optional, default=_UNSET
        Vertical alignment of the label. Selects which point of the text
        bbox is pinned to the anchor plus offset (``'top'``: text hangs
        below the anchor). If ``_UNSET``, ``'bottom'`` when ``h > 0`` and
        ``'top'`` otherwise, so the label sits outside the marker.
    label_offset_points : tuple[float, float] | _Unset, optional, default=_UNSET
        Signed ``(dx, dy)`` shift of the label in points, applied after
        anchoring. Positive ``dx`` moves right, positive ``dy`` moves up,
        independent of ``ha``, ``va`` and ``rotation``. If ``_UNSET``, ``dx``
        is ``config.spectral_line_marker.marker_label_offset_points[0]`` and
        ``dy`` is chosen from ``va`` (``'top'``: -4, ``'bottom'`` and
        ``'baseline'``: 4, ``'center'``: 0).
    label_kw : Mapping[str, Any] | None, optional, default=None
        Extra keyword arguments for ``ax.annotate``. Override label defaults.
    **kwargs
        Forwarded to :class:`matplotlib.collections.LineCollection`.

    Returns
    -------
    markers : list[matplotlib.collections.LineCollection]
        One per marker (group), in order of first appearance.
    annotations : list[matplotlib.text.Annotation]
        One per labeled marker, in the same order.
    """
    xs = [x] if _ndim(x) == 0 else list(x)
    n = len(xs)
    if n == 0:
        raise ValueError('At least one x value is required.')
    if unit is None:
        ensure_common_unit(xs)

    labs = [
        _normalize_label(lab)
        for lab in _broadcast_param(_tuple_to_list(labels), n, 'labels')
    ]
    ys = _broadcast_param(_tuple_to_list(y), n, 'y')
    hs = _broadcast_param(_tuple_to_list(h), n, 'h')
    ensure_common_unit([*ys, *hs, *([] if stem is None else [stem])])
    yv = [float(get_value(v)) for v in ys]
    hv = [float(get_value(v)) for v in hs]
    if any(v == 0 for v in hv):
        raise ValueError('`h` must be nonzero.')
    stem_v = 0.0 if stem is None else float(get_value(stem))
    if stem_v < 0:
        raise ValueError(f'`stem` must be >= 0; got {stem!r}.')

    params = _resolve_kwargs(
        kwargs,
        [
            _param('label_loc', label_loc, config.spectral_line_marker.label_ha),
            _param('color', color, config.axline.color),
            _param('unlabeled_color', unlabeled_color,
                   config.spectral_line_marker.unlabeled_color),
            _param('linewidth', linewidth, config.spectral_line_marker.linewidth),
            _param('linestyle', linestyle, config.spectral_line_marker.marker_linestyle),
            _param('alpha', alpha, config.axline.alpha),
            _param('rotation', rotation, config.spectral_line_marker.label_rotation),
            _param('rotation_mode', rotation_mode,
                   config.spectral_line_marker.label_rotation_mode),
            _param('fontsize', fontsize, config.fontsizes.resolve('text')),
        ],
    )
    if params.label_loc not in {'left', 'center', 'right'}:
        raise ValueError(
            f"`label_loc` must be 'left', 'center' or 'right'; got {params.label_loc!r}."
        )

    ax = get_ax(ax)
    xv = [float(v) for v in _x_to_values(xs, unit, equivalencies)]
    ext = _parse_extend(connector_extend, unit, xs)
    dx_default, _ = config.spectral_line_marker.marker_label_offset_points

    colors = _broadcast_param(
        _label_colors(params.color, labs, params.unlabeled_color), n, 'color'
    )
    widths = _broadcast_param(params.linewidth, n, 'linewidth')
    styles = _broadcast_param(params.linestyle, n, 'linestyle')
    alphas = _broadcast_param(params.alpha, n, 'alpha')

    markers: list[LineCollection] = []
    annotations: list[Annotation] = []

    for idx in _group_indices(labs, group):
        i0 = idx[0]
        if len({yv[i] for i in idx}) > 1 or len({hv[i] for i in idx}) > 1:
            raise ValueError(
                f'`y` and `h` must be identical within group {labs[i0]!r}; '
                'pass `group=False` to draw each line separately.'
            )
        lab = labs[i0]
        xg = [xv[i] for i in idx]
        x_lo, x_hi = min(xg), max(xg)
        y_tip, y_con = yv[i0], yv[i0] + hv[i0]
        ext_lo, ext_hi = ext[i0]
        c_lo, c_hi = (
            (x_lo - ext_lo, x_hi + ext_hi) if connector else (x_lo, x_hi)
        )
        x_anchor = {
            'left': c_lo, 'center': 0.5 * (c_lo + c_hi), 'right': c_hi
        }[params.label_loc]

        has_conn = connector and c_hi > c_lo
        segments = [[(xi, y_tip), (xi, y_con)] for xi in xg]
        if has_conn:
            segments.append([(c_lo, y_con), (c_hi, y_con)])

        y_lab = y_con
        if lab is not None and has_conn and stem_v and len(idx) > 1:
            y_lab = y_con + float(np.sign(hv[i0])) * stem_v
            segments.append([(x_anchor, y_con), (x_anchor, y_lab)])

        lc = LineCollection(
            segments,
            colors=colors[i0],
            linewidths=widths[i0],
            linestyles=styles[i0],
            alpha=alphas[i0],
            **kwargs,
        )
        ax.add_collection(lc)
        markers.append(lc)

        if lab is None:
            continue
        va_g = va if va is not _UNSET else ('bottom' if hv[i0] > 0 else 'top')
        ha_g = ha if ha is not _UNSET else params.label_loc
        annotations.append(
            _annotate_label(
                ax, lab, (x_anchor, y_lab), 'data',
                offset=label_offset_points,
                dx_default=dx_default,
                ha=ha_g, va=va_g,
                rotation=params.rotation,
                rotation_mode=params.rotation_mode,
                color=colors[i0],
                fontsize=params.fontsize,
                label_kw=label_kw,
            )
        )

    return markers, annotations


def spectral_elbow_markers(
    x: float | u.Quantity | Sequence[float | u.Quantity],
    labels: str | Sequence[str | None] | None = None,
    *,
    y: float | u.Quantity | Sequence[float | u.Quantity],
    h: float | u.Quantity | Sequence[float | u.Quantity],
    label_shift: float | u.Quantity | Sequence[float | u.Quantity | None] | None = None,
    elbow_frac: float | Sequence[float] = 0.5,
    half_elbow_gap: float | _Unset = _UNSET,
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
) -> tuple[list[LineCollection], list[Annotation]]:
    """
    Plot one elbow marker per spectral line, with the label displaced in x.

    Each marker is a path from the tip at ``y`` vertically to
    ``y + elbow_frac * h``, horizontally by ``label_shift``, then vertically
    to ``y + h`` (label anchor). With ``elbow_frac=1`` the path is an L: the
    horizontal run sits at ``y + h`` and the label is placed beside its end.

    Parameters
    ----------
    x : float | Quantity | Sequence[float | Quantity]
        Line positions. Scalar is a single line.
    labels : str | Sequence[str | None] | None, optional, default=None
        Scalar is broadcast; a sequence has length 1 or ``len(x)``. ``None``,
        ``''``, whitespace-only and ``'none'`` (case-insensitive) mark a line
        as unlabeled: drawn, but not annotated.
    y : float | Quantity | Sequence[float | Quantity]
        Data y-coordinate of the tips. Scalar is broadcast; a sequence has
        length 1 or ``len(x)``. ``Quantity`` is used as its bare value.
    h : float | Quantity | Sequence[float | Quantity]
        Signed marker height in data units; the label anchor sits at ``y + h``
        (``h > 0``: above the tips). Nonzero. Same handling as ``y``. Increase
        ``|h|`` across neighboring lines to stagger labels.
    label_shift : float | Quantity | Sequence[float | Quantity | None] | None, optional, default=None
        Signed x displacement of the label anchor in x data units. Scalar is
        broadcast; a sequence has length 1 or ``len(x)``. ``None`` means no
        shift. ``Quantity`` is converted to ``unit`` if given, without
        equivalencies.
    elbow_frac : float | Sequence[float], optional, default=0.5
        Fraction of ``h`` at which the horizontal run sits. Scalar is
        broadcast; a sequence has length 1 or ``len(x)``. Each value in
        (0, 1]; ``1`` selects the half-elbow (L) for that line.
    ax : matplotlib.axes.Axes | None, optional, default=None
        Target axes. If ``None``, uses ``plt.gca()``.
    color : ColorType | str | list[ColorType] | _Unset, optional, default=_UNSET
        Color of markers and label text. Lines with identical labels share a
        color.

        - single color (``str`` | RGB(A) ``tuple``): all markers.
        - colorset alias (``str``): palette assigned to unique labels in order
          of first appearance, cycled or cropped.
        - ``list`` | ``numpy.ndarray``: one color per line if
          ``len(color) == len(x)`` (verbatim); otherwise a palette assigned
          to unique labels, cycled or cropped.

        If ``_UNSET``, uses ``config.axline.color``.
    unlabeled_color : ColorType | _Unset, optional, default=_UNSET
        Color of unlabeled lines when ``color`` is a colorset or palette
        ``list``; ignored otherwise. If ``_UNSET``, uses
        ``config.spectral_line_marker.unlabeled_color``.
    linewidth : float | list[float] | _Unset, optional, default=_UNSET
        Scalar or per-line width. If ``_UNSET``, uses
        ``config.spectral_line_marker.linewidth``.
    linestyle : LineStyleType | list[LineStyleType] | _Unset, optional, default=_UNSET
        Scalar or per-line style. If ``_UNSET``, uses
        ``config.spectral_line_marker.marker_linestyle``.
    alpha : float | list[float] | _Unset, optional, default=_UNSET
        Scalar or per-line marker alpha (not applied to text). If ``_UNSET``,
        uses ``config.axline.alpha``.
    unit : astropy.units.UnitBase | None, optional, default=None
        Unit to convert ``x`` and ``label_shift`` to. If ``None``, ``Quantity``
        values are used as bare values.
    equivalencies : astropy.units.Equivalency | None, optional, default=None
        Equivalencies for the conversion of ``x``.
    fontsize : float | _Unset, optional, default=_UNSET
        Label font size. If ``_UNSET``, uses ``config.fontsizes.resolve('text')``.
    rotation : float | _Unset, optional, default=_UNSET
        Label rotation in degrees (CCW positive). If ``_UNSET``, uses
        ``config.spectral_line_marker.label_rotation``.
    rotation_mode : {'default', 'anchor'} | _Unset, optional, default=_UNSET
        Matplotlib ``rotation_mode``.

        - ``'default'``: rotate, then align the bbox with ``ha``/``va``.
        - ``'anchor'``: align with ``ha``/``va``, then rotate about the pinned
          point.

        If ``_UNSET``, uses ``config.spectral_line_marker.label_rotation_mode``.
    ha : {'left', 'center', 'right'} | _Unset, optional, default=_UNSET
        Horizontal label alignment. If ``_UNSET``, uses
        ``config.spectral_line_marker.label_ha``; for half-elbows with nonzero
        ``label_shift``, ``'left'`` if positive, ``'right'`` if negative.
    va : {'top', 'bottom', 'center', 'baseline'} | _Unset, optional, default=_UNSET
        Vertical label alignment (``'top'``: text hangs below the anchor). If
        ``_UNSET``, ``'bottom'`` if ``h > 0`` else ``'top'``; for half-elbows
        with nonzero ``label_shift``, ``'center'``.
    label_offset_points : tuple[float, float] | _Unset, optional, default=_UNSET
        Signed ``(dx, dy)`` label shift in points, applied after anchoring.
        ``+dx`` is right and ``+dy`` is up, independent of ``ha``, ``va`` and
        ``rotation``. If ``_UNSET``:

        - ``dx`` is ``config.spectral_line_marker.marker_label_offset_points[0]``,
          and ``dy`` follows ``va`` (``'top'``: -4, ``'bottom'``/``'baseline'``:
          4, ``'center'``: 0).
        - half-elbows with nonzero ``label_shift``:
          ``(sign(label_shift) * half_elbow_gap, 0)``.

    half_elbow_gap : float | _Unset, optional, default=_UNSET
        Gap in points between the end of the horizontal run and the label for
        half-elbows with nonzero ``label_shift``. Magnitude only; direction
        follows ``sign(label_shift)``. Ignored if ``label_offset_points`` is
        given. If ``_UNSET``, uses ``config.spectral_line_marker.half_elbow_label_gap``.
    label_kw : Mapping[str, Any] | None, optional, default=None
        Extra keyword arguments for ``ax.annotate``; override label defaults.
    **kwargs
        Forwarded to :class:`matplotlib.collections.LineCollection`. Do not
        pass ``colors``, ``linewidths`` or ``linestyles``.

    Returns
    -------
    markers : list[matplotlib.collections.LineCollection]
        One per line, in input order.
    annotations : list[matplotlib.text.Annotation]
        One per labeled line, in input order.
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
    hs = _broadcast_param(_tuple_to_list(h), n, 'h')
    shifts = _broadcast_param(_tuple_to_list(label_shift), n, 'label_shift')

    if unit is None:
        ensure_common_unit([*xs, *(v for v in shifts if v is not None)])
    ensure_common_unit([*ys, *hs])
    yv = [float(get_value(v)) for v in ys]
    hv = [float(get_value(v)) for v in hs]
    if any(v == 0 for v in hv):
        raise ValueError('`h` must be nonzero.')
    fracs = [float(f) for f in _broadcast_param(_tuple_to_list(elbow_frac), n, 'elbow_frac')]
    if any(not 0 <= f <= 1 for f in fracs):
        raise ValueError(f'`elbow_frac` must be in [0, 1]; got {elbow_frac!r}.')
    half = elbow_frac == 1

    params = _resolve_kwargs(
        kwargs,
        [
            _param('half_elbow_gap', half_elbow_gap,
                   config.spectral_line_marker.half_elbow_label_gap),
            _param('ha', ha, config.spectral_line_marker.label_ha),
            _param('color', color, config.axline.color),
            _param('unlabeled_color', unlabeled_color,
                   config.spectral_line_marker.unlabeled_color),
            _param('linewidth', linewidth, config.spectral_line_marker.linewidth),
            _param('linestyle', linestyle, config.spectral_line_marker.marker_linestyle),
            _param('alpha', alpha, config.axline.alpha),
            _param('rotation', rotation, config.spectral_line_marker.label_rotation),
            _param('rotation_mode', rotation_mode,
                   config.spectral_line_marker.label_rotation_mode),
            _param('fontsize', fontsize, config.fontsizes.resolve('text')),
        ],
    )

    ax = get_ax(ax)
    xv = [float(v) for v in _x_to_values(xs, unit, equivalencies)]
    sv = [
        0.0 if v is None else float(v)
        for v in _x_to_values(shifts, unit, None)
    ]
    dx_default, _ = config.spectral_line_marker.marker_label_offset_points

    colors = _broadcast_param(
        _label_colors(params.color, labs, params.unlabeled_color), n, 'color'
    )
    widths = _broadcast_param(params.linewidth, n, 'linewidth')
    styles = _broadcast_param(params.linestyle, n, 'linestyle')
    alphas = _broadcast_param(params.alpha, n, 'alpha')

    markers: list[LineCollection] = []
    annotations: list[Annotation] = []

    for i in range(n):
        y_tip, y_lab = yv[i], yv[i] + hv[i]
        x_lab = xv[i] + sv[i]
        half = fracs[i] == 1

        if half:
            path = [(xv[i], y_tip), (xv[i], y_lab), (x_lab, y_lab)]
        else:
            y_mid = y_tip + fracs[i] * hv[i]
            path = [(xv[i], y_tip), (xv[i], y_mid), (x_lab, y_mid), (x_lab, y_lab)]

        lc = LineCollection(
            [path],
            colors=colors[i],
            linewidths=widths[i],
            linestyles=styles[i],
            alpha=alphas[i],
            **kwargs,
        )
        ax.add_collection(lc)
        markers.append(lc)

        if labs[i] is None:
            continue
        ha_i, va_i, off_i = params.ha, va, label_offset_points
        if half and sv[i] != 0:
            sign = 1 if sv[i] > 0 else -1
            if ha is _UNSET:
                ha_i = 'left' if sign > 0 else 'right'
            if va is _UNSET:
                va_i = 'center'
            if label_offset_points is _UNSET:
                off_i = (sign * params.half_elbow_gap, 0.0)
        elif va is _UNSET:
            va_i = 'bottom' if hv[i] > 0 else 'top'
        annotations.append(
            _annotate_label(
                ax, labs[i], (x_lab, y_lab), 'data',
                offset=off_i,
                dx_default=dx_default,
                ha=ha_i, va=va_i,
                rotation=params.rotation,
                rotation_mode=params.rotation_mode,
                color=colors[i],
                fontsize=params.fontsize,
                label_kw=label_kw,
            )
        )

    return markers, annotations


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

        If ``_UNSET``, uses ``config.spectral_line_marker.label_rotation_mode``.
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
    """
    xs = [x] if _ndim(x) == 0 else list(x)
    n = len(xs)
    if n == 0:
        raise ValueError('At least one x value is required.')
    if unit is None:
        ensure_common_unit(xs)

    labs = [
        _normalize_label(lab)
        for lab in _broadcast_param(_tuple_to_list(labels), n, 'labels')
    ]
    ys = _broadcast_param(_tuple_to_list(y), n, 'y')
    ys_q = [v for v in ys if v is not None]
    if ys_q:
        ensure_common_unit(ys_q)

    params = _resolve_kwargs(
        kwargs,
        [
            _param('color', color, config.axline.color),
            _param('unlabeled_color', unlabeled_color,
                   config.spectral_line_marker.unlabeled_color),
            _param('ha', ha, config.spectral_line_marker.label_ha),
            _param('va', va, config.spectral_line_marker.label_va),
            _param('rotation', rotation, config.spectral_line_marker.label_rotation),
            _param('rotation_mode', rotation_mode,
                   config.spectral_line_marker.label_rotation_mode),
            _param('fontsize', fontsize, config.fontsizes.resolve('text')),
            _param('linewidth', linewidth, config.spectral_line_marker.linewidth),
            _param('linestyle', linestyle, config.spectral_line_marker.vline_linestyle),
        ],
    )
    dx_default, _ = config.spectral_line_marker.vline_label_offset_points

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
        annotations.append(
            _annotate_label(
                ax, lab, xy, xycoords,
                offset=label_offset_points,
                dx_default=dx_default,
                ha=params.ha, va=params.va,
                rotation=params.rotation,
                rotation_mode=params.rotation_mode,
                color=ln.get_color(),
                fontsize=params.fontsize,
                label_kw=label_kw,
            )
        )

    return lines, annotations


_VA_DEFAULT_DY = {'top': -4.0, 'bottom': 4.0, 'center': 0.0, 'baseline': 4.0}
_NO_LABEL = frozenset({'', 'none'})


def _annotate_label(
    ax: maxes.Axes,
    text: str,
    xy: tuple[float, float],
    xycoords: Any,
    *,
    offset: tuple[float, float] | _Unset,
    dx_default: float,
    ha: str,
    va: str,
    rotation: float,
    rotation_mode: str,
    color: ColorType,
    fontsize: float,
    label_kw: Mapping[str, Any] | None,
) -> Annotation:
    """
    Annotate a label at ``xy`` with an offset in points.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to annotate.
    text : str
        Label text.
    xy : tuple[float, float]
        Anchor position in ``xycoords``.
    xycoords : str | matplotlib.transforms.Transform
        Coordinate system of ``xy`` (``ax.annotate`` convention).
    offset : tuple[float, float] | _Unset
        Signed ``(dx, dy)`` shift in points. If ``_UNSET``, uses
        ``(dx_default, _VA_DEFAULT_DY[va])``.
    dx_default : float
        Default ``dx`` in points, used only if ``offset`` is ``_UNSET``.
    ha : str
        Horizontal alignment.
    va : str
        Vertical alignment.
    rotation : float
        Rotation in degrees.
    rotation_mode : str
        Matplotlib ``rotation_mode``.
    color : ColorType
        Text color.
    fontsize : float
        Font size.
    label_kw : Mapping[str, Any] | None
        Extra ``ax.annotate`` keyword arguments; override the defaults above.

    Returns
    -------
    matplotlib.text.Annotation
        The created annotation.
    """
    if offset is _UNSET:
        offset = (dx_default, _VA_DEFAULT_DY[va])
    kw = dict(
        xytext=offset, textcoords='offset points', ha=ha, va=va,
        rotation=rotation, rotation_mode=rotation_mode,
        color=color, fontsize=fontsize,
    )
    kw.update(label_kw or {})
    return ax.annotate(text, xy=xy, xycoords=xycoords, **kw)


def _x_to_values(coords, unit, equivalencies) -> list:
    """Strip units, converting to ``unit`` first if given."""
    out = []
    for c in coords:
        if isinstance(c, u.Quantity):
            if unit is not None:
                c = c.to(unit, equivalencies=equivalencies)
            c = c.value
        out.append(c)
    return out


def _parse_extend(
    extend: ConnectorExtendType | None,
    unit: u.UnitBase | None,
    xs: Sequence[float | u.Quantity],
) -> list[tuple[float, float]]:
    """
    Resolve ``connector_extend`` to a non-negative ``(left, right)`` pair per line.

    Parameters
    ----------
    extend : ConnectorExtendType | None
        - scalar: extends both sides, for every line.
        - ``tuple``: ``(left, right)``, for every line.
        - ``list``: length 1 or ``len(xs)``, one scalar or ``tuple`` per
          line. ``None`` entries mean no extension.
        - ``None``: no extension.

    unit : astropy.units.UnitBase | None
        Unit to convert ``Quantity`` values to, without equivalencies.
    xs : Sequence[float | astropy.units.Quantity]
        Line positions. Used for the length of the output and, when ``unit``
        is ``None``, to check unit consistency with ``extend``.

    Returns
    -------
    list[tuple[float, float]]
        Bare ``(left, right)`` extensions, one per line.

    Raises
    ------
    ValueError
        - ``extend`` is a ``list`` of length other than 1 or ``len(xs)``.
        - An entry is a ``tuple`` of length != 2.
        - Any extension is negative.
        - Units of ``extend`` and ``xs`` are inconsistent (``unit`` is ``None``).
    """
    entries = _broadcast_param(extend, len(xs), 'connector_extend')
    pairs = [
        None if e is None else (e if isinstance(e, tuple) else (e, e))
        for e in entries
    ]
    if any(p is not None and len(p) != 2 for p in pairs):
        raise ValueError(
            '`connector_extend` entries must be a scalar or a (left, right) tuple.'
        )
    if unit is None:
        given = [v for p in pairs if p is not None for v in p]
        if given:
            ensure_common_unit([*xs, *given])

    out = []
    for p in pairs:
        if p is None:
            out.append((0.0, 0.0))
            continue
        lo, hi = (float(v) for v in _x_to_values(list(p), unit, None))
        if lo < 0 or hi < 0:
            raise ValueError(f'`connector_extend` must be >= 0; got {p!r}.')
        out.append((lo, hi))
    return out


def _group_indices(labs, group) -> list[tuple[int, ...]]:
    """Index groups in order of first appearance. ``None`` labels stay single."""
    groups: list[list[int]] = []
    by_label: dict[str, list[int]] = {}
    for i, lab in enumerate(labs):
        if group and lab is not None:
            if lab in by_label:
                by_label[lab].append(i)
                continue
            by_label[lab] = [i]
            groups.append(by_label[lab])
        else:
            groups.append([i])
    return [tuple(g) for g in groups]


def _tuple_to_list(v: Any) -> Any:
    """Convert a ``tuple`` to a ``list`` so ``_broadcast_param`` treats it as a sequence."""
    return list(v) if isinstance(v, tuple) else v


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
