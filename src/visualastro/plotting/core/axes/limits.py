"""
Author: Elko Gerville-Reache
Date Created: 2026-10-28
Date Modified: 2026-11-07
Description:
    Module for setting axes limits.
"""

from typing import Any

import matplotlib.axes as maxes
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.path import Path
import numpy as np
from numpy.typing import NDArray

from visualastro.core.config import (
    config,
    _Unset,
    _UNSET,
    _resolve_default
)
from visualastro.core.data import get_value, _to_xy
from visualastro.core.kwargs import _params_from_explicit, _resolve_kwargs
from visualastro.plotting.core.axes.axes import get_ax


def set_limits(
    ax: maxes.Axes | None = None,
    *,
    limits: float | tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xpad: float | None | _Unset = _UNSET,
    ypad: float | None | _Unset = _UNSET,
    recompute_other_axis: bool | _Unset = _UNSET
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Set the limits of a Matplotlib Axes.

    If only one of ``xlim`` or ``ylim`` is provided, the limits of the other
    axis are recomputed from the data visible within the provided range.
    Matplotlib does not do this by itself. Call after all plotting calls.

    Parameters
    ----------
    ax : matplotlib.axes.Axes | None, optional, default=None
        Axes to modify. If ``None``, uses the current Axes.
    limits : float | tuple[float, float] | None, optional, default=None
        Limits applied to both axes. Cannot be combined with ``xlim`` or ``ylim``.

        - If a scalar, sets ``xlim = ylim = (-abs(limits), abs(limits))``.
        - If a ``tuple[float, float]``, sets ``xlim = ylim = limits`` as is.

    xlim : tuple[float, float] | None, optional, default=None
        X-axis limits, applied as given (inverted limits invert the axis).
        If only one of ``xlim``, ``ylim`` is provided and ``recompute_other_axis=True``,
        the other axis limits are computed from the data visible within the
        provided range. If both are provided, no computation is done.
    ylim : tuple[float, float] | None, optional, default=None
        Y-axis limits. See ``xlim``.
    xpad : float | None, optional, default=None
        Fractional padding of the computed X-axis limits. Must be finite
        and ``> -0.5``. If ``None``, uses ``ax.margins()[0]``.

        - Only used if ``ylim`` is provided without ``xlim``.
        - Ignored if ``recompute_other_axis=False``, or if ``limits`` is given.
        - Never applied to user-provided limits.

    ypad : float | None, optional, default=None
        Fractional padding of the computed Y-axis limits. Only used if ``xlim``
        is provided without ``ylim``. Otherwise behaves as ``xpad``. If ``None``,
        uses ``ax.margins()[1]``.

    recompute_other_axis : bool | _Unset, optional, default=_UNSET
        If ``True`` and only one of ``xlim`` or ``ylim`` is provided, the limits
        of the other axis are computed from the data visible within the
        provided range. If ``False``, only the provided limits are set and
        the other axis is autoscaled by Matplotlib. Ignored if ``limits`` is
        given or if both ``xlim`` and ``ylim`` are provided. If ``_UNSET``,
        uses ``config.axes.recompute_other_axis``.

    Returns
    -------
    xlim, ylim : tuple[float, float]
        Limits on ``ax`` after the call.

    Raises
    ------
    ValueError
        If ``limits`` is combined with ``xlim`` or ``ylim``.
        If ``limits`` is not a scalar or a ``tuple[float, float]``.
        If ``xpad`` or ``ypad`` is not finite or is ``<= -0.5``.

    Notes
    -----
    - If no limits are given, nothing is done.
    - Limits are interpreted in the units the data was plotted in.
    - Setting limits disables autoscaling on the affected axes.
    """
    xpad = _resolve_default(xpad, config.axes.xpad)
    ypad = _resolve_default(ypad, config.axes.ypad)
    recompute_other_axis = _resolve_default(
        recompute_other_axis, config.axes.recompute_other_axis
    )
    ax = get_ax(ax)

    for name, pad in (('xpad', xpad), ('ypad', ypad)):
        if pad is not None and not (np.isfinite(pad) and pad > -0.5):
            raise ValueError(f'{name} must be finite and > -0.5! got {pad}!')

    if limits is not None:
        return _set_symmetric_limits(ax, limits=limits, xlim=xlim, ylim=ylim)

    mx, my = ax.margins()
    _apply_limits(
        ax,
        xlim,
        ylim,
        xpad=mx if xpad is None else xpad,
        ypad=my if ypad is None else ypad,
        recompute_other_axis=recompute_other_axis,

    )

    return ax.get_xlim(), ax.get_ylim()


def _set_symmetric_limits(
    ax: maxes.Axes,
    limits: float | tuple[float, float] | None,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Helper function for setting symmetric limits in `set_limits`."""
    if xlim is not None or ylim is not None:
        raise ValueError('limits cannot be combined with xlim or ylim!')

    if isinstance(limits, (list, tuple)) and len(limits) == 2:
        lim = (float(get_value(limits[0])), float(get_value(limits[1])))
    elif (
        isinstance(limits, (int, float, np.integer, np.floating))
        and not isinstance(limits, (bool, np.bool_))
    ):
        lim = (-abs(float(limits)), abs(float(limits)))
    else:
        raise ValueError(
            'limits must be either a scalar or a tuple[float, float]! '
            f'got {type(limits).__name__}!'
        )
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    return ax.get_xlim(), ax.get_ylim()


def _apply_limits(
    ax: maxes.Axes,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    *,
    xpad: float,
    ypad: float,
    recompute_other_axis: bool = True,
) -> None:
    """
    Set axis limits, recomputing the unspecified axis from the data
    visible within the specified one.

    Helper function for `visualastro.plotting.core.axes.set_limits`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify. Must be populated before calling.
    xlim, ylim : tuple[float, float] | None, optional, default=None
        X and Y-axis limits, applied as given (inverted limits invert the axis).
        If only one is provided and `recompute_other_axis=True`, the limits of
        the other axis are computed from the data visible within the provided range.
        If both are provided, no computation is done.
    xpad, ypad : float
        Fractional padding applied to the computed X/Y limits.
    recompute_other_axis : bool, optional, default=True
        If `True`, the limits of the unspecified axis are computed from the
        data visible within the specified one. If `False`, only the
        specified limits are set and the other axis is left to Matplotlib
        autoscaling.


    Returns
    -------
    None

    Notes
    -----
    - If neither `xlim` nor `ylim` is provided, nothing is done and
      Matplotlib autoscaling is left untouched.
    - If no data lies within the provided limits, the other axis is
      left untouched.
    - Only `Line2D` segments are clipped to the window. All other
      artists contribute their vertices only.
    - In-window vertices are always included. `_clip_segments` only
      adds the points where segments cross the window edges.
    - If the computed range is singular, the limits are given by the
      `nonsingular` method of the axis major locator. `xpad`/`ypad`
      are not applied.
    - `Axes.set_xlim`/`set_ylim` disable autoscaling on the affected
      axis.
    """
    if xlim is None and ylim is None:
        ax.margins(x=xpad, y=ypad)
        ax.autoscale_view()
        return

    if xlim is not None:
        xlim = (float(get_value(xlim[0])), float(get_value(xlim[1])))
        ax.set_xlim(xlim)
    if ylim is not None:
        ylim = (float(get_value(ylim[0])), float(get_value(ylim[1])))
        ax.set_ylim(ylim)

    if not recompute_other_axis or (xlim is not None and ylim is not None):
        return

    w_is_x = xlim is not None
    lo, hi = sorted(xlim if w_is_x else ylim)
    w_log = (ax.get_xscale() if w_is_x else ax.get_yscale()) == 'log'
    t_log = (ax.get_yscale() if w_is_x else ax.get_xscale()) == 'log'
    t_axis = ax.yaxis if w_is_x else ax.xaxis
    set_t = ax.set_ylim if w_is_x else ax.set_xlim
    pad = ypad if w_is_x else xpad

    chunks = []
    for x, y, connected in _get_xy_pairs_from_ax(ax):
        w, t = (x, y) if w_is_x else (y, x)
        inside = (w >= lo) & (w <= hi) & np.isfinite(t)
        chunks.append(t[inside])
        if connected and len(w) > 1:
            chunks.append(
                _clip_segments(w, t, lo, hi, w_log=w_log, t_log=t_log)
            )

    if not chunks:
        return
    vals = np.concatenate(chunks)
    vals = vals[np.isfinite(vals)]
    if t_log:
        vals = vals[vals > 0]
    if vals.size == 0:
        return

    vmin, vmax = vals.min(), vals.max()
    if vmin == vmax:
        lim = t_axis.get_major_locator().nonsingular(vmin, vmax)
    else:
        lim = _pad_limits(vmin, vmax, pad, log=t_log)

    lim = (float(lim[0]), float(lim[1]))
    set_t(lim[::-1] if t_axis.get_inverted() else lim)


def _get_xy_pairs_from_ax(
    ax: maxes.Axes,
) -> list[tuple[NDArray, NDArray, bool]]:
    """
    Get 1D (x, y) arrays in data coordinates from the artists on an Axes.

    Includes `Line2D`, `PathCollection` (scatter), other collections
    drawn in data coordinates (e.g. `fill_between`, `errorbar` bars)
    and patches. Invisible artists and artists not drawn with
    `ax.transData` (e.g. `axhline`, `axvspan`) are skipped.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to read artists from.

    Returns
    -------
    list[tuple[NDArray, NDArray, bool]]
        One `(x, y, connected)` per artist (per path for generic
        collections). If `connected` is `True`, consecutive points are
        joined by straight segments and NaNs are preserved as gaps.
        Otherwise non-finite points have been dropped.
        Empty if `ax` has no valid artists.
    """
    out = []

    def _add(xy, connected=False):
        xy = _to_xy(xy)
        if xy is None:
            return
        if not connected:
            xy = xy[np.isfinite(xy).all(axis=1)]
        elif not np.isfinite(xy).all(axis=1).any():
            return
        if len(xy) > 0:
            out.append((xy[:, 0], xy[:, 1], connected))

    for line in ax.lines:
        if line.get_visible() and line.get_transform() == ax.transData:
            _add(line.get_xydata(), connected=_is_connected_line(line))

    for col in ax.collections:
        if not col.get_visible():
            continue
        if isinstance(col, PathCollection):
            if col.get_offset_transform() == ax.transData:
                _add(col.get_offsets())
        elif col.get_transform() == ax.transData:
            for path in col.get_paths():
                _add(_path_vertices(path))

    for patch in ax.patches:
        if patch.get_visible() and patch.get_data_transform() == ax.transData:
            verts = _path_vertices(patch.get_path())
            _add(patch.get_patch_transform().transform(verts))

    return out


def _path_vertices(path: Path) -> NDArray:
    """
    Get the vertices of a `Path`, excluding `CLOSEPOLY` dummy vertices.

    Helper for `_get_xy_pairs_from_ax`.

    Parameters
    ----------
    path : matplotlib.path.Path
        Path to read vertices from.

    Returns
    -------
    NDArray
        Array of shape (N, 2).
    """
    verts = np.asarray(path.vertices)
    if path.codes is not None:
        verts = verts[path.codes != Path.CLOSEPOLY]
    return verts


def _is_connected_line(line: Line2D) -> bool:
    """
    Whether the vertices of a `Line2D` are drawn as straight segments.

    Parameters
    ----------
    line : matplotlib.lines.Line2D
        Line to check.

    Returns
    -------
    bool
        `False` for marker-only lines and non-default drawstyles
        (e.g. `steps-*`), where the drawn path differs from the vertices.
    """
    return (
        line.get_drawstyle() == 'default'
        and line.get_linestyle() not in ('None', 'none', ' ', '')
    )


def _clip_segments(
    w: NDArray,
    t: NDArray,
    lo: float,
    hi: float,
    *,
    w_log: bool = False,
    t_log: bool = False,
) -> NDArray:
    """
    Get the `t` coordinates at the endpoints of polyline segments
    clipped to the window `lo <= w <= hi`.

    1D implementation of the Liang–Barsky clipping algorithm.
    Helper for `_apply_limits`.

    Parameters
    ----------
    w : NDArray
        Coordinates along the windowed axis, shape (N,). May contain NaN.
    t : NDArray
        Coordinates along the target axis, shape (N,). May contain NaN.
    lo, hi : float
        Window bounds on the `w` axis. Sorted internally.
    w_log, t_log : bool, optional, default=False
        If `True`, the respective axis has a log scale. Segments are
        interpolated in log10 space, matching how they are drawn.
        Non-positive coordinates on logarithmic axes are excluded.

    Returns
    -------
    NDArray
        Values of `t` in data space at the endpoints of the clipped
        segments. Empty if no segment intersects the window.

    Notes
    -----
    - Segments with a non-finite endpoint are discarded, preserving
      the gaps Matplotlib draws at NaNs.
    - Segments with constant `w` coordinates are discarded. Their
      endpoints must be handled separately by the caller.
    """
    lo, hi = sorted((lo, hi))
    if w_log:
        if hi <= 0:
            return np.empty(0)
        lo = np.log10(lo) if lo > 0 else -np.inf
        hi = np.log10(hi)

    ws, ts = _to_scale(w, w_log), _to_scale(t, t_log)
    # w0 & d0 are the starting coordinates of each segment
    # dw & dt are the displacements from each point to the next
    w0, dw = ws[:-1], np.diff(ws)
    t0, dt = ts[:-1], np.diff(ts)

    # coordinate of a point on a segment is given by:
    # x(s) = x0 + sΔx
    # where x0 is the starting point of the segment,
    # Δx is the total distance between the start and end points
    # of the segment, and s is the fraction of the way along the segment
    # of the point (i.e. s=0.5 is the midpoint).
    # plugging lo and hi for x(s) and solving for s we get:
    # s = (lo - x0) / Δx
    # sa and sb represent the s parameter required for each segment
    # to reach the low and hi clipping boundaries.
    # s = 0 is the start of the segment
    # s = 1 is the end of the segment
    # s < 0 & s > 1 lies outside the segment
    with np.errstate(divide='ignore', invalid='ignore'):
        sa = (lo - w0) / dw
        sb = (hi - w0) / dw
    # inner np.minimum and np.maximum of (sa, sb)
    # is to account for a flip in direction of the line
    # sa in one direction == sb in the opposite direction.
    # s0 ensures that all values are >= 0 while s1 <= 1.
    s0 = np.maximum(0.0, np.minimum(sa, sb))
    s1 = np.minimum(1.0, np.maximum(sa, sb))
    # keep finite, non-parallel segments whose clipped parameter
    # interval overlaps the segment's [0, 1] range. this restricts
    # the data to inside the window
    keep = np.isfinite(dw) & np.isfinite(dt) & (dw != 0) & (s0 <= s1)
    t0, dt, s0, s1 = t0[keep], dt[keep], s0[keep], s1[keep]

    # t(s) = t0 + sΔt
    # t-coordinates at the start and end of each clipped segment
    t_clipped = np.concatenate([t0 + s0 * dt, t0 + s1 * dt])

    return 10.0**t_clipped if t_log else t_clipped


def _to_scale(v: NDArray, log: bool) -> NDArray:
    """
    Transform values into axis scale space.

    Helper for `_clip_segments`.

    Parameters
    ----------
    v : NDArray
        Values in data space.
    log : bool
        If `True`, return `log10(v)` with non-positive values set to NaN.
        Otherwise return `v` unchanged.

    Returns
    -------
    NDArray
        Values in scale space.
    """
    if not log:
        return v
    return np.log10(np.where(v > 0, v, np.nan))


def _pad_limits(
    vmin: float,
    vmax: float,
    frac: float,
    *,
    log: bool = False,
) -> tuple[float, float]:
    """
    Pad a data range by a fraction of its extent.

    Helper for `_apply_limits`.

    Parameters
    ----------
    vmin, vmax : float
        Minimum and maximum of the data range. Must satisfy `vmin < vmax`,
        and be positive if `log=True`.
    frac : float
        Fractional padding applied to each side.
    log : bool, optional, default=False
        If `True`, pad in log10 space (multiplicative padding).

    Returns
    -------
    tuple[float, float]
        Padded (min, max) limits.
    """
    if log:
        vmin, vmax = np.log10(vmin), np.log10(vmax)

    pad = frac * (vmax - vmin)
    lo, hi = vmin - pad, vmax + pad

    if log:
        lo, hi = 10.0**lo, 10.0**hi

    return float(lo), float(hi)


def _configure_axes_scaling(
    ax: maxes.Axes,
    kwargs: dict[str, Any],
    **explicit: Any
) -> None:
    """
    Configure the autoscale state and axis scales of an Axes.

    Must be called before the main plotting call/loop in a plotting
    function, and before `visualastro.plotting.core.axes.set_limits`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes instance to configure.
    kwargs : dict[str, Any]
        Keyword arguments forwarded to matplotlib. Used to resolve
        the parameters below.
    **explicit : Any
        Any of the following keyword arguments. Values not given
        explicitly are taken from `kwargs`, then from `config.axes`.

        - autoscale : bool, optional
            Native Matplotlib autoscaling state, applied with
            `ax.set_autoscale_on(autoscale)`.
        - xlog, ylog : bool, optional
            If `True`, sets the respective axis to log scale.

    Returns
    -------
    None

    Notes
    -----
    - `autoscale=False` before plotting prevents Matplotlib from
      rescaling the view, so limits must then be provided to
      `set_limits`.
    - Axis scales are only set to `'log'` when requested. A `False`
      value does not reset an existing log scale to linear.
    """
    limits_defaults = {
        'autoscale': config.axes.autoscale,
        'xlog': config.axes.xlog,
        'ylog': config.axes.ylog,
    }
    params = _resolve_kwargs(
        kwargs,
        params=_params_from_explicit(explicit, limits_defaults)
    )

    ax.set_autoscale_on(bool(params.autoscale))

    if params.xlog:
        ax.set_xscale('log')
    if params.ylog:
        ax.set_yscale('log')
