"""
Author: Elko Gerville-Reache
Date Created: 2026-07-30
Date Modified: 2026-09-25
Description:
    Legend plotting functions.
"""

from collections.abc import Sequence
from typing import Literal

import matplotlib.axes as maxes
from matplotlib.transforms import Bbox, Transform

from visualastro.core.config import config, _resolve_default, _Unset, _UNSET
from visualastro.core.kwargs import _extract_kwargs, _param
from visualastro.core.numerical_utils import to_list
from visualastro.plotting.core.axes import get_ax


def legend(
    *args,
    ax: maxes.Axes | None = None,
    handles: Sequence | None = None,
    labels: Sequence | None = None,
    loc: str | int | _Unset = _UNSET,
    bbox_to_anchor: tuple | Bbox | _Unset = _UNSET,
    ncols: int | _Unset = _UNSET,
    fontsize: int | str | _Unset = _UNSET,
    labelcolor: str | Sequence | _Unset = _UNSET,
    numpoints: int | _Unset = _UNSET,
    scatterpoints: int | _Unset = _UNSET,
    markerscale: float | _Unset = _UNSET,
    markerfirst: bool | _Unset = _UNSET,
    reverse: bool | _Unset = _UNSET,
    frameon: bool | _Unset = _UNSET,
    fancybox: bool | _Unset = _UNSET,
    framealpha: float | _Unset = _UNSET,
    facecolor: str | _Unset = _UNSET,
    edgecolor: str | _Unset = _UNSET,
    mode: Literal['expand'] | _Unset = _UNSET,
    bbox_transform: Transform | _Unset = _UNSET,
    linewidth: float | Literal['spines'] | _Unset = _UNSET,
    title: str | _Unset = _UNSET,
    alignment: Literal['center', 'left', 'right'] | _Unset = _UNSET,
    borderpad: float | _Unset = _UNSET,
    labelspacing: float | _Unset = _UNSET,
    handlelength: float | _Unset = _UNSET,
    handleheight: float | _Unset = _UNSET,
    borderaxespad: float | _Unset = _UNSET,
    columnspacing: float | _Unset = _UNSET,
    zorder: float | _Unset = _UNSET,
    draggable: bool | _Unset = _UNSET,
    **kwargs
) -> None:
    """
    Create a legend on the specified axes with configuration defaults.

    Parameters
    ----------
    *args : tuple
        Positional arguments for legend specification:

        * If 1 arg: `labels` only
        * If 2 args: `handles`, `labels`

        Maximum of 2 positional arguments allowed.
    ax : matplotlib.axes.Axes | None, optional, default=None
        The axes object on which to place the legend. If `None`, uses
        `get_ax(ax)` to resolve the current axes.
    loc : str | int | _Unset, optional, default=_UNSET
        Legend location. If `_UNSET`, uses `config.legend.loc`.
    bbox_to_anchor : tuple | matplotlib.transforms.Bbox | _Unset, optional, default=_UNSET
        Bbox that the legend will be anchored to. If `_UNSET`, uses `loc`
        relative to the axes.
    ncols : int | _Unset, optional, default=_UNSET
        Number of columns. If `_UNSET`, uses `config.legend.ncols`.
    fontsize : int | str | _Unset, optional, default=_UNSET
        Fontsize for legend text in points. If `_UNSET`, uses
        `config.legend.fontsize`, resolved via
        `config.fontsizes.resolve('legend')` as a scaling factor relative
        to `config.fontsizes.size`.
    labelcolor : str | Sequence | _Unset, optional, default=_UNSET
        Text color for legend labels. Accepts a single color, a list of
        colors, or `'linecolor'` / `'markerfacecolor'` / `'markeredgecolor'`
        to match the corresponding artist property.
    numpoints : int | _Unset, optional, default=_UNSET
        Number of marker points in the legend for `Line2D` entries. If
        `_UNSET`, uses `config.legend.numpoints`.
    scatterpoints : int | _Unset, optional, default=_UNSET
        Number of marker points in the legend for scatter plot entries.
        If `_UNSET`, uses `config.legend.scatterpoints`.
    markerscale : float | _Unset, optional, default=_UNSET
        Relative size of legend markers compared to plotted markers. If
        `_UNSET`, uses `config.legend.markerscale`.
    markerfirst : bool | _Unset, optional, default=_UNSET
        If `True`, marker is placed to the left of the label. If `_UNSET`,
        uses `config.legend.markerfirst`.
    reverse : bool | _Unset, optional, default=_UNSET
        If `True`, reverses the legend entry order. If `_UNSET`, uses
        `config.legend.reverse`.
    frameon : bool | _Unset, optional, default=_UNSET
        Enable legend frame. If `_UNSET`, uses `config.legend.frameon`.
    fancybox : bool | _Unset, optional, default=_UNSET
        Enable rounded box frame. If `_UNSET`, uses `config.legend.fancybox`.
    framealpha : float | _Unset, optional, default=_UNSET
        Frame alpha transparency [0, 1]. If `_UNSET`, uses
        `config.legend.framealpha`.
    facecolor : str | _Unset, optional, default=_UNSET
        Frame background color. If `_UNSET`, uses `config.legend.facecolor`.
    edgecolor : str | _Unset, optional, default=_UNSET
        Frame edge color. If `_UNSET`, uses `config.legend.edgecolor`.
    mode : {'expand'} | _Unset, optional, default=_UNSET
        If `'expand'`, horizontally expands legend to fill axes width.
    bbox_transform : matplotlib.transforms.Transform | _Unset, optional, default=_UNSET
        Transform for `bbox_to_anchor`. If `_UNSET`, uses `ax.transAxes`.
    linewidth : float | {'spines'} | _Unset, optional, default=_UNSET
        Linewidth of the legend. If `'spines'`, uses the `linewidth` of
        the spines from `ax`. If `_UNSET`, does not set.
    title : str | _Unset, optional, default=_UNSET
        Legend title. If `_UNSET`, uses `config.legend.title`.
    alignment : {'center', 'left', 'right'} | _Unset, optional, default=_UNSET
        Legend alignment. If `_UNSET`, uses `config.legend.alignment`.
    borderpad : float | _Unset, optional, default=_UNSET
        Padding between legend content and frame, in units of fontsize.
        If `_UNSET`, uses `config.legend.borderpad`.
    labelspacing : float | _Unset, optional, default=_UNSET
        Vertical spacing between legend entries, in units of fontsize.
        If `_UNSET`, uses `config.legend.labelspacing`.
    handlelength : float | _Unset, optional, default=_UNSET
        Length of legend handles, in units of fontsize. If `_UNSET`,
        uses `2`.
    handleheight : float | _Unset, optional, default=_UNSET
        Height of legend handles, in units of fontsize. If `_UNSET`,
        uses `0.7`.
    borderaxespad : float | _Unset, optional, default=_UNSET
        Padding between legend and axes, in units of fontsize. If
        `_UNSET`, uses `config.legend.borderaxespad`.
    columnspacing : float | _Unset, optional, default=_UNSET
        Spacing between columns in units of fontsize. If `_UNSET`, uses
        `config.legend.columnspacing`.
    draggable : bool | _Unset, optional, default=_UNSET
        Enable legend dragging. If `_UNSET`, uses `config.legend.draggable`.
    zorder : float | _Unset, optional, default=_UNSET
        Legend zorder. If `_UNSET`, uses `config.zorder.legend`.

    Raises
    ------
    ValueError
        If more than 2 positional arguments provided.
    """
    legend_kwargs = _extract_kwargs(
        kwargs,
        params=[
            _param('loc', loc, config.legend.loc),
            _param('bbox_to_anchor', bbox_to_anchor, None),
            _param('ncols', ncols, config.legend.ncols),
            _param('fontsize', fontsize, config.legend.fontsize),
            _param('labelcolor', labelcolor, None),
            _param('numpoints', numpoints, config.legend.numpoints),
            _param('scatterpoints', scatterpoints, config.legend.scatterpoints),
            _param('markerscale', markerscale, config.legend.markerscale),
            _param('markerfirst', markerfirst, config.legend.markerfirst),
            _param('reverse', reverse, config.legend.reverse),
            _param('frameon', frameon, config.legend.frameon),
            _param('fancybox', fancybox, config.legend.fancybox),
            _param('framealpha', framealpha, config.legend.framealpha),
            _param('facecolor', facecolor, config.legend.facecolor),
            _param('edgecolor', edgecolor, config.legend.edgecolor),
            _param('linewidth', linewidth, config.legend.linewidth),
            _param('mode', mode, None),
            _param('bbox_transform', bbox_transform, None),
            _param('title', title, config.legend.title),
            _param('alignment', alignment, config.legend.alignment),
            _param('borderpad', borderpad, config.legend.borderpad),
            _param('labelspacing', labelspacing, config.legend.labelspacing),
            _param('handlelength', handlelength, 2),
            _param('handleheight', handleheight, 0.7),
            _param('borderaxespad', borderaxespad, config.legend.borderaxespad),
            _param('columnspacing', columnspacing, config.legend.columnspacing),
            _param('draggable', draggable, config.legend.draggable),
            _param('zorder', zorder, config.zorder.legend),
        ]
    )
    legend_kwargs['fontsize'] = _resolve_default(
        legend_kwargs['fontsize'], config.fontsizes.resolve('legend')
    )
    ax = get_ax(ax)

    # resolve handles and labels from *args
    if len(args) == 1:
        legend_kwargs['labels'] = args[0]
    elif len(args) == 2:
        handles, labels = args
        if handles is not None:
            legend_kwargs['handles'] = handles
        if labels is not None:
            legend_kwargs['labels'] = labels
    elif len(args) > 2:
        raise ValueError('legend() takes at most 2 positional arguments')

    if handles is not None:
        legend_kwargs['handles'] = handles
    if labels is not None:
        legend_kwargs['labels'] = labels

    for name in ('handles', 'labels'):
        value = legend_kwargs.get(name)
        if value:
            legend_kwargs[name] = to_list(value)
        else:
            legend_kwargs.pop(name, None)

    linewidth = legend_kwargs.pop('linewidth')
    zorder = legend_kwargs.pop('zorder')

    _resolve_vertical_loc(legend_kwargs)

    leg = ax.legend(**legend_kwargs)

    if linewidth:
        if linewidth == 'spines':
            spines = [a for a in ax.spines.values()]
            linewidth = spines[0].get_linewidth()

        leg.get_frame().set_linewidth(linewidth)

    leg.set_zorder(zorder)
