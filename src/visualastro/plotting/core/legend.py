"""
Author: Elko Gerville-Reache
Date Created: 2026-07-30
Date Modified: 2026-07-30
Description:
    Legend plotting functions.
"""

import matplotlib.axes as maxes

from visualastro.core.config import config
from visualastro.core.kwargs import _extract_kwargs, _kwarg
from visualastro.core.numerical_utils import to_list
from visualastro.plotting.core.axes import get_ax


def legend(*args, ax: maxes.Axes | None = None, **kwargs) -> None:
    """
    Create a legend on the specified axes with configuration defaults.

    Parameters
    ----------
    *args : tuple
        Positional arguments for legend specification:

        * If 1 arg: labels only
        * If 2 args: handles, labels

        Maximum of 2 positional arguments allowed.
    ax : matplotlib.axes.Axes
        The axes object on which to place the legend.
    handles : Sequence, optional
        Artists (lines, patches) to display in legend.
    labels : Sequence, optional
        Text labels corresponding to artists.
    loc : str, optional, default=config.legend.loc
        Legend location.
    ncols : int, optional, default=config.legend.ncols
        Number of columns.
    fontsize : int | str, optional, default=config.legend.fontsize
        Font size for legend text.
    fancybox : bool, optional, default=config.legend.fancybox
        Enable rounded box frame.
    framealpha : float, optional, default=config.legend.framealpha
        Frame alpha transparency [0, 1].
    facecolor : str, optional, default=config.legend.facecolor
        Frame background color.
    edgecolor : str, optional, default=config.legend.edgecolor
        Frame edge color.
    linewidth : float | {'spline'}, None, optional, default='spline'
        Linewidth of the legend. If `spline`, uses the `linewidth` of
        the splines from `ax`. If `None`, does not set.
    title : str, optional, default=config.legend.title
        Legend title.
    alignment : {'center', 'left', 'right'}, optional, default=config.legend.alignment
        Legend alignment.
    columnspacing : float, optional, default=config.legend.columnspacing
        Spacing between columns in units of fontsize.
    zorder : float, optional, default=config.zorder.legend
        Legend zorder.
    draggable : bool, optional, default=config.legend.draggable
        Enable legend dragging.

    Raises
    ------
    ValueError
        If more than 2 positional arguments provided.

    Returns
    -------
    None
    """
    legend_kwargs = _extract_kwargs(
        kwargs,
        additional_kwargs=[
            _kwarg('loc', config.legend.loc),
            _kwarg('ncols', config.legend.ncols),
            _kwarg('fontsize', config.legend.fontsize),
            _kwarg('numpoints', config.legend.numpoints),
            _kwarg('scatterpoints', config.legend.scatterpoints),
            _kwarg('markerscale', config.legend.markerscale),
            _kwarg('markerfirst', config.legend.markerfirst),
            _kwarg('reverse', config.legend.reverse),
            _kwarg('frameon', config.legend.frameon),
            _kwarg('fancybox', config.legend.fancybox),
            _kwarg('framealpha', config.legend.framealpha),
            _kwarg('facecolor', config.legend.facecolor),
            _kwarg('edgecolor', config.legend.edgecolor),
            _kwarg('title', config.legend.title),
            _kwarg('alignment', config.legend.alignment),
            _kwarg('borderpad', config.legend.borderpad),
            _kwarg('labelspacing', config.legend.labelspacing),
            _kwarg('borderaxespad', config.legend.borderaxespad),
            _kwarg('columnspacing', config.legend.columnspacing),
            _kwarg('draggable', config.legend.draggable),
        ]
    )
    ax = get_ax(ax)
    handles = None
    labels = None

    if len(args) == 1:
        labels = args[0]
    elif len(args) == 2:
        handles, labels = args
    elif len(args) > 2:
        raise ValueError('legend() takes at most 2 positional arguments')

    handles = kwargs.pop('handles', handles)
    labels = kwargs.pop('labels', labels)
    linewidth = kwargs.pop('linewidth', config.legend.linewidth)
    zorder = kwargs.pop('zorder', config.zorder.legend)

    if handles is not None:
        legend_kwargs['handles'] = to_list(handles)
    if labels is not None:
        legend_kwargs['labels'] = to_list(labels)

    leg = ax.legend(**legend_kwargs)

    if linewidth is not None:
        if linewidth == 'spines':
            spines = [a for a in ax.spines.values()]
            linewidth = spines[0].get_linewidth()

        leg.get_frame().set_linewidth(linewidth)

    leg.set_zorder(zorder)
