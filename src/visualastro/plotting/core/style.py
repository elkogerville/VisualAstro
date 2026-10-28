"""
Author: Elko Gerville-Reache
Date Created: 2025-07-13
Date Modified: 2026-10-25
Description:
    Functions related to setting the plotting style.
"""

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from importlib.resources import files
from pprint import pprint
from typing import Literal

from astropy.visualization.wcsaxes.core import WCSAxes
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType, LineStyleType, MarkerType

from visualastro.core.config import config
from visualastro.core.data import as_list
from visualastro.core.kwargs import _param, _resolve_kwargs
from visualastro.core.sequences import match_length


class VisualAstroStyles:
    """
    Container providing information on VisualAstro Matplotlib styles.

    Styles can be used directly with `matplotlib.pyplot.style.use`
    or `matplotlib.pyplot.style.context`.

    Attributes
    ----------
    available : list[str]
        All available VisualAstro styles registered with Matplotlib.
    axes : AxesStyles
        Collection of axes-related styles.
    fonts : FontStyles
        Collection of font-related styles.

    Examples
    --------
    >>> va.styles.available
    ['computer-modern-serif', 'computer-modern-mono', 'full', 'minimal' ...]

    >>> va.styles.fonts.serif
    ['computer-modern-serif', 'stix', 'times-serif', ...]

    >>> va.styles.axes.available
    ['full', 'minimal', ...]

    >>> plt.style.use(['full', 'computer-modern-serif'])
    """
    class Fonts:
        """
        Collection of font-related styles.

        Attributes
        ----------
        available : list[str]
            All available font styles.
        mono : list[str]
            Available monospace font styles.
        sans : list[str]
            Available sans-serif font styles.
        serif : list[str]
            Available serif font styles.
        """
        def __init__(self, fontstyles):
            self.available = _get_styles(fontstyles)

            for family in ('monospace', 'sans_serif', 'serif'):
                setattr(
                    self,
                    family.replace('_', ''),
                    _get_styles(fontstyles / family)
                )

        def __repr__(self):
            styles = ',\n'.join(f'    {item!r}' for item in self.available)
            return f'Fonts([\n{styles}\n])'

    class Axes:
        """
        Collection of axes-related styles.

        Attributes
        ----------
        available : list[str]
            All available axes styles.
        """
        def __init__(self, axesstyles):
            self.available = _get_styles(axesstyles)

        def __repr__(self):
            styles = ',\n'.join(f'    {item!r}' for item in self.available)
            return f'Axes([\n{styles}\n])'

    def __init__(self, available_styles):
        """VisualAstroStyles initialization."""
        stylelib = files('visualastro') / 'stylelib'

        self.available = available_styles
        self.axes = self.Axes(stylelib / 'axes_styles')
        self.fonts = self.Fonts(stylelib / 'fontstyles')

    def __repr__(self):
        styles = ',\n'.join(f'    {item!r}' for item in self.available)
        return f'Available Styles([\n{styles}\n])'


def _get_styles(directory) -> list[str]:
    """
    Helper function used by `VisualAstroStyles`.

    Returns a sorted list of style names in a given directory.
    """
    return sorted(list({p.stem for p in directory.rglob('*.mplstyle')}))


def _style_context(style: str | None = None) -> AbstractContextManager:
    """
    Return a context manager applying a Matplotlib style, or `nullcontext`
    if none is resolved.

    Parameters
    ----------
    style : str | None, optional, default=None
        Name of the style.

    Returns
    -------
    AbstractContextManager
        `plt.style.context(stylepath)` if a style path was resolved, otherwise
        `nullcontext()`.
    """
    style = style if style in plt.style.available else None
    return (
        plt.style.context(style) if style is not None else nullcontext()
    )


def reset_rcParams() -> None:
    """Reset all rcParams to their default values."""
    mpl.rcdefaults()


def print_rcParams(print_nondefault: bool = True) -> None:
    """
    Print all current rcParams.

    Paramters
    ---------
    print_nondefault : bool, optional, default=True
        If `True`, print only the non default values.
    """
    if print_nondefault:
        diffs = {k: v for k, v in mpl.rcParams.items() if v != mpl.rcParamsDefault[k]}
        pprint(diffs)
    else:
        print(mpl.rcParams)


def set_plot_cycle(
    colors: ColorType | Sequence[ColorType] | None = None,
    linestyles: LineStyleType | Sequence[LineStyleType] | None = None,
    markers: MarkerType | Sequence[MarkerType] | None = None,
    mode: Literal['broadcast', 'zip', 'product'] = 'broadcast',
    **kwargs
) -> None:
    """
    Set the matplotlib axes property cycle for color, linestyle, and/or marker.

    Only sets properties for the arguments passed in. Calling this function
    with no inputs does nothing.

    Parameters
    ----------
    colors : list[str] | None, optional, default=None
        Color sequence, resolved via `get_colors` if provided.
    linestyles : LineStyleType | Sequence[LineStyleType] | None, optional, default=None
        Linestyle sequence (e.g. '-', '--', '-.', ':').
    markers : MarkerType | Sequence[MarkerType] | None, optional, default=None
        Marker sequence (e.g. 'o', 's', '^', 'x').
    mode : {'broadcast', 'zip', 'product'}, optional, default='broadcast'

        * `'broadcast'` cycles shorter sequences to match the running combined length.
        * `'zip'` pairs sequences element-wise (requires equal length).
        * `'product'` generates the Cartesian product of all sequences.
    """
    from visualastro.plotting.core.colors import get_colors

    params = _resolve_kwargs(
        kwargs,
        params=[
            _param('color', colors, config.color),
            _param('linestyle', linestyles, config.linestyle),
            _param('marker', markers, config.marker),
        ]
    )

    cyclers = []
    if params.color is not None:
        cyclers.append(mpl.cycler(color=get_colors(params.color)))
    if params.linestyle is not None:
        cyclers.append(mpl.cycler(linestyle=as_list(params.linestyle)))
    if params.marker is not None:
        cyclers.append(mpl.cycler(marker=as_list(params.marker)))

    if not cyclers:
        return

    combined = cyclers[0]
    for c in cyclers[1:]:
        if mode == 'zip':
            combined = combined + c
        elif mode == 'broadcast':
            key = next(iter(c.keys))
            matched = match_length(c.by_key()[key], combined)
            combined = combined + mpl.cycler(**{key: matched})
        else:
            combined = combined * c

    mpl.rcParams['axes.prop_cycle'] = combined


STYLE_ALIASES = {
    'computer-modern': ('cm', 'cmu'),
    'computer-modern-mono': ('cm-mono', 'cmu-mono'),
    'computer-modern-sans': ('cm-sans', 'cmu-sans'),
    'dejavu': ('dejavu-serif',),
    'times-new-roman': ('tnr', 'timesnewroman', 'times'),
}


def apply_style_modifiers(ax, style: str):
    """
    Apply programmatic style modifiers based on underscore-separated suffixes.
    This updates an axes instance in place with stylistic modifiers.

    Modifiers are appended to the base style name with underscores and can be
    chained together in any order (e.g., 'astro_minimal_grid' or 'latex_bare').
    This function is mostly for internal use by plotting functions in the
    visualastro.plotting.ax module.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or astropy.visualization.wcsaxes.WCSAxes
        Axes object to apply style modifiers to.
    style : str
        Full style string including base style and optional modifiers.
        Format: 'basestyle_modifier1_modifier2_...'
        Example: 'astro_minimal_grid'

    Notes
    -----
    Supported modifiers:
        - minimal : Remove minor ticks and show ticks only on bottom-left axes.
                    For WCSAxes, uses coords positioning. For regular axes,
                    disables top and right ticks.
        - nominor : Remove minor tick marks only, keeping major ticks unchanged.
        - bare : Remove all frame elements including ticks, tick labels, and
                spines/frame. Creates a minimal plot with data only.
        - grid : Add a background grid. Uses config settings for color, alpha,
                and linestyle. Grid style differs between WCSAxes and regular axes.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> apply_style_modifiers(ax, 'astro_minimal')
    >>> apply_style_modifiers(ax, 'latex_grid_nominor')
    >>> apply_style_modifiers(ax, 'default_bare')
    """
    if style is None or '_' not in style:
        return

    parts = style.split('_')
    modifiers = parts[1:]

    for modifier in modifiers:

        modifier = modifier.lower()

        if modifier == 'minimal':
            # remove minor ticks
            ax.tick_params(which='minor', length=0)
            if isinstance(ax, WCSAxes):
                ax.coords['ra'].set_ticks_position('bl')
                ax.coords['dec'].set_ticks_position('bl')
            else:
                ax.tick_params(top=False, right=False)

        elif modifier == 'nominor':
            ax.tick_params(which='minor', length=0)

        elif modifier == 'bare':
            # remove the frame, ticks, and ticklabels
            if isinstance(ax, WCSAxes):
                ax.coords['ra'].set_ticklabel_visible(False)
                ax.coords['dec'].set_ticklabel_visible(False)
                ax.coords['ra'].set_ticks_visible(False)
                ax.coords['dec'].set_ticks_visible(False)
                ax.coords.frame.set_linewidth(0)
            else:
                ax.tick_params(which='both', length=0)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

        elif modifier == 'grid':
            if isinstance(ax, WCSAxes):
                 ax.coords.grid(
                     True,
                     color=config.wcsgrid.color,
                     alpha=config.wcsgrid.alpha,
                     ls=config.wcsgrid.linestyle
                 )
            else:
                ax.grid(
                    True,
                    color=config.gridlines.color,
                    alpha=config.gridlines.alpha,
                    ls=config.gridlines.linestyle
                )
