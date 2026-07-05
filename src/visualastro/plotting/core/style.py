"""
Author: Elko Gerville-Reache
Date Created: 2025-07-13
Date Modified: 2026-07-04
Description:
    Functions related to setting the plotting style.
"""

from contextlib import AbstractContextManager, contextmanager, nullcontext
from importlib.resources import files
import warnings

from astropy.visualization.wcsaxes.core import WCSAxes
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

from visualastro.core.config import (
    config,
    _Unset, _UNSET,
    _resolve_default
)


def _style_context(style: str | None = None) -> AbstractContextManager:
    """
    Return a context manager applying a Matplotlib style, or `nullcontext`
    if none is resolved.

    Parameters
    ----------
    style : str | None, optional, default=None
        Name of the style to resolve via `_get_stylepath`. Ignored if `path` is given.

    Returns
    -------
    AbstractContextManager
        `plt.style.context(stylepath)` if a style path was resolved, otherwise
        `nullcontext()`.
    """
    stylepath = _get_stylepath(style)
    return (
        plt.style.context(stylepath) if stylepath is not None else nullcontext()
    )


def _get_stylepath(style: str | list[str] | None) -> str | list[str] | None:
    """
    Returns the path to a visualastro mpl stylesheet for
    consistent plotting parameters.

    Matplotlib styles are also allowed (ex: 'classic').

    To add custom user defined mpl sheets, add files in:
    `VisualAstro/visualastro/stylelib/`
    Ensure the stylesheet follows the naming convention:
        `mystylesheet.mplstyle`

    If a style is unable to load due to missing fonts
    or other errors, `config.style_fallback` is used.

    Parameters
    ----------
    style : str | list[str] | None
        Name of the mpl stylesheet(s) without the extension.
        Stylesheets farther to the right take precedence.
        ie: `'astro'`, `'courier-new'`

    Returns
    -------
    style_path : str | list[str] | None
        Path to matplotlib stylesheet(s). Returns `None` if `style`
        is `None`.
    """
    if style is None:
        return None
    # if style is a default matplotlib stylesheet
    if style in mplstyle.available:
        return style

    if isinstance(style, list):
        if all(s in mplstyle.available for s in style):
            return style
        else:
            raise ValueError(
                'Invalid styles! To check available styles, '
                "run `plt.style.available`."
            )

    # if style is a visualastro stylesheet
    stylelib = files('visualastro').joinpath('stylelib')
    base_style = style.split('_')[0] if '_' in style else style
    style_path = stylelib.joinpath(f'{base_style}.mplstyle')

    # ensure that style works on computer, otherwise return default style
    try:
        with plt.style.context(str(style_path)):
            # pass if can load style successfully on computer
            pass
        return str(style_path)
    except Exception as e:
        warnings.warn(
            f"[visualastro] Could not apply style '{style}' ({e}). "
            f"Falling back to '{config.style_fallback}' style.",
            stacklevel=2
        )
        style = config.style_fallback
        base_style = style.split('_')[0] if '_' in style else style
        return str(stylelib.joinpath(f'{base_style}.mplstyle'))


@contextmanager
def style(
    name: str | _Unset = _UNSET,
    *additional_styles, rc: dict | None = None,
    **rc_kwargs
):
    """
    Context manager to temporarily apply a Matplotlib or VisualAstro style,
    with optional rcParams overrides.

    Parameters
    ----------
    name : str | _Unset, optional, default=_UNSET
        Matplotlib or VisualAstro style name. If `_UNSET`,
        uses `config.style`. Ex: 'astro' or 'latex'.
    rc : dict, optional
        Dictionary of rcParams overrides.
        Ex: {'font.size': 14}
    **rc_kwargs :
        Additional rcParams overrides supplied as keyword arguments.
        Use underscores in place of dots: font_size → font.size

    Examples
    --------
    >>> with style('latex', font_size=23, axes_labelsize=40):
    ...     plt.plot(x, y)

    >>> with style('paper', rc={'font.size': 14, 'lines.linewidth': 2}):
    ...     fig, ax = plt.subplots()

    >>> with style('astro', rc={'font.size': 12}, xtick_labelsize=10):
    ...     # rc dict and kwargs are merged (kwargs take precedence)
    ...     plt.plot(x, y)
    """
    name = _resolve_default(name, config.style)
    style_name = _get_stylepath(name)

    # update rcParams, with priority to kwargs
    rc_combined = {}
    if rc is not None:
        rc_combined.update(rc)
    if rc_kwargs:
        # replace '_' with '.' for rcParams
        rc_combined.update({
            k.replace('_', '.'): v for k, v in rc_kwargs.items()
        })
    styles = [style_name, rc_combined]
    modifiers = []
    for style in additional_styles:
        if isinstance(style, str):
            modifiers.append(style)
        if len(modifiers) > 0:
            styles += modifiers
    context = styles if rc_combined else style_name

    with plt.style.context(context):
        yield


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
                     color=config.wcs_grid_color,
                     alpha=config.grid_alpha,
                     ls=config.wcs_grid_linestyle
                 )
            else:
                ax.grid(
                    True,
                    color=config.grid_color,
                    alpha=config.grid_alpha,
                    ls=config.grid_linestyle
                )
