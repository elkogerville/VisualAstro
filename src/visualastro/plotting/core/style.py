"""
Author: Elko Gerville-Reache
Date Created: 2025-07-13
Date Modified: 2026-07-03
Description:
    Functions related to setting the plotting style.
"""

from contextlib import contextmanager, nullcontext
from importlib.resources import files
import warnings

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

from visualastro.core.config import (
    config,
    _Unset, _UNSET,
    _resolve_default
)


def _style_context(style: str | None = None, *, path: str | None = None):
    """
    Helper function to facilitate
    """
    stylepath = _get_stylepath(style) if path is None else path
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
