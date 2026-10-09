"""
Author: Elko Gerville-Reache
Date Created: 2026-04-10
Date Modified: 2026-10-06
Description:
    Color utility functions.
"""

from collections.abc import Sequence
from typing import Any, Literal, TypeGuard

import matplotlib as mpl
from matplotlib import colors as mcolors
from matplotlib.colors import is_color_like
from matplotlib.typing import ColorType
import numpy as np

from visualastro.core.config import config, _UNSET, _resolve_default
from visualastro.core.data import as_list
from visualastro.core.sequences import _cycle, _unwrap_if_single
from visualastro.plotting.core.colors.definitions import (
    RGBATuple, RGBTuple, COLORSET_ALIASES
)


def as_color(
    c: ColorType | Sequence[ColorType],
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> (
    str
    | RGBTuple
    | RGBATuple
    | list[str | RGBTuple | RGBATuple]
):
    """
    Convert a Matplotlib `ColorType` or a `list[ColorType]` into
    one of the following formats: `'hex'`, `'rgb'`, or `'rgba'`.

    Parameters
    ----------
    c : ColorType | List[ColorType]
        Matplotlib color(s). Can be named colors, rgb/rgba, hex, etc...
    fmt: {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.

    Returns
    -------
    str | list[str] :
        If `fmt='hex'`.
    tuple[float, float, float] | list[tuple[float, float, float]] :
        If `fmt='rgb'`.
    tuple[float, float, float, float] | list[tuple[float, float, float, float]] :
        If `fmt='rgba'`.
    """
    color_list = as_list(c)
    color_list = [_convert_color(c, fmt=fmt) for c in color_list]

    return _unwrap_if_single(color_list)


def _convert_color(
    c: ColorType,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
)-> str | tuple[float, float, float] | tuple[float, float, float, float]:
    """
    Convert a Matplotlib `ColorType` into one of the following
    formats: `'hex`'', `'rgb'`, or `'rgba'`.
    """
    return getattr(mcolors, f'to_{fmt}')(c)


def _find_colorset(name: str) -> tuple[str, bool]:
    """
    Resolve a colorset name to a `matplotlib.color_sequences` key.

    Exact matches (after alias lookup) take precedence over
    '_r' suffix stripping.

    Parameters
    ----------
    name : str
        Colorset name, optionally ending in '_r'.

    Returns
    -------
    key : str
        Key of `matplotlib.color_sequences`.
    reverse : bool
        `True` if the sequence must be reversed, i.e. `name` ended in
        '_r' and was not itself a colorset name.

    Raises
    ------
    ValueError
        If `name` (with or without the '_r' suffix) is not a known
        colorset.
    """
    def _resolve(name: str) -> str | None:
        name = COLORSET_ALIASES.get(name, name)
        return name if name in mpl.color_sequences else None

    resolved = _resolve(name)
    reverse = False

    if resolved is None and name.endswith('_r'):
        resolved = _resolve(name.removesuffix('_r'))
        reverse = True

    if resolved is None:
        raise ValueError(f"Unknown colorset: {name!r}")

    return resolved, reverse


def _is_colorset(name: str) -> bool:
    """Return `True` if `name` is a known colorset (optionally ending in '_r')."""
    try:
        _find_colorset(name)
    except ValueError:
        return False
    return True


def _is_color_like(color: Any) -> TypeGuard[ColorType]:
    """Check if an input is a valid Matplotlib color."""
    return mcolors.is_color_like(color)


def _get_single_color(color: Any) -> str | RGBTuple | RGBATuple:
    """
    Reduce a color or a sequence of colors to a single color.

    Parameters
    ----------
    color : str | tuple | list | np.ndarray
        A single Matplotlib color, or a non-empty sequence of colors.

    Returns
    -------
    color : str | RGBTuple | RGBATuple
        A single color.

    Raises
    ------
    ValueError
        If `color` is an empty sequence.
    TypeError
        If `color` is neither a color nor a sequence of colors.
    """
    if is_color_like(color):
        return color

    if isinstance(color, (Sequence, np.ndarray)) and not isinstance(color, str):
        if len(color) == 0:
            raise ValueError("Cannot select a color from an empty sequence.")
        selected = _cycle(color, 0)
        if not is_color_like(selected):
            raise TypeError(f"Invalid color: {selected!r}")
        return selected

    raise TypeError(f"Expected a color or a sequence of colors, got {color!r}")


def _is_cycled_colorset(colors) -> bool:
    """Whether `colors` resolves to a named colorset."""
    if colors is _UNSET:
        if _is_colorset(_resolve_default(colors, config.default_colorset)):
            return True
        return False
    return isinstance(colors, str) and _is_colorset(colors)
