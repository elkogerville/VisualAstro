"""
Author: Elko Gerville-Reache
Date Created: 2026-04-10
Date Modified: 2026-10-06
Description:
    Color related functions.
"""

from collections.abc import Sequence

from typing import Literal
import matplotlib as mpl
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType
import numpy as np

from visualastro.core.config import (
    config, _resolve_default, _Unset, _UNSET
)
from visualastro.core.data import as_list
from visualastro.core.sequences import _roll
from visualastro.plotting.core.colormaps import get_cmap
from visualastro.plotting.core.colors.definitions import (
    RGBATuple,
    RGBTuple,
    COLORSETS,
)
from visualastro.plotting.core.colors.transforms import (
    simulate_colorblindness,
    _transform_colors,
)
from visualastro.plotting.core.colors.utils import (
    as_color,
    _convert_color,
    _get_single_color,
    _find_colorset,
    _is_color_like,
    _is_colorset,
    _is_cycled_colorset,
)


def get_color(
    color: ColorType,
    *,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex',
    transform: Literal['lighten', 'darken', 'saturate', 'desaturate'] | None | _Unset = _UNSET,
    factor: float | _Unset = _UNSET,
    cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] | None = None,
    severity: int = 100,
) -> str | RGBTuple | RGBATuple:
    """
    Convert a single Matplotlib color to the requested format.

    Named colors (including bare xkcd names) are resolved first. All other
    inputs are passed to `as_color`. Colorset names are not supported, see
    `get_colorset`. Modifiers are applied once.

    Parameters
    ----------
    color : ColorType
        A single color: named, hex, grayscale string, or RGB(A) tuple.
    fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.
    transform : {'lighten', 'darken', 'saturate', 'desaturate'} | None | _Unset, optional, default=_UNSET
        Method to modify the color. If `None`, the color is unchanged.
        If `_UNSET`, uses `config.color_transform`.
    factor : float | _Unset, optional, default=_UNSET
        Modification strength, see `get_colors`. If `_UNSET`, uses
        `config.color_transform_factor`.
    cvd_type : {'deuteranomaly', 'protanomaly', 'tritanomaly'} | None, optional, default=None
        If not `None`, apply a color vision deficiency
        simulation after `transform`.
    severity : int, optional, default=100
        CVD severity in [0, 100]. 100 = complete colorblindness.
        Ignored if `cvd_type` is `None`.

    Returns
    -------
    str | RGBTuple | RGBATuple
        `str` for `fmt='hex'`, `RGBTuple` for `fmt='rgb'`,
        `RGBATuple` for `fmt='rgba'`.

    Raises
    ------
    ValueError
        If `color` is not a valid Matplotlib color.
    """
    if isinstance(color, str) and (named := _get_namedcolor(color)) is not None:
        color = named

    if not _is_color_like(color):
        raise ValueError(f"Invalid color: {color!r}")

    return _get_single_color(_apply_color_modifiers(
        as_color(color, fmt),
        fmt=fmt,
        transform=transform,
        factor=factor,
        cvd_type=cvd_type,
        severity=severity,
    ))


def get_colors(
    colors: ColorType | int | Sequence[ColorType] | _Unset = _UNSET,
    *,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex',
    cmap: mcolors.Colormap | str | _Unset = _UNSET,
    cmap_range: tuple[float, float] = (0, 1),
    transform: Literal['lighten', 'darken', 'saturate', 'desaturate'] | None | _Unset = _UNSET,
    factor: float | _Unset = _UNSET,
    cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] | None = None,
    severity: int = 100,
) -> list[str | RGBTuple | RGBATuple]:
    """
    Get colors from a colorset name, colormap sampling, or explicit colors.

    Modifiers are applied once, in the order `transform` -> CVD simulation.
    Named colorsets are rotated left by `config.color_cycle_idx`.

    Parameters
    ----------
    colors : ColorType | int | Sequence[ColorType] | _Unset, optional, default=_UNSET
        - `_UNSET`: default colorset (`config.default_colorset`).
        - `str`: VisualAstro colorset name (optional '_r' suffix to reverse),
          named color, or `'random'`.
        - `ColorType`: a single explicit color.
        - `int`: number of colors to sample from `cmap`.
        - `Sequence[ColorType]`: explicit list of colors.

    fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.
    cmap : Colormap | str | _Unset, optional, default=_UNSET
        Colormap sampled when `colors` is an `int`. If `_UNSET`, uses
        `config.sample_cmap`.
    cmap_range : tuple[float, float], optional, default=(0, 1)
        Normalized range of `cmap` to sample. Used only if `colors` is an
        `int`.
    transform : {'lighten', 'darken', 'saturate', 'desaturate'} | None | _Unset, optional, default=_UNSET
        Method to modify the colors. If `None`, colors are unchanged.
        If `_UNSET`, uses `config.color_transform`.
    factor : float | _Unset, optional, default=_UNSET
        `transform` Modification strength. If `_UNSET`, uses
        `config.color_transform_factor`.

        - `'lighten'`: blending ratio with white (0 = original, 1 = white).
        - `'darken'`: blending ratio with black (0 = original, 1 = black).
        - `'saturate'`: saturation level in HSL space (0 = grayscale,
          1 = maximum saturation).
        - `'desaturate'`: desaturation amount (0 = original, 1 = grayscale).

    cvd_type : {'deuteranomaly', 'protanomaly', 'tritanomaly'} | None, optional, default=None
        If not None, apply a color vision deficiency simulation after `transform`.
    severity : int, optional, default=100
        CVD severity in [0, 100]. 100 = complete colorblindness. Ignored
        if `cvd_type` is None.

    Returns
    -------
    list[str] | list[RGBTuple] | list[RGBATuple]
        `list[str]` for `fmt='hex'`, `list[RGBTuple]` for `fmt='rgb'`,
        `list[RGBATuple]` for `fmt='rgba'`.

    Raises
    ------
    TypeError
        If `colors` is not a supported type.
    ValueError
        If `colors` contains an invalid color.
    """
    colorname = colors
    if colors is None or isinstance(colors, str) and colors in {'face', 'none'}:
         return [colors]

    colors = _get_colors(colors, fmt=fmt, cmap=cmap, cmap_range=cmap_range)
    colors = _apply_color_modifiers(
        colors=colors,
        fmt=fmt,
        transform=transform,
        factor=factor,
        cvd_type=cvd_type,
        severity=severity,
    )

    if _is_cycled_colorset(colorname):
        colors = _roll(colors, -config.color_cycle_idx)

    return colors


def _get_colors(
    colors: ColorType | int | Sequence[ColorType] | _Unset = _UNSET,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex',
    cmap: mcolors.Colormap | str | _Unset = _UNSET,
    cmap_range: tuple[float, float] = (0, 1),
) -> list[str | RGBTuple | RGBATuple]:
    """Helper function for `get_colors`"""
    if colors is _UNSET:
        colorset = COLORSETS.get(
            config.default_colorset, COLORSETS['visualastro']
        )
        return as_list(as_color(colorset, fmt=fmt))

    if isinstance(colors, str):
        if colors == 'random':
            n = np.random.randint(1, config.random_colors_max_N)
            return random_colors(int(n), fmt=fmt)

        if _is_colorset(colors):
            return _get_colorset(colors, fmt)

        return as_list(as_color(_get_namedcolor(colors) or colors, fmt))

    if _is_color_like(colors):
        return as_list(as_color(colors, fmt))

    if isinstance(colors, (np.ndarray, list)):
        return [
            _get_colors(c, fmt=fmt, cmap=cmap, cmap_range=cmap_range)[0] for c in colors
        ]

    # if user passes an integer N, sample a cmap for N colors
    if isinstance(colors, int):
        cmap = get_cmap(
            _resolve_default(cmap, config.sample_cmap), cmap_range=cmap_range
        )
        return as_list(sample_cmap(int(colors), cmap=cmap, fmt=fmt))

    raise TypeError(
        'colors must be unset, a str colorset name, a str color, '
        'a tuple, a list of colors, or an integer! '
        f"got: {type(colors).__name__}"
    )


def get_colorset(
    colorset_name: str,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex',
    transform: Literal['lighten', 'darken', 'saturate', 'desaturate'] | None | _Unset = _UNSET,
    factor: float | _Unset = _UNSET,
    cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] | None = None,
    severity: int = 100,
) -> list[str | RGBTuple | RGBATuple]:
    """
    Return a color sequence by name, converted to the requested format.

    Looks up `colors` via `_find_colorset` (VisualAstro aliases and
    `matplotlib.color_sequences`). A trailing '_r' reverses the sequence.

    Parameters
    ----------
    colorset_name : str
        Colorset name. Append '_r' to reverse the order.
    fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.
    transform : {'lighten', 'darken', 'saturate', 'desaturate'} | None | _Unset, optional, default=_UNSET
        Method to modify the colors. If `None`, colors are unchanged.
        If `_UNSET`, uses `config.color_transform`.
    factor : float | _Unset, optional, default=_UNSET
        Modification strength, see `get_colors`. If `_UNSET`, uses
        `config.color_transform_factor`.
    cvd_type : {'deuteranomaly', 'protanomaly', 'tritanomaly'} | None, optional, default=None
        If not None, apply a color vision deficiency simulation after
        `transform`.
    severity : int, optional, default=100
        CVD severity in [0, 100]. Ignored if `cvd_type` is None.

    Returns
    -------
    list[str] | list[RGBTuple] | list[RGBATuple]
        Colors in sequence order (reversed if '_r' suffix was used and the
        full name is not itself a colorset).

    Raises
    ------
    TypeError
        If `colors` is not a `str`.
    ValueError
        If `colors` (with or without '_r') is not a known colorset.
    """
    colorset = _get_colorset(colorset_name)

    return _apply_color_modifiers(
        colors=colorset,
        fmt=fmt,
        transform=transform,
        factor=factor,
        cvd_type=cvd_type,
        severity=severity
    )


def _get_colorset(
    colorset_name: str,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex',
) -> list[str | RGBTuple | RGBATuple]:
    """Retrieve a colorset from a colorset name."""
    if not isinstance(colorset_name, str):
        raise TypeError(
            f"colors must be a str! got {type(colorset_name).__name__}"
        )
    name, reverse = _find_colorset(colorset_name)
    colorset = mpl.color_sequences[name]
    if reverse:
        colorset = colorset[::-1]

    return as_list(as_color(colorset, fmt))


def get_namedcolor(
    name: str,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex',
    transform: Literal['lighten', 'darken', 'saturate', 'desaturate'] | None | _Unset = _UNSET,
    factor: float | _Unset = _UNSET,
    cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] | None = None,
    severity: int = 100,
) -> str | RGBTuple | RGBATuple | None:
    """
    Resolve a named color to the requested format.

    Lookup order:

    1. `matplotlib.colors.get_named_colors_mapping()` (CSS4, base,
       tableau, prefixed xkcd).
    2. Bare xkcd name (`'xkcd:' + name`).

    Parameters
    ----------
    name : str
        Color name.
    fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.
    transform : {'lighten', 'darken', 'saturate', 'desaturate'} | None | _Unset, optional, default=_UNSET
        Method to modify the color. If `None`, the color is unchanged.
        If `_UNSET`, uses `config.color_transform`.
    factor : float | _Unset, optional, default=_UNSET
        Modification strength, see `get_colors`. If `_UNSET`, uses
        `config.color_transform_factor`.
    cvd_type : {'deuteranomaly', 'protanomaly', 'tritanomaly'} | None, optional, default=None
        If not None, apply a color vision deficiency simulation after
        `transform`.
    severity : int, optional, default=100
        CVD severity in [0, 100]. Ignored if `cvd_type` is None.

    Returns
    -------
    str | RGBTuple | RGBATuple | None
        Color in the requested format, or `None` if `name` is not a known
        named color.
    """
    color = _get_namedcolor(name)
    if color is not None:
        return _get_single_color(_apply_color_modifiers(
            colors=color,
            fmt=fmt,
            transform=transform,
            factor=factor,
            cvd_type=cvd_type,
            severity=severity,
        ))
    return None


def _get_namedcolor(name: str) -> str | None:
    """Retrieve a named color."""
    named = mcolors.get_named_colors_mapping()
    if name in named:
        return name
    xkcd_name = f'xkcd:{name}'
    if xkcd_name in mcolors.XKCD_COLORS and name not in mcolors.CSS4_COLORS:
        return xkcd_name
    return None


def sample_cmap(
    N: int,
    cmap: str | mcolors.Colormap | _Unset = _UNSET,
    cmap_range: tuple[float, float] = (0, 1),
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex',
) -> list[str | RGBTuple | RGBATuple]:
    """
    Sample N distinct colors from a given Matplotlib colormap
    returned as a list of colors in a specified format.

    Parameters
    ----------
    N : int
        Number of colors to sample.
    cmap : str | Colormap | _Unset, optional, default=_UNSET
        Name of the Matplotlib colormap or `Colormap` object. If
        `_UNSET` uses `config.cmap`.
    cmap_range : tuple[float, float], optional, default=(0,1)
        The normalized value range in the colormap from which colors
        should be taken. By default, the entire colormap is used.
    fmt: {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.

    Returns
    -------
    list[str] :
        If `fmt='hex'`.
    list[tuple[float, float, float]] :
        If `fmt='rgb'`.
    list[tuple[float, float, float, float]] :
        If `fmt='rgba'`.
    """
    cmap = _resolve_default(cmap, config.cmap)
    colors = plt.get_cmap(cmap)(np.linspace(cmap_range[0], cmap_range[1], N))

    return [_convert_color(c, fmt) for c in colors]


def random_colors(
    N: int,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> list[str | RGBTuple | RGBATuple]:
    """
    Generate N random colors

    Parameters
    ----------
    N : int
        Number of colors to generate

    Returns
    -------
    colors : list[tuple[float, float, float]]
    """
    random_colors = np.random.rand(N, 3)
    return as_color(
        [tuple([float(c[0]), float(c[1]), float(c[2])]) for c in random_colors],  # type: ignore
        fmt=fmt
    )
