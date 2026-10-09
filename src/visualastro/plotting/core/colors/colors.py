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
from visualastro.core.sequences import _unwrap_if_single
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
    if isinstance(color, str):
        named = get_namedcolor(color, fmt)
        if named is not None:
            return named

    if not mcolors.is_color_like(color):
        raise ValueError(f"Invalid color: {color!r}")

    return _unwrap_if_single(as_color(color, fmt))


def get_colors(
    colors: ColorType | int | Sequence[ColorType] | _Unset = _UNSET,
    cmap: mcolors.Colormap | str | _Unset = _UNSET,
    cmap_range: tuple[float, float] = (0, 1),
    transform: Literal['lighten', 'darken', 'saturate', 'desaturate'] | None | _Unset = _UNSET,
    factor: float | _Unset = _UNSET,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex',
    cvd_type: Literal['deuteranomaly', 'protanomaly', 'tritanomaly'] | None = None,
    severity: int = 100
) -> list[str | RGBTuple | RGBATuple]:
    """
    Get colors from colorset name, colormap sampling, or explicit colors.

    Parameters
    ----------
    colors : ColorType | int | Sequence[ColorType] | _Unset, optional, default=_UNSET

        - `UNSET`: Use default colorset
        - `str`:  VisualAstro colorset name (with optional '_r' suffix) or single color
        - `ColorType`: Explicit color
        - `int`: Number of colors to sample from cmap
        - `Sequence[ColorType]`: Explicit list of colors
        - `random`: Random sequence of colors

        If `_UNSET`, uses `config.default_colorset`.
    cmap : Colormap | str | _Unset, optional, default=_UNSET
        Colormap for sampling when colors is int. If `_UNSET`,
        uses `config.cmap`.
    cmap_range : tuple[float, float], optional, default=(0, 1)
        The normalized range of the colormap. By default, is `(0,1)`,
        meaning the returned colormap has its entire range. Ignored
        if `cmap` is an `int`.
    transform : str | None | _Unset, optional, default=_UNSET
        Method to modify the color. Can be one of `'lighten'`, `'darken'`,
        `'saturate'`, or `'desaturate'`. If `None`, returns `color` unchanged.
        If `_UNSET`, uses `config.color_transform`.
    factor : float | _Unset, optional, default=_UNSET
        Modification strength.

        - If `transform='lighten'`: Blending ratio with white.

            - `factor=0`: Original color
            - `factor=1`: Pure white

        - If `transform='darken'`: Blending ratio with black.

            - `factor=0`: Original color
            - `factor=1`: Pure black

        - If `transform='saturate'`: Saturation level in hsl space.

            - `factor=1`: Maximum saturation for each given color
            - `factor=0`: Grayscale

        - If `transform='desaturate'`: Desaturation amount.

            - `factor=0`: Original color
            - `factor=1`: Grayscale

        If `_UNSET`, uses `config.color_transform_factor`.
    fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output format.
    cvd_type : {'deuteranomaly', 'protanomaly', 'tritanomaly'} | None, optional, default=None
        If not None, return the list of colors with a colorblind simulation applied.
    severity : int, optional, default=100
        Severity level (0-100). 100 = complete colorblindness.
        Only used if `cvd_type` is not None.

    Returns
    -------
    list[str] :
        If `fmt='hex'`.
    list[tuple[float, float, float]] :
        If `fmt='rgb'`.
    list[tuple[float, float, float, float]] :
        If `fmt='rgba'`.
    """
    transform = _resolve_default(transform, config.color_transform)
    factor = _resolve_default(factor, config.color_transform_factor)
    colorname = colors
    if colors is None or isinstance(colors, str) and colors in {'face', 'none'}:
         return [colors]

    colors = _get_colors(colors, cmap, fmt=fmt, cmap_range=cmap_range)
    colors = as_list(
        _transform_colors(
            colors,
            transform=transform,
            factor=factor,
            fmt=fmt
        )
    )
    if cvd_type is not None:
        colors = simulate_colorblindness(
            colors,
            cvd_type=cvd_type,
            severity=severity,
            fmt=fmt
        )
    if isinstance(colorname, str) and colorname.removesuffix('_r') in COLORSETS:
        modulo_idx = config.color_cycle_idx % len(colors)
        colors = colors[modulo_idx:] + colors[:modulo_idx]

    return colors


def _get_colors(
    colors: ColorType | int | Sequence[ColorType] | _Unset = _UNSET,
    cmap: mcolors.Colormap | str | _Unset = _UNSET,
    cmap_range: tuple[float, float] = (0, 1),
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
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
            return get_colorset(colors, fmt)

        named_color = get_namedcolor(colors)
        if named_color is not None:
            return as_list(named_color)

        return as_list(as_color(colors, fmt))

    if isinstance(colors, tuple):
        return as_list(as_color(colors, fmt))

    if isinstance(colors, (np.ndarray, list)):
        return [_get_colors(c, fmt=fmt)[0] for c in colors]

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
    colors: str,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex',
) -> list[str | RGBTuple | RGBATuple]:
    """
    Return a color sequence by name, converted to the requested format.

    Looks up `colors` in `COLORSET_ALIASES` and
    `matplotlib.color_sequences`. A trailing '_r' reverses the
    sequence.

    Parameters
    ----------
    colors : str
        Name of the colorset. Must be a key of `COLORSET_ALIASES` or
        `matplotlib.color_sequences`. Append '_r' to reverse the
        order.
    fmt : Literal['hex', 'rgb', 'rgba'], default='hex'
        Output color format, forwarded to `as_color`.

    Returns
    -------
    list[str | RGBTuple | RGBATuple]
        Colors in the requested format, in sequence order (reversed if
        `colors` ends with '_r' and is not itself a colorset name).
        Element type is `str` for `fmt='hex'`, `RGBTuple` for
        `fmt='rgb'`, `RGBATuple` for `fmt='rgba'`.

    Raises
    ------
    TypeError
        If `colors` is not a `str`.
    ValueError
        If `colors` (with or without the '_r' suffix) is not a known
        colorset.
    """
    if not isinstance(colors, str):
        raise TypeError(
            f"colors must be a str! got {type(colors).__name__}"
        )

    name, reverse = _find_colorset(colors)
    colorset = mpl.color_sequences[name]
    if reverse:
        colorset = colorset[::-1]

    return as_list(as_color(colorset, fmt))


def get_namedcolor(
    name: str,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex',
) -> str | RGBTuple | RGBATuple | None:
    """
    Resolve a named color to the requested format.

    Lookup order:

    1. `matplotlib.colors.get_named_colors_mapping()` (CSS4, base,
       tableau, prefixed xkcd).
    2. Bare xkcd name (`'xkcd:' + name`), only if `name` is not a
       CSS4 color.

    Parameters
    ----------
    name : str
        Color name.
    fmt : Literal['hex', 'rgb', 'rgba'], default='hex'
        Output color format, forwarded to `as_color`.

    Returns
    -------
    str | RGBTuple | RGBATuple | None
        Color in the requested format, or `None` if `name` is not a
        known named color.
    """
    named = mcolors.get_named_colors_mapping()

    if name in named:
        return _unwrap_if_single(as_color(named[name], fmt))

    xkcd_name = f'xkcd:{name}'
    if xkcd_name in mcolors.XKCD_COLORS and name not in mcolors.CSS4_COLORS:
        return _unwrap_if_single(as_color(xkcd_name, fmt))

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
