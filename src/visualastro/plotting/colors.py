"""
Author: Elko Gerville-Reache
Date Created: 2026-04-10
Date Modified: 2026-04-11
Description:
    Functions related to colors and colormaps in plotting.
Dependencies:
    - matplotlib
    - numpy
"""

from collections.abc import Sequence
import colorsys
from typing import Literal, TypeAlias
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType
import numpy as np
import tol_colors as tc

from visualastro.core.config import config, resolve_default, _Unset, _UNSET
from visualastro.core.numerical_utils import as_list, to_list, _unwrap_if_single
from visualastro.core.validation import _type_name


RGBTuple: TypeAlias = tuple[float, float, float]
RGBATuple: TypeAlias = tuple[float, float, float, float]


COLORSETS: dict[str, list[str]] = {
    'va': ['#483D8B', '#DC267F', '#648FFF', '#FFB000', '#26DCBA'],
    'ibm': ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000'],
    'ibm_contrast': ['#648FFF', '#DC267F', '#785EF0', '#26DCBA', '#FFB000', '#FE6100'],
    'astro': ['#9FB7FF', '#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000', '#CFE23C', '#26DCBA'],
    'MSG': ['#483D8B', '#DC267F', '#DBB0FF', '#26DCBA', '#7D7FF3'],
    'smplot': ['k', '#FF0000', '#0000FF', '#00FF00', '#00FFFF', '#FF00FF', '#FFFF00'],
    'bright': [mcolors.to_hex(c) for c in tc.bright],
    'vibrant': [mcolors.to_hex(c) for c in tc.vibrant],
    'muted': [mcolors.to_hex(c) for c in tc.muted],
    'light': [mcolors.to_hex(c) for c in tc.light],
    'dark': [mcolors.to_hex(c) for c in tc.dark],
    'medium_contrast': [mcolors.to_hex(c) for c in tc.medium_contrast][1:],
    'high_contrast': [mcolors.to_hex(c) for c in tc.high_contrast],
    'land_cover': [mcolors.to_hex(c) for c in tc.land_cover],
    'okabe_ito': ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
}
COLORNAMES = [key for key in COLORSETS.keys()]


def create_colormap(
    colors: list[ColorType],
    positions: list[float] | None = None,
    name: str = 'continous_cmap'
) -> mcolors.LinearSegmentedColormap:
    """
    Creates a colormap from colors with optional position control.

    Parameters
    ----------
    colors : list[ColorType]
        Color specifications (hex, named colors, RGB tuples, etc.).
    positions : list[float] | None, optional
        Positions in [0, 1] for each color. Must start with 0 and end with 1.
        If None, colors are evenly spaced.

    Returns
    -------
    LinearSegmentedColormap
    """
    rgb_list = [mcolors.to_rgb(color) for color in colors]

    if positions is None:
        positions = list(np.linspace(0, 1, len(rgb_list)))

    cdict = {channel: [[positions[i], rgb_list[i][idx], rgb_list[i][idx]]
                       for i in range(len(positions))]
             for idx, channel in enumerate(['red', 'green', 'blue'])}

    return mcolors.LinearSegmentedColormap(name, segmentdata=cdict, N=256) # type: ignore


def sample_cmap(
    N: int,
    cmap: str | mcolors.Colormap | _Unset = _UNSET,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> list[str | RGBTuple | RGBATuple]:
    """
    Sample N distinct colors from a given matplotlib colormap
    returned as a list of colors in a specified format.

    Parameters
    ----------
    N : int
        Number of colors to sample.
    cmap : str | Colormap | _Unset, optional, default=_UNSET
        Name of the matplotlib colormap or Colormap object. If
        ``_UNSET`` uses the default value in ``config.cmap``.
    fmt: {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.

    Returns
    -------
    list[str] :
        If ``fmt='hex'``.
    list[tuple[float, float, float]] :
        If ``fmt='rgb'``.
    list[tuple[float, float, float, float]] :
        If ``fmt='rgba'``.
    """
    cmap = resolve_default(cmap, config.cmap)
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, N))

    return [_convert_color(c, fmt) for c in colors]


def get_colors(
    colors: str | ColorType | int | Sequence[ColorType] | _Unset = _UNSET,
    cmap: mcolors.Colormap | str | _Unset = _UNSET,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
) -> list[str | RGBTuple | RGBATuple]:
    """
    Get colors from colorset name, colormap sampling, or explicit colors.

    Parameters
    ----------
    colors : ColorType | int | Sequence[ColorType] | _Unset, default=_UNSET
        - ``UNSET``: Use default colorset
        - ``str``:  visualastro colorset name (with optional '_r' suffix) or single color
        - ``ColorType``: Explicit color
        - ``int``: Number of colors to sample from cmap
        - ``Sequence[ColorType]``: Explicit list of colors

        If ``_UNSET``, uses the default value from ``config.default_colorset``.
    cmap : Colormap | str | _Unset, optional, default=_UNSET
        Colormap for sampling when colors is int. If ``_UNSET``,
        uses the default value from ``config.cmap``.
    fmt : {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output format.

    Returns
    -------
    list[str] :
        If ``fmt='hex'``.
    list[tuple[float, float, float]] :
        If ``fmt='rgb'``.
    list[tuple[float, float, float, float]] :
        If ``fmt='rgba'``.
    """

    cmap = resolve_default(cmap, config.cmap)

    if colors is _UNSET:
        colorset = COLORSETS.get(config.default_colorset, COLORSETS['va'])
        return as_list(as_color(colorset, fmt=fmt))

    if isinstance(colors, str):
        # if colorset in visualastro colorsets
        # return a reversed colorset if colorset
        # ends with '_r'
        if colors.removesuffix('_r') in COLORSETS:
            base_name = colors.removesuffix('_r')
            colorset = COLORSETS[base_name]
            # if '_r', reverse colorset
            if colors.endswith('_r'):
                colorset = colorset[::-1]
            return as_list(as_color(colorset, fmt))

        else:
            return as_list(as_color(colors, fmt))

    if isinstance(colors, (np.ndarray, Sequence)):
        return as_list(as_color(colors, fmt))

    # if user passes an integer N, sample a cmap for N colors
    if isinstance(colors, int):
        return as_list(sample_cmap(colors, cmap=cmap, fmt=fmt))

    raise TypeError(
        'colors must be None, a str colorset name, a str color, '
        f'a list of colors, or an integer! got {_type_name(colors)}'
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
    Convert a matplotlib ``ColorType`` or a ``list[ColorType]`` into
    one of the following formats: ``'hex`'', ``'rgb'``, or ``'rgba'``.

    Parameters
    ----------
    c : ColorType | List[ColorType]
        Matplotlib color(s). Can be named colors, rgb/rgba, hex, etc...
    fmt: {'hex', 'rgb', 'rgba'}, optional, default='hex'
        Output color format.

    Returns
    -------
    str | list[str] :
        If ``fmt='hex'``.
    tuple[float, float, float] | list[tuple[float, float, float]] :
        If ``fmt='rgb'``.
    tuple[float, float, float, float] | list[tuple[float, float, float, float]] :
        If ``fmt='rgba'``.
    """
    color_list = as_list(c)
    color_list = [_convert_color(c) for c in color_list] # type: ignore

    return _unwrap_if_single(color_list)


def _convert_color(
    c: ColorType,
    fmt: Literal['hex', 'rgb', 'rgba'] = 'hex'
)-> str | tuple[float, float, float] | tuple[float, float, float, float]:
    """
    Convert a matplotlib ``ColorType`` into one of the following
    formats: ``'hex`'', ``'rgb'``, or ``'rgba'``.
    """
    return getattr(mcolors, f'to_{fmt}')(c)


def _lighten_color(color: ColorType, mix: float = 0.5) -> 'str':
    """
    Lightens the given matplotlib color by mixing it with white.

    Parameters
    ----------
    color : ColorType
        Matplotlib named color, hex color, html color or rgb tuple.
    mix : float or int
        Ratio of color to white in mix.
        ``mix=0`` returns the original color,
        ``mix=1`` returns pure white.
    """

    # convert to rgb
    rgb = np.array(mcolors.to_rgb(color))
    white = np.array([1, 1, 1])
    # mix color with white
    mixed = (1 - mix) * rgb + mix * white

    return mcolors.to_hex(tuple(mixed))


def _desaturate_color(color: ColorType, factor: float = 0.5) -> str:
    """
    Desaturate a color by moving it toward gray.

    Parameters
    ----------
    color : ColorType
        Matplotlib named color, hex color, html color or rgb tuple.
    factor : float
        Desaturation amount. 0=original, 1=full gray.

    Returns
    -------
    str
        Hex color string.
    """
    rgb = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    s_new = s * (1 - factor)
    rgb_new = colorsys.hls_to_rgb(h, l, s_new)

    return mcolors.to_hex(rgb_new)
